"""Augmentations matched to this data (GPU, torch). All parameters are drawn once per clip
(temporally consistent) with small per-frame jitter.

- mirror-swap: flip crops + swap L/R hand streams (pose x is mirrored at train time)
- chroma-key background replacement for TTRS (blue studio → real backgrounds)
- crop jitter (detector error), photometric (webcam colour/exposure),
  down-up rescale (small, low-res hands like 640×360 phone/webcam video), blur, sensor noise
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from .utils import ART


def clip_rng(clip_id: str, version: str) -> np.random.RandomState:
    h = int(hashlib.md5(f"{clip_id}|{version}".encode()).hexdigest()[:8], 16)
    return np.random.RandomState(h)


def flip_swap(x: torch.Tensor) -> torch.Tensor:
    """x [T,3,S,S,C] (hand_l, hand_r, face) → mirrored with hands swapped."""
    x = x.flip(-2)
    return x[:, [1, 0, 2]]


class BackgroundPool:
    def __init__(self, path=None, device="cuda"):
        path = path or ART / "backgrounds" / "bg_pool.npy"
        self.imgs = np.load(path, mmap_mode="r") if path.exists() else None
        self.device = device

    def sample(self, rng) -> torch.Tensor:
        if self.imgs is None or rng.rand() < 0.2:
            return synthetic_bg(rng, self.device)
        img = torch.from_numpy(np.ascontiguousarray(self.imgs[rng.randint(len(self.imgs))])).to(self.device)
        if rng.rand() < 0.5:
            img = img.flip(1)
        return img  # [H,W,3] uint8 BGR


def synthetic_bg(rng, device, h=360, w=640) -> torch.Tensor:
    base = torch.tensor(rng.randint(0, 256, 3), dtype=torch.float32, device=device)
    gy = torch.linspace(0, 1, h, device=device)[:, None, None]
    gx = torch.linspace(0, 1, w, device=device)[None, :, None]
    grad = torch.tensor(rng.uniform(-80, 80, 3), dtype=torch.float32, device=device)
    img = base + grad * (gx * rng.rand() + gy * rng.rand())
    lowres = torch.rand(1, 3, 9, 16, device=device) * 60 - 30
    img = img + F.interpolate(lowres, (h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    return img.clamp(0, 255).to(torch.uint8)


def chroma_replace(x: torch.Tensor, mask: torch.Tensor, geom: np.ndarray, frame_hw, bg: torch.Tensor) -> torch.Tensor:
    """x [T,3,S,S,3] uint8, mask [T,3,S,S] uint8 (255 = key), geom [T,3,3] (cx,cy,side) in frame px."""
    T, K, S = x.shape[:3]
    H, W = frame_hw
    bgf = bg.permute(2, 0, 1)[None].float()
    bgf = F.interpolate(bgf, (H, W), mode="bilinear", align_corners=False)
    g = torch.as_tensor(geom.reshape(-1, 3), dtype=torch.float32, device=x.device)
    half = g[:, 2:3] / 2
    boxes = torch.cat([torch.zeros_like(half), g[:, :2] - half, g[:, :2] + half], 1)
    patches = roi_align(bgf, boxes, (S, S), spatial_scale=1.0, sampling_ratio=2, aligned=True)  # [T*K,3,S,S]
    patches = patches.permute(0, 2, 3, 1).reshape(T, K, S, S, 3)
    m = mask.float().reshape(T * K, 1, S, S) / 255.0
    m = F.avg_pool2d(m, 3, 1, 1).reshape(T, K, S, S, 1)  # soft key edge
    return (x.float() * (1 - m) + patches * m).clamp(0, 255)


def photometric(x: torch.Tensor, rng, crop_size: int) -> torch.Tensor:
    """x [T,3,S,S,3] float 0..255 → augmented float 0..255 (same params for the whole clip)."""
    T, K, S = x.shape[:3]
    dev = x.device
    y = x.permute(0, 1, 4, 2, 3).reshape(T * K, 3, S, S)
    # crop jitter: clip-level scale/shift + per-frame wobble
    sc = rng.uniform(0.87, 1.15)
    tx, ty = rng.uniform(-0.10, 0.10, 2)
    wob = torch.randn(T, 1, 2, device=dev) * 0.02
    theta = torch.zeros(T, 2, 3, device=dev)
    theta[:, 0, 0] = sc
    theta[:, 1, 1] = sc
    theta[:, :, 2] = torch.tensor([tx, ty], device=dev) + wob[:, 0]
    theta = theta.repeat_interleave(K, 0)
    grid = F.affine_grid(theta, y.shape, align_corners=False)
    y = F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # colour: per-channel gain (white balance), brightness, contrast, saturation
    gain = torch.tensor(rng.uniform(0.85, 1.15, 3), dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    y = y * gain * rng.uniform(0.65, 1.35)
    mean = y.mean(dim=(1, 2, 3), keepdim=True)
    y = (y - mean) * rng.uniform(0.7, 1.3) + mean
    gray = y.mean(1, keepdim=True)
    y = (y - gray) * rng.uniform(0.6, 1.4) + gray
    # low resolution: down-up to r px (hands in 640x360 video are ~30–60 px)
    if rng.rand() < 0.75:
        r = int(rng.uniform(24, 72))
        y = F.interpolate(y, (r, r), mode="area")
        y = F.interpolate(y, (S, S), mode="bilinear", align_corners=False)
    if rng.rand() < 0.4:
        k = 5
        sig = rng.uniform(0.5, 1.5)
        ax = torch.arange(k, device=dev) - k // 2
        g1 = torch.exp(-(ax.float() ** 2) / (2 * sig * sig))
        g1 = g1 / g1.sum()
        ker = (g1[:, None] * g1[None, :]).expand(3, 1, k, k)
        y = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="replicate"), ker, groups=3)
    if rng.rand() < 0.5:
        y = y + torch.randn_like(y) * rng.uniform(2, 8)
    return y.clamp(0, 255).reshape(T, K, 3, S, S).permute(0, 1, 3, 4, 2)
