"""Experimental training-only hand reconstruction. No production inference changes.

observed is mandatory: confidence alone cannot distinguish detected and imputed joints.
Inputs and observed masks must already share all temporal/geometric transforms.
"""
import torch
from torch import nn
from torch.nn import functional as F


class HandReconstruction(nn.Module):
    def __init__(self, features=768):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, 84))

    def forward(self, frames):
        return self.head(frames).reshape(*frames.shape[:2], 42, 2)


def mask_hands(parts, valid, observed, probability=0.15, generator=None):
    if not 0 < probability <= 1:
        raise ValueError('probability must be in (0, 1]')
    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError('valid must be bool [B,T]')
    inputs = {k: v.clone() for k, v in parts.items()}
    targets, selected = [], []
    for name in ('left', 'right'):
        x = parts[name]
        obs = observed[name]
        if x.shape != (*valid.shape, 21, 3) or obs.shape != x.shape[:-1] or obs.dtype != torch.bool:
            raise ValueError('hand/observed shape mismatch or non-boolean observed mask')
        eligible = obs & obs[..., :1] & valid[..., None] & (x[..., 2] > 0.3) & torch.isfinite(x).all(-1)
        # Mask complete hand frames; roots are hidden along with finger coordinates.
        draw = torch.rand(valid.shape, device=x.device, generator=generator) < probability
        chosen = eligible & draw[..., None]
        inputs[name][draw & valid] = 0
        targets.append(torch.nan_to_num(x[..., :2].detach()))
        selected.append(chosen)
    return inputs, torch.cat(targets, 2), torch.cat(selected, 2)


def reconstruction_loss(prediction, target, selected):
    if prediction.shape != target.shape or selected.shape != target.shape[:-1]:
        raise ValueError('unaligned reconstruction target; do not silently resample labels')
    if not selected.any():
        return prediction.sum() * 0
    return F.smooth_l1_loss(prediction[selected].float(), target[selected].float())
