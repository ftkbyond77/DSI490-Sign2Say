"""v4 L2: pretrained sign encoder — Uni-Sign pose branch (ICLR 2025, arXiv:2501.15187).

Part-wise ST-GCN over body(9) · left hand(21) · right hand(21) · face(18) keypoints from RTMW, pretrained on
CSL-News (1,985 h) and fine-tuned on WLASL-2000 (isolated ASL signs, ~100 signers). Optionally the mT5-base
encoder that sat on top of it in Uni-Sign is kept as a temporal context model.

Keypoint normalisation reproduces `datasets.load_part_kp` of the official code exactly."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "unisign"))
from stgcn_layers import Graph, get_stgcn_chain  # noqa: E402

HF_SNAP = Path.home() / ".cache/huggingface/hub/models--ZechengLi19--Uni-Sign/snapshots/eab251b7fe7e8521afc0e67be98add670ea40a0d"
CKPTS = {"wlasl": HF_SNAP / "wlasl_pose_only_islr.pth", "csl_stage1": HF_SNAP / "csl_stage1_weight.pth"}
MODES = ("body", "left", "right", "face_all")
BODY_IDX = [0] + list(range(3, 11))
FACE_IDX = list(range(23, 23 + 17))[::2] + list(range(83, 83 + 8)) + [53]


# ----------------------------------------------------------------------------- keypoint preparation
def _crop_scale(motion, thr):
    result = copy.deepcopy(motion)
    valid = motion[motion[..., 2] > thr][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape), 0
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0
    xs, ys = (xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale


def prepare_parts(kp: np.ndarray, sc: np.ndarray, thr=0.3, scale_ref=None):
    """kp [T,133,2] in [0,1] (x/W,y/H), sc [T,133] → {part: float32 [T,V,3]} (official load_part_kp).
    scale_ref: (kp, sc) of the whole session so windows share one body scale (continuous video)."""
    kp = kp.astype(np.float64); sc = sc.astype(np.float64)
    out = {}
    body = np.concatenate([kp[:, BODY_IDX], sc[:, BODY_IDX, None]], -1)
    if scale_ref is not None:
        rk, rs = scale_ref
        ref = np.concatenate([rk[:, BODY_IDX].astype(np.float64), rs[:, BODY_IDX, None].astype(np.float64)], -1)
        _, scale = _crop_scale(ref, thr)
        valid = ref[ref[..., 2] > thr][:, :2]
        if scale:
            xs = (valid[:, 0].min() + valid[:, 0].max() - scale) / 2
            ys = (valid[:, 1].min() + valid[:, 1].max() - scale) / 2
            r = body.copy()
            r[..., :2] = ((body[..., :2] - [xs, ys]) / scale - 0.5) * 2
            r = np.clip(r, -1, 1); r[r[..., 2] <= thr] = 0
            out["body"] = r
    if "body" not in out:
        out["body"], scale = _crop_scale(body, thr)
    for part, sl, root in (("left", slice(91, 112), 0), ("right", slice(112, 133), 0), ("face_all", FACE_IDX, -1)):
        k = kp[:, sl]
        k = k - k[:, [root] if isinstance(root, int) and root >= 0 else [-1]]
        r = np.concatenate([k, sc[:, sl, None]], -1)
        if scale == 0:
            r = np.zeros(r.shape)
        else:
            r[..., :2] = r[..., :2] / scale
            r = np.clip(r, -1, 1)
            r[r[..., 2] <= thr] = 0
        out[part] = r
    return {p: out[p].astype(np.float32) for p in MODES}


def upsample(kp: np.ndarray, sc: np.ndarray, factor: int = 2):
    """Linear temporal interpolation (12.5 fps corpus → 25 fps, the frame rate Uni-Sign was trained on)."""
    T = len(kp)
    if factor == 1 or T < 2:
        return kp, sc
    t_new = np.linspace(0, T - 1, (T - 1) * factor + 1)
    i0 = np.floor(t_new).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (t_new - i0)[:, None, None]
    kpu = kp[i0] * (1 - w) + kp[i1] * w
    scu = np.minimum(sc[i0], sc[i1])
    return kpu.astype(np.float32), scu.astype(np.float32)


def collate_parts(list_of_parts, max_len=None):
    L = max(len(p["body"]) for p in list_of_parts)
    if max_len:
        L = min(L, max_len)
    B = len(list_of_parts)
    out = {m: torch.zeros(B, L, list_of_parts[0][m].shape[1], 3) for m in MODES}
    mask = torch.zeros(B, L, dtype=torch.bool)
    for b, p in enumerate(list_of_parts):
        n = min(len(p["body"]), L)
        for m in MODES:
            x = torch.from_numpy(p[m][:n])
            out[m][b, :n] = x
            if n < L:  # official collate repeats the last frame
                out[m][b, n:] = x[-1]
        mask[b, :n] = True
    return out, mask


# ----------------------------------------------------------------------------- model
class UniSignPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph, A = {}, []
        self.proj_linear = nn.ModuleDict()
        for m in MODES:
            self.graph[m] = Graph(layout=m, strategy="distance", max_hop=1)
            A.append(torch.tensor(self.graph[m].A, dtype=torch.float32))
            self.proj_linear[m] = nn.Linear(3, 64)
        self.gcn_modules, self.fusion_gcn_modules = nn.ModuleDict(), nn.ModuleDict()
        k = A[0].size(0)
        for i, m in enumerate(MODES):
            self.gcn_modules[m], fd = get_stgcn_chain(64, "spatial", (1, k), A[i].clone(), True)
            self.fusion_gcn_modules[m], _ = get_stgcn_chain(fd, "temporal", (5, k), A[i].clone(), True)
        self.gcn_modules["left"] = self.gcn_modules["right"]
        self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
        self.proj_linear["left"] = self.proj_linear["right"]
        self.part_para = nn.Parameter(torch.zeros(256 * 4))
        self.pose_proj = nn.Linear(256 * 4, 768)

    def forward(self, parts: dict, return_parts=False):
        feats, body = [], None
        for m in MODES:
            x = self.proj_linear[m](parts[m]).permute(0, 3, 1, 2)
            g = self.gcn_modules[m](x)
            if m == "body":
                body = g
            elif m == "left":
                g = g + body[..., -2][..., None].detach()
            elif m == "right":
                g = g + body[..., -1][..., None].detach()
            else:
                g = g + body[..., 0][..., None].detach()
            g = self.fusion_gcn_modules[m](g)
            feats.append(g.mean(-1).transpose(1, 2))
        cat = torch.cat(feats, -1) + self.part_para
        out = self.pose_proj(cat)
        return (out, cat) if return_parts else out


class UniSignEncoder(nn.Module):
    """pose GCN → (optional) mT5-base encoder with the Uni-Sign task prefix."""

    def __init__(self, ckpt="wlasl", use_mt5=True, device="cuda"):
        super().__init__()
        sd = torch.load(CKPTS.get(ckpt, ckpt), map_location="cpu", weights_only=False)["model"]
        self.pose = UniSignPose()
        missing = self.pose.load_state_dict({k: v for k, v in sd.items() if not k.startswith("mt5_model")}, strict=False)
        assert not [k for k in missing.missing_keys if "left" not in k], missing.missing_keys
        self.use_mt5 = use_mt5
        if use_mt5:
            from transformers import MT5Config, MT5EncoderModel, T5Tokenizer
            cfg = MT5Config.from_pretrained("google/mt5-base")
            self.mt5 = MT5EncoderModel(cfg)
            enc_sd = {k[len("mt5_model."):]: v for k, v in sd.items() if k.startswith("mt5_model.encoder.") or k == "mt5_model.shared.weight"}
            self.mt5.load_state_dict(enc_sd, strict=False)
            tok = T5Tokenizer.from_pretrained("google/mt5-base", legacy=False)
            ids = tok(["Translate sign language video to English: "], return_tensors="pt")
            with torch.no_grad():
                self.register_buffer("prefix", self.mt5.shared(ids["input_ids"]).detach()[0])
            self.mt5.shared = nn.Embedding(1, 1)  # drop the 250k-token table (≈192 M params) after taking the prefix
            self.mt5.encoder.embed_tokens = self.mt5.shared
        del sd
        self.to(device)
        self.device = device

    @torch.no_grad()
    def frames(self, parts: dict, mask: torch.Tensor, level="both"):
        """→ dict gcn [B,T,768] and ctx [B,T,768] (mT5 hidden states at frame positions)."""
        parts = {k: v.to(self.device) for k, v in parts.items()}
        mask = mask.to(self.device)
        g = self.pose(parts)
        out = {"gcn": g}
        if self.use_mt5 and level in ("both", "ctx"):
            B = g.shape[0]
            pre = self.prefix[None].expand(B, -1, -1).to(g.dtype)
            emb = torch.cat([pre, g], 1)
            am = torch.cat([torch.ones(B, pre.shape[1], dtype=torch.long, device=self.device), mask.long()], 1)
            h = self.mt5(inputs_embeds=emb, attention_mask=am).last_hidden_state
            out["ctx"] = h[:, pre.shape[1]:]
        return out
