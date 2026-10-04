"""v4 L2: pretrained sign encoder — Uni-Sign pose branch (ICLR 2025, arXiv:2501.15187).

Part-wise ST-GCN over body(9) · left hand(21) · right hand(21) · face(18) keypoints from RTMW, pretrained on
CSL-News (1,985 h) and fine-tuned on WLASL-2000 (isolated ASL signs, ~100 signers). Optionally the mT5-base
encoder that sat on top of it in Uni-Sign is kept as a temporal context model.

Keypoint normalisation reproduces `datasets.load_part_kp` of the official code exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .parts import BODY_IDX, FACE_IDX, MODES, prepare_parts  # noqa: F401  (re-exported: keypoint preparation is NumPy-only)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "unisign"))
from stgcn_layers import Graph, get_stgcn_chain  # noqa: E402

HF_SNAP = Path.home() / ".cache/huggingface/hub/models--ZechengLi19--Uni-Sign/snapshots/eab251b7fe7e8521afc0e67be98add670ea40a0d"
CKPTS = {"wlasl": HF_SNAP / "wlasl_pose_only_islr.pth", "csl_stage1": HF_SNAP / "csl_stage1_weight.pth"}


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
