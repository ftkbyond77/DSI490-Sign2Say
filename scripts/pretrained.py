"""Rebuild artifacts/pretrained/ (Uni-Sign base encoders used for TRAINING; inference needs only models/).

  python main.py pretrained            # downloads ZechengLi19/Uni-Sign (Hugging Face) and writes unisign_wlasl_enc.pt + unisign_csl_enc.pt

Not stored in git (2 × 353 MB, public checkpoints): GitHub's free LFS quota is 1 GB and models/ already uses ~0.9 GB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from huggingface_hub import hf_hub_download
    from modules.utils import PRETRAINED
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    for f in ("wlasl_pose_only_islr.pth", "csl_stage1_weight.pth"):
        hf_hub_download("ZechengLi19/Uni-Sign", f)            # cached under ~/.cache/huggingface (modules.unisign reads it there)
    from modules.unisign import UniSignEncoder
    for name, ck in (("wlasl", "wlasl"), ("csl", "csl_stage1")):
        enc = UniSignEncoder(ck, use_mt5=True, device="cpu")
        torch.save(enc.state_dict(), PRETRAINED / f"unisign_{name}_enc.pt")
        (PRETRAINED / "mt5_config.json").write_text(json.dumps(enc.mt5.config.to_dict()), encoding="utf-8")
        print("wrote", PRETRAINED / f"unisign_{name}_enc.pt")


if __name__ == "__main__":
    main()
