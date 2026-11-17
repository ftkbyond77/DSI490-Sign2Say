"""v12 release folder (prompt-opt-v12 4A): models/model-v12 = models/model-v10 + the files of the KEPT ladder rungs.

  python scripts/v12_build.py --cand v12-B2b [--live] [--name model-v12]

Copied from the candidate folder (artifacts/v12/<name>) when present: bank.npz, hand_bank.npz, lm_bank.npz, spotting.json,
segment_calibration.json, onnx/lmenc.onnx; the evaluation banks go to <release>/eval/ (harness only — the runtime never reads them).
class_prior.npz is recomputed from the release's production bank (same rule as v10: leave-one-signer-out reliability per word).
--live writes the v12 live-mode rules into config.json ("live"; the web reads them through /v1/info — model-v10 has none = v10 rules).
Rollback: THAISLM_MODELS_DIR=./models/model-v10 (untouched).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

V10 = ROOT / "models" / "model-v10"
LIVE_V12 = dict(preroll_ms=1500, tail_ms=800, wrist_grace_ms=800, cap_cut=True,
                evidence="scripts/v12_live_rules.py: web-rate TSL51 TUNE sentences, live-cut WER v10 rule 0.375 -> 0.337 (whole clip 0.336); "
                         "LOG: 79 / 218 runaway utterances end earlier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--live", action="store_true"); ap.add_argument("--name", default="model-v12")
    ap.add_argument("--rungs", default="", help="text for the model card, e.g. 'B1b+B2 (+B6 live)'")
    a = ap.parse_args()
    from decision import class_prior
    from modules.runtime import variant_lut
    from modules.utils import read_json, write_json
    src = ROOT / "artifacts" / "v12" / a.cand.replace("v12-", "")
    dst = ROOT / "models" / a.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(V10, dst)
    changed = []
    for f in ("bank.npz", "hand_bank.npz", "lm_bank.npz", "spotting.json", "segment_calibration.json"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f); changed.append(f)
    if (src / "onnx" / "lmenc.onnx").exists():
        shutil.copy2(src / "onnx" / "lmenc.onnx", dst / "onnx" / "lmenc.onnx"); changed.append("onnx/lmenc.onnx")
    if (src / "eval").exists():
        shutil.copytree(src / "eval", dst / "eval")
    if "bank.npz" in changed:
        concepts = read_json(dst / "concepts.json")
        vc = read_json(dst / "vocab_conversation.json")
        allowed = {concepts.index(c) for c in vc["concepts"] if c in concepts}
        fold = variant_lut(concepts, vc.get("variants", {}))
        cp = class_prior(dst / "bank.npz", concepts, allowed, fold)
        np.savez(dst / "class_prior.npz", rel=cp["rel"].astype(np.float32), depth=cp["depth"].astype(np.float32)); changed.append("class_prior.npz")
    cfg = read_json(dst / "config.json")
    cfg.update(version="v12", built=time.strftime("%Y-%m-%d %H:%M"), built_from=dict(base="model-v10", candidate=a.cand, files=changed, rungs=a.rungs))
    if a.live:
        cfg["live"] = LIVE_V12
    bz = np.load(dst / "bank.npz"); cfg["bank"] = int(len(bz["c"]))
    write_json(dst / "config.json", cfg)
    write_json(dst / "model_card.json", dict(version="v12", base="v10", rungs=a.rungs, changed=changed, live=cfg.get("live"),
                                             rollback="THAISLM_MODELS_DIR=./models/model-v10", built=cfg["built"]))
    print("built", dst, "changed", changed, "live", bool(a.live))


if __name__ == "__main__":
    main()
