"""Build the ensemble bundle  candidates/face-model-ensemble-v2ferplus-b2/  (v2-ferplus 48 px  +  HSEmotion EfficientNet-B2 on a 96 px crop)
and check it end to end with modules/face_fer.py (the code the API runs):
  * parity: its probabilities = the offline evaluation (hse_eval.py) on RAVDESS actors 3-4
  * the app path (unknown rule + smoother) on the 18 signer clips for a few `min_conf` values -> the one written to config.json

  python build_ensemble.py [--prior 0.75]
HSEmotion weights: research / education use (trained on AffectNet); see model_card.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import CLASSES, ExpressionSmoother, FaceFER   # noqa: E402
import real_check                                                    # noqa: E402
from video_eval import scores                                         # noqa: E402

V2 = ROOT / "Face-Lab" / "candidates" / "face-model-efficientnetb0-v2-ferplus"
B2 = ROOT / "data_test" / "hsemotion" / "enet_b2_8.onnx"
OUT = ROOT / "Face-Lab" / "candidates" / "face-model-ensemble-v2ferplus-b2"
HSE_ORDER = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def build(prior, min_conf):
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(V2 / "face_fer.onnx", OUT / "face_fer.onnx"); shutil.copy(B2, OUT / "enet_b2_8.onnx")
    v2 = json.loads((V2 / "config.json").read_text(encoding="utf-8"))
    cfg = dict(name="face-model-ensemble-v2ferplus-b2", task=v2["task"], classes=CLASSES,
               members=[dict(onnx="face_fer.onnx", kind="fer48", temperature=v2["temperature"], note="our v2-ferplus (EfficientNet-B0, FER2013 images + FER+ labels), 48x48 gray"),
                        dict(onnx="enet_b2_8.onnx", kind="hse", size=260, order=HSE_ORDER, temperature=1.0,
                             note="HSEmotion enet_b2_8 (Savchenko 2022, AffectNet), 96x96 gray crop; contempt dropped")],
               input=dict(g="48x48 uint8 gray (2304 bytes)", g96="96x96 uint8 gray of the same box (9216 bytes); without it only the 48 px member answers"),
               prior_shift=dict(neutral=prior), crop=v2["crop"],
               unknown=dict(min_conf=min_conf, min_margin=0.10, min_quality=0.25),
               smoother={**v2["smoother"], "min_conf": min_conf},
               brow=dict(thr=2.0, span=1.5, w_max=0.5, unit=0.02, base_pct=85, min_base=6, keep=60, target="angry",
                         note="brow-furrow cue from the 68 face points (brow_eval.py; RAVDESS a3-4 F1 0.614 -> 0.675, angry recall 0.25 -> 0.75)"))
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    return cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prior", type=float, default=0.75); a = ap.parse_args()
    build(a.prior, 0.45)
    m = FaceFER(OUT)
    # parity on RAVDESS actors 3-4 (clip level, as video_eval / hse_eval)
    R = np.load(ROOT / "Face-Lab" / "cache" / "ravdess_faces.npz"); ok = (R["q"] >= 0.25) & np.isin(R["actor"], [3, 4])
    clip, lab = R["clip"][ok], R["label"][ok]
    g96 = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY), (96, 96), interpolation=cv2.INTER_AREA) for c in R["col"][ok]])
    P = m.probs(R["g"][ok], list(g96)); clips = sorted(set(clip))
    s = scores(np.array([lab[clip == c][0] for c in clips]), np.stack([P[clip == c].mean(0) for c in clips]).argmax(1))
    P48 = m.probs(R["g"][ok], None)
    s48 = scores(np.array([lab[clip == c][0] for c in clips]), np.stack([P48[clip == c].mean(0) for c in clips]).argmax(1))
    rep = dict(prior=a.prior, ravdess_a34=dict(ensemble=s, only_48px_member=s48), signers={})
    print(f"RAVDESS a3-4: ensemble F1 {s['macro_f1']:.3f} exprRec {s['expression_recall']:.3f} | without 96 px crops F1 {s48['macro_f1']:.3f}")
    # the app path on signer clips
    S = np.load(real_check.CACHE, allow_pickle=True)
    jl = real_check.judged(S["clip"], S["t"], S["q"]); n, h = jl == 6, jl == 3
    s96 = list(np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]]))
    Ps = m.probs(S["g"], s96)
    for mc in (0.40, 0.45, 0.50, 0.55, 0.63):
        shown = np.empty(len(Ps), object)
        for c in sorted(set(S["clip"])):
            sm = ExpressionSmoother(m.classes, **{**json.loads((OUT / "config.json").read_text(encoding="utf-8"))["smoother"], "min_conf": mc})
            for i in np.where(S["clip"] == c)[0]:
                p = Ps[i]; top = np.argsort(-p)[:2]; valid = S["q"][i] >= 0.25 and p[top[0]] >= mc and p[top[0]] - p[top[1]] >= 0.10
                shown[i] = sm.update(dict(t=float(S["t"][i]), valid=bool(valid), probs={k: float(p[j]) for j, k in enumerate(m.classes)}))["expression"]
        row = dict(shown_neutral=round(float((shown[n] == "neutral").mean()), 3), wrong=round(float(((shown[n] != "neutral") & (shown[n] != "unknown")).mean()), 3),
                   unknown=round(float((shown[n] == "unknown").mean()), 3), smiles=f"{int((shown[h] == 'happy').sum())}/{int(h.sum())}")
        rep["signers"][f"min_conf_{mc}"] = row
        print(f"signers (app path) min_conf {mc}: {row}")
    (ROOT / "Face-Lab" / "results" / "ensemble_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
