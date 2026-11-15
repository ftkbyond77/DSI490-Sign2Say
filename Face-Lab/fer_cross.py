"""The other direction of the comparison: every model on the official FER2013 test folder (7,178 48x48 gray still images, never used to
train or choose anything by any of them — B0 v1 trained on FER2013 train, v2-ferplus on FER+ labels of FER2013 train, B2 on AffectNet,
the RAVDESS head on RAVDESS + neutral signing).
 Our bundle needs 96 px crops: the 48 px image is enlarged to 96 px (bicubic), and the head sees a "window" of one image (no 2 s video) —
 both against it. Shown without a prior and with each bundle's deployed neutral prior.
  python fer_cross.py → results/fer_cross.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER, softmax   # noqa: E402
from data import CLASSES, load_raw              # noqa: E402
from evaluate import classification             # noqa: E402


def report(name, P, y, rep):
    r = classification(y, P)
    rep[name] = dict(accuracy=r["accuracy"], macro_f1=r["macro_f1"], uar=r["uar"],
                     per_class_recall={c: r["per_class"][c]["recall"] for c in CLASSES}, confusion=r["confusion"])
    print(f"{name:45s} acc {r['accuracy']:.3f} macro-F1 {r['macro_f1']:.3f}", {c[:3]: round(r['per_class'][c]['recall'], 2) for c in CLASSES}, flush=True)


def main():
    X, y, _ = load_raw()["test"]
    rep = dict(note=__doc__, n=int(len(y)), per_class_n={c: int((y == i).sum()) for i, c in enumerate(CLASSES)})

    m = FaceFER(ROOT / "models" / "face-model-efficientnetb0-v1"); o = [m.classes.index(c) for c in CLASSES]
    Z = np.concatenate([m.logits(X[i:i + 512]) for i in range(0, len(X), 512)])
    shipped = float((m.cfg.get("prior_shift") or {}).get("neutral", 0.0))
    for p in sorted({0.0, shipped}):
        b = np.array([p if c == "neutral" else 0.0 for c in m.classes])
        report(f"main: B0 v1 (prior {p})", softmax(Z + m.T * b, m.T)[:, o], y, rep)

    del m, Z
    h = FaceFER(ROOT / "models" / "face-model-ravhead-v1"); hd = h.head; o = [h.classes.index(c) for c in CLASSES]
    X96 = [cv2.resize(g, (96, 96), interpolation=cv2.INTER_CUBIC) for g in X]
    V = []
    for i in range(0, len(X), 64):
        h.probs(X[i:i + 64], X96[i:i + 64]); V.append(h._vecs)
        print(i, flush=True) if i % 2048 == 0 else None
    V = np.concatenate(V)
    k = len(h.classes)
    report("v2-ferplus alone (member, T 0.7)", V[:, 2 * k:][:, hd.to_cls][:, o], y, rep)
    report("HSEmotion B2 alone (member)", V[:, k:2 * k][:, hd.to_cls][:, o], y, rep)
    shipped = float(hd.bias[h.classes.index("neutral")]); keep = hd.bias.copy()
    for p in sorted({0.0, shipped}):
        hd.bias = np.array([p if c == "neutral" else 0.0 for c in h.classes])
        report(f"ours: RAVDESS head (prior {p})", np.array([hd.window_probs(v[None]) for v in V])[:, o], y, rep)
    hd.bias = keep
    (ROOT / "Face-Lab" / "results" / "fer_cross.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
