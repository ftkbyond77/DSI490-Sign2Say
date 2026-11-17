"""Brow-furrow cue: which measure separates RAVDESS angry from the rest, relative to the person's own resting face.
  python brow_eval.py   (needs cache/ravdess_brows.npz from brow_harvest.py)"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.face_fer import CLASSES, brow_features   # noqa: E402

BS = {"browDownLeft": 1, "browDownRight": 2, "browInnerUp": 3}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    D = np.load(ROOT / "Face-Lab" / "cache" / "ravdess_brows.npz")
    px = D["face"] * np.array([1280, 720], np.float32)                     # RAVDESS 1280x720: the measures need pixel geometry
    F = np.stack([brow_features(p) for p in px])                          # [N, 2]: inner-brow gap, inner-brow height (both / eye distance)
    bd = D["bs"][:, [BS["browDownLeft"], BS["browDownRight"]]].mean(1)
    lab, act, rel = D["label"], D["actor"], D["rel"]
    mid = (rel > 0.25) & (rel < 0.9)
    A, N = CLASSES.index("angry"), CLASSES.index("neutral")
    out = {}
    for name, x in (("gap", F[:, 0]), ("height", F[:, 1]), ("browDown_bs", -bd)):
        z = np.zeros_like(x)
        for a in set(act):                                             # relative to the actor's resting face (median of the neutral clips)
            m = act == a
            base = np.median(x[m & (lab == N)]); sd = np.std(x[m & (lab == N)]) + 1e-6
            z[m] = (base - x[m]) / sd                                  # > 0: brows closer together / lower / more 'down' than at rest
        out[name] = z
        r = {CLASSES[c]: round(auc(z[mid & (lab == A)], z[mid & (lab == c)]), 3) for c in range(len(CLASSES)) if c != A}
        print(f"{name:12s} AUC angry vs: {r}")
    comb = out["gap"] + out["height"]
    r = {CLASSES[c]: round(auc(comb[mid & (lab == A)], comb[mid & (lab == c)]), 3) for c in range(len(CLASSES)) if c != A}
    print(f"{'gap+height':12s} AUC angry vs: {r}")
    for thr in (1.5, 2.0, 2.5, 3.0):
        print(f"gap+height z>{thr}: angry frames {np.mean(comb[mid & (lab == A)] > thr):.2f} | neutral {np.mean(comb[mid & (lab == N)] > thr):.2f} | "
              + " ".join(f"{CLASSES[c][:3]} {np.mean(comb[mid & (lab == c)] > thr):.2f}" for c in range(len(CLASSES)) if c not in (A, N)))


if __name__ == "__main__":
    main()
