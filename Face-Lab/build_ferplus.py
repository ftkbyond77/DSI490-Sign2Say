"""Attach FER+ crowd votes (Barsoum et al. 2016, 10 annotators per image) to our FER2013 images.

  python build_ferplus.py            -> cache/ferplus.npz  (votes aligned to data.load_raw() order) + results/ferplus_match.json

Our images come from the Kaggle folder copy of FER2013 (JPEG re-encoded, file names do not give the csv row), while FER+ votes follow the row
order of the original fer2013.csv. So each image is matched to the nearest csv image (pixel L2) among rows with the same Usage group and the
same original label; the match distance and the gap to the second-best row are kept so bad matches can be dropped.
Inputs: ../FERPlus-Data/x/fer2013/fer2013.csv (md5 f8428a1e...) and ../FERPlus-Data/fer2013new.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "FERPlus-Data"
# FER+ vote columns -> our class ids (data.CLASSES order); contempt / unknown / NF have no class of ours
FP_COLS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]
TO_OURS = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "sadness": 4, "surprise": 5, "neutral": 6}


def main():
    raw = data.load_raw()
    f = ROOT / "cache" / "fer2013_csv_pixels.npy"
    df = pd.read_csv(SRC / "x" / "fer2013" / "fer2013.csv")
    if f.exists():
        P = np.load(f)
    else:
        P = np.stack([np.array(s.split(), np.uint8) for s in df.pixels]).reshape(-1, 48, 48); np.save(f, P)
    fp = pd.read_csv(SRC / "fer2013new.csv")
    assert len(fp) == len(df) and (fp.Usage.values == df.Usage.values).all(), "FER+ rows do not follow fer2013.csv"
    V = fp[FP_COLS].fillna(0).values.astype(np.float32)               # [35887, 10] raw vote counts
    emo, use = df.emotion.values, df.Usage.values
    groups = {"train": ["Training"], "test": ["PublicTest", "PrivateTest"]}
    out, rep = {}, {}
    for s, (X, y, n) in raw.items():
        row, dist, gap = np.full(len(X), -1), np.zeros(len(X), np.float32), np.zeros(len(X), np.float32)
        for c in range(len(data.CLASSES)):
            qi = np.where(y == c)[0]
            ci = np.where(np.isin(use, groups[s]) & (emo == c))[0]
            Q = torch.from_numpy(X[qi].reshape(len(qi), -1).astype(np.float32))
            C = torch.from_numpy(P[ci].reshape(len(ci), -1).astype(np.float32))
            cn = (C * C).sum(1)
            for a in range(0, len(Q), 1024):
                q = Q[a:a + 1024]
                d2 = (q * q).sum(1, keepdim=True) + cn[None] - 2 * q @ C.T
                v, k = d2.clamp(min=0).topk(min(2, len(ci)), dim=1, largest=False)
                r = np.sqrt(v.numpy() / 2304)                          # RMS pixel error, 0..255
                row[qi[a:a + 1024]] = ci[k[:, 0].numpy()]
                dist[qi[a:a + 1024]] = r[:, 0]
                gap[qi[a:a + 1024]] = r[:, 1] - r[:, 0] if r.shape[1] > 1 else 99
        uniq = len(np.unique(row))
        out[s] = dict(row=row, dist=dist, gap=gap, votes=V[row], names=n)          # names: to check the alignment with load_raw() elsewhere
        rep[s] = dict(n=int(len(X)), unique_rows=int(uniq), dist_pct={p: round(float(np.percentile(dist, p)), 2) for p in (50, 90, 99, 99.9, 100)},
                      ambiguous_gap_lt1=int((gap < 1).sum()), dist_gt8=int((dist > 8).sum()))
    np.savez_compressed(ROOT / "cache" / "ferplus.npz", **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ferplus_match.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def load(split_names):
    """FER+ votes aligned to load_raw(); split_names = dict(train=names, test=names) to check the order is the same as when it was built."""
    z = np.load(ROOT / "cache" / "ferplus.npz", allow_pickle=False)
    for s, n in split_names.items():
        assert (z[f"{s}_names"] == n).all(), "cache/ferplus.npz does not follow this FER2013 folder order - rerun build_ferplus.py"
    return {s: z[f"{s}_votes"] for s in split_names}


def soft_targets(votes, min_votes=1):
    """FER+ votes [N,10] -> (probabilities over our 7 classes [N,7], usable mask). Contempt / unknown / not-a-face votes are dropped and the rest
    renormalised (Barsoum's 'probability' scheme); an image with fewer than `min_votes` votes on our classes, or a not-a-face majority, is not usable."""
    t = np.zeros((len(votes), 7), np.float32)
    for j, name in enumerate(FP_COLS):
        if name in TO_OURS:
            t[:, TO_OURS[name]] = votes[:, j]
    s = t.sum(1)
    ok = (s >= min_votes) & (votes[:, FP_COLS.index("NF")] < s)
    t[ok] /= s[ok, None]
    return t, ok


if __name__ == "__main__":
    main()
