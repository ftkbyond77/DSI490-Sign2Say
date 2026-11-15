"""Which face-point / blendshape measures separate SMILES (happy) and RAISED BROWS (surprise) from neutral on RAVDESS 24 actors
(cache/ravdess24, 10 fps). Every measure relative to the actor's own level (a percentile of all their frames — live: the recent history),
clip score = 90th percentile over the clip. AUC per measure: target vs neutral, and target vs every other emotion.
  python cue_eval.py → results/cue_eval.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.mediapipe_pose import BLENDSHAPES   # noqa: E402
from compare24 import EMO                        # noqa: E402

B = {n: i for i, n in enumerate(BLENDSHAPES)}


def point_feats(p):
    """68 points in pixels → dict of measures (÷ eye distance, eye-line frame; y down the face)."""
    el, er = p[36:42].mean(0), p[42:48].mean(0); d = er - el; iod = np.hypot(*d)
    if iod < 1e-6:
        return None
    ux = d / iod; uy = np.array([-ux[1], ux[0]])
    Y = lambda q: float(np.dot(q, uy)) / iod; X = lambda q: float(np.dot(q, ux)) / iod
    eye_y = (Y(el) + Y(er)) / 2
    lipc = (p[51] + p[57]) / 2
    return dict(mouth_width=X(p[54]) - X(p[48]),
                corner_lift=Y(lipc) - (Y(p[48]) + Y(p[54])) / 2,            # corners above the lip centre → +
                corner_rise=-((Y(p[48]) + Y(p[54])) / 2 - eye_y),             # corners closer to the eyes → + (cheeks pushed up)
                brow_inner=np.mean([Y(el) - Y(p[21]), Y(er) - Y(p[22])]),
                brow_outer=np.mean([Y(el) - Y(p[17]), Y(er) - Y(p[26])]),
                brow_all=np.mean([Y(el if i < 22 else er) - Y(p[i]) for i in range(17, 27)]),
                eye_open=np.mean([Y(p[41]) - Y(p[37]), Y(p[40]) - Y(p[38]), Y(p[47]) - Y(p[43]), Y(p[46]) - Y(p[44])]),
                jaw=Y(p[57]) - Y(p[51]),
                brow_nose_all=np.mean([Y(p[30]) - Y(p[i]) for i in range(17, 27)]),
                brow_nose_outer=np.mean([Y(p[30]) - Y(p[i]) for i in (17, 18, 25, 26)]),
                brow_nose_inner=np.mean([Y(p[30]) - Y(p[i]) for i in (20, 21, 22, 23)]))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    rows, lab = [], []
    for f in sorted((ROOT / "Face-Lab" / "cache" / "ravdess24").glob("actor_*.npz")):
        z = np.load(f); face = z["face"].astype(np.float64); bs = z["bs"].astype(np.float64)
        F = []
        for k in range(len(face)):
            pf = point_feats(face[k] * [1280, 720]) if np.abs(face[k]).sum() > 0 else None
            if pf is None or bs[k].sum() == 0:
                F.append(None); continue
            pf.update(bs_smile=(bs[k, B["mouthSmileLeft"]] + bs[k, B["mouthSmileRight"]]) / 2,
                      bs_brow_outer=(bs[k, B["browOuterUpLeft"]] + bs[k, B["browOuterUpRight"]]) / 2, bs_brow_inner=bs[k, B["browInnerUp"]],
                      bs_eye_wide=(bs[k, B["eyeWideLeft"]] + bs[k, B["eyeWideRight"]]) / 2, bs_cheek_squint=(bs[k, B["cheekSquintLeft"]] + bs[k, B["cheekSquintRight"]]) / 2)
            F.append(pf)
        keys = list(next(x for x in F if x).keys())
        M = np.array([[x[k] if x else np.nan for k in keys] for x in F])
        base = {pct: np.nanpercentile(M, pct, axis=0) for pct in (30, 50)}
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1]); m = M[a0:a1]
            if np.isnan(m).all():
                continue
            rows.append({f"{k}@p{pct}": float(np.nanpercentile(m[:, i] - base[pct][i], 90)) for pct in base for i, k in enumerate(keys)})
            lab.append(EMO[int(z["emotion"][ci])])
    lab = np.array(lab); out = {}
    for tgt in ("happy", "surprise"):
        res = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows])
            res[k] = dict(vs_neutral=round(auc(v[lab == tgt], v[lab == "neutral"]), 3), vs_rest=round(auc(v[lab == tgt], v[lab != tgt]), 3))
        out[tgt] = dict(sorted(res.items(), key=lambda kv: -kv[1]["vs_neutral"]))
        print(tgt); [print(f"  {k:28s} {v}") for k, v in list(out[tgt].items())[:10]]
    (ROOT / "Face-Lab" / "results" / "cue_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
