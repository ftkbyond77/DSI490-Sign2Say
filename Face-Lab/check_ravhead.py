"""Check the bundle candidates/face-model-ravhead-v1 end to end with modules/face_fer.py (the code the API runs):
  * parity: PictureHead over a clip's crops = rav_head.py's head on the same sample (picfeat → fit_lr head)
  * the app path on the 18 signer clips (frame_results with the carried window + unknown rule + smoother), against the deployed
    ensemble, for a few neutral prior shifts of the head
  python check_ravhead.py   → results/ravhead_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab")); sys.path.insert(0, str(ROOT / "scripts"))
from modules.face_fer import ExpressionSmoother, FaceFER   # noqa: E402
from modules.face_model import EMOTIONS, FaceEmotion                    # noqa: E402
import real_check                                                       # noqa: E402
from rav_head import member_probs, picfeat, HSE                         # noqa: E402

OUT = ROOT / "Face-Lab" / "candidates" / "face-model-ravhead-v1"
ENS = ROOT / "Face-Lab" / "candidates" / "face-model-ensemble-v2ferplus-b2"


def parity(m):
    z = np.load(ROOT / "Face-Lab" / "cache" / "ravdess24" / "actor_03.npz")
    h = np.load(OUT / "rav_head.npz"); ref = FaceEmotion(h["mu"], h["sd"], h["W"], h["b"])
    to7 = [HSE.index(c) for c in m.classes]; e_idx = [m.classes.index(e) for e in EMOTIONS]
    worst = 0.0
    for ci in range(6):
        sel = (z["crop_clip"] == ci) & (z["crop_q"] >= 0.25)
        G, G96 = z["crop_g"][sel], z["crop_g96"][sel]
        m.probs(G, list(G96)); V = m._vecs
        p_app = m.head.window_probs(V)
        feat = m.members[1].last_feat
        zb8 = m.members[1].sess.run(None, {"input": _prep(G96)})[0]
        pb, pv = member_probs(zb8, m.members[0].logits(G, None), to7, e_idx)
        L = ref.logits(picfeat(feat, pb, pv)[None])[0]
        p_ref = np.exp(L - L.max()); p_ref /= p_ref.sum()
        p_ref = p_ref[[EMOTIONS.index(c) for c in m.classes]]
        worst = max(worst, float(np.abs(p_app - p_ref).max()))
    return worst


def _prep(G96):
    from modules.face_fer import _IMNET_MEAN, _IMNET_STD
    x = np.stack([cv2.resize(g, (260, 260), interpolation=cv2.INTER_CUBIC) for g in G96]).astype(np.float32) / 255.0
    return np.ascontiguousarray(((np.repeat(x[..., None], 3, -1) - _IMNET_MEAN) / _IMNET_STD).transpose(0, 3, 1, 2))


def app_path(m, S, s96, jl):
    n, h = jl == 6, jl == 3
    shown = np.empty(len(S["g"]), object)
    for c in sorted(set(S["clip"])):
        idx = np.where(S["clip"] == c)[0]; idx = idx[np.argsort(S["t"][idx])]
        sm = ExpressionSmoother(m.classes, **m.cfg["smoother"]); hist = []
        for i in idx:                                                    # one crop per call, as the live client (order kept)
            fr = m.frame_results(S["g"][i:i + 1], [S["q"][i]], [float(S["t"][i])], [s96[i]], hist)[0]
            shown[i] = sm.update(fr)["expression"]
    return dict(shown_neutral=round(float((shown[n] == "neutral").mean()), 3),
                wrong=round(float(((shown[n] != "neutral") & (shown[n] != "unknown")).mean()), 3),
                unknown=round(float((shown[n] == "unknown").mean()), 3), smiles=f"{int((shown[h] == 'happy').sum())}/{int(h.sum())}",
                wrong_as={k: int((shown[n] == k).sum()) for k in m.classes if k != "neutral" and (shown[n] == k).any()})


def main():
    m = FaceFER(OUT)
    rep = dict(parity_max_abs_diff=parity(m))
    print("parity (max |Δp|):", rep["parity_max_abs_diff"], flush=True)
    S = np.load(real_check.CACHE, allow_pickle=True)
    jl = real_check.judged(S["clip"], S["t"], S["q"])
    s96 = list(np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]]))
    rep["ensemble (deployed)"] = app_path(FaceFER(ENS), S, s96, jl); print("ensemble", rep["ensemble (deployed)"], flush=True)
    for ps in (0.0, 0.5, 1.0):
        m.head.bias = np.array([ps if c == "neutral" else 0.0 for c in m.classes])
        rep[f"ravhead neutral_prior {ps}"] = r = app_path(m, S, s96, jl); print(f"ravhead prior {ps}", r, flush=True)
    (ROOT / "Face-Lab" / "results" / "ravhead_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
