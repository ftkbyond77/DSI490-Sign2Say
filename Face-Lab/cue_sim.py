"""Round 2 tuning, two ideas measured on BOTH sets before anything ships:
  smile  : add the old ensemble's 'happy' back into the head (p[happy] += a · p_ens[happy], renormalised) — the ensemble caught 2/3 subtle smiles
  surprise: eye-widening cue from the 68 points (eye opening above the person's own level; RAVDESS AUC 0.88, and it stays quiet on the team's
           question brows, unlike brow height) — only on frames the model calls neutral, like the brow-furrow cue
RAVDESS: out-of-fold head logits (cache/rav_oof.npz from rav_prior.py, prior 0.5) + ensemble probabilities per sample, cue per sample.
Team clips: the app path (one crop per call, carried window, cue history, unknown rule, smoother).   python cue_sim.py → results/cue_sim.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import ExpressionSmoother, FaceFER, softmax   # noqa: E402
from modules.face_model import EMOTIONS                             # noqa: E402
from compare24 import EMO, metrics, spans                           # noqa: E402
from cue_eval import point_feats                                    # noqa: E402
import rav_head as R                                                # noqa: E402
import real_check                                                   # noqa: E402

MODEL = ROOT / "models" / "face-model-ravhead-v1"
GRID_A = (0.0, "ovr0.35", "ovr0.4", "ovr0.5", "ovr0.6")
GRID_EYE = (None,)
UNIT, SPAN, WMAX, PCT, KEEP, MINB = 0.01, 1.5, 0.5, 30, 60, 6


def mix(P, Pe, a, hi):
    """a = float: p[happy] += a·p_ens[happy] (renormalised); "ovrT": the ensemble's answer replaces the head's where the ensemble's top
    class is happy with p ≥ T (the head, adapted to neutral signing, reads subtle smiles as neutral; the ensemble does not)."""
    if isinstance(a, str):
        T = float(a[3:]); P = np.array(P, np.float64); Pe = np.asarray(Pe)
        if P.ndim == 1:
            return Pe.copy() if int(np.argmax(Pe)) == hi and Pe[hi] >= T else P
        m = (Pe.argmax(1) == hi) & (Pe[:, hi] >= T); P[m] = Pe[m]
        return P
    P = P.copy(); P[..., hi] += a * Pe[..., hi]
    return P / P.sum(-1, keepdims=True)


def eye_push(P, s, thr, tgt, neu):
    if thr is None or s is None or not np.isfinite(s) or s <= thr or int(np.argmax(P)) != neu:
        return P
    w = WMAX * min(1.0, (s - thr) / SPAN); P = P * (1 - w); P[tgt] += w
    return P


def ravdess():
    o = np.load(ROOT / "Face-Lab" / "cache" / "rav_oof.npz")
    L = o["L"].copy(); L[:, 0] += 0.5; Ph = softmax(L)
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8")); cls = cfg["classes"]
    bias = np.array([cfg["prior_shift"].get(c, 0.0) for c in cls]); to7 = [R.HSE.index(c) for c in cls]; to_e = [cls.index(e) for e in EMOTIONS]
    Pe, S = [], []
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = np.load(f); e = np.load(R.FEAT / f.name); fps = float(z["fps"])
        P = (softmax(e["v2"] / 0.7 + bias) + softmax(e["b2"][:, to7] + bias)) / 2
        face = z["face"].astype(np.float64)
        eo = np.array([(lambda d: d["eye_open"] if d else np.nan)(point_feats(face[k] * [1280, 720]) if np.abs(face[k]).sum() > 0 else None) for k in range(len(face))])
        base = np.nanpercentile(eo, PCT)
        t_all, cq, ct, cc = z["t"], z["crop_q"], z["crop_t"], z["crop_clip"]
        for ci in range(len(z["name"])):
            a0, a1 = int(z["off"][ci]), int(z["off"][ci + 1]); tt = t_all[a0:a1]
            for k, (s, en) in enumerate(spans(a1 - a0, fps)):
                sel = (cc == ci) & (cq >= 0.25) & (ct >= tt[s] - 1e-6) & (ct <= tt[min(en, len(tt)) - 1] + 1e-6)
                if not sel.any():
                    sel = (cc == ci) & (cq >= 0.25)
                if not sel.any():
                    sel = cc == ci
                Pe.append(P[sel].mean(0)[to_e])
                x = eo[a0:a1][s:en]; S.append(np.nanpercentile(x - base, 90) / UNIT if np.isfinite(x).any() else np.nan)
    Pe, S = np.array(Pe), np.array(S)
    assert len(Pe) == len(Ph)
    hi, su, ne = EMOTIONS.index("happy"), EMOTIONS.index("surprise"), EMOTIONS.index("neutral")
    out = {}
    for a in GRID_A:
        for thr in GRID_EYE:
            P = mix(Ph, Pe, a, hi)
            P = np.array([eye_push(p, s, thr, su, ne) for p, s in zip(P, S)])
            out[(a, thr)] = metrics(P, o["y"], o["whole"], o["inten"])
    return out


def team():
    m = FaceFER(MODEL); cls = m.classes
    S = np.load(real_check.CACHE, allow_pickle=True); pts = np.load(ROOT / "Face-Lab" / "cache" / "real_pts.npz")["pts"].astype(np.float64)
    jl = real_check.judged(S["clip"], S["t"], S["q"])
    import cv2
    s96 = [cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]]
    hi, su, ne = cls.index("happy"), cls.index("surprise"), cls.index("neutral")
    # per frame, once: ensemble probs + head probs (carried window, one crop per call as live)
    Pe, Ph = np.zeros((len(jl), len(cls))), np.zeros((len(jl), len(cls)))
    order = []
    for c in sorted(set(S["clip"])):
        idx = np.where(S["clip"] == c)[0]; idx = idx[np.argsort(S["t"][idx])]; hist = []; order.append(idx)
        for i in idx:
            pe = m.probs(S["g"][i:i + 1], [s96[i]]); Pe[i] = pe[0]
            Ph[i] = m.head.apply(pe, m._vecs, [float(S["t"][i])], [S["q"][i]], hist)[0]
    eo = np.array([point_feats(p)["eye_open"] for p in pts])
    exp = {(c, t): lab for c, t, lab in real_check.EXPRESSIVE}
    out = {}
    for a in GRID_A:
        for thr in GRID_EYE:
            shown = np.empty(len(jl), object)
            for idx in order:
                sm = ExpressionSmoother(cls, **m.cfg["smoother"]); h = []
                for i in idx:
                    p = mix(Ph[i], Pe[i], a, hi)
                    s = (eo[i] - np.percentile(h, PCT)) / UNIT if len(h) >= MINB else None
                    h.append(eo[i]); del h[:-KEEP]
                    p = eye_push(p, s, thr, su, ne)
                    shown[i] = sm.update(dict(t=float(S["t"][i]), **m.judge(p, float(S["q"][i]))))["expression"]
            n, hh = jl == 6, jl == 3
            ex = {f"{lab}@{c}:{t}": shown[np.where((S["clip"] == c) & (np.abs(S["t"] - t) < 0.26))[0][0]] for (c, t), lab in exp.items()}
            out[(a, thr)] = dict(shown_neutral=round(float((shown[n] == "neutral").mean()), 3), wrong=round(float(((shown[n] != "neutral") & (shown[n] != "unknown")).mean()), 3),
                                 smiles=f"{int((shown[hh] == 'happy').sum())}/{int(hh.sum())}",
                                 wrong_as={k: int((shown[n] == k).sum()) for k in cls if k != "neutral" and (shown[n] == k).any()}, expressive=ex)
    return out


def main():
    rv, tm = ravdess(), team()
    rep = {}
    print(f"{'a':>5} {'eye':>5} | {'F1':>6} {'happy':>6} {'surpr':>6} {'neut':>6} | team wrong  smiles  wrong_as")
    for k in rv:
        r, t = rv[k], tm[k]
        print(f"{k[0]:>5} {str(k[1]):>5} | {r['clips_macro_f1']:.3f} {r['per_class_recall']['happy']:.3f} {r['per_class_recall']['surprise']:.3f} {r['per_class_recall']['neutral']:.3f} | "
              f"{t['wrong']:.3f}  {t['smiles']}  {t['wrong_as']}")
        rep[f"happy_mix_{k[0]}__eye_thr_{k[1]}"] = dict(ravdess=r, team=t)
    print("expressive frames (a=0, no eye / a=0.5, eye 3):", tm[(0.0, None)]["expressive"], "\n", tm[(0.5, 3.0)]["expressive"])
    (ROOT / "Face-Lab" / "results" / "cue_sim.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
