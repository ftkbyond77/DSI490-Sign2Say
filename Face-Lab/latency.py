"""How long the LIVE label (/v1/face: head over a trailing window → unknown rule → ExpressionSmoother) takes to show an expression, and what
faster settings cost in false alarms.
 RAVDESS (24 actors, out-of-fold heads: each actor is run through a head trained without its fold, as rav_prior.py): for every strong-
   intensity emotion clip, a neutral clip of the same actor is played first, then the emotion clip (crops every 0.2 s, as the web's
   ~180 ms). Latency = time from the emotion clip's first frame to the first frame the shown label is (a) that emotion (b) any emotion.
 Team clips (check_ravhead.app_path, deployed head): false emotion on neutral frames.
 Server time (~0.25–0.35 s per call) comes on top of the numbers here.
  python latency.py → results/latency.json"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER, ExpressionSmoother   # noqa: E402
from modules.face_model import EMOTIONS                    # noqa: E402
from face_model import fit_lr                              # noqa: E402
from compare24 import EMO                                  # noqa: E402
import rav_head as R                                       # noqa: E402
from rav_prior import load_samples                         # noqa: E402
import check_ravhead as C                                  # noqa: E402
import real_check                                          # noqa: E402

BUNDLE = ROOT / "models" / "face-model-ravhead-v1"
VARIANTS = {  # name: (window_s, neutral prior, smoother overrides)
    "now (2 s, prior 0.5, tau 0.35, hold 0.3)": (2.0, 0.5, {}),
    "smoother faster (tau 0.2, hold 0.15)": (2.0, 0.5, dict(tau_s=0.2, min_hold_s=0.15)),
    "window 1 s": (1.0, 0.5, {}),
    "window 1 s + smoother faster": (1.0, 0.5, dict(tau_s=0.2, min_hold_s=0.15)),
    "window 1.5 s + smoother faster": (1.5, 0.5, dict(tau_s=0.2, min_hold_s=0.15)),
}


def set_head(m, h):
    m.head.mu, m.head.sd, m.head.W, m.head.b = (np.asarray(getattr(h, k), np.float64) for k in ("mu", "sd", "W", "b"))


def main():
    m = FaceFER(BUNDLE); base_head = copy.deepcopy(m.head); base_sm = dict(m.cfg["smoother"])
    assert list(m.classes) == ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    hse_idx = [mm for mm in m.members if mm.kind == "hse"][0].idx
    # out-of-fold heads (same training as the deployed one, minus the fold's actors)
    XP, y, actor, *_ , to7, e_idx = load_samples(); Xd = R.sign_neutral(to7, e_idx)
    acts = np.unique(actor); folds = [acts[i::6] for i in range(6)]
    heads = {}
    for te in folds:
        k = np.isin(actor, te); h = fit_lr(np.r_[XP[~k], Xd], np.r_[y[~k], np.zeros(len(Xd), int)], 0.1)
        for a in te:
            heads[int(a)] = h
        print("fold", flush=True)
    # per actor: rows of every crop
    seqs = []                                                        # (actor, rows [n,21] neutral+emotion, t, q, onset_t, target)
    for f in sorted(R.DATA.glob("actor_*.npz")):
        z = np.load(f); e = np.load(R.FEAT / f.name); a = int(z["actor"])
        V = base_head.crop_vecs(e["feat"].astype(np.float32), e["b2"][:, hse_idx], e["v2"])
        cc, ct, cq = z["crop_clip"], z["crop_t"], z["crop_q"]
        emo = [EMO[int(v)] for v in z["emotion"]]
        neu = [ci for ci in range(len(emo)) if int(z["emotion"][ci]) == 1]
        for ci in range(len(emo)):
            if emo[ci] == "neutral" or int(z["intensity"][ci]) != 2:
                continue
            n = neu[ci % len(neu)]; a_i, b_i = np.where(cc == n)[0], np.where(cc == ci)[0]
            dur = float(ct[a_i].max()) + 0.2
            seqs.append((a, np.r_[V[a_i], V[b_i]], np.r_[ct[a_i], ct[b_i] + dur], np.r_[cq[a_i], cq[b_i]], dur, emo[ci]))
    print(len(seqs), "neutral→emotion sequences", flush=True)
    S = np.load(real_check.CACHE, allow_pickle=True); jl = real_check.judged(S["clip"], S["t"], S["q"])
    s96 = list(np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]]))
    rep = dict(note=__doc__, n_sequences=len(seqs), variants={})
    for name, (win, prior, sm_over) in VARIANTS.items():
        sm_cfg = {**base_sm, **sm_over}
        lat_ok, lat_any, false_before = [], [], 0
        for a, V, t, q, onset, target in seqs:
            set_head(m, heads[a]); m.head.win = win; m.head.bias = np.array([prior if c == "neutral" else 0.0 for c in m.classes])
            sm = ExpressionSmoother(m.classes, **sm_cfg); hist = []
            P = m.head.apply(np.zeros((len(V), 7)), V, t, q, hist)
            hit_ok = hit_any = None; fb = False
            for i in range(len(V)):
                s = sm.update(dict(t=float(t[i]), **m.judge(P[i], float(q[i]))))["expression"]
                if t[i] < onset:
                    fb |= s not in ("neutral", "unknown")
                    continue
                if hit_any is None and s not in ("neutral", "unknown"):
                    hit_any = t[i] - onset
                if hit_ok is None and s == target:
                    hit_ok = t[i] - onset
            lat_ok.append(hit_ok); lat_any.append(hit_any); false_before += fb
        ok = np.array([v for v in lat_ok if v is not None]); an = np.array([v for v in lat_any if v is not None])
        m.head = copy.deepcopy(base_head); m.head.win = win; m.cfg["smoother"] = sm_cfg
        team = C.app_path(m, S, s96, jl); m.cfg["smoother"] = base_sm
        rep["variants"][name] = r = dict(
            window_s=win, prior=prior, smoother=sm_cfg,
            correct_shown=round(len(ok) / len(seqs), 3), median_s_to_correct=round(float(np.median(ok)), 2), p90_s_to_correct=round(float(np.percentile(ok, 90)), 2),
            any_emotion_shown=round(len(an) / len(seqs), 3), median_s_to_any=round(float(np.median(an)), 2),
            ravdess_false_on_neutral_lead_in=round(false_before / len(seqs), 3), team=team)
        print(f"{name:42s} correct {r['correct_shown']:.0%} median {r['median_s_to_correct']}s p90 {r['p90_s_to_correct']}s | any {r['median_s_to_any']}s"
              f" | team wrong {team['wrong']} {team['wrong_as']} smiles {team['smiles']}", flush=True)
    (ROOT / "Face-Lab" / "results" / "latency.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
