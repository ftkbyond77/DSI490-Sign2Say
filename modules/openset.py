"""v11 decision layer (prompt-opt-v11 §2B / §3C): open-set head · per-group conformal prediction sets · three-way decision.

Applied by SignPipeline.decide when segment_calibration.json carries a "v11" block (fitted by `python main.py v11-decision fit`):

    segment ─► P(known) < bound ? ─► UNKNOWN (show nothing)
                  │ no
                  ▼
           conformal set of its group (words whose calibrated probability ≥ 1 − q_group)
                  │ size 1 ─────────────────────────────► KNOWN-SURE   (show + speak; status "accepted")
                  │ size 2–3, one confusion group ─► verifier confident ─► KNOWN-SURE
                  │                                   └ not ─────────► KNOWN-AMBIGUOUS (alternatives, never spoken; "uncertain")
                  │ larger / empty ───────────────────────────────────► UNKNOWN ("unknown")
Extended-tier words additionally need set size 1 AND a larger margin than Core (tier_margin). The class-reliability prior never takes
part in the accept decision (it may still rank). Every segment keeps `status` (accepted / uncertain / unknown — face, speech and the
web read it) and gains `decision` (KNOWN-SURE / KNOWN-AMBIGUOUS / UNKNOWN), `alternatives` and `p_known`.

Open-set head = logistic regression (NumPy, no sklearn at run time) on per-segment features of the reading (OPEN_FEATS), trained with
full outlier exposure: positives = Core / Extended segments of TUNE signers, negatives = Tail signs of TUNE signers (known unknowns) and
idle / not-a-sign windows; folds grouped by signer.
"""
from __future__ import annotations

import numpy as np

OPEN_FEATS = ["s1", "s2", "s5", "gap12", "energy", "z", "null_gap", "p_sign", "hand_frac", "log_len", "lm_agree", "hand_agree", "prob1",
              "prob_margin", "tail_gap"]
LEN_BINS = (12, 24)


def length_bin(n):
    return "<=12" if n <= LEN_BINS[0] else "13-24" if n <= LEN_BINS[1] else ">24"


def open_features(s, T=0.05):
    c = s.get("candidates") or []
    sc = np.array([x.get("support", x.get("score", 0.0)) for x in c[:10]] + [0.0] * max(0, 5 - len(c)), float)
    e = float(T * np.log(np.exp((sc[:10] - sc.max()) / T).sum()) + sc.max()) if len(sc) else 0.0
    L = max(1, s["f1"] - s["f0"])
    return [sc[0], sc[1], sc[4], sc[0] - sc[1], e, s.get("z", 0.0), s.get("null_gap", 0.0), s.get("p_sign", 0.0), s.get("hand_frac", 1.0),
            np.log(L), s.get("lm_agree", 0.0), s.get("hand_agree", 0.0), s.get("prob1", 0.0), s.get("prob_margin", 0.0), s.get("tail_gap", 0.0)]


class Logit:
    """weighted logistic regression (L2), NumPy only"""

    def __init__(self, w=None, b=0.0, mu=None, sd=None):
        self.w, self.b, self.mu, self.sd = w, b, mu, sd

    def fit(self, X, y, sw=None, l2=1e-2, iters=800, lr=0.5):
        X = np.asarray(X, float); y = np.asarray(y, float)
        sw = np.ones(len(y)) if sw is None else np.asarray(sw, float)
        sw = sw / sw.mean()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1]); b = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = sw * (p - y)
            w -= lr * (Z.T @ g / len(y) + l2 * w); b -= lr * g.mean()
        self.w, self.b = w, b
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))

    def to_dict(self):
        return dict(w=self.w.tolist(), b=float(self.b), mu=self.mu.tolist(), sd=self.sd.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["w"]), float(d["b"]), np.array(d["mu"]), np.array(d["sd"]))


def group_of(s, tiers, source="live"):
    w = s.get("nearest")
    return f"{source}|{length_bin(s['f1'] - s['f0'])}|{tiers.get(w, 'tail')}"


def word_probs(s, pc=None):
    """calibrated distribution over the candidates: p(top-1) = calibrated P(correct); the rest (1 − P) spread over the other candidates
    by their temperature softmax ("prob")"""
    c = s.get("candidates") or []
    if not c:
        return {}
    pc = float(s.get("p_correct", 0.0) if pc is None else pc)
    rest = np.array([x.get("prob", 0.0) for x in c[1:]], float)
    rest = rest / rest.sum() if rest.sum() > 0 else rest
    out = {c[0]["word"]: pc}
    for x, r in zip(c[1:], rest):
        out[x["word"]] = out.get(x["word"], 0.0) + (1 - pc) * float(r)
    return out


def conformal_set(s, q, pc=None):
    """LAC prediction set: words whose calibrated probability ≥ 1 − q (q = the group's nonconformity quantile)"""
    wp = word_probs(s, pc)
    return [w for w, p in sorted(wp.items(), key=lambda kv: -kv[1]) if p >= 1.0 - q]


def three_way(segs, cfg, groups, verifier=None, source="live"):
    """cfg = segment_calibration.json "v11" block; groups = list of confusion groups (lists of words).
    KNOWN-SURE = P(known) >= bound AND calibrated P(correct) >= tau of the segment's group (per-group selective risk control: tau is the
    lowest threshold whose out-of-fold accepted error (errors + 1) / (accepted + 1) <= 5 % on TUNE; small groups take the strictest tau)
    AND, for Extended words, prob_margin >= the Extended margin. Otherwise the LAC conformal set (level alpha) decides: 1-3 words ->
    KNOWN-AMBIGUOUS with alternatives (the verifier may resolve a pair inside one confusion group), else UNKNOWN.
    (The pure conformal-singleton rule of the prompt was measured on TUNE: <= 5 % accepted error left only 12-14 % of the correct
    words accepted - artifacts/reports/v11_decision_fit_*.json - so acceptance uses the per-group risk-controlled threshold.)"""
    head = Logit.from_dict(cfg["open_head"])
    tiers = cfg.get("tiers", {})
    gidx = {}
    for k, g in enumerate(groups):
        for w in g:
            gidx.setdefault(w, set()).add(k)
    for s in segs:
        if s.get("status") == "transition" or s.get("kind") == "letter":
            continue
        pk = float(head.predict([open_features(s, cfg.get("energy_T", 0.05))])[0])
        s["p_known"] = round(pk, 4)
        g = group_of(s, tiers, source)
        q = cfg["q"].get(g, cfg["q_strict"])
        tau = cfg.get("tau", {}).get(g, cfg.get("tau_strict", 1.01))
        cs = conformal_set(s, q)
        s["alternatives"] = cs[:3] if cs else [c["word"] for c in (s.get("candidates") or [])[:3]]
        tier = tiers.get(s.get("nearest"), "tail")
        pc = float(s.get("p_correct", 0.0))
        v10_mode = cfg.get("accept_mode") == "v10"       # acceptance = the calibrated v10 policy already applied (status); v11 adds the
        if pk < cfg["known_bound"]:                       # open-set veto, conformal alternatives and the verifier on top
            dec = "UNKNOWN"
        elif (s.get("status") == "accepted") if v10_mode else (pc >= tau and s.get("prob_margin", 1.0) >= cfg.get("tier_margin", {}).get(tier, 0.0)):
            dec = "KNOWN-SURE"
        elif v10_mode and s.get("status") == "unknown":
            dec = "UNKNOWN"
        elif v10_mode and s.get("status") == "uncertain" and cfg.get("keep_uncertain", False):
            dec = "KNOWN-AMBIGUOUS"                       # 4G fix: v10's "uncertain" stays an offered alternative (live words, [?] menu)
            if verifier is not None and len(cs) >= 2 and set.intersection(*[gidx.get(w, set()) for w in cs]):
                pick = verifier(s, cs)
                if pick is not None and pick != s["nearest"]:
                    s["visual_top1"] = s["nearest"]; s["nearest"] = pick
                    s["candidates"] = sorted(s["candidates"], key=lambda x: x["word"] != pick); s["verified"] = True
        elif 1 <= len(cs) <= 3:
            dec = "KNOWN-AMBIGUOUS"
            if verifier is not None and len(cs) >= 2 and set.intersection(*[gidx.get(w, set()) for w in cs]):
                pick = verifier(s, cs)
                if pick is not None:
                    if pick != s["nearest"]:
                        s["visual_top1"] = s["nearest"]; s["nearest"] = pick
                        s["candidates"] = sorted(s["candidates"], key=lambda x: x["word"] != pick)
                    dec = "KNOWN-SURE" if (tier != "extended" and cfg.get("verifier_upgrade", True)) else "KNOWN-AMBIGUOUS"
                    s["verified"] = True
        else:
            dec = "UNKNOWN"
        s["decision"] = dec
        s["status"] = {"KNOWN-SURE": "accepted", "KNOWN-AMBIGUOUS": "uncertain", "UNKNOWN": "unknown"}[dec]
        s["reason"] = f"v11:{dec.lower()}"
    return segs
