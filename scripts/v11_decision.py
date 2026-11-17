"""v11 decision layer fit (prompt-opt-v11 §2B, refit in §3C on the final encoder).

  python main.py v11-decision fit --tag v10 [--bank_v11 '{"lam":0.08,"which":["reader"]}'] [--spot_override '{...}'] --out <version dir>
        [--alpha 0.05] [--fa 0.05]

TUNE only (+ the TUNE Tail and idle sets). Steps:
 1. raw segments of every TUNE set with the version's reader;
 2. temperature (NLL of the true word) → candidate probabilities;
 3. signer-balanced logistic calibration of P(correct) (no class-reliability feature — it hid under-resourced words), 5-fold CV grouped
    by signer, isotonic kept only if CV ECE falls; weights cap every person at 20 % of the TUNE mass;
 4. open-set head (modules/openset.Logit on OPEN_FEATS): positives = segments on in-vocabulary signs (TUNE sets), negatives = Tail-sign
    segments (tail_tune) + idle segments (idle_tune); out-of-fold by signer → AUROC; bound = the lowest P(known) that lets ≤ `fa` of the
    out-of-fold unknown segments through;
 5. per-group conformal quantile q (nonconformity 1 − p(true word), miss rate ≤ alpha), out-of-fold by signer; groups with < 500
    calibration segments take the strictest (largest) q of the large groups (or the pooled q);
 6. Extended-tier margin: the prob_margin at which Extended singletons reach the Core singletons' precision;
→ writes <out>/segment_calibration.json ("conversation" block + "v11") and artifacts/reports/v11_decision_fit_<name>.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "pitty" / "scripts"))

import numpy as np

from decision import TUNE_SETS, Harness, _iou, ece, fit_temperature, load_set, score, REP, _p  # noqa: E402
from modules.decode import FEATS_V9, LogReg, candidate_probs, seg_features  # noqa: E402
from modules.openset import OPEN_FEATS, Logit, group_of, open_features, word_probs  # noqa: E402
from modules.utils import PREP, read_json, write_json  # noqa: E402

FEATS_V11 = [f for f in FEATS_V9 if f not in ("cls_rel", "log_depth")]       # the class-reliability prior is out of the accept decision


def folds_by(groups, k=5, seed=0):
    ug = np.unique(groups); rng = np.random.RandomState(seed); rng.shuffle(ug)
    f = {g: i % k for i, g in enumerate(ug)}
    return np.array([f[g] for g in groups])


def signer_weights(signers, cap=0.2):
    s = np.asarray(signers)
    u, cnt = np.unique(s, return_counts=True)
    share = dict(zip(u, cnt / cnt.sum()))
    w = np.array([min(1.0, cap / share[x]) for x in s])
    return w / w.mean()


def collect(h, names):
    """raw segments of the TUNE sets with gold matching → list of dict(seg, set, signer, true word, correct, in_vocab)"""
    out = []
    for n in names:
        for r in load_set(n):
            segs = h.back(r["st"], r["bank"], raw=True)
            gold = r.get("gold") or []
            sg = str(r.get("signer") or (r["id"].split("@")[0] if n.startswith("t51") else r["id"]))
            for s in segs:
                if s.get("status") == "transition":
                    continue
                best = max(((_iou((s["f0"], s["f1"]), g), j) for j, g in enumerate(gold)), default=(0, None))
                w = r["ref"][best[1]] if best[0] >= 0.3 else None
                out.append(dict(seg=s, set=n, signer=sg, true=w, correct=float(w is not None and s.get("nearest") == w),
                                known=(w is not None and w in h.vocab), utt=r["id"]))
    return out


def cmd_fit(a):
    h = Harness()
    if a.bank_v11:
        h.bank_v11 = json.loads(a.bank_v11); h.bank_v11_forced = True
    b = h.use(a.tag)
    if a.spot_override:
        h.configure(spot=dict(h.pipe.spot, **json.loads(a.spot_override)))
    tiers_all = read_json(PREP / "manifest" / "tiers.json")["words"]
    tiers = {w: v["tier"] for w, v in tiers_all.items()}
    from v11 import GROUPS
    gf = PREP / "manifest" / "confusion_groups_tune.json"
    groups = GROUPS + (read_json(gf)["groups"] if gf.exists() else [])
    D = collect(h, TUNE_SETS)
    U = collect(h, ["tail_tune", "idle_tune"])
    for d in D + U:
        d["raw_seg"] = copy.deepcopy(d["seg"])
    segs = [d["seg"] for d in D]
    _p("TUNE segments %d (known %d) · unknown segments (Tail + idle) %d", len(D), sum(d["known"] for d in D), len(U))
    # 2. temperature on segments whose true word is among the candidates
    gold_rank = np.array([[c["word"] for c in d["seg"]["candidates"]].index(d["true"]) if d["true"] in [c["word"] for c in d["seg"]["candidates"]] else -1
                          for d in D])
    T, nll = fit_temperature(segs, gold_rank)
    candidate_probs([d["seg"] for d in U], T)
    # 3. signer-balanced calibration
    X = seg_features(segs, FEATS_V11); y = np.array([d["correct"] for d in D])
    sg = np.array([d["signer"] for d in D]); w = signer_weights(sg)
    f = folds_by(sg)
    P = np.zeros(len(y))
    for k in range(5):
        m = f == k
        P[m] = Logit().fit(X[~m], y[~m], w[~m]).predict(X[m])
    lr = Logit().fit(X, y, w)
    cv_ece = ece(P, y)
    from decision import _isotonic_fit
    Q = np.zeros_like(P)
    for k in range(5):
        m = f == k
        t = _isotonic_fit(P[~m], y[~m]); Q[m] = np.interp(P[m], t["x"], t["y"])
    use_iso = ece(Q, y) < cv_ece - 0.005
    iso = _isotonic_fit(P, y) if use_iso else None
    _p("calibration: T %.4f · CV ECE %.4f (isotonic %.4f → %s)", T, cv_ece, ece(Q, y), "kept" if use_iso else "not kept")
    # 4. open-set head
    allD = D + U
    Xo = np.array([open_features(d["seg"]) for d in allD]); yo = np.array([float(d["known"]) for d in D] + [0.0] * len(U))
    so = np.array([d["signer"] for d in allD]); wo = signer_weights(so)
    fo = folds_by(so)
    Po = np.zeros(len(yo))
    for k in range(5):
        m = fo == k
        Po[m] = Logit().fit(Xo[~m], yo[~m], wo[~m]).predict(Xo[m])
    head = Logit().fit(Xo, yo, wo)
    pos, neg = Po[yo == 1], Po[yo == 0]
    auroc = float((pos[:, None] > neg[None, :]).mean())
    unk = np.sort(Po[len(D):])
    bound = float(np.quantile(unk, 1 - a.fa)) if len(unk) else 0.5
    _p("open-set head: OOF AUROC %.3f · bound %.3f (≤ %.0f%% of OOF unknown segments pass) · known kept %.3f", auroc, bound, 100 * a.fa,
       float((pos >= bound).mean()))
    # 5. conformal per group, CONDITIONAL on the true word being in the candidate list (top-5 readings: an unconditional 95 % is
    #    impossible when R@5 < 95 %; the unconditional miss is reported); α and the open-set bound are then chosen JOINTLY by
    #    simulating the whole three-way decision out-of-fold (signer folds) on TUNE
    known = [d for d in D if d["known"]]
    inlist = np.array([d["true"] in [c["word"] for c in d["seg"]["candidates"]] for d in known])
    for d, pc in zip(D, P):                                         # out-of-fold calibrated P(correct) of every TUNE segment
        d["pc"] = float(pc)
    Pu = lr.predict(seg_features([d["seg"] for d in U], FEATS_V11)) if U else np.zeros(0)
    for d, pc in zip(U, Pu):
        d["pc"] = float(pc)
    nc = np.array([1.0 - word_probs(d["seg"], d["pc"]).get(d["true"], 0.0) for d in known])
    gk = [group_of(d["seg"], tiers, d["set"].split("_")[0]) for d in known]
    def quantiles(alpha, mask=None):
        mask = np.ones(len(nc), bool) if mask is None else mask
        qs, big = {}, []
        for g in sorted(set(gk)):
            m = np.array([x == g for x in gk]) & inlist & mask
            n = int(m.sum())
            q = float(np.quantile(nc[m], min(1.0, np.ceil((n + 1) * (1 - alpha)) / max(n, 1)))) if n else 1.0
            qs[g] = dict(n=n, q=q)
            if n >= 500:
                big.append(q)
        mm = inlist & mask
        pooled = float(np.quantile(nc[mm], min(1.0, np.ceil((mm.sum() + 1) * (1 - alpha)) / max(mm.sum(), 1))))
        strict = max(big) if big else pooled
        return {g: (v["q"] if v["n"] >= 500 else strict) for g, v in qs.items()}, strict, qs, pooled
    Pk_D, Pk_U = Po[:len(D)], Po[len(D):]
    sgD = np.array([d["signer"] for d in D]); fD = folds_by(sgD)
    gD = [group_of(d["seg"], tiers, d["set"].split("_")[0]) for d in D]
    yD = np.array([d["correct"] for d in D]); pcD = np.array([d["pc"] for d in D])

    def taus(mask, bound, min_n=100, target=0.05):
        """per-group tau: lowest threshold with (errors + 1) / (accepted + 1) <= 5 % among segments that pass the open-set bound"""
        out, big = {}, []
        for g in sorted(set(gD)):
            m = np.array([x == g for x in gD]) & mask & (Pk_D >= bound)
            t = 1.01
            if m.sum() >= min_n:
                for th in np.unique(np.round(pcD[m], 3)):              # lowest threshold whose accepted set (>= 20) meets the bound
                    sel = m & (pcD >= th)
                    if sel.sum() >= 20 and ((1 - yD[sel]).sum() + 1) / (sel.sum() + 1) <= target:
                        t = float(th); break
                big.append(t)
            out[g] = (t, int(m.sum()))
        strict = max(big) if big else 1.01
        return {g: (t if n >= min_n else strict) for g, (t, n) in out.items()}, strict
    best, grid = None, []
    for target, bound in [(t_, b_) for t_ in (0.05, 0.04, 0.03, 0.02) for b_ in np.unique(np.r_[np.quantile(Pk_U, [0.5, 0.8, 0.9, 0.95]), 0.0])]:
        acc = acc_ok = 0; per_g = {}
        for k in range(5):                                           # tau fitted on 4 signer folds, applied to the 5th
            tmap, ts = taus(fD != k, bound, target=target)
            for i_ in np.flatnonzero(fD == k):
                if Pk_D[i_] >= bound and pcD[i_] >= tmap.get(gD[i_], ts):
                    acc += 1; acc_ok += yD[i_]
                    e = per_g.setdefault(gD[i_], [0, 0]); e[0] += 1; e[1] += 1 - yD[i_]
        tmap, ts = taus(np.ones(len(D), bool), bound, target=target)
        gU = [group_of(d["seg"], tiers, "tail") for d in U]
        unk = sum(pk >= bound and d["pc"] >= tmap.get(g, ts) for d, pk, g in zip(U, Pk_U, gU))
        row = dict(target=target, bound=float(bound), correct_accepted=acc_ok / max(yD.sum(), 1), accepted_error=1 - acc_ok / max(acc, 1),
                   unknown_accepted=unk / max(len(U), 1), max_group_error=max([e[1] / e[0] for e in per_g.values() if e[0] >= 30], default=0.0))
        grid.append(row)
        if row["accepted_error"] <= 0.05 and row["unknown_accepted"] <= a.fa and (best is None or row["correct_accepted"] > best["correct_accepted"]):
            best = row
    if best is None:
        best = min(grid, key=lambda r: (r["unknown_accepted"] > a.fa, r["accepted_error"], -r["correct_accepted"]))
        _p("WARNING: no bound meets accepted error <= 5%% and unknown accepted <= %.0f%% on TUNE - safest point kept", 100 * a.fa)
    bound = best["bound"]
    tmap, tau_strict = taus(np.ones(len(D), bool), bound, target=best.get("target", 0.05))
    qmap, q_strict, qs, pooled = quantiles(a.alpha)
    best["alpha"] = a.alpha
    cover = dict(conditional=float(np.mean(nc[inlist] <= np.array([qmap.get(g, q_strict) for g, il in zip(gk, inlist) if il]))),
                 unconditional=float(np.mean(inlist & (nc <= np.array([qmap.get(g, q_strict) for g in gk])))))
    _p("three-way chosen on TUNE (OOF): bound %.3f -> correct accepted %.3f, accepted error %.3f, max group error %.3f, unknown accepted %.3f",
       bound, best["correct_accepted"], best["accepted_error"], best["max_group_error"], best["unknown_accepted"])
    _p("per-group tau %s (strict %.3f), conformal (alpha %.2f) coverage conditional %.3f unconditional %.3f", {g: round(t, 3) for g, t in tmap.items()},
       tau_strict, a.alpha, cover["conditional"], cover["unconditional"])
    # 5b. v10-acceptance mode (C5: the strict per-group rule above cuts coverage beyond tolerance): KNOWN-SURE = the v10 policy's accepted
    #     words; the open-set head only vetoes, with the largest bound whose OOF correct-accepted loss is <= 1 point
    from modules.decode import class_features, decide as v10_decide

    def v10_status(seg_list):
        """the exact v10 decision path on COPIES of the raw segments (v10 temperature, class features, LR, isotonic, decode.decide)"""
        cp = copy.deepcopy(seg_list)
        cal10 = h.pipe.cal
        if cal10.get("temperature"):
            candidate_probs(cp, cal10["temperature"])
        class_features(cp, h.pipe.concepts, h.pipe.class_prior)
        Pv = h.pipe.lr.predict(seg_features(cp, getattr(h.pipe.lr, "feats", None))) if cp else np.zeros(0)
        if cal10.get("isotonic") and len(Pv):
            Pv = np.interp(Pv, cal10["isotonic"]["x"], cal10["isotonic"]["y"])
        v10_decide(cp, Pv, cal10)
        return np.array([x["status"] == "accepted" for x in cp])
    raw_D = [d.get("raw_seg", d["seg"]) for d in D]
    acc10, accU = v10_status(raw_D), v10_status([d.get("raw_seg", d["seg"]) for d in U])
    base_ca = float((acc10 & (yD == 1)).sum() / max(yD.sum(), 1)); base_ua = float(accU.mean()) if len(accU) else 0.0
    base_err = float(1 - yD[acc10].mean()) if acc10.any() else 0.0
    v10_rows, v10_best = [], 0.0
    for b_ in np.unique(np.r_[0.0, np.quantile(Pk_D, np.linspace(0.01, 0.3, 30))]):
        ca = float((acc10 & (Pk_D >= b_) & (yD == 1)).sum() / max(yD.sum(), 1)); ua = float((accU & (Pk_U >= b_)).mean()) if len(accU) else 0.0
        er = float(1 - yD[acc10 & (Pk_D >= b_)].mean()) if (acc10 & (Pk_D >= b_)).any() else 0.0
        v10_rows.append(dict(bound=float(b_), correct_accepted=ca, unknown_accepted=ua, accepted_error=er))
        if base_ca - ca <= 0.01:
            v10_best = max(v10_best, float(b_))
    vb = next(r for r in v10_rows if r["bound"] == v10_best)
    _p("v10-acceptance mode: base correct accepted %.3f (error %.3f, unknown accepted %.3f) → open-set veto bound %.3f: correct accepted %.3f, error %.3f, unknown accepted %.3f",
       base_ca, base_err, base_ua, v10_best, vb["correct_accepted"], vb["accepted_error"], vb["unknown_accepted"])
    # 6. Extended margin: smallest prob_margin where Extended singletons reach Core precision
    tier_margin = {"core": 0.0, "extended": 0.0}
    cfg = dict(open_head=head.to_dict(), known_bound=bound, q=qmap, q_strict=q_strict, tau=tmap, tau_strict=tau_strict, tiers=tiers,
               groups=groups, tier_margin=tier_margin, energy_T=0.05, alpha=a.alpha, fa=a.fa)
    if a.accept_mode == "v10":
        cfg.update(accept_mode="v10", known_bound=v10_best, strict_rule_measured=best)
    core_ok = [d["correct"] for d in D if tiers.get(d["seg"].get("nearest")) == "core" and d["seg"].get("prob1", 0) >= 1 - q_strict]
    ext = [(d["seg"].get("prob_margin", 0.0), d["correct"]) for d in D if tiers.get(d["seg"].get("nearest")) == "extended" and d["seg"].get("prob1", 0) >= 1 - q_strict]
    target = float(np.mean(core_ok)) if core_ok else 0.95
    for mgn in np.linspace(0, 0.9, 19):
        sel = [c for pm, c in ext if pm >= mgn]
        if len(sel) >= 20 and np.mean(sel) >= target:
            tier_margin["extended"] = float(mgn); break
    cal = copy.deepcopy(h.pipe.cal)
    cal["tau_accept"] = read_json(b["dir"] / "segment_calibration.json" if (b["dir"] / "segment_calibration.json").exists() else ROOT / "models" / "model-v10" / "segment_calibration.json")["conversation"].get("tau_accept", cal.get("tau_accept"))   # keep the SHIPPED threshold (the pipeline floors it at accept_min)
    if a.accept_mode == "v10":                      # the v10 calibration (temperature, LR, isotonic, thresholds) stays exactly as shipped
        cal.update(v11=cfg)
    else:
        cal.update(temperature=T, model=dict(lr_to_logreg(lr, FEATS_V11)), isotonic=iso, v11=cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc_path = out / "segment_calibration.json"
    full = read_json(sc_path) if sc_path.exists() else {}
    full["conversation"] = cal
    write_json(sc_path, full)
    rep = dict(tag=a.tag, out=str(out), T=T, nll=nll, cv_ECE=cv_ece, isotonic=bool(use_iso), open_auroc=auroc, known_bound=bound,
               known_kept=float((pos >= bound).mean()), conformal=dict(pooled=pooled, strict=q_strict, groups=qs, coverage=cover),
               chosen=best, grid=grid, v10_mode=dict(base_correct_accepted=base_ca, base_error=base_err, base_unknown_accepted=base_ua, bound=v10_best, rows=v10_rows),
               tier_margin=tier_margin, n_segments=len(D), n_unknown=len(U))
    write_json(REP / f"v11_decision_fit_{out.name}.json", rep)
    _p("written %s (v11 block) + report", sc_path)


def lr_to_logreg(lr, feats):
    """modules.openset.Logit → the dict format of modules.decode.LogReg (so SignPipeline loads it unchanged)"""
    m = LogReg()
    m.w = np.asarray(lr.w, float); m.b = float(lr.b); m.mu = np.asarray(lr.mu, float); m.sd = np.asarray(lr.sd, float)
    return m.to_dict(feats)


def cmd_segtune(a):
    """Path-B decoder settings on TUNE by coordinate descent (the semi-Markov decoder's thresholds were tuned for the v10 tagger / span head;
    a new front end needs its own). One Harness, Path-B front cached in memory. Objective = mean TUNE WER; every row is logged."""
    from decision import aggregate, evaluate
    h = Harness()
    b = h.use(a.tag)
    rows = {n: load_set(n) for n in TUNE_SETS}
    spot0 = dict(b["spot"])
    grid = json.loads(a.grid) if a.grid else {"act_thr": [0.3, 0.4, 0.5, 0.6, 0.7], "lam": [1.0, 2.0, 4.0, 6.0, 8.0], "mu_null": [0.0, 1.0, 2.5, 5.0],
                                              "min_len": [6, 10, 14], "gap_cost": [0.0, 0.5, 1.0]}
    log_rows = []

    def run(sp):
        h.configure(spot=sp)
        res = {n: aggregate(evaluate(h, rows[n])) for n in TUNE_SETS}
        obj = float(np.mean([res[n]["WER"] for n in TUNE_SETS]))
        log_rows.append(dict(spot={k: sp.get(k) for k in grid}, obj=obj, **{n: round(res[n]["WER"], 4) for n in TUNE_SETS}))
        _p("[segtune] %s obj %.4f | %s", {k: sp.get(k) for k in grid}, obj, " ".join(f"{n.replace('_tune', '')}:{res[n]['WER']:.3f}" for n in TUNE_SETS))
        return obj
    best_sp, best = dict(spot0), run(spot0)
    for _round in range(a.rounds):
        for k, vals in grid.items():
            for v in vals:
                if best_sp.get(k) == v:
                    continue
                sp = dict(best_sp, **{k: v})
                o = run(sp)
                if o < best - 0.002:
                    best, best_sp = o, sp
    out = Path(a.out) / "spotting.json"
    full = read_json(out); full["conversation"] = best_sp; write_json(out, full)
    write_json(REP / f"v11_segtune_{Path(a.out).name}.json", dict(tag=a.tag, best=best, best_spot={k: best_sp.get(k) for k in grid}, rows=log_rows))
    _p("[segtune] chosen %s obj %.4f → %s", {k: best_sp.get(k) for k in grid}, best, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--tag", default="v10"); f.add_argument("--out", required=True)
    f.add_argument("--bank_v11", default=""); f.add_argument("--spot_override", default="")
    f.add_argument("--alpha", type=float, default=0.10, help="conformal level of the AMBIGUOUS sets (alternatives)")
    f.add_argument("--fa", type=float, default=0.05)
    f.add_argument("--accept_mode", default="v10", choices=["v10", "strict"], help="v10 = keep the v10 acceptance policy (C5 coverage guard)")
    g = sub.add_parser("segtune"); g.add_argument("--tag", required=True); g.add_argument("--out", required=True)
    g.add_argument("--grid", default=""); g.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    dict(fit=cmd_fit, segtune=cmd_segtune)[a.cmd](a)


if __name__ == "__main__":
    main()
