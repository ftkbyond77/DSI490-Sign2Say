"""L3b: spotting decoder over window tables (CPU) + metrics + segment-correctness calibration.

gain(window) = z(top-1 concept vs the whole vocabulary) − λ
             − μ · max(0, sim(window, null prototype) − top-1 score + δ)        ← "looks more like rest/transition than any word"
             + β · P_begin(tagger, window start)                                  ← boundary evidence from the tagger
Windows are only proposed where the tagger says "signing" (and a signer is visible). A semi-Markov DP picks non-overlapping
windows maximising Σ gain − γ·(uncovered signing frames); neighbours with the same concept are merged; each segment's
candidates are a temporal ensemble (Σ scores of every window inside it).

Segment status uses P(correct) from a logistic model on held-out tune data (features: z, score, margin, null gap, tagger
sign probability, ensemble agreement, length): accepted ≥ τ_acc · uncertain ≥ τ_unc · otherwise "[?] ≈ nearest".
"""
from __future__ import annotations

import numpy as np


def active_from_prob(prob, thr=0.5, present=None, off=None):
    """Signing frames = smoothed tagger P(sign) > thr. off < thr (v8): hysteresis — a signing run starts above thr and continues
    while P(sign) stays above off, so a dip inside one sign does not open and close a segment."""
    p = np.convolve(prob[:, 1].astype(np.float32) + prob[:, 2].astype(np.float32), np.ones(5) / 5, "same")
    act = p > thr
    if off is not None and off < thr and act.any():
        keep = p > off
        idx = np.flatnonzero(np.diff(np.r_[0, keep.astype(np.int8), 0]))
        for a, b in zip(idx[::2], idx[1::2]):          # runs of p > off that contain at least one frame above thr
            if act[a:b].any():
                act[a:b] = True
    if present is not None:
        act &= present
    return act, p


def dp_select(wins, gain, act, gap_cost, stride=2):
    T = len(act)
    nodes = sorted(set(range(0, T + 1, stride)) | {T} | {b for _, b in wins} | {a for a, _ in wins})
    pos = {n: i for i, n in enumerate(nodes)}
    ending = {}
    for w, (a, b) in enumerate(wins):
        ending.setdefault(b, []).append((a, w))
    cum = np.concatenate([[0], np.cumsum(act)])
    f = np.full(len(nodes), -1e18); f[0] = 0.0
    back = [None] * len(nodes)
    for i in range(1, len(nodes)):
        n, pn = nodes[i], nodes[i - 1]
        f[i] = f[i - 1] - gap_cost * (cum[n] - cum[pn]); back[i] = ("gap", i - 1, None)
        for a, w in ending.get(n, []):
            v = f[pos[a]] + gain[w]
            if v > f[i]:
                f[i] = v; back[i] = ("win", pos[a], w)
    picks, i = [], len(nodes) - 1
    while i > 0:
        kind, j, w = back[i]
        if kind == "win":
            picks.append(w)
        i = j
    return picks[::-1]


def apply_csls(table, hub, alpha, prior=None, gamma=0.0):
    """Re-rank each window's stored top-K candidates (K = 50) by
         score − α·hub[concept]  (CSLS hubness correction: concepts that attract everything are discounted)
               + γ·prior[concept] (Thai word-frequency prior: everyday words beat rare dictionary entries on near-ties)."""
    use_h = hub is not None and alpha
    use_p = prior is not None and gamma
    if not (use_h or use_p) or len(table["wins"]) == 0:
        return table
    ti, ts = table["top_idx"], table["top_sc"].astype(np.float32)
    adj = ts.copy()
    shift = 0.0
    if use_h:
        adj -= alpha * hub[ti]; shift -= alpha * float(hub.mean())
    if use_p:
        adj += gamma * prior[ti]; shift += gamma * float(prior.mean())
    o = np.argsort(-adj, axis=1)
    t = dict(table)
    t["top_idx"], t["top_sc"] = np.take_along_axis(ti, o, 1), np.take_along_axis(adj, o, 1)
    t["mu"] = table["mu"] + shift
    return t


def concept_prior(freq_prior, depth, gamma=0.0, delta=0.0):
    """Word prior added to every concept score: γ·(Thai corpus log-frequency) + δ·(log bank depth = how many signers know it),
    both scaled to [0, 1]. Everyday, widely-signed words win near-ties against rare dictionary entries; visual evidence still decides."""
    pr = np.zeros(len(freq_prior) if freq_prior is not None else len(depth), np.float32)
    if freq_prior is not None and gamma:
        pr += gamma * freq_prior
    if depth is not None and delta:
        pr += delta * (np.log1p(depth) / np.log1p(max(float(depth.max()), 1.0))).astype(np.float32)
    return pr if (gamma or delta) else None


def frequency_prior(concepts, groups, freq):
    """Per-concept prior in [0, 1] = log(1 + corpus frequency of the most frequent member of the synonym group) / log(1 + max)."""
    f = np.zeros(len(concepts), np.float32)
    for i, c in enumerate(concepts):
        members = groups.get(c, [c]) + [c]
        f[i] = max(freq.get(m, 0) for m in members)
    return (np.log1p(f) / np.log1p(max(f.max(), 1))).astype(np.float32)


def decode(table, prob, concepts, cfg, present=None, depth=None, rank_table=None):
    """→ list of segments dict(f0, f1, nearest, candidates[{word, score, support}], z, score, margin, null_gap, p_sign, agree, depth).
    table       drives SEGMENTATION (where signs are): visual scores (+ CSLS) only.
    rank_table  drives RECOGNITION (which word): the same windows with word priors added; default = table.
                Keeping priors out of segmentation stops them from inflating window gains (over-splitting signs).
    depth [C] = number of signers behind each concept in the bank (calibration feature: shallow words are less reliable)."""
    wins_all = table["wins"]
    rt = rank_table if rank_table is not None else table
    if len(wins_all) == 0:
        return []
    T = len(prob)
    act, p_sign = active_from_prob(prob, cfg["act_thr"], present, cfg.get("act_off"))
    W = np.asarray(wins_all)
    L = W[:, 1] - W[:, 0]
    cum = np.concatenate([[0], np.cumsum(act)])
    inside = (cum[W[:, 1]] - cum[W[:, 0]]) / np.maximum(L, 1)
    keep = (L >= cfg["min_len"]) & (inside >= cfg.get("min_inside", 0.6))
    if not keep.any():
        return []
    top1, top2 = table["top_sc"][:, 0], table["top_sc"][:, 1]
    z = (top1 - table["mu"]) / table["sd"]
    null_gap = table["null"] - top1
    gain = z - cfg["lam"] - cfg["mu_null"] * np.maximum(0.0, null_gap + cfg.get("delta", 0.0))
    if cfg.get("beta", 0):
        pb = np.maximum.reduce([np.roll(prob[:, 1].astype(np.float32), k) for k in (-2, -1, 0, 1, 2)])
        gain = gain + cfg["beta"] * pb[W[:, 0]]
    idx = np.flatnonzero(keep)
    picks = [int(idx[j]) for j in dp_select([wins_all[i] for i in idx], gain[idx], act, cfg["gap_cost"])]
    segs = []
    for w in picks:
        a, b = wins_all[w]
        wi = int(table["top_idx"][w, 0])
        if segs and segs[-1]["_wi"] == wi and a - segs[-1]["f1"] <= 2:
            s = segs[-1]; s["f1"] = int(b); s["wins"].append((int(a), int(b)))
            if z[w] > s["z"]:
                s.update(z=float(z[w]), _w=w)
            continue
        segs.append(dict(f0=int(a), f1=int(b), _wi=wi, _w=w, z=float(z[w]), wins=[(int(a), int(b))]))
    out = []
    for s in segs:
        w = s.pop("_w"); s.pop("_wi"); wi = int(rt["top_idx"][w, 0])
        cands = ensemble(rt, s["f0"], s["f1"], cfg["min_len"], concepts, w)
        s.update(nearest=cands[0]["word"], candidates=cands, score=float(table["top_sc"][w, 0]), margin=float(table["top_sc"][w, 0] - table["top_sc"][w, 1]),
                 null_gap=float(null_gap[w]), p_sign=float(p_sign[s["f0"]:s["f1"]].mean()), agree=float(concepts[wi] == cands[0]["word"]),
                 support_ratio=float(cands[0]["support"] / max(cands[1]["support"], 1e-6)) if len(cands) > 1 else 5.0, window_top1=concepts[wi],
                 depth=float(depth[concepts.index(cands[0]["word"])]) if depth is not None and cands[0]["word"] in concepts else 0.0)
        out.append(s)
    return out


def ensemble(rt, f0, f1, min_len, concepts, w=None):
    """Candidates of a segment = temporal ensemble of every window inside it: Σ scores per concept (support = mean), best score."""
    W = np.asarray(rt["wins"]).reshape(-1, 2)
    L = W[:, 1] - W[:, 0]
    ti, ts = rt["top_idx"][:, :5], rt["top_sc"][:, :5]
    m = (W[:, 0] >= f0 - 2) & (W[:, 1] <= f1 + 2) & (L >= min_len)
    n_in = int(m.sum())
    if not n_in:
        if w is None:
            return []
        return [dict(word=concepts[int(j)], score=round(float(x), 4), support=round(float(x), 4)) for j, x in zip(ti[w], ts[w])]
    ids, vals = ti[m].ravel(), ts[m].ravel()
    uniq, inv = np.unique(ids, return_inverse=True)
    acc = np.zeros(len(uniq)); np.add.at(acc, inv, vals)
    best = np.full(len(uniq), -9.0); np.maximum.at(best, inv, vals)
    o = np.argsort(-acc)[:5]
    return [dict(word=concepts[int(uniq[k])], score=round(float(best[k]), 4), support=round(float(acc[k]) / n_in, 4)) for k in o]


MERGE_FEATS = ["pb", "rest", "no_cut", "speed_dip", "len_short", "len_long", "member", "d_best", "d_member", "same", "gap", "s_short", "s_long"]


def pair_features(a, b, row, concepts, pb, rest, speed):
    """Features of two neighbouring segments for "is this one sign cut in two?": tagger begin probability and rest probability at
    the cut, the hand-speed dip there (signers slow down between words), part lengths, whether the union's reading is one of the
    parts' words, how much the union score drops vs the parts, same word, gap."""
    T = len(pb)
    c = min(b["f0"], T - 1)
    lo, hi = max(0, a["f1"] - 3), min(T, b["f0"] + 4)
    u = concepts[int(np.argmax(row))]
    su = float(row.max())
    mem = a if u == a["nearest"] else b if u == b["nearest"] else None
    sp_med = float(np.median(speed)) + 1e-3
    la, lb = a["f1"] - a["f0"], b["f1"] - b["f0"]
    r_ = float(rest[lo:hi].max()) if hi > lo else 0.0
    sh, lg = (a, b) if la <= lb else (b, a)
    return [float(pb[c]), r_, (1.0 - float(pb[c])) * (1.0 - r_), float(speed[lo:hi].min() / sp_med) if hi > lo else 1.0,
            min(la, lb) / 25.0, max(la, lb) / 25.0, float(mem is not None), su - max(a["score"], b["score"]),
            su - mem["score"] if mem is not None else -0.5, float(a["nearest"] == b["nearest"]), (b["f0"] - a["f1"]) / 25.0,
            float(sh["score"]), float(lg["score"])]


def merge_split(segs, embed, bank, null_proto, concepts, prob, rt, cfg, speed=None, collect=None):
    """(v8) One sign cut into two segments ("แมว" → "กิน แมว": the onset of แมว near the mouth looked like กิน). Every pair of
    neighbouring segments (gap ≤ max_gap, union ≤ max_len frames) gets P(same sign) from a small logistic model (MERGE_FEATS: the
    tagger's begin / rest evidence at the cut, the hand-speed dip, lengths, and how the union span — embedded as one piece — reads
    compared to the parts), fitted on tune sentences + whole isolated clips (scripts/decision.py). Pairs above thr merge, best
    first, until none is left; the merged segment takes the union's reading. Repeated words of a sentence are NOT merged by name:
    two full signs with a pause / new onset between them score low.
    collect: a list → every candidate pair's features are appended (no merging) — the training data."""
    mc = cfg.get("merge")
    if not mc or not mc.get("on", True) or len(segs) < 2 or (collect is None and not mc.get("model")):
        return segs
    T = len(prob)
    pb = np.maximum.reduce([np.roll(prob[:, 1].astype(np.float32), k) for k in (-2, -1, 0, 1, 2)])
    rest = prob[:, 0].astype(np.float32)
    speed = np.ones(T, np.float32) if speed is None else np.asarray(speed, np.float32)
    _, p_sign = active_from_prob(prob, cfg["act_thr"], None, cfg.get("act_off"))
    lr = LogReg.from_dict(mc["model"]) if mc.get("model") else None
    while len(segs) >= 2:
        idx = [i for i in range(len(segs) - 1)
               if segs[i + 1]["f0"] - segs[i]["f1"] <= mc.get("max_gap", 6) and segs[i + 1]["f1"] - segs[i]["f0"] <= mc.get("max_len", 72)]
        if not idx:
            break
        Z = embed([(segs[i]["f0"], segs[i + 1]["f1"]) for i in idx])
        S = bank.scores(Z)
        X = np.array([pair_features(segs[i], segs[i + 1], S[k], concepts, pb, rest, speed) for k, i in enumerate(idx)], np.float32)
        if collect is not None:
            for k, i in enumerate(idx):
                collect.append(dict(x=X[k].tolist(), a=(segs[i]["f0"], segs[i]["f1"]), b=(segs[i + 1]["f0"], segs[i + 1]["f1"])))
            return segs
        P = lr.predict(X)
        k = int(np.argmax(P))
        if P[k] < mc.get("thr", 0.5):
            break
        i = idx[k]
        a, b = segs[i], segs[i + 1]
        row, valid = S[k], S[k] > -1.5
        cix = {c: j for j, c in enumerate(concepts)}
        ja, jb = cix[a["nearest"]], cix[b["nearest"]]
        o = np.argsort(-row)[:5]
        top = ja if row[ja] >= row[jb] else jb            # the merged sign is one of its parts' words: whichever the union reads better
        o = np.r_[top, [j for j in o if j != top]][:5]
        mu, sd = float(row[valid].mean()), float(row[valid].std() + 1e-6)
        f0, f1 = a["f0"], b["f1"]
        u = concepts[int(o[0])]
        cands = ensemble(rt, f0, f1, cfg["min_len"], concepts)
        lead = next((c for c in cands if c["word"] == u), None)
        if lead is None:
            lead = dict(word=u, score=round(float(row[o[0]]), 4), support=round(float(row[o[0]]), 4))
        else:
            lead = dict(lead, support=max(lead["support"], max([c["support"] for c in cands])))   # the merged reading leads the ranking
        cands = [lead] + [c for c in cands if c["word"] != u][:4]
        if len(cands) < 2:
            cands += [dict(word=concepts[int(j)], score=round(float(row[j]), 4), support=round(float(row[j]), 4)) for j in o[1:5]]
        segs = segs[:i] + [dict(a, f0=int(f0), f1=int(f1), wins=a.get("wins", []) + b.get("wins", []), nearest=u, candidates=cands,
                                score=max(float(row[o[0]]), a["score"] if a["nearest"] == u else 0.0, b["score"] if b["nearest"] == u else 0.0),
                                margin=float(row[o[0]] - row[o[1]]), z=max(float((row[o[0]] - mu) / sd), a["z"] if a["nearest"] == u else -9.0,
                                b["z"] if b["nearest"] == u else -9.0), null_gap=float(Z[k] @ null_proto - row[o[0]]),
                                p_sign=float(p_sign[f0:f1].mean()), agree=1.0, window_top1=u, merged=a.get("merged", 1) + b.get("merged", 1),
                                p_merge=float(P[k]), support_ratio=float(cands[0]["support"] / max(cands[1]["support"], 1e-6)))] + segs[i + 2:]
    return segs


SPLIT_FEATS = ["pb", "rest", "speed_dip", "len_a", "len_b", "s_whole", "s_a", "s_b", "gain", "differ", "len_whole"]


def _reading(row, f0, f1, Zrow, null_proto, concepts, rt, cfg, p_sign):
    """A segment dict for span [f0, f1) read from its bank-score row (used by split_long)."""
    valid = row > -1.5
    o = np.argsort(-row)[:5]
    mu, sd = float(row[valid].mean()), float(row[valid].std() + 1e-6)
    cands = ensemble(rt, f0, f1, cfg["min_len"], concepts) or [dict(word=concepts[int(j)], score=round(float(row[j]), 4), support=round(float(row[j]), 4)) for j in o]
    return dict(f0=int(f0), f1=int(f1), wins=[(int(f0), int(f1))], nearest=cands[0]["word"], candidates=cands, score=float(row[o[0]]),
                margin=float(row[o[0]] - row[o[1]]), z=float((row[o[0]] - mu) / sd), null_gap=float(Zrow @ null_proto - row[o[0]]),
                p_sign=float(p_sign[f0:f1].mean()), agree=float(concepts[int(o[0])] == cands[0]["word"]), window_top1=concepts[int(o[0])],
                support_ratio=float(cands[0]["support"] / max(cands[1]["support"], 1e-6)) if len(cands) > 1 else 5.0, split=True)


def split_long(segs, embed, bank, null_proto, concepts, prob, rt, cfg, speed=None, collect=None):
    """(v8.1) The opposite of merge_split: two fluent signs read as one segment (no rest between them, so the tagger saw no
    boundary — fast / natural signing). For every long segment, the 3 cut points with the most boundary evidence (tagger begin
    probability, hand-speed dip) are tried: both parts and the whole are embedded (span head) and read against the bank. A small
    logistic model (SPLIT_FEATS, fitted on tune data by scripts/decision.py) gives P(two signs); the best cut above thr splits the
    segment (one split per segment). collect: a list → every candidate cut's features are appended (no splitting)."""
    sc = cfg.get("split")
    if not sc or not segs or (collect is None and not sc.get("model")) or len(segs) < sc.get("min_segments", 1):
        return segs                      # min_segments 2: a one-sign utterance (isolated word) is never split
    T = len(prob)
    pb = np.maximum.reduce([np.roll(prob[:, 1].astype(np.float32), k) for k in (-2, -1, 0, 1, 2)])
    rest = prob[:, 0].astype(np.float32)
    speed = np.ones(T, np.float32) if speed is None else np.asarray(speed, np.float32)
    sp_med = float(np.median(speed)) + 1e-3
    _, p_sign = active_from_prob(prob, cfg["act_thr"], None, cfg.get("act_off"))
    lr = LogReg.from_dict(sc["model"]) if sc.get("model") else None
    m_part, m_len = sc.get("min_part", 8), sc.get("min_len", 24)
    out = []
    for s in segs:
        f0, f1 = s["f0"], s["f1"]
        if f1 - f0 < m_len:
            out.append(s); continue
        cuts = np.arange(f0 + m_part, f1 - m_part + 1, 2)
        dip = np.array([speed[max(0, c - 2):c + 3].min() / sp_med for c in cuts])
        top = cuts[np.argsort(-(pb[cuts] - 0.5 * dip))[:3]]
        Z = embed([(f0, int(c)) for c in top] + [(int(c), f1) for c in top] + [(f0, f1)])
        S = bank.scores(Z)
        n = len(top)
        sw = float(S[-1].max())
        X = []
        for k, c in enumerate(top):
            sa, sb = float(S[k].max()), float(S[n + k].max())
            X.append([float(pb[c]), float(rest[max(0, c - 3):c + 4].max()), float(speed[max(0, c - 2):c + 3].min() / sp_med),
                      (c - f0) / 25.0, (f1 - c) / 25.0, sw, sa, sb, (sa + sb) / 2 - sw, float(np.argmax(S[k]) != np.argmax(S[n + k])), (f1 - f0) / 25.0])
        X = np.array(X, np.float32)
        if collect is not None:
            collect += [dict(x=X[k].tolist(), a=(f0, int(c)), b=(int(c), f1)) for k, c in enumerate(top)]
            out.append(s); continue
        P = lr.predict(X)
        k = int(np.argmax(P))
        if P[k] < sc.get("thr", 0.5):
            out.append(s); continue
        c = int(top[k])
        out += [dict(_reading(S[k], f0, c, Z[k], null_proto, concepts, rt, cfg, p_sign), p_split=float(P[k])),
                dict(_reading(S[n + k], c, f1, Z[n + k], null_proto, concepts, rt, cfg, p_sign), p_split=float(P[k]))]
    return out


# ----------------------------------------------------------------------------- metrics
def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(max(a[1], b[1]) - min(a[0], b[0]), 1)


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(len(ref), 1)


def sentence_metrics(segs, gt_segs, gt_words, vocab_set=None):
    """gt_segs [(f0,f1)] (may be pseudo), gt_words [concept]. Segment F1@IoU0.3, word recall/precision (IoU-matched), WER on sequences."""
    ps = [(s["f0"], s["f1"]) for s in segs]
    used, tp = set(), 0
    match = [None] * len(segs)
    for i, p in enumerate(ps):
        best = max(((j, iou(p, g)) for j, g in enumerate(gt_segs) if j not in used), key=lambda x: x[1], default=(None, 0))
        if best[1] >= 0.3:
            used.add(best[0]); tp += 1; match[i] = best[0]
    seg_p, seg_r = tp / max(len(ps), 1), tp / max(len(gt_segs), 1)
    correct = [m is not None and segs[i]["nearest"] == gt_words[m] for i, m in enumerate(match)]
    inv = [j for j, w in enumerate(gt_words) if vocab_set is None or w in vocab_set]
    hit1 = sum(any(m == j and segs[i]["nearest"] == gt_words[j] for i, m in enumerate(match)) for j in inv)
    hit5 = sum(any(m == j and gt_words[j] in [c["word"] for c in segs[i]["candidates"]] for i, m in enumerate(match)) for j in inv)
    bag_hit = sum(w in {s["nearest"] for s in segs} for w in gt_words)
    return dict(n_pred=len(ps), n_gt=len(gt_segs), seg_tp=tp, seg_f1=2 * seg_p * seg_r / max(seg_p + seg_r, 1e-9), count_err=abs(len(ps) - len(gt_segs)),
                n_inv=len(inv), hit1=hit1, hit5=hit5, n_correct=int(sum(correct)), correct=correct, bag_hit=bag_hit,
                wer=wer(list(gt_words), [s["nearest"] for s in segs]))


def aggregate(rows):
    n_inv = sum(r["n_inv"] for r in rows); n_pred = sum(r["n_pred"] for r in rows); n_gt = sum(r["n_gt"] for r in rows)
    rec1 = sum(r["hit1"] for r in rows) / max(n_inv, 1); prec1 = sum(r["n_correct"] for r in rows) / max(n_pred, 1)
    return dict(n_sentences=len(rows), seg_f1=float(np.mean([r["seg_f1"] for r in rows])), count_abs_err=float(np.mean([r["count_err"] for r in rows])),
                exact_count_rate=float(np.mean([r["count_err"] == 0 for r in rows])), words_pred=n_pred / max(len(rows), 1), words_gt=n_gt / max(len(rows), 1),
                word_recall_top1=rec1, word_recall_top5=sum(r["hit5"] for r in rows) / max(n_inv, 1), word_precision_top1=prec1,
                spot_f1=2 * rec1 * prec1 / max(rec1 + prec1, 1e-9), bag_recall=sum(r["bag_hit"] for r in rows) / max(n_gt, 1),
                WER=float(np.mean([r["wer"] for r in rows])))


# ----------------------------------------------------------------------------- calibration
FEATS = ["z", "score", "margin", "null_gap", "p_sign", "agree", "support_ratio", "len"]
FEATS_DEPTH = FEATS + ["log_depth", "z_x_log_depth"]      # per-bank-depth calibration (prompt_7 bottleneck 3)
# v8: + temperature-scaled candidate probabilities (top-1, top-1 − top-2) and the word's own reliability in the bank
FEATS_V8 = FEATS + ["prob1", "prob_margin", "cls_rel", "log_depth", "merged"]
# v8.1: + the hand-shape cue's opinion of the chosen word (z over the vocabulary, and whether the cue alone ranks it first)
FEATS_V81 = FEATS_V8 + ["hand_z", "hand_agree"]
# v9: + whether the second encoder (Thai landmark encoder) alone ranks the chosen word first
FEATS_V9 = FEATS_V81 + ["lm_agree"]


def seg_features(segs, feats=None):
    feats = feats or FEATS
    rows = []
    for s in segs:
        ld = float(np.log1p(s.get("depth", 0.0)))
        v = dict(z=s["z"], score=s["score"], margin=s["margin"], null_gap=s["null_gap"], p_sign=s["p_sign"], agree=s["agree"],
                 support_ratio=min(s["support_ratio"], 5.0), len=(s["f1"] - s["f0"]) / 25.0, log_depth=ld, z_x_log_depth=s["z"] * ld,
                 prob1=s.get("prob1", 0.0), prob_margin=s.get("prob_margin", 0.0), cls_rel=s.get("cls_rel", 0.5), merged=float(s.get("merged", 1) > 1),
                 hand_z=min(max(s.get("hand_z", 0.0), -3.0), 6.0), hand_agree=s.get("hand_agree", 0.0), lm_agree=s.get("lm_agree", 0.5))
        rows.append([v[f] for f in feats])
    return np.array(rows, np.float32).reshape(-1, len(feats))


def candidate_probs(segs, temperature):
    """Softmax over a segment's candidate scores (cosine similarities to the bank) at a temperature fitted on tune data (NLL of the
    true word) → each candidate gets "prob"; the segment gets prob1, prob2 and prob_margin. Raw cosine scores (0.6 vs 0.58) do not
    say how far apart two readings are; these do — a near-tie (0.41 / 0.39) and a clear lead (0.70 / 0.10) read differently."""
    for s in segs:
        c = s.get("candidates") or []
        if not c:
            continue
        x = np.array([float(k["support"]) for k in c]) / max(float(temperature), 1e-3)     # support = the ranking quantity
        e = np.exp(x - x.max()); pr = e / e.sum()
        for k, p in zip(c, pr):
            k["prob"] = round(float(p), 4)
        s["prob1"], s["prob2"] = float(pr[0]), float(pr[1]) if len(pr) > 1 else 0.0
        s["prob_margin"] = s["prob1"] - s["prob2"]
    return segs


def candidate_matrix(s, concepts=None, prior=None):
    """[K, F] features of a segment's candidates for the learned re-ranker: best window score, ensemble support (both relative to
    the leader), rank one-hot, the word's reliability and signers in the bank."""
    c = s.get("candidates") or []
    if not c:
        return np.zeros((0, 10), np.float32)
    sc0, su0 = float(c[0]["score"]), float(c[0]["support"])
    cidx = {w: i for i, w in enumerate(concepts)} if concepts is not None else {}
    rows = []
    for k, x in enumerate(c[:5]):
        i = cidx.get(x["word"])
        rel = float(prior["rel"][i]) if prior and i is not None else 0.5
        dep = float(np.log1p(prior["depth"][i])) if prior and i is not None else 0.0
        oh = [float(k == j) for j in range(1, 5)]
        rows.append([float(x["score"]) - sc0, float(x["support"]) - su0, float(x["score"]), *oh, rel, dep, rel * float(x["support"])])
    return np.array(rows, np.float32)


def rerank_candidates(segs, w, concepts, prior):
    """v8 learned re-ranker (conditional logit over the top-5, fitted on tune segments): may move a lower candidate to the top
    when its evidence pattern says so (e.g. clearly better ensemble support, a word that is reliable in the bank)."""
    if w is None:
        return segs
    w = np.asarray(w, np.float32)
    for s in segs:
        X = candidate_matrix(s, concepts, prior)
        if len(X) < 2:
            continue
        u = X @ w
        k = int(np.argmax(u))
        if k:
            c = s["candidates"]
            s["visual_top1"] = c[0]["word"]; s["reranked"] = True
            s["candidates"] = [c[k]] + [x for j, x in enumerate(c) if j != k]
            s["nearest"] = s["candidates"][0]["word"]
    return segs


def class_features(segs, concepts, prior):
    """prior = {"rel": [C] leave-one-signer-out top-1 accuracy of each word inside the bank (smoothed), "depth": [C] signers}."""
    if not prior:
        return segs
    cidx = {c: i for i, c in enumerate(concepts)}
    for s in segs:
        i = cidx.get(s["nearest"])
        if i is not None:
            s["cls_rel"] = float(prior["rel"][i]); s["depth"] = float(prior["depth"][i])
    return segs


def decide(segs, P, cal):
    """Accept / Uncertain / Reject (status accepted / uncertain / unknown) with the reason, from the calibrated P(correct):
      accept    P ≥ τ_accept and no warning sign: the segment is not abnormally short and the top-2 readings are not a near-tie
      uncertain τ_reject ≤ P, or a warning sign → "อาจหมายถึง …" with the alternatives (the user may repeat the sign)
      reject    P < τ_reject: no word is reliable (outside the vocabulary, or a hand transition) → "ไม่สามารถรู้จำได้"."""
    d = cal.get("decision", {})
    for s, p in zip(segs, P):
        s["p_correct"] = float(p)
        flags = []
        if d.get("short_frames") and (s["f1"] - s["f0"]) < d["short_frames"]:
            flags.append("short")
        if d.get("margin_min") and s.get("prob_margin", 1.0) < d["margin_min"]:
            flags.append("near_tie")
        if p >= cal["tau_accept"] and not flags:
            s["status"], s["reason"] = "accepted", "confident"
        elif p >= cal["tau_uncertain"]:
            s["status"], s["reason"] = "uncertain", "+".join(flags) if p >= cal["tau_accept"] else "below_accept"
        else:
            s["status"], s["reason"] = "unknown", "no_reliable_word"
        s["flags"] = flags
        s["alternatives"] = [c["word"] for c in (s.get("candidates") or [])[1:3]]
    return segs


class LogReg:
    def fit(self, X, y, l2=1e-2, iters=3000, lr=0.1):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        self.w, self.b = np.zeros(X.shape[1]), float(np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)))
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Xn @ self.w + self.b)))
            g = p - y
            self.w -= lr * (Xn.T @ g / len(y) + l2 * self.w); self.b -= lr * g.mean()
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.zeros(0)
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))

    def to_dict(self, feats=None):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), w=self.w.tolist(), b=self.b, feats=feats or FEATS)

    @classmethod
    def from_dict(cls, d):
        m = cls(); m.mu, m.sd, m.w, m.b = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"]), d["b"]
        m.feats = d.get("feats", FEATS)
        return m


def status_of(p, cal):
    return "accepted" if p >= cal["tau_accept"] else ("uncertain" if p >= cal["tau_uncertain"] else "unknown")


# ----------------------------------------------------------------------------- v7 post-processing (prompt_9 P2)
def postprocess(segs, cfg):
    """After calibration. (a) a segment that is short, looks like the null prototype or has a weak tagger P(sign), and is not an
    accepted word, is a hand transition between signs → status 'transition' (kept in the timeline, never sent to the sentence);
    (b) a word that repeats in several non-adjacent uncertain segments of one utterance is a hub symptom (เด็ก / ศูนย์ in v6) →
    only its most probable occurrence keeps it, the others fall back to their next candidate or become unknown."""
    pp = cfg.get("post", {})
    min_f, null_max, psign_min = pp.get("min_frames", 10), pp.get("null_gap_max", 0.0), pp.get("p_sign_min", 0.55)
    for s in segs:
        if s["status"] == "accepted":
            continue
        short = (s["f1"] - s["f0"]) < min_f
        nullish = s["null_gap"] > null_max
        weak = s["p_sign"] < psign_min
        if (short and (nullish or weak or s["status"] == "unknown")) or (nullish and weak):
            s["status"], s["reason"] = "transition", "hand_transition"
    if pp.get("repeat_penalty", True):
        # v8.1: a word may really occur twice in a sentence ("ฉัน … ฉัน"); an occurrence whose own evidence is strong
        # (P(correct) ≥ repeat_keep_p, chosen on tune data) is a new sign, not a hub symptom, and is left alone
        keep_p = pp.get("repeat_keep_p", 2.0)
        live = [s for s in segs if s["status"] in ("uncertain", "unknown") and s.get("p_correct", 0.0) < keep_p]
        by = {}
        for i, s in enumerate(live):
            by.setdefault(s["nearest"], []).append(i)
        used = {s["nearest"] for s in segs if s["status"] == "accepted"}
        for w, idx in by.items():
            if len(idx) < 2 and w not in used:
                continue
            keep = None if w in used else max(idx, key=lambda i: live[i].get("p_correct", 0))
            for i in idx:
                if i == keep:
                    continue
                s = live[i]
                alt = next((c for c in s["candidates"][1:4] if c["word"] not in by and c["word"] not in used), None)
                s["repeat_of"] = w
                if alt is not None and alt["support"] >= 0.8 * s["candidates"][0]["support"]:
                    s["nearest"] = alt["word"]
                    s["candidates"] = [alt] + [c for c in s["candidates"] if c["word"] != alt["word"]]
                else:
                    s["status"], s["reason"] = "unknown", "repeated_word_guard"
    return segs
