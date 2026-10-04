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


def active_from_prob(prob, thr=0.5, present=None):
    p = np.convolve(prob[:, 1].astype(np.float32) + prob[:, 2].astype(np.float32), np.ones(5) / 5, "same")
    act = p > thr
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
    act, p_sign = active_from_prob(prob, cfg["act_thr"], present)
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
    ti, ts = rt["top_idx"][:, :5], rt["top_sc"][:, :5]
    for s in segs:
        w = s.pop("_w"); s.pop("_wi"); wi = int(rt["top_idx"][w, 0])
        m = (W[:, 0] >= s["f0"] - 2) & (W[:, 1] <= s["f1"] + 2) & (L >= cfg["min_len"])
        n_in = int(m.sum())
        if n_in:
            ids, vals = ti[m].ravel(), ts[m].ravel()
            uniq, inv = np.unique(ids, return_inverse=True)
            acc = np.zeros(len(uniq)); np.add.at(acc, inv, vals)
            best = np.full(len(uniq), -9.0); np.maximum.at(best, inv, vals)
            o = np.argsort(-acc)[:5]
            cands = [dict(word=concepts[int(uniq[k])], score=round(float(best[k]), 4), support=round(float(acc[k]) / n_in, 4)) for k in o]
        else:
            cands = [dict(word=concepts[int(j)], score=round(float(x), 4), support=round(float(x), 4)) for j, x in zip(ti[w], ts[w])]
        s.update(nearest=cands[0]["word"], candidates=cands, score=float(table["top_sc"][w, 0]), margin=float(table["top_sc"][w, 0] - table["top_sc"][w, 1]),
                 null_gap=float(null_gap[w]), p_sign=float(p_sign[s["f0"]:s["f1"]].mean()), agree=float(concepts[wi] == cands[0]["word"]),
                 support_ratio=float(cands[0]["support"] / max(cands[1]["support"], 1e-6)) if len(cands) > 1 else 5.0, window_top1=concepts[wi],
                 depth=float(depth[concepts.index(cands[0]["word"])]) if depth is not None and cands[0]["word"] in concepts else 0.0)
        out.append(s)
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


def seg_features(segs, feats=None):
    feats = feats or FEATS
    rows = []
    for s in segs:
        ld = float(np.log1p(s.get("depth", 0.0)))
        v = dict(z=s["z"], score=s["score"], margin=s["margin"], null_gap=s["null_gap"], p_sign=s["p_sign"], agree=s["agree"],
                 support_ratio=min(s["support_ratio"], 5.0), len=(s["f1"] - s["f0"]) / 25.0, log_depth=ld, z_x_log_depth=s["z"] * ld)
        rows.append([v[f] for f in feats])
    return np.array(rows, np.float32).reshape(-1, len(feats))


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
            s["status"] = "transition"
    if pp.get("repeat_penalty", True):
        live = [s for s in segs if s["status"] in ("uncertain", "unknown")]
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
                    s["status"] = "unknown"
    return segs
