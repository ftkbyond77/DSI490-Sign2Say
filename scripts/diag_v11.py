"""`python main.py diagnose v11` — the v11 metric catalogue (prompt-opt-v11 C6) through the WHOLE chain, for any version the decision
harness can load (v10, a v11 candidate folder) on cached sets (default: TUNE). Imported by scripts/diagnose.py.

  python main.py diagnose v11 --tag v10 [--sets tune|test|a,b] [--bank_v11 '{"lam":0.05}'] [--out name] [--openset] [--folds 4]

Per set: WER top-1 and displayed WER (accepted words only) · insertions / deletions / substitutions · seg F1 · boundary F1 · count error
· merged / split / missed shares · extra segments per utterance · segmentation tax (in-vocabulary reference words read right on the
gold span but wrong in the chain) · short-sign (≤ 12 frames) recall · reader ablation on gold spans (Uni-Sign span, hand cue,
landmark encoder, fusion, any-of-three: rank of the true word → R@1 / R@5) · per-signer R@1 (gold spans and chain; min / quartiles)
· focus-word and confusion-group error (gold spans) + confusion-group margin · attractor counts (how often each word is the WRONG
top-1) · decision: accepted precision, correct-accepted share, false accept, AUROC / ECE of P(correct), status shares for correct /
wrong · risk–coverage per group (source × segment-length bin × tier) · in-sentence gap (gold-span R@1 isolated − in fluent sentences,
same words + signers). --openset: open-set simulation (4 folds of isolated-set words removed from vocabulary + bank → share of their
clips with ≥ 1 accepted word). Idle words per 10 s on the idle set (TSL51 null clips).
"""
from __future__ import annotations

import collections
import copy
import json
import time

import numpy as np

from decision import (TEST_SETS, TUNE_SETS, Harness, _iou, aggregate, ece, load_set, score, REP, _p, cache_path)  # noqa: F401
from modules.decode import wer
from modules.utils import PREP, read_json, write_json

FOCUS = ["กิน", "คุณ", "ยาย", "ไป", "อะไร", "เหงา", "น้อง", "น้า", "เที่ยว", "ข้าว", "อา", "ย่า"]


def groups():
    from v11 import GROUPS
    f = PREP / "manifest" / "confusion_groups_tune.json"
    return GROUPS + (read_json(f)["groups"] if f.exists() else [])


def tiers():
    f = PREP / "manifest" / "tiers.json"
    return read_json(f)["words"] if f.exists() else {}


def align_ops(ref, hyp):
    """Levenshtein alignment → (substitutions, deletions, insertions)"""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int); D[:, 0] = np.arange(n + 1); D[0] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            Dl += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, Dl, I


def ledger(segs, row):
    live = [s for s in segs if s["status"] != "transition"]
    ref, gold = row["ref"], row["gold"]
    cls = []
    for j, g in enumerate(gold):
        L = max(g[1] - g[0], 1)
        ov = [(i, min(s["f1"], g[1]) - max(s["f0"], g[0])) for i, s in enumerate(live)]
        ov = [(i, o) for i, o in ov if o > 0.3 * L or o > 0.5 * max(live[i]["f1"] - live[i]["f0"], 1)]
        if not ov:
            cls.append("missed"); continue
        if len(ov) >= 2:
            cls.append("split"); continue
        s = live[ov[0][0]]
        if [k for k, h in enumerate(gold) if k != j and min(s["f1"], h[1]) - max(s["f0"], h[0]) > 0.5 * max(h[1] - h[0], 1)]:
            cls.append("merged"); continue
        cls.append("correct" if s["nearest"] == ref[j] else "sub")
    extra = sum(all(_iou((s["f0"], s["f1"]), g) < 0.1 for g in gold) for s in live)
    return cls, extra


def readers_on_gold(h, row):
    """rank of the true word for each reader on the gold spans → dict reader → list of ranks (-1 = not ranked), + fused rows"""
    st = {k: (v.astype(np.float32) if k in ("H", "H2") and v is not None else v) for k, v in row["st"].items()}
    if st.get("H") is None or not row.get("gold"):
        return None
    bank = h.banks[row["bank"]]
    spans = [(int(a), int(max(b, a + 1))) for a, b in row["gold"]]
    from modules.handshape import descriptor
    rows = {}
    rows["unisign"] = (getattr(bank, "reader", None) or bank).scores(h.pipe.span(st["H"], spans))
    if getattr(bank, "hand", None) is not None:
        rows["hand"] = bank.hand.scores(np.stack([descriptor(st["kpi"], st["sc"], a, b) for a, b in spans]))
    if getattr(bank, "lm", None) is not None and h.pipe.lm is not None:
        rows["lmenc"] = bank.lm.scores(h.pipe.lm.spans(st["kpi"], st["sc"], spans))
    segs = [dict(f0=a, f1=b, candidates=[]) for a, b in spans]
    segs = h.pipe.hand_rescore(segs, dict(st, f=1.0), bank)
    cidx = {c: i for i, c in enumerate(h.pipe.concepts)}
    out = {k: [] for k in list(rows) + ["fusion"]}
    top = {k: [] for k in out}
    for j, w in enumerate(row["ref"]):
        if w not in h.vocab:
            for k in out:
                out[k].append(None); top[k].append(None)
            continue
        c = cidx[w]
        for k, R in rows.items():
            r = R[j]
            out[k].append(int((r > r[c]).sum()) if r[c] > -1.5 else -1)
            top[k].append(h.pipe.concepts[int(np.argmax(r))])
        cw = [x["word"] for x in segs[j].get("candidates", [])]
        out["fusion"].append(cw.index(w) if w in cw else 5)
        top["fusion"].append(segs[j].get("nearest"))
    return out, top, segs


def _r(ranks, k):
    v = [x for x in ranks if x is not None]
    return dict(n=len(v), R1=float(np.mean([0 <= x < 1 for x in v])) if v else None, R5=float(np.mean([0 <= x < 5 for x in v])) if v else None)


def _boot_ci(vals, groups_, n=2000, seed=0):
    """95 % CI of a mean by resampling groups (utterances or signers)"""
    vals, groups_ = np.asarray(vals, float), np.asarray(groups_)
    ug = np.unique(groups_)
    idx = {g: np.flatnonzero(groups_ == g) for g in ug}
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        pick = ug[rng.randint(0, len(ug), len(ug))]
        bs.append(np.concatenate([vals[idx[g]] for g in pick]).mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def diag_set(h, name, rows, G, T):
    """everything per set; returns (metrics, per-item records for cross-set analyses)"""
    sc_all, segs_all = [], []
    for r in rows:
        segs = h.back(r["st"], r["bank"])
        segs_all.append(segs); sc_all.append(score(segs, r, h.vocab))
    m = aggregate(sc_all)
    m["WER_ci_utt"] = _boot_ci([x["wer"] for x in sc_all], np.arange(len(sc_all)))
    if any(r.get("signer") for r in rows):
        m["WER_ci_signer"] = _boot_ci([x["wer"] for x in sc_all], [str(r.get("signer")) for r in rows])
    # displayed WER: only KNOWN-SURE (accepted) words are shown
    disp = []
    S = Dl = I = 0
    for r, x, segs in zip(rows, sc_all, segs_all):
        live = [s for s in segs if s["status"] != "transition"]
        shown = [s["nearest"] for s in live if s["status"] == "accepted"]
        disp.append(wer(r["ref"], shown) if r["ref"] else float(len(shown) > 0))
        s_, d_, i_ = align_ops(r["ref"], x["hyp"]); S += s_; Dl += d_; I += i_
    n_ref = max(sum(len(r["ref"]) for r in rows), 1)
    m.update(displayed_WER=float(np.mean(disp)), subs=S / n_ref, dels=Dl / n_ref, ins=I / n_ref, ins_per_utt=I / max(len(rows), 1))
    # ledger + segmentation tax + reader ablation (gold sets)
    rec = []
    if rows and rows[0].get("gold"):
        L = collections.Counter(); extra = 0; tax = n_inv = 0
        short_hit = short_n = 0
        ranks = collections.defaultdict(list)
        for r, segs, x in zip(rows, segs_all, sc_all):
            cls, e = ledger(segs, r); L.update(cls); extra += e
            ro = readers_on_gold(h, r)
            if ro is None:
                continue
            rk, top, _ = ro
            for k, v in rk.items():
                ranks[k] += v
            for j, w in enumerate(r["ref"]):
                if w not in h.vocab:
                    continue
                n_inv += 1
                gold_ok = rk["fusion"][j] == 0
                tax += gold_ok and cls[j] != "correct"
                g0, g1 = r["gold"][j]
                if g1 - g0 <= 12:
                    short_n += 1; short_hit += cls[j] == "correct"
                rec.append(dict(set=name, utt=r["id"], signer=str(r.get("signer", r["id"].split("@")[0])), word=w, chain=cls[j],
                                gold_top={k: top[k][j] for k in top}, gold_rank={k: rk[k][j] for k in rk}, len=g1 - g0,
                                tier=T.get(w, {}).get("tier", "tail")))
        tot = max(sum(L.values()), 1)
        m.update(ledger={k: round(v / tot, 4) for k, v in L.items()}, extra_per_utt=extra / max(len(rows), 1),
                 segmentation_tax=tax / max(n_inv, 1), short_sign_recall=short_hit / max(short_n, 1), short_n=short_n)
        m["readers_gold"] = {k: _r(v, k) for k, v in ranks.items()}
        anyk = [any(ranks[k][i] == 0 for k in ("unisign", "hand", "lmenc") if k in ranks and ranks[k][i] is not None)
                for i in range(len(ranks["fusion"])) if ranks["fusion"][i] is not None]
        m["readers_gold"]["any_of_three"] = dict(n=len(anyk), R1=float(np.mean(anyk)) if anyk else None)
    # attractors: wrong top-1 words of the chain
    att = collections.Counter()
    for x in sc_all:
        for t, hyp_, ok in zip(x["true"], x["hyp"], x["correct"]):
            if t is not None and not ok:
                att[hyp_] += 1
    m["attractors"] = att.most_common(20)
    # decision quality
    P, Y, ST, LEN, SRC = [], [], [], [], []
    for r, segs, x in zip(rows, segs_all, sc_all):
        live = [s for s in segs if s["status"] != "transition"]
        for s, ok in zip(live, x["correct"]):
            P.append(s.get("p_correct", 0.0)); Y.append(float(ok)); ST.append(s["status"]); LEN.append(s["f1"] - s["f0"])
            SRC.append(r.get("dataset") or name.split("_")[0])
    P, Y = np.array(P), np.array(Y)
    if len(P):
        pos, neg = P[Y == 1], P[Y == 0]
        m["AUROC"] = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else None
        m["ECE"] = ece(P, Y)
        m["status_share"] = {k: {st_: float(np.mean([s == st_ for s, y in zip(ST, Y) if y == v])) if (Y == v).any() else None
                                 for st_ in ("accepted", "uncertain", "unknown")} for k, v in (("correct", 1), ("wrong", 0))}
        m["correct_accepted"] = float(np.mean([s == "accepted" for s, y in zip(ST, Y) if y == 1])) if (Y == 1).any() else None
    return m, rec, (P, Y, ST, LEN, SRC, segs_all, sc_all)


def risk_coverage(blocks, T, h):
    """per group (source × segment-length bin × tier of the predicted word): coverage = accepted share of segments, accepted error"""
    out = {}
    for name, (P, Y, ST, LEN, SRC, segs_all, sc_all) in blocks.items():
        words = [s["nearest"] for segs in segs_all for s in segs if s["status"] != "transition"]
        for p, y, st_, L, src, w in zip(P, Y, ST, LEN, SRC, words):
            g = f"{src}|{'<=12' if L <= 12 else '13-24' if L <= 24 else '>24'}|{T.get(w, {}).get('tier', 'tail')}"
            d = out.setdefault(g, dict(n=0, acc=0, acc_err=0, P=[], Y=[]))
            d["n"] += 1; d["acc"] += st_ == "accepted"; d["acc_err"] += (st_ == "accepted") and not y
            d["P"].append(float(p)); d["Y"].append(float(y))
    res = {}
    for g, d in out.items():
        o = np.argsort(-np.array(d["P"])); y = np.array(d["Y"])[o]
        cov = np.arange(1, len(y) + 1) / len(y); risk = 1 - np.cumsum(y) / np.arange(1, len(y) + 1)
        pts = [dict(coverage=float(cov[k]), risk=float(risk[k])) for k in np.unique(np.linspace(0, len(y) - 1, min(10, len(y))).astype(int))]
        res[g] = dict(n=d["n"], coverage=d["acc"] / d["n"], accepted_error=d["acc_err"] / max(d["acc"], 1), curve=pts)
    return res


def openset(h, folds=4, seed=0, set_name="iso_tune"):
    """open-set simulation: words of one fold removed from vocabulary + banks → share of their clips with ≥ 1 accepted word"""
    rows = load_set(set_name)
    words = sorted({r["ref"][0] for r in rows if r["ref"][0] in h.vocab})
    rng = np.random.RandomState(seed); rng.shuffle(words)
    vocab0, variants0 = list(h.vocab_words), dict(getattr(h, "_variants", {}))
    fa = n = 0
    for k in range(folds):
        out = set(words[k::folds])
        h.set_vocab([w for w in vocab0 if w not in out], variants0)
        for r in rows:
            if r["ref"][0] in out:
                segs = h.back(r["st"], r["bank"])
                n += 1; fa += any(s["status"] == "accepted" for s in segs)
    h.set_vocab(vocab0, variants0)
    return dict(n_clips=n, false_accept=fa / max(n, 1), folds=folds)


def cmd_v11(a):
    h = Harness()
    if a.bank_v11:
        h.bank_v11 = json.loads(a.bank_v11); h.bank_v11_forced = True
    h.use(a.tag)
    if a.spot_override:
        h.configure(spot=dict(h.pipe.spot, **json.loads(a.spot_override)))
    h._variants = h.fold is not None and {}
    G, T = groups(), tiers()
    names = {"tune": TUNE_SETS + ["idle_tune"], "test": TEST_SETS + ["idle_test"]}.get(a.sets, a.sets.split(","))
    out = dict(tag=a.tag, bank_v11=a.bank_v11 or None, sets={}, created=time.strftime("%Y-%m-%d %H:%M"))
    recs, blocks = [], {}
    for name in names:
        if not cache_path(name).exists():
            _p("%s: not cached (python main.py decision cache --sets %s)", name, name); continue
        t = time.time()
        rows = load_set(name)
        if name.startswith(("tail", "w_tail")):          # (v12: the web-rate Tail / idle sets are counted the same way)                 # known unknowns: share of Tail-sign clips with >= 1 accepted word (G7, real Tail signs)
            n_fa = 0
            for r in rows:
                segs = h.back(r["st"], r["bank"])
                n_fa += any(s_["status"] == "accepted" for s_ in segs)
            out["sets"][name] = dict(n=len(rows), tail_false_accept=n_fa / max(len(rows), 1))
            _p("%-16s Tail signs accepted as a word: %.3f (%d clips)", name, n_fa / max(len(rows), 1), len(rows))
            continue
        if name.startswith(("idle", "w_idle")):
            n_words = n_sec = 0
            for r in rows:
                segs = h.back(r["st"], r["bank"])
                n_words += sum(s["status"] == "accepted" for s in segs); n_sec += r["T"] / 25.0
            out["sets"][name] = dict(n=len(rows), idle_accepted_words_per_10s=10 * n_words / max(n_sec, 1e-6), seconds=n_sec)
            _p("%-16s idle accepted words / 10 s %.3f (%d clips, %.0f s)", name, out["sets"][name]["idle_accepted_words_per_10s"], len(rows), n_sec)
            continue
        m, rec, blk = diag_set(h, name, rows, G, T)
        out["sets"][name] = m; recs += rec; blocks[name] = blk
        rg = m.get("readers_gold", {})
        _p("%-16s WER %.3f [%s] disp %.3f | R1 %.3f | tax %s | ins %.3f | readers %s | corr-acc %s | AUROC %s (%.0f s)", name, m["WER"],
           ",".join(f"{x:.3f}" for x in m["WER_ci_utt"]), m["displayed_WER"], m["R1"], f"{m.get('segmentation_tax', float('nan')):.3f}",
           m["ins"], {k: round(v.get("R1") or 0, 3) for k, v in rg.items()}, f"{m.get('correct_accepted') or 0:.3f}",
           f"{m.get('AUROC') or 0:.3f}", time.time() - t)
    # per-signer gold-span R1 (fusion) and chain R1
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["set"].split("_")[0], r["signer"])].append(r)
    per = {}
    for (fam, sg), rr in by.items():
        if len(rr) >= 8:
            per[f"{fam}|{sg}"] = dict(n=len(rr), gold_R1=float(np.mean([x["gold_rank"]["fusion"] == 0 for x in rr])),
                                     chain_R1=float(np.mean([x["chain"] == "correct" for x in rr])))
    vals = [v["gold_R1"] for v in per.values()]
    out["per_signer"] = dict(signers=per, min=min(vals) if vals else None, quartiles=np.percentile(vals, [25, 50, 75]).tolist() if vals else None)
    # focus words + confusion groups (gold spans, fusion)
    fw = {}
    for w in FOCUS:
        rr = [x for x in recs if x["word"] == w]
        if rr:
            fw[w] = dict(n=len(rr), error=float(np.mean([x["gold_rank"]["fusion"] != 0 for x in rr])),
                         confused_with=collections.Counter(x["gold_top"]["fusion"] for x in rr if x["gold_rank"]["fusion"] != 0).most_common(3))
    out["focus_words"] = fw
    out["focus_mean_error"] = float(np.mean([v["error"] for v in fw.values()])) if fw else None
    gr = []
    for g in G:
        rr = [x for x in recs if x["word"] in g]
        if rr:
            within = [x["gold_top"]["fusion"] in g and x["gold_top"]["fusion"] != x["word"] for x in rr]
            gr.append(dict(group=g, n=len(rr), error=float(np.mean([x["gold_rank"]["fusion"] != 0 for x in rr])), within_group_error=float(np.mean(within))))
    out["confusion_groups"] = gr
    out["group_mean_within_error"] = float(np.mean([g_["within_group_error"] for g_ in gr])) if gr else None
    # in-sentence gap: same words + signers, isolated vs fluent (gold spans, fusion)
    iso = {(x["signer"], x["word"]): x for x in recs if x["set"].startswith("iso")}
    flu = [x for x in recs if x["set"].startswith("flu") and (x["signer"], x["word"]) in iso]
    if flu:
        out["in_sentence_gap"] = dict(n=len(flu), iso_R1=float(np.mean([iso[(x["signer"], x["word"])]["gold_rank"]["fusion"] == 0 for x in flu])),
                                      flu_R1=float(np.mean([x["gold_rank"]["fusion"] == 0 for x in flu])))
        out["in_sentence_gap"]["gap"] = out["in_sentence_gap"]["iso_R1"] - out["in_sentence_gap"]["flu_R1"]
    out["risk_coverage"] = risk_coverage(blocks, T, h)
    acc_err_groups = {g: v["accepted_error"] for g, v in out["risk_coverage"].items() if v["n"] >= 30}
    out["max_group_accepted_error"] = max(acc_err_groups.values()) if acc_err_groups else None
    if a.openset:
        out["openset"] = openset(h, a.folds, set_name=getattr(a, "openset_set", "iso_tune") or "iso_tune")
        _p("open-set false accept %.3f (%d clips)", out["openset"]["false_accept"], out["openset"]["n_clips"])
    # TUNE confusion matrix → extra confusion groups (written once, from TUNE only)
    if a.write_groups:
        conf = collections.Counter((x["word"], x["gold_top"]["fusion"]) for x in recs if x["gold_rank"]["fusion"] not in (0, None))
        known = {w for g in G for w in g}
        extra = []
        for (t, p_), n in conf.most_common(60):
            if n >= 3 and not ({t, p_} <= known):
                extra.append([t, p_])
        write_json(PREP / "manifest" / "confusion_groups_tune.json", dict(rule="TUNE gold-span confusions ≥ 3 not already grouped (diagnose v11, v10)", groups=extra))
        _p("extra confusion groups from TUNE: %s", extra)
    _p("per-signer gold R1 min %s quartiles %s | focus mean error %s | group within-error %s | gap %s | max group accepted error %s",
       out["per_signer"]["min"], out["per_signer"]["quartiles"], out["focus_mean_error"], out["group_mean_within_error"],
       out.get("in_sentence_gap", {}).get("gap"), out["max_group_accepted_error"])
    write_json(REP / f"diag_v11_{a.out or a.tag}.json", out)
    return out


def cmd_paired(a):
    """paired v10-vs-<new> comparison on the same items: per-utterance WER, bootstrap 2,000 resamples by utterance AND by signer"""
    from decision import evaluate
    h = Harness()
    names = a.sets.split(",")
    res = {}
    rng = np.random.RandomState(0)
    for name in names:
        rows = load_set(name)
        W = {}
        for tag in (a.old, a.tag):
            h.use(tag)
            W[tag] = np.array([x["wer"] for x in evaluate(h, rows)])
        d = W[a.tag] - W[a.old]
        sg = np.array([str(r.get("signer", r["id"].split("@")[0])) for r in rows])
        bs_u = d[rng.randint(0, len(d), (2000, len(d)))].mean(1)
        us = np.unique(sg); idx = {u: np.flatnonzero(sg == u) for u in us}
        bs_s = [np.concatenate([d[idx[u]] for u in us[rng.randint(0, len(us), len(us))]]).mean() for _ in range(2000)]
        res[name] = dict(n=len(d), signers=len(us), old=float(W[a.old].mean()), new=float(W[a.tag].mean()), diff=float(d.mean()),
                         ci_utt=[float(np.percentile(bs_u, 2.5)), float(np.percentile(bs_u, 97.5))],
                         ci_signer=[float(np.percentile(bs_s, 2.5)), float(np.percentile(bs_s, 97.5))], better=int((d < 0).sum()), worse=int((d > 0).sum()))
        r_ = res[name]
        _p("%-16s %s %.3f → %s %.3f  dWER %+.3f  CI utt [%+.3f, %+.3f]  CI signer [%+.3f, %+.3f]  (n %d, %d signers)", name, a.old, r_["old"], a.tag, r_["new"],
           r_["diff"], *r_["ci_utt"], *r_["ci_signer"], r_["n"], r_["signers"])
    write_json(REP / f"paired_{a.old}_{a.tag}_{a.out or 'sets'}.json", res)
