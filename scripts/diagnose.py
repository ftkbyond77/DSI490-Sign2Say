"""Forward error diagnosis (v9 → v10 stage 1): where does each error come from — segmentation, recognition, the landmarks, the data?

    python main.py diagnose ledger [--tag v8.1] [--sets a,b]   # error ledger + oracle bounds on the cached tune / test sets
                                                               #   → artifacts/reports/diagnosis_<tag>.json
    python main.py diagnose deep [--tag v9]                    # v10 stage 1 (prompt_14 §3.3 A–G) → artifacts/reports/diagnosis_v10.json
                                                               #   + result_reporting/figures_v10/diag_*.png

Needs the front-pass cache of scripts/decision.py (`python main.py decision cache`). For every utterance with gold boundaries:
  * each reference word gets one class: correct · sub_top5 (right word 2nd–5th) · sub_out (not in top-5) · merged (one predicted
    segment covers it and a neighbour) · split (two predictions inside it) · missed (no prediction overlaps it)
  * extra = predicted segments that overlap no reference word
  * oracle segmentation: the same reader (span head + hand cue) on the gold spans → the recognition ceiling of this encoder / bank
  * oracle recognition: the predicted segments with every matched one named correctly → the segmentation floor of WER
deep adds (A) the ledger per source / signer, (B) landmark quality of every reference word's span (modules/posequality.py) against
its error class, (C) the most confused word pairs and the phonological parameter they differ in (handshape / location / movement /
orientation / two-handedness, measured on training-signer bank clips), (D) embedding geometry (cross-signer spread, hub words, the
isolated-vs-in-sentence gap), (E) calibration per group, (F) real-video drill-down, (G) the web's saved live utterances.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from decision import TEST_SETS, TUNE_SETS, Harness, _iou, load_set, score, REP, _p  # noqa: E402
from modules.decode import wer  # noqa: E402
from modules.utils import write_json  # noqa: E402


def ledger_row(segs, row):
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
        i = ov[0][0]; s = live[i]
        others = [k for k, h in enumerate(gold) if k != j and min(s["f1"], h[1]) - max(s["f0"], h[0]) > 0.5 * max(h[1] - h[0], 1)]
        if others:
            cls.append("merged"); continue
        top = [c["word"] for c in s["candidates"][:5]]
        cls.append("correct" if s["nearest"] == ref[j] else "sub_top5" if ref[j] in top else "sub_out")
    extra = sum(all(_iou((s["f0"], s["f1"]), g) < 0.1 for g in gold) for s in live)
    return cls, extra


def oracle_read(h, row, bank):
    """Gold spans read like any segment (h.pipe.hand_rescore: span head over the frame states + hand cue)."""
    st = {k: (v.astype(np.float32) if k in ("H", "H2") and v is not None else v) for k, v in row["st"].items()}
    if st.get("H") is None:
        return [None] * len(row["gold"])
    segs = [dict(f0=int(a), f1=int(max(b, a + 1)), candidates=[]) for a, b in row["gold"]]
    if h.pipe.spot.get("hand_w", 0) and getattr(h.banks[bank], "hand", None) is not None:
        segs = h.pipe.hand_rescore(segs, st, h.banks[bank])
    else:
        E = h.pipe.span(st["H"], [(s["f0"], s["f1"]) for s in segs])
        S = h.banks[bank].scores(E)
        for s, r in zip(segs, S):
            o = np.argsort(-r)[:5]
            s["nearest"] = h.pipe.concepts[int(o[0])]; s["candidates"] = [dict(word=h.pipe.concepts[int(k)]) for k in o]
    return segs


def cmd_ledger(a):
    h = Harness()
    h.use(a.tag)
    out = {}
    names = a.sets.split(",") if a.sets else TEST_SETS + TUNE_SETS
    for name in names:
        rows = load_set(name)
        if not rows[0].get("gold"):
            continue
        C = collections.Counter(); n_extra = 0
        conf = collections.Counter()
        o_hit1 = o_hit5 = o_n = 0; o_wer, f_wer, wers = [], [], []
        for row in rows:
            segs = h.back(row["st"], row["bank"])
            sc = score(segs, row, h.vocab)
            wers.append(sc["wer"])
            cls, extra = ledger_row(segs, row)
            C.update(cls); n_extra += extra
            live = [s for s in segs if s["status"] != "transition"]
            for s, t in zip(live, sc["true"]):
                if t is not None and s["nearest"] != t:
                    conf[(t, s["nearest"])] += 1
            # segmentation floor: matched predictions named correctly, everything else as is
            hyp = [t if t is not None else s["nearest"] for s, t in zip(live, sc["true"])]
            f_wer.append(wer(row["ref"], hyp))
            orc = oracle_read(h, row, row["bank"])
            if orc[0] is not None:
                hyp = []
                for s, w in zip(orc, row["ref"]):
                    top = [c["word"] for c in s["candidates"][:5]]
                    if w in h.vocab:
                        o_n += 1; o_hit1 += s["nearest"] == w; o_hit5 += w in top
                    hyp.append(s["nearest"])
                o_wer.append(wer(row["ref"], hyp))
        tot = sum(C.values())
        out[name] = dict(n_utt=len(rows), ref_words=tot, classes={k: round(v / tot, 4) for k, v in C.most_common()}, counts=dict(C),
                         extra_per_utt=n_extra / len(rows), WER=float(np.mean(wers)),
                         oracle_seg=dict(R1=o_hit1 / max(o_n, 1), R5=o_hit5 / max(o_n, 1), WER=float(np.mean(o_wer)) if o_wer else None),
                         oracle_recog_WER=float(np.mean(f_wer)),
                         top_confusions=[(f"{t}→{p}", n) for (t, p), n in conf.most_common(25)])
        r = out[name]
        _p("%-15s WER %.3f | oracle-seg WER %.3f R1 %.3f R5 %.3f | oracle-recog WER %.3f | %s extra/utt %.2f", name, r["WER"],
           r["oracle_seg"]["WER"] or -1, r["oracle_seg"]["R1"], r["oracle_seg"]["R5"], r["oracle_recog_WER"],
           " ".join(f"{k}={v:.3f}" for k, v in r["classes"].items()), r["extra_per_utt"])
        _p("    confusions: %s", ", ".join(f"{k}({n})" for k, n in r["top_confusions"][:12]))
    write_json(REP / f"diagnosis_{a.tag}.json", out)


# ----------------------------------------------------------------------------- v10 deep diagnosis
QBINS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]


def _sources():
    """signer → dataset (fluent synthetic sentences carry only the signer) and concept → number of training signers."""
    import pandas as pd
    from modules.utils import PREP
    C = pd.read_parquet(PREP / "shards" / "clips.parquet", columns=["signer", "dataset", "concept", "concept_signers"])
    return dict(zip(C.signer, C.dataset)), C.groupby("concept").concept_signers.max().to_dict()


def _src(row, signer_ds):
    if str(row["id"]).startswith("t51s:"):
        return "tsl51"
    return row.get("dataset") or signer_ds.get(row.get("signer"), "?")


def _q(st, f0, f1):
    from modules.posequality import segment_quality
    q = segment_quality(st["kpi"], st["sc"], f0, f1)
    return dict(dom_seen=q["dom_seen"], any_seen=q["any_seen"], nd_seen=q["L"]["seen"], gap_s=q["R"]["gap_s"], jitter=q["R"]["jitter"],
                bone_cv=q["R"]["bone_cv"], wrist_gap=q["R"]["wrist_gap"], swap=q["R"]["swap"])


def _collect(h, names, signer_ds, csig):
    """One pass over cached sets → per reference word records (class + span quality) and per predicted segment records."""
    words, segs_out, conf = [], [], collections.Counter()
    for name in names:
        rows = load_set(name)
        for row in rows:
            segs = h.back(row["st"], row["bank"])
            sc = score(segs, row, h.vocab)
            live = [s for s in segs if s["status"] != "transition"]
            src = _src(row, signer_ds)
            st = row["st"]
            if row.get("gold"):
                cls, _ = ledger_row(segs, row)
                for w, c, (f0, f1) in zip(row["ref"], cls, row["gold"]):
                    words.append(dict(set=name, source=src, signer=row.get("signer", "t51"), word=w, cls=c, frames=int(f1 - f0),
                                      signers=int(csig.get(w, 0)), **_q(st, f0, f1)))
            for s, ok, t in zip(live, sc["correct"], sc["true"]):
                segs_out.append(dict(set=name, source=src, status=s["status"], p=float(s.get("p_correct", 0.0)), correct=bool(ok),
                                     frames=int(s["f1"] - s["f0"]), word=s["nearest"], signers=int(csig.get(s["nearest"], 0)),
                                     **_q(st, s["f0"], s["f1"])))
                if t is not None and s["nearest"] != t and name in TUNE_SETS:
                    conf[(t, s["nearest"])] += 1
    return words, segs_out, conf


def _err_by_quality(W, key="dom_seen"):
    """P(error) per quality bin, overall and per source, + the share of errors 'attributable' to low quality: Σ over bins of
    n_bin × (error rate of the bin − error rate of the best bin)⁺ ÷ errors (an upper bound of what better hands could fix)."""
    def table(rows):
        out = []
        for a_, b_ in QBINS:
            g = [r for r in rows if a_ <= r[key] < b_]
            out.append(dict(bin=f"[{a_:.2f},{min(b_, 1):.2f})", n=len(g), err=float(np.mean([r["cls"] != "correct" for r in g])) if g else None))
        best = out[-1]["err"] if out[-1]["n"] >= 20 else min(x["err"] for x in out if x["n"] >= 20)
        n_err = sum(r["cls"] != "correct" for r in rows)
        attr = sum(x["n"] * max(0.0, x["err"] - best) for x in out if x["err"] is not None) / max(n_err, 1)
        return dict(bins=out, attributable=float(attr), n=len(rows), errors=int(n_err))
    res = dict(all=table(W))
    for s in sorted({r["source"] for r in W}):
        g = [r for r in W if r["source"] == s]
        if len(g) >= 40:
            res[s] = table(g)
    return res


def _quality_model(W):
    """Out-of-fold AUC of error ~ landmark quality alone, of error ~ source + word data alone, and of both (logistic regression,
    folds by signer): how much the hands' quality predicts an error beyond who / which source / how many signers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    qk = ["dom_seen", "any_seen", "nd_seen", "gap_s", "jitter", "bone_cv", "wrist_gap", "swap"]
    srcs = sorted({r["source"] for r in W})
    Xq = np.nan_to_num(np.array([[r[k] for k in qk] for r in W], float), nan=0.0)
    Xs = np.array([[r["source"] == s for s in srcs] + [np.log1p(r["signers"]), np.log1p(r["frames"])] for r in W], float)
    y = np.array([r["cls"] != "correct" for r in W], int)
    g = np.array([str(r["signer"]) for r in W])
    out = {}
    for name, X in (("quality", Xq), ("source+data", Xs), ("both", np.c_[Xq, Xs])):
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(X, y, g):
            p[te] = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out[name] = float(roc_auc_score(y, p))
    return out


def _calib_groups(S):
    """ECE / false-accept / accepted precision per group of predicted segments."""
    from decision import ece
    def stats(g):
        if len(g) < 30:
            return None
        p = np.array([x["p"] for x in g]); y = np.array([x["correct"] for x in g], float)
        acc = [x for x in g if x["status"] == "accepted"]; bad = [x for x in g if not x["correct"]]
        return dict(n=len(g), ECE=float(ece(p, y)), acc_prec=float(np.mean([x["correct"] for x in acc])) if acc else None,
                    false_accept=float(np.mean([x["status"] == "accepted" for x in bad])) if bad else None,
                    coverage=float(np.mean([x["status"] == "accepted" and x["correct"] for x in g])), top1=float(y.mean()))
    groups = {
        "set": lambda x: x["set"], "source": lambda x: x["source"],
        "speed": lambda x: x["set"].split("_x")[-1] if "_x" in x["set"] else "1.0",
        "signers_per_word": lambda x: "1" if x["signers"] <= 1 else "2-3" if x["signers"] <= 3 else "4-8" if x["signers"] <= 8 else "9+",
        "length": lambda x: "<12f" if x["frames"] < 12 else "12-24f" if x["frames"] < 24 else "24-40f" if x["frames"] < 40 else "40f+",
        "hand_seen": lambda x: next(f"[{a_:.2f},{min(b_, 1):.2f})" for a_, b_ in QBINS if a_ <= x["dom_seen"] < b_),
    }
    return {k: {v: stats([x for x in S if f(x) == v]) for v in sorted({f(x) for x in S})} for k, f in groups.items()}


def _phonology_params(load, clip):
    """One bank clip → phonological descriptors of its sign span (canonical, dominant = right):
    handshape (finger bends + tip distances, middle third), location (dominant hand centre vs nose / chest, shoulder widths),
    movement (net displacement + path length + main direction), orientation (wrist → middle MCP direction), two-handed
    (non-dominant hand seen + its motion)."""
    from modules.handshape import hand_shape
    from modules.schema import LH, RH, SH_L, SH_R
    x = load(clip)
    if x is None:
        return None
    kp, sc, f0, f1 = x
    f0 = 0 if f0 is None else int(f0); f1 = len(kp) if f1 is None else int(f1)
    k, s = kp[f0:f1], sc[f0:f1]
    okR = s[:, RH].mean(1) > 0.3
    okS = (s[:, SH_L] > 0.3) & (s[:, SH_R] > 0.3)
    if okR.sum() < 3 or not okS.any():
        return None
    sw = float(np.median(np.linalg.norm(k[okS, SH_L] - k[okS, SH_R], axis=1)))
    mid = np.median((k[okS, SH_L] + k[okS, SH_R]) / 2, 0)
    nose = np.median(k[s[:, 0] > 0.3, 0], 0) if (s[:, 0] > 0.3).any() else mid
    hR = k[okR][:, RH]
    n = len(hR); m3 = hR[n // 3: max(n // 3 + 1, 2 * n // 3)]
    shape = hand_shape(m3).mean(0)[:24]
    cen = hR.mean(1)
    loc = np.r_[(cen.mean(0) - mid) / sw, (cen.mean(0) - nose) / sw]
    d = np.diff(cen, axis=0) / sw
    path = float(np.linalg.norm(d, axis=1).sum())
    net = (cen[-1] - cen[0]) / sw
    u, sv, vt = np.linalg.svd(cen - cen.mean(0), full_matrices=False) if n >= 3 else (None, np.zeros(2), np.eye(2))
    mdir = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
    mov = np.r_[net, path, mdir * (sv[0] / (sv.sum() + 1e-6))]
    ori = ((hR[:, 9] - hR[:, 0]) / (np.linalg.norm(hR[:, 9] - hR[:, 0], axis=1, keepdims=True) + 1e-6)).mean(0)
    okL = s[:, LH].mean(1) > 0.3
    if okL.sum() >= 3:
        cL = k[okL][:, LH].mean(1)
        two = np.r_[okL.mean(), np.linalg.norm(np.diff(cL, axis=0), axis=1).sum() / sw, np.linalg.norm(cL.mean(0) - cen.mean(0)) / sw]
    else:
        two = np.r_[okL.mean(), 0.0, 0.0]
    return dict(handshape=shape, location=loc, movement=mov, orientation=ori, two_handed=two)


def _phonology(conf, top=30, seed=0):
    """Top confused pairs (tune) → per pair, which phonological parameter separates the two words. Separation of parameter p for
    words a, b = ||μa − μb|| ÷ pooled within-word spread (training-signer bank clips), expressed as a percentile among random word
    pairs of the vocabulary (≥ 3 clips each). A pair 'differs in' the parameters with percentile ≥ 50 and is 'same' in those < 25."""
    from decision import bank_clip_loader
    from modules.utils import ART, read_json
    z = np.load(ART / "eval" / "bank_eval.npz")
    concepts = read_json(ROOT / "models" / "model-v9" / "concepts.json")
    clip_c = [(str(cl), concepts[int(c)]) for cl, c in zip(z["clip"], z["c"])]
    pairs = [p for p, _ in conf.most_common(top)]
    words = sorted({w for p in pairs for w in p})
    rng = np.random.RandomState(seed)
    vocab_words = sorted({w for _, w in clip_c})
    extra = list(rng.choice(vocab_words, min(80, len(vocab_words)), replace=False))
    need = set(words) | set(extra)
    load = bank_clip_loader()
    per = collections.defaultdict(list)
    for cl, w in clip_c:
        if w in need and len(per[w]) < 12:
            d = _phonology_params(load, cl)
            if d is not None:
                per[w].append(d)
    P = ["handshape", "location", "movement", "orientation", "two_handed"]
    stats = {}
    for w, L in per.items():
        if len(L) >= 3:
            stats[w] = {p: (np.mean([d[p] for d in L], 0), np.mean(np.std([d[p] for d in L], 0)) + 1e-3) for p in P}

    def sep(a, b):
        return {p: float(np.linalg.norm(stats[a][p][0] - stats[b][p][0]) / np.sqrt((stats[a][p][1] ** 2 + stats[b][p][1] ** 2) / 2)) for p in P}
    ref_words = [w for w in extra if w in stats]
    ref = collections.defaultdict(list)
    for i in range(len(ref_words)):
        for j in range(i + 1, len(ref_words)):
            for p, v in sep(ref_words[i], ref_words[j]).items():
                ref[p].append(v)
    out = []
    for (a, b), n in conf.most_common(top):
        if a not in stats or b not in stats:
            out.append(dict(pair=f"{a}→{b}", n=n, note="too few bank clips")); continue
        s = sep(a, b)
        pct = {p: float(np.mean(np.array(ref[p]) < v) * 100) for p, v in s.items()}
        out.append(dict(pair=f"{a}→{b}", n=n, separation=s, percentile=pct, differs=[p for p in P if pct[p] >= 50],
                        same=[p for p in P if pct[p] < 25], main=max(P, key=lambda p: pct[p])))
    return out


def _geometry(conf, top=30):
    """(D) Uni-Sign + landmark-encoder bank geometry: cross-signer similarity of the same word vs similarity to its confusion
    partner, hub words (k-occurrence of bank clips as nearest neighbours of other words' clips), and the isolated-vs-in-sentence gap
    (the TSL51 signer's own isolated words vs their spans inside tune sentences)."""
    from modules.utils import ART, read_json
    concepts = read_json(ROOT / "models" / "model-v9" / "concepts.json")
    cidx = {c: i for i, c in enumerate(concepts)}
    out = {}
    for name, f in (("unisign", ART / "eval" / "bank_eval.npz"), ("lmenc", ART / "eval" / "lm_bank_eval.npz")):
        z = np.load(f); zb = np.load(ART / "eval" / "bank_eval.npz")
        ok = z["ok"].astype(bool) if "ok" in z.files else np.ones(len(z["c"]), bool)
        Z = z["Z"][ok].astype(np.float32); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        c, sg = z["c"][ok], zb["signer"][ok]
        rows = []
        for (a, b), n in conf.most_common(top):
            ia, ib = np.flatnonzero(c == cidx.get(a, -1)), np.flatnonzero(c == cidx.get(b, -1))
            if len(ia) < 2 or len(ib) < 1:
                continue
            Saa = Z[ia] @ Z[ia].T; diff = sg[ia][:, None] != sg[ia][None]
            Sab = Z[ia] @ Z[ib].T
            rows.append(dict(pair=f"{a}→{b}", n=n, intra_cross_signer=float(Saa[diff].mean()) if diff.any() else None, inter=float(Sab.mean()),
                             margin=float(Saa[diff].mean() - Sab.mean()) if diff.any() else None))
        # hubness: every clip's nearest clip of ANOTHER word — which words attract most (k = 1 over the whole bank, 4k sample)
        rng = np.random.RandomState(0)
        q = rng.choice(len(Z), min(4000, len(Z)), replace=False)
        S = Z[q] @ Z.T
        S[np.arange(len(q)), q] = -2
        S[c[q][:, None] == c[None]] = -2
        nn = S.argmax(1)
        occ = collections.Counter(concepts[int(c[j])] for j in nn)
        k = np.bincount(nn, minlength=len(Z)).astype(float)
        out[name] = dict(pairs=rows, hub_words=occ.most_common(15),
                         k_occurrence_skew=float(((k - k.mean()) ** 3).mean() / (k.std() ** 3 + 1e-9)),
                         mean_margin=float(np.nanmean([r["margin"] for r in rows if r["margin"] is not None])) if rows else None)
    # isolated-vs-in-sentence gap (Uni-Sign; span head on cached frame states of TSL51 tune sentences)
    from decision import Harness  # noqa: F401  (already imported above; keeps the dependency explicit)
    zp = np.load(ROOT / "models" / "model-v9" / "bank.npz")
    clip = np.asarray(zp["clip"]).astype(str)
    m = np.array([x.startswith("t51u:") for x in clip])
    Zi = zp["Z"][m].astype(np.float32); Zi /= np.linalg.norm(Zi, axis=1, keepdims=True) + 1e-8
    ci = zp["c"][m]
    zo = np.load(ART / "eval" / "bank_eval.npz")
    Zo = zo["Z"].astype(np.float32); Zo /= np.linalg.norm(Zo, axis=1, keepdims=True) + 1e-8
    co = zo["c"]
    return out, (Zi, ci, Zo, co)


def _iso_cont_gap(h, iso_bank):
    """For every TSL51 tune-sentence word: cos(in-sentence span, the same signer's isolated clips of that word) vs cos(the same
    signer's isolated clip, other signers' isolated clips of that word) — how far a sign inside a sentence drifts from its citation
    form, compared with how far another person's citation form is."""
    Zi, ci, Zo, co = iso_bank
    concepts = h.pipe.concepts
    cidx = {c: i for i, c in enumerate(concepts)}
    sent_iso, sent_other, iso_other = [], [], []
    for row in load_set("t51_tune_x1.0"):
        H = row["st"]["H"]
        if H is None:
            continue
        E = h.pipe.span(H.astype(np.float32), [(int(a), int(max(b, a + 1))) for a, b in row["gold"]])
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
        for e, w in zip(E, row["ref"]):
            j = cidx.get(w, -1)
            a_, b_ = Zi[ci == j], Zo[co == j]
            if len(a_) and len(b_):
                sent_iso.append(float((a_ @ e).mean())); sent_other.append(float((b_ @ e).mean()))
                iso_other.append(float((a_ @ b_.T).mean()))
    return dict(n=len(sent_iso), sentence_vs_own_isolated=float(np.mean(sent_iso)), sentence_vs_other_signers=float(np.mean(sent_other)),
                own_isolated_vs_other_signers=float(np.mean(iso_other)))


def _real(h):
    """(F) per real clip: what went wrong — out-of-vocabulary words, repeated performances, count error, in-vocabulary
    substitutions, hands leaving the frame (wrist at the bottom edge) — and data_test v8.1 vs v9 per clip (E10)."""
    from decision import REAL_SETS, _reload_raw, performances
    out = {}
    for name in ("data_test", "team"):
        rows = load_set(name)
        per = []
        res = {}
        for tag in ("v8.1", "v9"):
            h.use(tag); h.configure(prior=h.prod_prior)
            res[tag] = [score(h.back(r["st"], r["bank"]), r, h.vocab) for r in rows]
        h.use("v9"); h.configure(prior=h.prod_prior)
        for r, a8, a9 in zip(rows, res["v8.1"], res["v9"]):
            raw = _reload_raw(r)
            k = performances(raw)
            oov = [w for w in r["ref"] if w not in h.vocab]
            inv_ref = [w for w in r["ref"] if w in h.vocab] * k
            wr = raw.get("rh"); wl = raw.get("lh"); pose = raw["pose"]
            # hands out of frame: the body wrist is at / below the bottom 5 % of the picture while the hand points are missing
            low = ((pose[:, 15, 1] > 0.95) | (pose[:, 16, 1] > 0.95))
            lost = (np.abs(wr).sum((1, 2)) == 0) & (np.abs(wl).sum((1, 2)) == 0)
            sh = pose[:, [11, 12], 1].mean(1); sw = np.abs(pose[:, 11, 0] - pose[:, 12, 0]) + 1e-3
            room = float(np.median((1.0 - sh) / sw))                  # frame below the shoulders, in shoulder widths
            n_inv_sub = sum(1 for w in a9["hyp"] if w not in r["ref"])
            per.append(dict(id=r["id"], ref=r["ref"], hyp_v9=a9["hyp"], hyp_v81=a8["hyp"], wer_v9=a9["wer"], wer_v81=a8["wer"],
                            performances=k, oov=oov, count_err=a9["n_pred"] - len(r["ref"]) * k, wrong_words=n_inv_sub,
                            hands_out_of_frame=float((low & lost).mean()), room_below_shoulders_sw=room,
                            wer_in_vocab_per_perf=float(wer(inv_ref, a9["hyp"])) if inv_ref else None))
        out[name] = per
    return out


def _live():
    """(G) the web's saved live utterances (cache/live_requests/*.json.gz, landmarks only): capture fps, framing (frame below the
    shoulders), hand presence while the hands are up, how many words were accepted / uncertain / unknown."""
    import gzip
    import json
    rows = []
    for f in sorted((ROOT / "cache" / "live_requests").glob("*.json.gz")):
        try:
            d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        except Exception:                                   # noqa: BLE001 — a truncated save
            continue
        fr = d.get("frames") or []
        if not fr:
            continue
        pose = np.array([x.get("pose") or [[0, 0, 0, 0]] * 33 for x in fr], dtype=object)
        try:
            P = np.array([np.array(x["pose"], float) for x in fr if x.get("pose")])
        except Exception:                                   # noqa: BLE001
            P = np.zeros((0, 33, 4))
        hands = np.array([bool(x.get("lh")) or bool(x.get("rh")) for x in fr])
        room = None
        if len(P) and P.shape[1:] == (33, 4):
            sh = P[:, [11, 12], 1].mean(1); sw = np.abs(P[:, 11, 0] - P[:, 12, 0]) + 1e-3
            room = float(np.median((1.0 - sh) / sw))
        segs = [s for s in (d.get("result") or {}).get("segments", []) if s.get("status") != "transition"]
        rows.append(dict(file=f.name, fps=d.get("fps_actual"), n_frames=d.get("n_frames"), vocab=d.get("vocab"), alphabet=d.get("alphabet"),
                         user_signs=d.get("n_user_signs"), hands_frac=float(hands.mean()), room_below_shoulders_sw=room,
                         model=(d.get("result") or {}).get("model_version"), n_seg=len(segs),
                         accepted=sum(s.get("status") == "accepted" for s in segs), uncertain=sum(s.get("status") == "uncertain" for s in segs),
                         unknown=sum(s.get("status") == "unknown" for s in segs)))
        del pose
    if not rows:
        return dict(n=0)
    import pandas as pd
    D = pd.DataFrame(rows)
    num = lambda c: D[c].dropna().astype(float)              # noqa: E731
    return dict(n=len(D), fps_median=float(num("fps").median()), fps_p10=float(num("fps").quantile(0.1)),
                hands_frac_median=float(num("hands_frac").median()), room_median=float(num("room_below_shoulders_sw").median()),
                room_lt_1_share=float((num("room_below_shoulders_sw") < 1.0).mean()),
                words_per_utt=float(D.n_seg.mean()), accepted_share=float(D.accepted.sum() / max(D.n_seg.sum(), 1)),
                unknown_share=float(D.unknown.sum() / max(D.n_seg.sum(), 1)), vocab=D.vocab.value_counts().to_dict(),
                models=D.model.value_counts().to_dict(), per_request=rows)


def cmd_deep(a):
    import pandas as pd
    from decision import TEST_SETS as TS
    from modules.utils import read_json
    h = Harness()
    h.use(a.tag)
    signer_ds, csig = _sources()
    t = __import__("time").time()
    W_tune, S_tune, conf = _collect(h, TUNE_SETS, signer_ds, csig)
    W_test, S_test, _ = _collect(h, TS, signer_ds, csig)
    _p("collected %d tune words / %d segments, %d test words (%.0f s)", len(W_tune), len(S_tune), len(W_test), __import__("time").time() - t)
    out = dict(tag=a.tag)
    # A. ledger per set / source / signer (+ oracle bounds per set, from the v9 ledger)
    led = read_json(REP / f"diagnosis_{a.tag}.json") if (REP / f"diagnosis_{a.tag}.json").exists() else {}
    D = pd.DataFrame(W_tune + W_test)
    D["split"] = np.where(D.set.isin(TUNE_SETS), "tune", "test")
    out["A_ledger"] = dict(
        per_set={k: led.get(k) for k in TUNE_SETS + TS if k in led},
        per_source={f"{sp}|{s}": dict(n=len(g), **{c: float((g.cls == c).mean()) for c in ("correct", "sub_top5", "sub_out", "merged", "split", "missed")})
                    for (sp, s), g in D.groupby(["split", "source"]) if len(g) >= 20},
        per_signer={f"{sp}|{s}|{sg}": dict(n=len(g), err=float((g.cls != "correct").mean()))
                    for (sp, s, sg), g in D[D.source != "tsl51"].groupby(["split", "source", "signer"]) if len(g) >= 15},
        perfect_calibration={k: dict(top1=float(np.mean([x["correct"] for x in S_tune if x["set"] == k])))
                             for k in TUNE_SETS})
    # B. landmark quality ↔ error
    out["B_pose"] = dict(tune=_err_by_quality(W_tune), test=_err_by_quality(W_test),
                         tune_jitter=_err_by_quality([dict(r, jq=1.0 - min(1.0, (r["jitter"] if np.isfinite(r["jitter"]) else 0.0) / 0.2)) for r in W_tune], "jq"),
                         auc=_quality_model(W_tune),
                         quality_by_source={s: dict(n=len(g), dom_seen=float(g.dom_seen.median()), any_seen=float(g.any_seen.median()),
                                                    gap_s=float(g.gap_s.median()), jitter=float(g.jitter.median()), bone_cv=float(g.bone_cv.median()),
                                                    wrist_gap=float(g.wrist_gap.median()), swap=float(g.swap.mean()),
                                                    dom_seen_lt_80=float((g.dom_seen < 0.8).mean()))
                                            for s, g in D.groupby("source")})
    _p("B: attributable share of tune errors to low hand presence %.3f; AUC %s", out["B_pose"]["tune"]["all"]["attributable"], out["B_pose"]["auc"])
    # C. confusions + phonology
    out["C_confusion"] = _phonology(conf, top=30)
    # D. embedding geometry
    geo, iso_bank = _geometry(conf, top=30)
    geo["isolated_vs_continuous"] = _iso_cont_gap(h, iso_bank)
    out["D_geometry"] = geo
    # E. calibration per group
    out["E_calibration"] = dict(tune=_calib_groups(S_tune), test=_calib_groups(S_test))
    # F. real videos, G. live utterances
    out["F_real"] = _real(h)
    out["G_live"] = _live()
    write_json(REP / "diagnosis_v10.json", out)
    D.to_parquet(ROOT / "cache" / "v10" / "diag_words.parquet")
    pd.DataFrame(S_tune + S_test).to_parquet(ROOT / "cache" / "v10" / "diag_segments.parquet")
    _p("written %s", REP / "diagnosis_v10.json")


def cmd_labels(a):
    """(v10 W1) Label-noise candidates in the training-signer bank by cross-signer agreement: every clip is ranked against the OTHER
    signers' clips (leave-signer-out, Uni-Sign space of the v10 bank); a clip whose own word is ranked > 20 while the word has ≥ 3
    other signers that agree with each other is a candidate (wrong label, or a different sign variant under the same name).
    → cloud_s3/data_prep/qc/label_noise_candidates.csv + an ablation: tune WER with the candidates removed from the bank."""
    import pandas as pd
    from decision import ART, _decide_all, _objective, _raw
    from modules.runtime import Bank
    from modules.utils import PREP, read_json
    src = ART / "eval" / "v10" / "bank_eval.npz" if (ART / "eval" / "v10" / "bank_eval.npz").exists() else ART / "eval" / "bank_eval.npz"
    z = np.load(src)
    Z = z["Z"].astype(np.float32); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    c, sg = z["c"].astype(int), z["signer"].astype(str)
    concepts = read_json(ROOT / "models" / "model-v9" / "concepts.json")
    n_sig = pd.Series(sg).groupby(c).nunique()
    rows = []
    for s in np.unique(sg):
        q, ref = np.flatnonzero(sg == s), np.flatnonzero(sg != s)
        S = Z[q] @ Z[ref].T
        cr = c[ref]
        order = np.argsort(cr, kind="stable"); cr = cr[order]; S = S[:, order]
        starts = np.r_[0, np.flatnonzero(np.diff(cr)) + 1]
        best = np.maximum.reduceat(S, starts, axis=1)          # [q, concepts present among the other signers]
        cols = cr[starts]
        for i, k in enumerate(q):
            if c[k] not in set(cols) or n_sig.get(c[k], 0) < 4:
                continue
            j = int(np.flatnonzero(cols == c[k])[0])
            rank = int((best[i] > best[i, j]).sum())
            if rank > 20:
                rows.append(dict(clip=str(z["clip"][k]), signer=s, concept=concepts[c[k]], rank=rank, top1=concepts[int(cols[np.argmax(best[i])])],
                                 sim_own=float(best[i, j]), sim_top=float(best[i].max())))
    D = pd.DataFrame(rows).sort_values("rank", ascending=False)
    out = PREP / "qc" / "label_noise_candidates.csv"
    D.to_csv(out, index=False, encoding="utf-8-sig")
    _p("%d candidates of %d clips (%s) → %s", len(D), len(Z), D.groupby(D["clip"].str.split(":").str[0]).size().to_dict(), out)
    h = Harness(); h.use("v10")
    rows_ = {n: load_set(n) for n in TUNE_SETS}
    base = _decide_all(h, _raw(h, TUNE_SETS, rows_))
    eb = h.banks["eval"]
    drop = set(D["clip"])
    keep = np.array([str(x) not in drop for x in z["clip"]])
    nb = Bank(z["Z"][keep].astype(np.float32), h.fold[z["c"][keep]], len(h.pipe.concepts), set(eb.concepts_present.tolist()))
    nb.hand, nb.lm = eb.hand, eb.lm
    h.banks["eval"] = nb
    abl = _decide_all(h, _raw(h, TUNE_SETS, rows_))
    res = dict(candidates=len(D), by_source=D.groupby(D["clip"].str.split(":").str[0]).size().to_dict(),
               tune_with=dict(objective=_objective(base, TUNE_SETS), **{n: base[n]["WER"] for n in TUNE_SETS}),
               tune_without=dict(objective=_objective(abl, TUNE_SETS), **{n: abl[n]["WER"] for n in TUNE_SETS}))
    _p("ablation (tune WER): with %s | without %s", {k: round(v, 4) for k, v in res["tune_with"].items()}, {k: round(v, 4) for k, v in res["tune_without"].items()})
    rep = read_json(REP / "diagnosis_v10.json")
    rep["W1_label_noise"] = res
    write_json(REP / "diagnosis_v10.json", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ledger", "deep", "labels"])
    ap.add_argument("--tag", default="v8.1")
    ap.add_argument("--sets", default="")
    a = ap.parse_args()
    dict(ledger=cmd_ledger, deep=cmd_deep, labels=cmd_labels)[a.step](a)


if __name__ == "__main__":
    main()
