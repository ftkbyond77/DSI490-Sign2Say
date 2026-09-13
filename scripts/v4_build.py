"""ThaiSLM v4 build / optimise / evaluate.

  python scripts/v4_build.py embed_ttrs          # Uni-Sign ctx embeddings of every usable TTRS sign (+ augmented views)
  python scripts/v4_build.py embed_titles        # single-sign YouTube clips labelled by title
  python scripts/v4_build.py mine                # speech-word → verified sign instances from lessons
  python scripts/v4_build.py bank                # vocabulary bank (deployment + train-only variants)
  python scripts/v4_build.py eval                # E1 / E2 (full galleries) + lesson-domain queries
  python scripts/v4_build.py calib               # open-set thresholds (accept / uncertain / unknown)
  python scripts/v4_build.py seg_train           # segmentation model
  python scripts/v4_build.py infer               # data_test end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from modules.utils import ART, get_logger, read_json, seed_all, write_json

log = get_logger("v4b")
V4 = ART / "v4"; V4.mkdir(parents=True, exist_ok=True)
MAN = ART / "manifests"; MAN4 = ART / "manifests_v4"; REP = ART / "reports"


def enc_():
    from modules.unisign import UniSignEncoder
    return UniSignEncoder("wlasl", use_mt5=True).eval()


def _aug_view(kp, sc, rng):
    from modules.segment import _affine, _resample
    k, s = _resample(kp, sc, rng.uniform(1.2, 2.5))
    k = _affine(k, rng) + rng.normal(0, 0.002, k.shape).astype(np.float32)
    return k, s


# ----------------------------------------------------------------------------- TTRS
def cmd_embed_ttrs(a):
    from modules.recognize import active_span, embed, load_pose
    from modules.unisign import prepare_parts, upsample
    from modules.wholebody import pose_path
    cv4 = pd.read_parquet(MAN4 / "clips_v4.parquet")
    t = cv4[(cv4.source == "ttrs") & (cv4.usable_for.str.contains("vocab"))]
    enc = enc_()
    rng = np.random.RandomState(0)
    recs, parts, aug_parts, aug_owner = [], [], [], []
    for r in t.itertuples():
        p = pose_path(r.clip_id)
        if not p.exists():
            continue
        kp, sc, hw, _ = load_pose(p)
        hand_conf = float(sc[:, 91:133].mean())
        f0, f1 = active_span(kp, sc, hw)
        if f1 - f0 < 3 or hand_conf < 0.3:
            continue
        recs.append(dict(clip_id=r.clip_id, word=r.label, sign_variant_id=r.sign_variant_id, signer=r.signer_id, f0=f0, f1=f1,
                         hand_conf=hand_conf, source="ttrs"))
        parts.append(prepare_parts(*upsample(kp[f0:f1], sc[f0:f1], 2)))
        for _ in range(a.n_aug):
            k, s = _aug_view(kp[f0:f1], sc[f0:f1], rng)
            aug_parts.append(prepare_parts(*upsample(k, s, 2))); aug_owner.append(len(recs) - 1)
    log.info("embedding %d TTRS signs (+%d augmented)", len(parts), len(aug_parts))
    Z = embed(enc, parts, "ctx", bs=128)
    df = pd.DataFrame(recs)
    df.to_parquet(V4 / "emb_ttrs.parquet", index=False); np.save(V4 / "emb_ttrs.npy", Z.astype(np.float16))
    if aug_parts:
        Za = embed(enc, aug_parts, "ctx", bs=128)
        np.save(V4 / "emb_ttrs_aug.npy", Za.astype(np.float16)); np.save(V4 / "emb_ttrs_aug_owner.npy", np.array(aug_owner))
    log.info("done")


# ----------------------------------------------------------------------------- YouTube title clips
def cmd_embed_titles(a):
    from modules.recognize import embed, load_pose
    from modules.segment import sign_span
    from modules.unisign import prepare_parts, upsample
    from modules.wholebody import pose_path
    cv4 = pd.read_parquet(MAN4 / "clips_v4.parquet")
    t = cv4[cv4.label_source == "title"]
    enc = enc_()
    recs, parts = [], []
    for r in t.itertuples():
        p = pose_path(r.clip_id)
        if not p.exists():
            continue
        kp, sc, hw, _ = load_pose(p)
        sp = sign_span(kp, sc, hw, pad=1) or (0, len(kp))
        recs.append(dict(clip_id=r.clip_id, word=r.label, sign_variant_id=None, signer=r.signer_id, f0=sp[0], f1=sp[1],
                         hand_conf=float(sc[:, 91:133].mean()), source="title"))
        parts.append(prepare_parts(*upsample(kp[sp[0]:sp[1]], sc[sp[0]:sp[1]], 2)))
    Z = embed(enc, parts, "ctx") if parts else np.zeros((0, 768), np.float32)
    pd.DataFrame(recs).to_parquet(V4 / "emb_title.parquet", index=False); np.save(V4 / "emb_title.npy", Z.astype(np.float16))
    log.info("title clips embedded: %d", len(recs))


# ----------------------------------------------------------------------------- mining
def cmd_mine(a):
    from pythainlp.corpus import thai_stopwords
    from modules.mining import mine_clip
    from modules.recognize import VocabBank, load_pose
    from modules.wholebody import pose_path
    df = pd.read_parquet(V4 / "emb_ttrs.parquet"); Z = np.load(V4 / "emb_ttrs.npy").astype(np.float32)
    bank = VocabBank(Z, df.word.tolist())
    sw = pd.read_parquet(MAN4 / "speech_words.parquet")
    stop = frozenset(thai_stopwords()) | frozenset({"ภาษามือ", "มือ", "คำ", "ภาษา", "คนหูหนวก", "ท่า", "สวัสดี_ครับ"})
    enc = enc_()
    recs, embs, stats = [], [], []
    for cid, g in sw.groupby("clip_id"):
        p = pose_path(cid)
        if not p.exists():
            continue
        d = np.load(p)
        kp, sc, hw = d["kp"].astype(np.float32), d["sc"].astype(np.float32), tuple(int(x) for x in d["hw"])
        t_start = float(d["t0"])
        t0 = time.time()
        got = mine_clip(enc, bank, kp, sc, hw, g.to_dict("records"), t_start, rank_k=a.rank_k, stop_words=stop)
        for o in got:
            embs.append(o.pop("emb")); recs.append(dict(clip_id=cid, source="mined", **o))
        stats.append(dict(clip_id=cid, hits=len(g), accepted=len(got), sec=round(time.time() - t0, 1)))
        log.info("%s: %d hits → %d instances (%.1fs)", cid, len(g), len(got), time.time() - t0)
    m = pd.DataFrame(recs)
    m.to_parquet(V4 / "mined.parquet", index=False); np.save(V4 / "mined.npy", np.stack(embs).astype(np.float16) if embs else np.zeros((0, 768), np.float16))
    write_json(REP / "v4_mining.json", dict(videos=len(stats), hits=int(sum(s["hits"] for s in stats)), instances=len(m),
                                             words=int(m.word.nunique()) if len(m) else 0, per_video=stats))
    log.info("mined %d instances of %d words from %d videos", len(m), m.word.nunique() if len(m) else 0, len(stats))


# ----------------------------------------------------------------------------- bank
def load_sources(train_only=False):
    sp = read_json(MAN / "splits.json")
    parts = []
    t = pd.read_parquet(V4 / "emb_ttrs.parquet"); Zt = np.load(V4 / "emb_ttrs.npy").astype(np.float32)
    if train_only:
        keep = t.clip_id.isin(set(sp["train_islr"])).values
        t, Zt = t[keep], Zt[keep]
    parts.append((t, Zt))
    for name in ("title", "mined"):
        pq, npy = V4 / (f"emb_{name}.parquet" if name == "title" else "mined.parquet"), V4 / (f"emb_{name}.npy" if name == "title" else "mined.npy")
        if pq.exists():
            d = pd.read_parquet(pq); Z = np.load(npy).astype(np.float32)
            if len(d):
                parts.append((d, Z))
    df = pd.concat([p[0][["clip_id", "word", "source"] + (["signer"] if "signer" in p[0] else [])] for p in parts], ignore_index=True)
    if "signer" not in df:
        df["signer"] = None
    df["signer"] = df["signer"].fillna(df["clip_id"])  # YouTube instances: one video ≈ one signer
    Z = np.concatenate([p[1] for p in parts])
    return df, Z


def cmd_bank(a):
    for train_only, name in ((False, "bank.npz"), (True, "bank_train.npz")):
        df, Z = load_sources(train_only)
        np.savez(V4 / name, Z=Z.astype(np.float32), words=np.array(df.word.tolist()), source=np.array(df.source.tolist()),
                 signer=np.array(df.signer.astype(str).tolist()), clip=np.array(df.clip_id.tolist()))
        log.info("%s: %d instances, %d words, sources %s", name, len(df), df.word.nunique(), df.source.value_counts().to_dict())


# ----------------------------------------------------------------------------- evaluation
def e1_eval(bank, Zq, q_words, topm=2):
    S = bank.word_scores(Zq, topm)
    tgt = np.array([bank.widx.get(w, -1) for w in q_words])
    ok = tgt >= 0
    S, tgt = S[ok], tgt[ok]
    rank = (S > S[np.arange(len(tgt)), tgt][:, None]).sum(1)
    n = len(tgt)
    ci = lambda p: 1.96 * np.sqrt(p * (1 - p) / max(n, 1))
    r1, r5 = float((rank < 1).mean()), float((rank < 5).mean())
    return dict(R1=r1, R1_ci95=float(ci(r1)), R5=r5, R5_ci95=float(ci(r5)), R10=float((rank < 10).mean()),
                MRR=float((1 / (rank + 1)).mean()), median_rank=float(np.median(rank) + 1), n=n, vocab=len(bank.vocab))


def cmd_eval(a):
    from modules.recognize import VocabBank
    sp = read_json(MAN / "splits.json")
    t = pd.read_parquet(V4 / "emb_ttrs.parquet"); Zt = np.load(V4 / "emb_ttrs.npy").astype(np.float32)
    row = {c: i for i, c in enumerate(t.clip_id)}
    res = {}
    # lemma-level words (TTRS variants of the same lemma count as the same word: that is what translation needs)
    for variant in ("ttrs_train_only", "ttrs_train+youtube"):
        df, Z = load_sources(train_only=True)
        if variant == "ttrs_train_only":
            keep = (df.source == "ttrs").values
            df, Z = df[keep], Z[keep]
        bank = VocabBank(Z, df.word.tolist())
        for part in ("val", "test"):
            qs = [q for q in sp[f"seen_{part}_queries"] if q in row]
            r = e1_eval(bank, Zt[[row[q] for q in qs]], [t.word.iloc[row[q]] for q in qs])
            res[f"E1|{part}|{variant}"] = r
            log.info("E1 %-5s %-22s R1 %.3f±%.2f R5 %.3f R10 %.3f MRR %.3f med %.0f (n=%d, vocab=%d)", part, variant, r["R1"],
                     r["R1_ci95"], r["R5"], r["R10"], r["MRR"], r["median_rank"], r["n"], r["vocab"])
    # E5: YouTube single-sign clips labelled by title (new signers, real cameras) vs the full TTRS dictionary
    if (V4 / "emb_title.parquet").exists():
        from modules.recognize import VocabBank as VB
        d = pd.read_parquet(V4 / "emb_title.parquet"); Zd = space(np.load(V4 / "emb_title.npy").astype(np.float32))
        dft = pd.read_parquet(V4 / "emb_ttrs.parquet"); Zb = space(np.load(V4 / "emb_ttrs.npy").astype(np.float32))
        bank = VB(Zb, dft.word.tolist())
        keep = d.word.isin(bank.widx).values
        r = e1_eval(bank, Zd[keep], d.word[keep].tolist())
        res["E5|youtube_title_clips_vs_full_TTRS"] = r
        S = bank.word_scores(Zd)
        res["E5|per_clip"] = [dict(word=w, in_ttrs=bool(w in bank.widx), top5=[bank.vocab[j] for j in np.argsort(-S[i])[:5]]) for i, w in enumerate(d.word)]
        log.info("E5 youtube title clips: R1 %.3f R5 %.3f R10 %.3f med %.0f (n=%d in-vocab of %d)", r["R1"], r["R5"], r["R10"], r["median_rank"], r["n"], len(d))
    # lesson-domain queries: mined instances, leave-one-video-out, against TTRS-only bank (silver labels)
    if (V4 / "mined.parquet").exists():
        m = pd.read_parquet(V4 / "mined.parquet"); Zm = np.load(V4 / "mined.npy").astype(np.float32)
        if len(m):
            dft, Zbt = load_sources(train_only=False)
            keep = (dft.source == "ttrs").values
            bank = VocabBank(Zbt[keep], dft[keep].word.tolist())
            r = e1_eval(bank, Zm, m.word.tolist())
            res["LESSON|mined_vs_ttrs_bank (silver, selection-biased)"] = r
            log.info("lesson-domain silver: R1 %.3f R5 %.3f (n=%d) — biased: mining required rank<K", r["R1"], r["R5"], r["n"])
    write_json(REP / "v4_eval.json", res)


# ----------------------------------------------------------------------------- adapter fine-tuning
def space(Z):
    """Embedding space used everywhere downstream: raw Uni-Sign, or the fine-tuned adapter if selected."""
    cfg = read_json(V4 / "config.json") if (V4 / "config.json").exists() else {}
    if not cfg.get("use_adapter"):
        return Z
    from modules.recognize import Adapter
    ad = Adapter().cuda(); ad.load_state_dict(torch.load(V4 / "adapter.pt", map_location="cuda"))
    return ad.apply_np(Z)


def cmd_adapter(a):
    from modules.recognize import VocabBank, train_adapter
    seed_all(0)
    sp = read_json(MAN / "splits.json")
    t = pd.read_parquet(V4 / "emb_ttrs.parquet"); Zt = np.load(V4 / "emb_ttrs.npy").astype(np.float32)
    row = {c: i for i, c in enumerate(t.clip_id)}
    df, Z = load_sources(train_only=True)
    aug_Z = np.load(V4 / "emb_ttrs_aug.npy").astype(np.float32) if (V4 / "emb_ttrs_aug.npy").exists() else None
    aug_owner_ttrs = np.load(V4 / "emb_ttrs_aug_owner.npy") if aug_Z is not None else None
    # map augmented owners (indices in emb_ttrs) to indices in the train-only source table
    src_row = {c: i for i, c in enumerate(df.clip_id) if df.source.iloc[i] == "ttrs"}
    if aug_Z is not None:
        keep = [j for j, o in enumerate(aug_owner_ttrs) if t.clip_id.iloc[o] in src_row]
        aug_Z = aug_Z[keep]; aug_owner = [src_row[t.clip_id.iloc[aug_owner_ttrs[j]]] for j in keep]
    else:
        aug_owner = None
    qs = {p: [q for q in sp[f"seen_{p}_queries"] if q in row] for p in ("val", "test")}

    def e1(model, part):
        Zb = model.apply_np(Z) if model is not None else Z
        Zq = Zt[[row[q] for q in qs[part]]]
        Zq = model.apply_np(Zq) if model is not None else Zq
        return e1_eval(VocabBank(Zb, df.word.tolist()), Zq, [t.word.iloc[row[q]] for q in qs[part]])

    base = {p: e1(None, p) for p in ("val", "test")}
    model, hist = train_adapter(Z, df.word.tolist(), df.signer.astype(str).tolist(), aug_Z, aug_owner, steps=a.steps,
                                eval_fn=lambda m: e1(m, "val"))
    tuned = {p: e1(model, p) for p in ("val", "test")}
    use = (tuned["val"]["MRR"] > base["val"]["MRR"] + 0.01) and (tuned["val"]["median_rank"] <= base["val"]["median_rank"])
    torch.save(model.state_dict(), V4 / "adapter.pt")
    write_json(V4 / "config.json", dict(use_adapter=bool(use)))
    write_json(REP / "v4_adapter.json", dict(zero_shot=base, adapter=tuned, selected=bool(use), hist=hist, rule="use adapter iff val MRR +0.01 AND val median rank not worse; E5 must not be scored through an adapter trained on title clips"))
    for p in ("val", "test"):
        log.info("E1 %s zero-shot R1 %.3f R5 %.3f MRR %.3f med %.0f | adapter R1 %.3f R5 %.3f MRR %.3f med %.0f", p, base[p]["R1"], base[p]["R5"],
                 base[p]["MRR"], base[p]["median_rank"], tuned[p]["R1"], tuned[p]["R5"], tuned[p]["MRR"], tuned[p]["median_rank"])
    log.info("adapter selected: %s", use)


# ----------------------------------------------------------------------------- calibration
def cmd_calib(a):
    """Three-tier open-set decision from TTRS E1/E2 **val** score distributions (quantiles only), then the empirical
    precision of each tier is measured on E1 test and E5 (independent YouTube signers) — never used for thresholds.
      likely     score ≥ q_high(in-vocab correct) and margin ≥ m_high
      candidate  sign detected, top-5 shown, no single word asserted
      unknown    score < q_low(in-vocab correct): "there is a sign — nearest known word is ___" """
    from modules.recognize import VocabBank
    sp = read_json(MAN / "splits.json")
    t = pd.read_parquet(V4 / "emb_ttrs.parquet"); Zt = np.load(V4 / "emb_ttrs.npy").astype(np.float32)
    row = {c: i for i, c in enumerate(t.clip_id)}
    df, Z = load_sources(train_only=True)

    def feats_for(qs, words, Zq, bank_df, bank_Z, oov_sim=True):
        out = []
        for q, w, zq in zip(qs, words, Zq):
            for oov in ((False, True) if oov_sim else (False,)):
                keep = ~((bank_df.clip_id == q).values) & ((~(bank_df.word == w).values) if oov else True)
                bank = VocabBank(bank_Z[keep], bank_df[keep].word.tolist())
                if not oov and w not in bank.widx:
                    continue
                s = bank.word_scores(zq[None])[0]; o = np.argsort(-s)
                out.append(dict(score=float(s[o[0]]), margin=float(s[o[0]] - s[o[1]]), oov=oov,
                                correct=(bank.vocab[o[0]] == w) and not oov, top5=(w in [bank.vocab[i] for i in o[:5]]) and not oov))
        return pd.DataFrame(out)

    sets = {}
    for part in ("val", "test"):
        qs = [q for q in sp[f"seen_{part}_queries"] if q in row] + [q for q in sp[f"{part}_clips"] if q in row]
        sets[part] = feats_for(qs, [t.word.iloc[row[q]] for q in qs], Zt[[row[q] for q in qs]], df, Z)
    v = sets["val"]
    corr = v[v.correct]
    q_high = float(np.quantile(corr.score, 0.5)); m_high = float(np.quantile(corr.margin, 0.5))
    q_low = float(np.quantile(corr.score, 0.05))
    cal = dict(accept=dict(tau_score=q_high, tau_margin=m_high), uncertain=dict(tau_score=q_low, tau_margin=0.0),
               rule="thresholds = quantiles of val in-vocab CORRECT matches: likely ≥ median score & median margin; unknown < 5th pct score")

    def tiers(f):
        likely = (f.score >= q_high) & (f.margin >= m_high)
        unknown = f.score < q_low
        cand = ~likely & ~unknown
        rep = {}
        for name, m in (("likely", likely), ("candidate", cand), ("unknown", unknown)):
            inv = m & ~f.oov
            rep[name] = dict(share=float(m.mean()), n=int(m.sum()), top1_precision_in_vocab=float(f.correct[inv].mean()) if inv.sum() else None,
                             top5_hit_in_vocab=float(f.top5[inv].mean()) if inv.sum() else None, oov_share=float(f.oov[m].mean()) if m.sum() else None)
        return rep
    cal["empirical"] = {"val": tiers(sets["val"]), "test": tiers(sets["test"])}
    if (V4 / "emb_title.parquet").exists():
        d = pd.read_parquet(V4 / "emb_title.parquet"); Zd = np.load(V4 / "emb_title.npy").astype(np.float32)
        full = pd.read_parquet(V4 / "emb_ttrs.parquet")
        cal["empirical"]["E5_youtube_title"] = tiers(feats_for(d.clip_id.tolist(), d.word.tolist(), Zd, full, Zt, oov_sim=False))
    write_json(V4 / "calibration.json", cal); write_json(REP / "v4_calibration.json", cal)
    log.info("calibration: %s", json.dumps(cal, ensure_ascii=False)[:1500])
# ----------------------------------------------------------------------------- segmentation
def cmd_seg_train(a):
    from modules.recognize import load_pose
    from modules.segment import SentenceSynth, evaluate_segmenter, sign_span, train_segmenter_cached
    from modules.wholebody import pose_path
    seed_all(0)
    t = pd.read_parquet(V4 / "emb_ttrs.parquet")
    clips_tr, clips_va = [], []
    for r in t.itertuples():
        kp, sc, hw, _ = load_pose(pose_path(r.clip_id))
        sp = sign_span(kp, sc, hw) or (r.f0, r.f1)
        if sp[1] - sp[0] < 3:
            continue
        c = dict(kp=kp, sc=sc, hw=hw, span=sp, signer=r.signer, word=r.word)
        (clips_va if r.signer == a.val_signer else clips_tr).append(c)
    log.info("segmentation data: train clips %d, val clips %d (held-out signer %s)", len(clips_tr), len(clips_va), a.val_signer)
    enc = enc_()
    model, hist, val = train_segmenter_cached(enc, SentenceSynth(clips_tr, 0), SentenceSynth(clips_va, 1), n_train=a.n_train, epochs=a.epochs)
    torch.save(model.state_dict(), V4 / "segmenter.pt")
    heur = evaluate_segmenter(model, val, heuristic=True)
    learned = evaluate_segmenter(model, val)
    write_json(REP / "v4_segmentation.json", dict(hist=hist, val_learned=learned, val_heuristic=heur, val_signer=a.val_signer))
    log.info("segmentation val (held-out signer): learned %s | heuristic %s", learned, heur)


# ----------------------------------------------------------------------------- spotting (segmentation × recognition)
def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(max(a[1], b[1]) - min(a[0], b[0]), 1)


def spot_metrics(pred, gt, words, vocab_set, tol=2):
    """pred: [dict(f0,f1,nearest,candidates)], gt: [(f0,f1)], words: gt words (same order)."""
    from modules.evalkit import gloss_wer
    from modules.segment import boundary_f1
    ps = [(p["f0"], p["f1"]) for p in pred]
    # segment detection F1 at IoU ≥ 0.3 (word identity ignored)
    used, tp = set(), 0
    for p in ps:
        best = max(((j, _iou(p, g)) for j, g in enumerate(gt) if j not in used), key=lambda x: x[1], default=(None, 0))
        if best[1] >= 0.3:
            used.add(best[0]); tp += 1
    seg_p, seg_r = tp / max(len(ps), 1), tp / max(len(gt), 1)
    inv = [j for j, w in enumerate(words) if w in vocab_set]
    hit1 = hit5 = 0
    for j in inv:
        ov = [p for p in pred if _iou((p["f0"], p["f1"]), gt[j]) > 0]
        hit1 += any(p["nearest"] == words[j] for p in ov)
        hit5 += any(words[j] in [c["word"] for c in p["candidates"]] for p in ov)
    correct = []
    for p in pred:
        j, v = max(((j, _iou((p["f0"], p["f1"]), g)) for j, g in enumerate(gt)), key=lambda x: x[1], default=(None, 0))
        correct.append(bool(v > 0 and words[j] in vocab_set and p["nearest"] == words[j]))
    n_pred_inv = sum(1 for p in pred if max((_iou((p["f0"], p["f1"]), g) for g in gt), default=0) > 0)
    return dict(seg_f1=2 * seg_p * seg_r / max(seg_p + seg_r, 1e-9), boundary_f1=boundary_f1(ps, gt, tol), count_err=abs(len(ps) - len(gt)),
                n_pred=len(ps), n_gt=len(gt), n_inv=len(inv), hit1=hit1, hit5=hit5, n_correct=int(sum(correct)), correct=correct,
                wer=gloss_wer(list(words), [p["nearest"] for p in pred]))


def _agg(rows):
    n_inv = sum(r["n_inv"] for r in rows); n_pred = sum(r["n_pred"] for r in rows)
    rec1 = sum(r["hit1"] for r in rows) / max(n_inv, 1); prec1 = sum(r["n_correct"] for r in rows) / max(n_pred, 1)
    return dict(seg_f1=float(np.mean([r["seg_f1"] for r in rows])), boundary_f1=float(np.mean([r["boundary_f1"] for r in rows])),
                count_err=float(np.mean([r["count_err"] for r in rows])), words_per_sentence_pred=n_pred / len(rows),
                words_per_sentence_gt=sum(r["n_gt"] for r in rows) / len(rows), word_recall_top1=rec1,
                word_recall_top5=sum(r["hit5"] for r in rows) / max(n_inv, 1), word_precision_top1=prec1,
                spot_f1_top1=2 * rec1 * prec1 / max(rec1 + prec1, 1e-9), WER=float(np.mean([r["wer"] for r in rows])), n_sentences=len(rows))


def cmd_spot_eval(a):
    """Held-out signer synthetic sentences (bank WITHOUT that signer): old tagger decoder vs recognition-driven spotting.
    λ/γ tuned on split 'tune', reported on split 'test' (different sentences, same held-out signer)."""
    from modules.pipeline_v4 import load_bank
    from modules.recognize import VocabBank, load_pose
    from modules.segment import SentenceSynth, Segmenter, decode_segments, gt_segments, sequence_features, sign_span
    from modules.spotting import active_frames, decode, window_scores, window_table
    from modules.wholebody import pose_path
    seed_all(0)
    t = pd.read_parquet(V4 / "emb_ttrs.parquet")
    clips = []
    for r in t[t.signer == a.signer].itertuples():
        kp, sc, hw, _ = load_pose(pose_path(r.clip_id))
        sp = sign_span(kp, sc, hw) or (r.f0, r.f1)
        if sp[1] - sp[0] >= 3:
            clips.append(dict(kp=kp, sc=sc, hw=hw, span=sp, signer=r.signer, word=r.word))
    full = load_bank()
    keep = np.array([m["signer"] != a.signer for m in full.meta])
    bank = VocabBank(full.Z[keep], [w for w, k in zip(full.words, keep) if k])
    vocab_set = set(bank.vocab)
    log.info("spot eval: signer %s, %d clips; bank without signer: %d instances / %d words; clip words in vocab %.2f",
             a.signer, len(clips), keep.sum(), len(bank.vocab), np.mean([c["word"] in vocab_set for c in clips]))
    enc = enc_()
    seg = Segmenter().cuda(); seg.load_state_dict(torch.load(V4 / "segmenter.pt")); seg.eval()
    import pickle
    cache_p = V4 / f"spot_eval_cache_{a.signer}_{a.n_sent}.pkl"
    data = pickle.load(open(cache_p, "rb")) if cache_p.exists() else {}
    for split, seed in (("tune", 101), ("test", 202)):
        if split in data:
            continue
        synth = SentenceSynth(clips, seed)
        rows = []
        t0 = time.time()
        for i in range(a.n_sent):
            s = synth.sample()
            x = sequence_features(enc, s["kp"], s["sc"], s["hw"])
            with torch.no_grad():
                prob = seg(torch.from_numpy(x)[None].cuda()).softmax(-1)[0].cpu().numpy()
            act = active_frames(prob)
            base_segs = decode_segments(prob)
            base = []
            if base_segs:
                _, S = window_scores(enc, bank, s["kp"], s["sc"], base_segs)
                for (f0, f1), row in zip(base_segs, S):
                    o = np.argsort(-row)[:5]
                    base.append(dict(f0=f0, f1=f1, nearest=bank.vocab[o[0]], score=float(row[o[0]]), margin=float(row[o[0]] - row[o[1]]),
                                     candidates=[dict(word=bank.vocab[j], score=float(row[j])) for j in o]))
            rows.append(dict(gt=gt_segments(s["y"]), words=s["words"], act=act, prob=prob.astype(np.float32), table=window_table(enc, bank, s["kp"], s["sc"], act), base=base))
            if (i + 1) % 20 == 0:
                log.info("%s %d/%d (%.0fs)", split, i + 1, a.n_sent, time.time() - t0)
        data[split] = rows
        pickle.dump(data, open(cache_p, "wb"))
    res = {}
    for split in ("tune", "test"):
        res[f"{split}|tagger_decoder"] = _agg([spot_metrics(r["base"], r["gt"], r["words"], vocab_set) for r in data[split]])
        log.info("%s tagger decoder: %s", split, res[f"{split}|tagger_decoder"])
    grid = []
    # β = 0 transfers to real video (the tagger's begin peaks vanish on fluent signing); β = 2 (tagger begin prior) is reported for reference
    for beta in (0.0, 2.0):
        for min_len in (8, 10, 12, 14, 16):
            for lam in (8.0, 9.0, 10.0, 12.0):
                for gap in (0.3, 0.6, 1.0, 2.0):
                    m = _agg([spot_metrics(decode(r["table"], r["act"], bank.vocab, lam, gap, min_len=min_len, prob=r["prob"], beta=beta), r["gt"], r["words"], vocab_set) for r in data["tune"]])
                    m["objective"] = m["spot_f1_top1"] + 0.5 * m["seg_f1"] + 0.5 * m["boundary_f1"] - 0.05 * m["count_err"]
                    grid.append(dict(beta=beta, min_len=min_len, lam=lam, gap_cost=gap, **m))
    g = pd.DataFrame(grid).sort_values("objective", ascending=False)
    cols = ["beta", "min_len", "lam", "gap_cost", "seg_f1", "boundary_f1", "count_err", "words_per_sentence_pred", "word_recall_top1", "word_recall_top5", "word_precision_top1", "objective"]
    log.info("spotting grid (tune, top 6 per β):\n%s", g.groupby("beta").head(6)[cols].round(3).to_string())
    gb = g[g.beta == 2.0].iloc[0]
    res["test|spotting+tagger_begin_prior"] = _agg([spot_metrics(decode(r["table"], r["act"], bank.vocab, gb.lam, gb.gap_cost, min_len=int(gb.min_len), prob=r["prob"], beta=2.0), r["gt"], r["words"], vocab_set) for r in data["test"]])
    log.info("test spotting + begin prior: %s", res["test|spotting+tagger_begin_prior"])
    best = g[g.beta == 0.0].iloc[0]
    lam, gap, min_len = float(best.lam), float(best.gap_cost), int(best.min_len)
    # tiers for spotted segments: fitted on tune picks (z of correct vs wrong), measured on test
    feats = {}
    for split in ("tune", "test"):
        fs, rows_m = [], []
        for r in data[split]:
            segs = decode(r["table"], r["act"], bank.vocab, lam, gap, min_len=min_len)
            m = spot_metrics(segs, r["gt"], r["words"], vocab_set); rows_m.append(m)
            fs += [dict(z=s_["z"], score=s_["score"], margin=s_["margin"], correct=c) for s_, c in zip(segs, m["correct"])]
        feats[split] = pd.DataFrame(fs)
        res[f"{split}|spotting"] = _agg(rows_m)
        log.info("%s spotting (λ=%.1f γ=%.2f min_len=%d): %s", split, lam, gap, min_len, res[f"{split}|spotting"])
    ft = feats["tune"]
    z_like = float(ft[ft.correct].z.quantile(0.5)) if ft.correct.sum() >= 5 else float(ft.z.quantile(0.8))
    z_unk = float(ft[ft.correct].z.quantile(0.1)) if ft.correct.sum() >= 5 else float(ft.z.quantile(0.2))
    tiers = {}
    for split, f in feats.items():
        tier = np.where(f.z >= z_like, "likely", np.where(f.z >= z_unk, "candidate", "unknown"))
        tiers[split] = {k: dict(share=float((tier == k).mean()), n=int((tier == k).sum()), top1_precision=float(f.correct[tier == k].mean()) if (tier == k).any() else None)
                        for k in ("likely", "candidate", "unknown")}
    cfg = dict(lam=lam, gap_cost=gap, min_len=min_len, agg="sum", lens=list(range(6, 31, 2)), stride=2, act_thr=0.5, z_likely=z_like, z_unknown=z_unk,
               tuned_on=f"synthetic sentences, held-out signer {a.signer}, bank without that signer", tiers_empirical=tiers)
    write_json(V4 / "spotting.json", cfg)
    write_json(REP / "v4_spotting.json", dict(results=res, grid=grid, config=cfg))
    log.info("spotting config: %s", {k: v for k, v in cfg.items() if k != "tiers_empirical"}); log.info("tiers: %s", tiers)


# ----------------------------------------------------------------------------- inference on data_test
REFS = {"ฉันปลอบเพื่อนร้องไห้": ["ฉัน", "ปลอบใจ", "เพื่อน", "ร้องไห้"],
        "ไปทานข้าวด้วยกันมั้ย": ["ไป", "กิน", "ข้าว", "ด้วยกัน", "ไหม"]}
SYN = {"ทาน": "กิน", "รับประทาน": "กิน", "มั้ย": "ไหม", "ปลอบ": "ปลอบใจ", "ร่วมกัน": "ด้วยกัน"}


def cmd_infer(a):
    from modules.evalkit import chrf, gloss_wer
    from modules.pipeline_v4 import SignPipelineV4
    pipe = SignPipelineV4(use_llm=not a.no_llm, spotting=a.decoder == "spotting")
    out = {}
    for stem, ref in REFS.items():
        od = ROOT / "result_reporting" / "inference_v4" / (stem if a.tag == "final" else f"_{a.tag}" + "/" + stem)
        r = pipe.run(ROOT / "data_test" / f"{stem}.mp4", od, tts=a.tts)
        canon = lambda w: SYN.get(w, w)
        refc = [canon(w) for w in ref]
        acc_words = [canon(s["nearest"]) for s in r["segments"] if s["status"] == "accepted"]
        top5_hit = {w: any(w in [canon(c["word"]) for c in s["candidates"]] for s in r["segments"]) for w in refc}
        rank_of = {}
        for w in refc:
            best = None
            for s in r["segments"]:
                ws = [canon(c["word"]) for c in s["candidates"]]
                if w in ws:
                    best = min(best or 99, ws.index(w) + 1)
            rank_of[w] = best
        fusion = r.get("fusion") or {}
        chosen = [canon(w) for w in fusion.get("chosen", []) if w != "[?]"]
        rec = dict(ref=refc, n_segments=len(r["segments"]), n_ref=len(refc),
                   segments=[(s["t0"], s["t1"], s["status"], [c["word"] for c in s["candidates"]], round(s["score"], 3)) for s in r["segments"]],
                   accepted_words=acc_words, WER_accepted=gloss_wer(refc, acc_words), WER_llm_chosen=gloss_wer(refc, chosen) if fusion else None,
                   ref_in_top5_any_segment=top5_hit, best_rank=rank_of, sentence=fusion.get("sentence_th"),
                   chrF=chrf("".join(ref), fusion.get("sentence_th", "") or "") if fusion else None, fusion=fusion, face=r["face"],
                   tentative_sentence=fusion.get("tentative_sentence_th"), evidence_gloss=fusion.get("evidence_gloss"),
                   timings_ms=r["timings_ms"])
        rec["decoder"] = a.decoder
        out[stem] = rec
        log.info("%s: segs=%d (ref %d) accepted=%s top5=%s rank=%s sentence=%s", stem, len(r["segments"]), len(refc), acc_words, top5_hit, rank_of, rec["sentence"])
    write_json(REP / f"v4_infer_{a.tag}.json", out)


def cmd_lm_eval(a):
    """Does LLM decoding over top-K candidates recover real sentences better than top-1 — or does it guess?
    Sentences: GPT writes natural Thai sentences using only words that TTRS has from ≥2 signers (it never sees the
    recogniser output). Each word is 'signed' with a real TTRS clip; the bank excludes that clip AND every clip of the
    same signer for that word (cross-signer). Oracle segments → isolates recognition + language decoding."""
    import os
    from openai import OpenAI
    from modules.env import load_env
    from modules.recognize import VocabBank
    from modules.translate import compose
    load_env()
    rng = np.random.RandomState(a.seed)
    t = pd.read_parquet(V4 / "emb_ttrs.parquet"); Zt = np.load(V4 / "emb_ttrs.npy").astype(np.float32)
    sig = t.groupby("word").signer.nunique()
    W = sorted([w for w in sig[sig >= 2].index if isinstance(w, str) and 1 < len(w) <= 12])
    sent_path = V4 / "lm_eval_sentences.json"
    if sent_path.exists():
        sents = read_json(sent_path)
    else:
        listing = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(W))
        prompt = ("รายการ gloss ภาษามือไทยที่อนุญาต (คัดลอกคำให้ตรงตัวอักษรทุกตัว):\n" + listing +
                  f"\n\nจงแต่ง {a.n_sent} ประโยคสั้นที่มีความหมายเป็นธรรมชาติ แต่ละประโยคใช้ gloss 3–5 คำ ทุกคำต้องคัดลอกจากรายการด้านบนเท่านั้น "
                  "(ห้ามผันคำ ห้ามเติมคำเชื่อม ห้ามใช้คำนอกรายการ) เรียงแบบภาษามือได้ "
                  "ตอบ JSON: {\"sentences\": [{\"glosses\": [\"คำ1\", \"คำ2\", \"คำ3\"], \"thai\": \"ประโยคภาษาไทยปกติ\"}]}")
        r = OpenAI(timeout=120).chat.completions.create(model="gpt-4.1", temperature=0.4,
                                                        response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
        raw = json.loads(r.choices[0].message.content)["sentences"]
        sents = []
        for s in raw:
            g = [x.strip() for x in s.get("glosses", []) if x.strip() in W]
            if len(g) >= 3:
                sents.append(dict(glosses=g, thai=s.get("thai"), dropped=[x for x in s.get("glosses", []) if x.strip() not in W]))
        write_json(sent_path, sents)
        write_json(V4 / "lm_eval_sentences_raw.json", raw)
    log.info("LM eval: %d valid sentences over %d cross-signer words", len(sents), len(W))
    res, wers = [], {"top1": [], "llm": [], "oracle_top5": []}
    from modules.evalkit import gloss_wer
    viol = n_words = 0
    for s in sents:
        segs, ref = [], s["glosses"]
        for k, g in enumerate(ref):
            cand_rows = t.index[t.word == g].tolist()
            qi = cand_rows[rng.randint(len(cand_rows))]
            qsig = t.signer.iloc[qi]
            keep = ~((t.word == g) & (t.signer == qsig)).values
            bank = VocabBank(Zt[keep], t.word[keep].tolist())
            sc = bank.word_scores(Zt[qi][None])[0]
            o = np.argsort(-sc)[:5]
            segs.append(dict(t0=float(k), t1=float(k + 1), status="uncertain",
                             candidates=[dict(word=bank.vocab[j], score=float(sc[j])) for j in o]))
        top1 = [x["candidates"][0]["word"] for x in segs]
        fusion = compose(segs, dict(emotion="neutral", probs={}, nmm={}))
        chosen = fusion.get("chosen", [])
        viol += len(fusion.get("guard_violations", [])); n_words += len(ref)
        oracle = [g if g in [c["word"] for c in x["candidates"]] else x["candidates"][0]["word"] for g, x in zip(ref, segs)]
        wers["top1"].append(gloss_wer(ref, top1)); wers["llm"].append(gloss_wer(ref, [c for c in chosen if c != "[?]"]))
        wers["oracle_top5"].append(gloss_wer(ref, oracle))
        acc = lambda hyp: float(np.mean([h == r_ for h, r_ in zip(hyp, ref)])) if len(hyp) == len(ref) else float("nan")
        res.append(dict(ref=ref, thai=s.get("thai"), top1=top1, llm=chosen, llm_sentence=fusion.get("sentence_th"), tentative=fusion.get("tentative_sentence_th"),
                        acc_top1=acc(top1), acc_llm=acc(chosen), acc_oracle5=acc(oracle), in_top5=[g in [c["word"] for c in x["candidates"]] for g, x in zip(ref, segs)]))
    df = pd.DataFrame(res)
    summary = dict(n_sentences=len(df), n_words=n_words, word_acc_top1=float(df.acc_top1.mean()), word_acc_llm=float(df.acc_llm.mean()),
                   word_acc_oracle_top5=float(df.acc_oracle5.mean()), WER_top1=float(np.mean(wers["top1"])), WER_llm=float(np.mean(wers["llm"])),
                   WER_oracle_top5=float(np.mean(wers["oracle_top5"])), llm_out_of_candidate_words=viol,
                   llm_unknown_rate=float(np.mean([c == "[?]" for r_ in res for c in r_["llm"]])) if res else 0.0,
                   llm_precision_resolved_words=float(np.mean([h == g for r_ in res for h, g in zip(r_["llm"], r_["ref"]) if h != "[?]"] or [np.nan])),
                   top1_precision=float(np.mean([h == g for r_ in res for h, g in zip(r_["top1"], r_["ref"])])),
                   sentences_withheld=float(np.mean([not r_["llm_sentence"] for r_ in res])), model=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    write_json(REP / "v4_lm_eval.json", dict(summary=summary, sentences=res))
    log.info("LM decoding eval: %s", summary)


def cmd_mined_sheet(a):
    """Audit sheet: random mined instances (lesson video frames) next to the TTRS reference clip of the same word."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from modules.data import decode_frames
    m = pd.read_parquet(V4 / "mined.parquet")
    c = pd.read_parquet(MAN / "clips_signers.parquet").set_index("clip_id")
    t = pd.read_parquet(V4 / "emb_ttrs.parquet")
    rng = np.random.RandomState(a.seed)
    rows, labels = [], []
    for i in rng.choice(len(m), min(a.n_show, len(m)), replace=False):
        r = m.iloc[i]
        fr = list(decode_frames(ROOT / c.loc[r.clip_id, "media_path"], fps=12.5, start_s=float(r.t0), max_s=float(r.t1 - r.t0) + 0.1))
        if not fr:
            continue
        fr = np.concatenate(fr)
        ims = [cv2.resize(fr[j], (160, 90)) for j in np.linspace(0, len(fr) - 1, 4).astype(int)]
        ref = t[t.word == r.word].iloc[0]
        rf = np.concatenate(list(decode_frames(ROOT / c.loc[ref.clip_id, "media_path"], fps=12.5)))
        ims += [cv2.resize(rf[j], (160, 90)) for j in np.linspace(ref.f0, ref.f1 - 1, 3).astype(int)]
        rows.append(np.concatenate(ims, 1)); labels.append(f"{r.word}  rank={r['rank']} score={r.score:.2f}  {r.clip_id} {r.t0:.1f}s  | right: TTRS reference")
    sheet = np.concatenate([np.pad(x, ((20, 0), (0, 0), (0, 0)), constant_values=255) for x in rows], 0)
    img = Image.fromarray(sheet[..., ::-1]); d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 13)
    for k, lab in enumerate(labels):
        d.text((4, k * 110 + 2), lab, fill=(0, 0, 0), font=font)
    out = ROOT / "result_reporting" / "v4_mined_examples.jpg"; out.parent.mkdir(exist_ok=True)
    img.save(out, quality=88)
    log.info("mined audit sheet → %s", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed_ttrs"); e.add_argument("--n_aug", type=int, default=2)
    sub.add_parser("embed_titles")
    m = sub.add_parser("mine"); m.add_argument("--rank_k", type=int, default=3)
    sub.add_parser("bank"); sub.add_parser("eval")
    ad = sub.add_parser("adapter"); ad.add_argument("--steps", type=int, default=3000)
    le = sub.add_parser("lm_eval"); le.add_argument("--n_sent", type=int, default=40); le.add_argument("--seed", type=int, default=0)
    ms = sub.add_parser("mined_sheet"); ms.add_argument("--n_show", type=int, default=16); ms.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("calib"); c.add_argument("--precision", type=float, default=0.8); c.add_argument("--top5_precision", type=float, default=0.5)
    s = sub.add_parser("seg_train"); s.add_argument("--n_train", type=int, default=2400); s.add_argument("--epochs", type=int, default=15); s.add_argument("--val_signer", default="S_ttrs_00")
    se = sub.add_parser("spot_eval"); se.add_argument("--signer", default="S_ttrs_00"); se.add_argument("--n_sent", type=int, default=120)
    i = sub.add_parser("infer"); i.add_argument("--tag", default="v4"); i.add_argument("--no_llm", action="store_true"); i.add_argument("--tts", action="store_true"); i.add_argument("--decoder", default="spotting")
    a = ap.parse_args()
    dict(spot_eval=cmd_spot_eval, embed_ttrs=cmd_embed_ttrs, embed_titles=cmd_embed_titles, mine=cmd_mine, bank=cmd_bank, eval=cmd_eval, calib=cmd_calib, adapter=cmd_adapter, mined_sheet=cmd_mined_sheet, lm_eval=cmd_lm_eval,
         seg_train=cmd_seg_train, infer=cmd_infer)[a.cmd](a)


if __name__ == "__main__":
    main()
