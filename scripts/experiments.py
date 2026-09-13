"""Build → test → optimize loop. Every stage writes JSON into artifacts/reports/ for result_reporting.

  python scripts/experiments.py pack --set ttrs
  python scripts/experiments.py signers
  python scripts/experiments.py baselines
  python scripts/experiments.py ssl --steps 6000 --tag ssl_v1
  python scripts/experiments.py islr --init ssl_v1 --tag islr_v1
  python scripts/experiments.py continuous --tag islr_v1
  python scripts/experiments.py realworld --tag islr_v1
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

log = get_logger("exp")
REP = ART / "reports"
MAN = ART / "manifests"


# ----------------------------------------------------------------------------- shared context
class Ctx:
    def __init__(self, need_store=True):
        self.clips = pd.read_parquet(MAN / "clips_signers.parquet") if (MAN / "clips_signers.parquet").exists() \
            else pd.read_parquet(MAN / "clips.parquet")
        self.splits = read_json(MAN / "splits.json")
        self.lex = pd.read_parquet(MAN / "lexicon.parquet")
        if need_store:
            from modules.encoder import FeatureStore
            self.store = FeatureStore("ttrs")
        t = self.clips[self.clips.source == "ttrs"]
        if need_store:
            t = t[t.clip_id.isin(self.store.row)]
        self.ttrs = t
        unit = self.splits["unit"]
        self.clip_unit = dict(zip(t.clip_id, t[unit]))
        self.clip_signer = {c: s for c, s in zip(t.clip_id, t.signer_id) if isinstance(s, str)}
        self.unit_lemma = dict(zip(t[unit], t.lemma))
        self.train_clips = [c for c in self.splits["train_islr"] if c in self.clip_unit]
        self.unit_clips_train: dict = {}
        for c in self.train_clips:
            self.unit_clips_train.setdefault(self.clip_unit[c], []).append(c)

    def eval_units(self, part):
        units = self.splits[f"{part}_units"]
        uc = {}
        for c in self.splits[f"{part}_clips"]:
            if c in self.clip_signer and c in self.clip_unit:
                uc.setdefault(self.clip_unit[c], []).append(c)
        units = [u for u in units if u in uc and len({self.clip_signer[c] for c in uc[u]}) >= 2]
        return units, uc

    def retrieval_seen(self, embed_fn, part="val", version="base", csls=(0,), ks=(1, 5, 10), center=(False,),
                       extra_session_clips=0):
        """E1 dictionary retrieval: query = held-out clip of a new signer; gallery = prototypes of ALL training signs
        (the sign itself is represented by the other signer's training clip).
        center=True: session centering in embedding space — bank prototypes minus their signer's mean prototype,
        query minus the mean embedding of the same signer's session (the other eval queries of that signer)."""
        from modules.evalkit import embed_clips, knn_mean_sim, unit_prototypes
        qs = [c for c in self.splits[f"seen_{part}_queries"] if c in self.clip_unit]
        zq = embed_clips(embed_fn, self.store, qs, version)
        et = dict(zip(self.train_clips, embed_clips(embed_fn, self.store, self.train_clips, version)))
        keys = sorted(self.unit_clips_train)
        P0 = unit_prototypes(et, self.unit_clips_train, keys)
        kidx = {k: i for i, k in enumerate(keys)}
        tgt = np.array([kidx[self.clip_unit[c]] for c in qs])
        out = {}
        for cen in center:
            P, Q = P0, zq
            if cen:
                psig = []
                for k_ in keys:
                    s = [self.clip_signer.get(c) for c in self.unit_clips_train[k_]]
                    s = [x for x in s if x]
                    psig.append(max(set(s), key=s.count) if s else "unk")
                psig = np.array(psig)
                P = P0.copy()
                for s in set(psig):
                    if (psig == s).sum() >= 3: P[psig == s] -= P0[psig == s].mean(0)
                P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-6
                qsig = np.array([self.clip_signer[c] for c in qs])
                Q = zq.copy()
                for s in set(qsig):
                    if (qsig == s).sum() >= 3: Q[qsig == s] -= zq[qsig == s].mean(0)
                Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-6
            for k in csls:
                S = Q @ P.T
                if k:
                    S = S - knn_mean_sim(P, P, k, self_mask=True)[None]
                rank = (S > S[np.arange(len(qs)), tgt][:, None]).sum(1)
                r = {f"R@{kk}": float((rank < kk).mean()) for kk in ks}
                r["MRR"] = float((1 / (rank + 1)).mean())
                for kk in (1, 5):
                    p = r[f"R@{kk}"]; r[f"R@{kk}_ci95"] = 1.96 * float(np.sqrt(p * (1 - p) / max(len(qs), 1)))
                r.update(n_queries=len(qs), gallery_size=len(keys), median_rank=float(np.median(rank) + 1))
                out[f"csls{k}" + ("_center" if cen else "")] = r
        return out

    def eval_all(self, embed_fn, part="val", version="base"):
        e1s = self.retrieval_seen(embed_fn, part, version, center=(False, True))
        e2 = self.retrieval(embed_fn, part, with_distractors=False, version=version)
        r = dict(e1s["csls0"])
        r.update({f"C_{k}": v for k, v in e1s["csls0_center"].items() if k in ("R@1", "R@5", "R@10", "MRR", "median_rank")})
        r.update({f"E2_{k}": v for k, v in e2.items() if "small" in k})
        return r

    def retrieval_multi(self, embed_fn, part="val", csls=(0, 10), version="base"):
        """Embed once, score with and without CSLS."""
        from modules.evalkit import embed_clips, retrieval_eval, unit_prototypes
        units, uc = self.eval_units(part)
        qc = sorted({c for u in units for c in uc[u]})
        emb = dict(zip(qc, embed_clips(embed_fn, self.store, qc, version)))
        et = dict(zip(self.train_clips, embed_clips(embed_fn, self.store, self.train_clips, version)))
        P = unit_prototypes(et, self.unit_clips_train, sorted(self.unit_clips_train))
        return {f"csls{k}": retrieval_eval(emb, units, uc, self.clip_signer, None, P, csls_k=k) for k in csls}

    def retrieval(self, embed_fn, part="val", with_distractors=True, version="base"):
        from modules.evalkit import embed_clips, retrieval_eval, unit_prototypes
        units, uc = self.eval_units(part)
        qc = sorted({c for u in units for c in uc[u]})
        z = embed_clips(embed_fn, self.store, qc, version)
        emb = dict(zip(qc, z))
        P = None
        if with_distractors:
            tr = self.train_clips
            zt = embed_clips(embed_fn, self.store, tr, version)
            et = dict(zip(tr, zt))
            P = unit_prototypes(et, self.unit_clips_train, sorted(self.unit_clips_train))
        return retrieval_eval(emb, units, uc, self.clip_signer, None, P)


# ----------------------------------------------------------------------------- stages
def st_pack(a):
    from modules.encoder import pack
    from modules.utils import load_cfg
    clips = pd.read_parquet(MAN / "clips.parquet")
    cfg = load_cfg("data")
    if a.set == "ttrs":
        ids = clips[clips.source == "ttrs"].clip_id.tolist()
        pack("ttrs", ids, cfg["cache"]["versions_ttrs"], mode_for=lambda c: "clip")
    else:
        sp = read_json(MAN / "splits.json")
        ids = clips[(clips.source == "youtube") & clips.clip_id.isin(sp["ssl_pool"])].clip_id.tolist()
        pack("youtube", ids, cfg["cache"]["versions_other"], mode_for=lambda c: "stream")


def st_signers(a):
    from modules.data import assign_signers, make_splits
    from modules.encoder import FeatureStore
    clips = pd.read_parquet(MAN / "clips.parquet")
    st = FeatureStore("ttrs", ["base"])
    ff = {}
    for cid, s, n in st.index.itertuples(index=False):
        v = st.valid[s:s + n, 2].astype(bool)
        if v.sum():
            ff[cid] = np.asarray(st.feat["base"][s:s + n][v, 2, :384], np.float32).mean(0)
    c, rep = assign_signers(clips, ff, min_prob=a.min_prob, cluster_thr=a.cluster_thr)
    c.to_parquet(MAN / "clips_signers.parquet", index=False)
    sp = make_splits(c)
    rep["signer_counts"] = c[c.source == "ttrs"].signer_id.value_counts(dropna=False).to_dict()
    rep["n_val_units"], rep["n_test_units"] = len(sp["val_units"]), len(sp["test_units"])
    rep["n_val_clips"], rep["n_test_clips"] = len(sp["val_clips"]), len(sp["test_clips"])
    write_json(REP / "signers.json", rep)
    log.info("%s", rep)


def st_baselines(a):
    from modules.evalkit import frozen_embed_fn
    from modules.normalize import NAP
    ctx = Ctx()
    res = {}
    st = ctx.store

    def run(name, fn, active=True, part="val"):
        from modules.evalkit import embed_clips, retrieval_eval, unit_prototypes
        t0 = time.time()
        units, uc = ctx.eval_units(part)
        qc = sorted({c for u in units for c in uc[u]})
        emb = dict(zip(qc, embed_clips(fn, st, qc, "base", active_only=active)))
        et = dict(zip(ctx.train_clips, embed_clips(fn, st, ctx.train_clips, "base", active_only=active)))
        P = unit_prototypes(et, ctx.unit_clips_train, sorted(ctx.unit_clips_train))
        r = retrieval_eval(emb, units, uc, ctx.clip_signer, None, P)
        # nearest-neighbour same-signer rate (identity leakage diagnostic)
        keys = list(emb); Z = np.stack([emb[k] for k in keys]); S = Z @ Z.T; np.fill_diagonal(S, -9)
        nn = S.argmax(1)
        r["nn_same_signer"] = float(np.mean([ctx.clip_signer[keys[i]] == ctx.clip_signer[keys[j]] for i, j in enumerate(nn)]))
        r["sec"] = round(time.time() - t0, 1)
        e1 = ctx.retrieval_seen(fn, part)["csls0"]
        r.update({f"E1_{k}": v for k, v in e1.items()})
        res[f"{name}|{part}"] = r
        log.info("%-40s %s E1 R@1 %.3f R@5 %.3f R@10 %.3f | E2 small R@1 %.3f R@5 %.3f | nn_same %.2f", name, part, r["E1_R@1"],
                 r["E1_R@5"], r["E1_R@10"], r["R@1_small"], r["R@5_small"], r["nn_same_signer"])

    run("frozen_meanpool_allframes", frozen_embed_fn(), active=False)
    run("frozen_meanpool_active", frozen_embed_fn(), active=True)
    run("frozen_active_cls_only", frozen_embed_fn(part="cls"), active=True)
    run("frozen_active_hands_only", frozen_embed_fn(streams=(0, 1)), active=True)
    run("frozen_active_+pose", frozen_embed_fn(use_pose=True), active=True)
    # NAP fitted on train-split signers, active frames
    by = {}
    for c in ctx.train_clips:
        s = ctx.clip_signer.get(c)
        if s is None:
            continue
        f, _, v, act = st.clip(c, "base", active_only=True)
        by.setdefault(s, []).append(f[::4])
    by = {s: np.concatenate(v) for s, v in by.items() if len(v) >= 20}
    log.info("NAP fit signers: %s", {s: len(v) for s, v in by.items()})
    best_d, best = 0, -1
    for d in range(1, len(by)):
        nap = NAP.fit(by, d)
        run(f"frozen_active_nap_d{d}", frozen_embed_fn(nap.N), active=True)
        key = res[f"frozen_active_nap_d{d}|val"]["R@5_small"] + res[f"frozen_active_nap_d{d}|val"]["R@1_small"]
        if key > best:
            best, best_d = key, d
    nap = NAP.fit(by, best_d)
    nap.save(ART / "nap" / "nap.npz")
    res["nap_best_d"] = best_d
    run(f"frozen_active_nap_d{best_d}", frozen_embed_fn(nap.N), active=True, part="test")
    run("frozen_meanpool_active", frozen_embed_fn(), active=True, part="test")
    write_json(REP / "baselines.json", res)


def st_ssl(a):
    from modules.encoder import FeatureStore
    from modules.normalize import NAP
    from modules.ssl import SSLSampler, train_ssl
    seed_all(a.seed)
    ctx = Ctx()
    sp = ctx.splits
    pool = set(sp["ssl_pool"])
    stores = {"ttrs": ctx.store}
    clip_lists = {"ttrs": [c for c in ctx.store.row if c in pool]}
    weights = {"ttrs": 0.35}
    yt_dir = ART / "cache" / "dinov2s_112" / "packed" / "youtube"
    if (yt_dir / "index.parquet").exists() and not a.no_youtube:
        ys = FeatureStore("youtube")
        stores["youtube"] = ys
        clip_lists["youtube"] = [c for c in ys.row if c in pool]
        weights["youtube"] = 0.65
    nap_N = NAP.load().N if a.nap else None
    sampler = SSLSampler(stores, weights, clip_lists, seed=a.seed)

    def eval_fn(teacher):
        from modules.heads import pooled
        teacher.eval()
        fn = lambda f, p, v, m: pooled(teacher(f, p, v, m), m, mode="cls")
        return ctx.eval_all(fn, "val")

    kw = dict(d=a.d, layers=a.layers, heads=a.heads)
    teacher, hist = train_ssl(sampler, nap_N, steps=a.steps, batch=a.batch, lr=a.lr, K=a.K, model_kw=kw,
                              ckpt_path=ART / "checkpoints" / "ssl" / f"{a.tag}.pt", eval_fn=eval_fn, eval_every=a.eval_every)
    from modules.heads import pooled
    fn = lambda f, p, v, m: pooled(teacher(f, p, v, m), m, mode="cls")
    r = dict(val=ctx.eval_all(fn, "val"), test=ctx.eval_all(fn, "test"))
    write_json(REP / f"ssl_{a.tag}.json", dict(args=vars(a), hist=hist, knn_val=r))
    log.info("SSL %s kNN val: %s", a.tag, r)


def build_islr(ctx, a):
    from modules.heads import ISLRModel, ISLRSampler
    from modules.normalize import NAP
    from modules.ssl import SignEncoder, load_encoder
    units = sorted(ctx.unit_clips_train)
    uidx = {u: i for i, u in enumerate(units)}
    signers = sorted({s for c, s in ctx.clip_signer.items() if c in set(ctx.train_clips)})
    sidx = {s: i for i, s in enumerate(signers)}
    if a.init and a.init != "scratch":
        enc = load_encoder(ART / "checkpoints" / "ssl" / f"{a.init}.pt")
        enc.train()
    else:
        nap_N = NAP.load().N if a.nap else None
        enc = SignEncoder(d=a.d, layers=a.layers, heads=a.heads, nap_N=nap_N).cuda()
    lora = None
    if a.lora:
        enc.enable_lora(a.lora, 2 * a.lora)
        lora = dict(r=a.lora, alpha=2 * a.lora)
    elif a.freeze:
        for p in enc.parameters():
            p.requires_grad_(False)
    streams = tuple(a.streams.split(","))
    model = ISLRModel(enc, len(units), len(signers), pool=a.pool, streams=streams).cuda()
    versions = tuple(a.versions.split(","))
    sampler = ISLRSampler(ctx.store, ctx.unit_clips_train, uidx, ctx.clip_signer, sidx, versions=versions, seed=a.seed,
                          speed=(0.8, a.speed_max))
    return model, sampler, units, signers, lora, streams


def st_islr(a):
    from modules.heads import PrototypeBank, train_islr
    seed_all(a.seed)
    ctx = Ctx()
    model, sampler, units, signers, lora, streams = build_islr(ctx, a)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("ISLR %s: %d classes, %d signers, trainable params %.2fM", a.tag, len(units), len(signers), ntrain / 1e6)

    def eval_fn(m):
        m.eval()
        with torch.no_grad():
            return ctx.eval_all(m.embed, "val")

    model, hist = train_islr(model, sampler, steps=a.steps, P=a.P, lr=a.lr, enc_lr=a.enc_lr, w_adv=a.w_adv,
                             w_arc=a.w_arc, eval_fn=eval_fn, eval_every=a.eval_every, center_train=a.center_train)
    model.eval()
    with torch.no_grad():
        val = ctx.eval_all(model.embed, "val")
        test = ctx.eval_all(model.embed, "test")
        val_aug = ctx.eval_all(model.embed, "val", version="aug1"); test_aug = ctx.eval_all(model.embed, "test", version="aug1")
    ck = ART / "checkpoints" / "islr" / f"{a.tag}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), enc_cfg=model.encoder.cfg, n_classes=len(units), n_signers=len(signers),
                    pool=a.pool, streams=streams, lora=lora, units=units, args=vars(a)), ck)
    write_json(REP / f"islr_{a.tag}.json", dict(args=vars(a), hist=hist, val=val, test=test, val_aug1_queries=val_aug, test_aug1_queries=test_aug,
                                               trainable_params=ntrain))
    log.info("ISLR %s val %s", a.tag, val)
    log.info("ISLR %s test %s", a.tag, test)


def st_evalckpt(a):
    """Re-score saved checkpoints: E1 (dictionary retrieval) × CSLS k × session centering × query version
    (base = studio clip, aug1 = simulated webcam: new background, low-res, colour); E2 small gallery."""
    from modules.pipeline import load_islr
    ctx = Ctx()
    out = {}
    for tag in a.tags.split(","):
        m = load_islr(ART / "checkpoints" / "islr" / f"{tag}.pt")
        for part in ("val", "test"):
            for ver in ("base", "aug1"):
                r = ctx.retrieval_seen(m.embed, part, ver, csls=(0, 10), center=(False, True))
                e2 = ctx.retrieval(m.embed, part, with_distractors=False, version=ver)
                for k, v in r.items():
                    out[f"{tag}|{part}|{ver}|E1|{k}"] = v
                    log.info("%s %s %s E1 %-14s R@1 %.3f R@5 %.3f R@10 %.3f med_rank %.0f", tag, part, ver, k,
                             v["R@1"], v["R@5"], v["R@10"], v["median_rank"])
                out[f"{tag}|{part}|{ver}|E2"] = e2
                log.info("%s %s %s E2 small R@1 %.3f R@5 %.3f", tag, part, ver, e2["R@1_small"], e2["R@5_small"])
    write_json(REP / f"evalckpt_{a.out}.json", out)


def st_bank(a):
    """Prototype bank over ALL TTRS sign variants (train + val + test: at deployment every dictionary clip is enrolled)."""
    from modules.evalkit import embed_clips
    from modules.heads import PrototypeBank
    from modules.pipeline import load_islr
    ctx = Ctx()
    model = load_islr(ART / "checkpoints" / "islr" / f"{a.tag}.pt")
    cids = sorted(ctx.store.row)
    z = embed_clips(model.embed, ctx.store, cids, "base")
    if a.with_flip:
        z = z + embed_clips(model.embed, ctx.store, cids, "flip")
    by = {}
    for c, zz in zip(cids, z):
        by.setdefault(ctx.clip_unit[c], []).append(zz)
    keys = sorted(by)
    P = np.stack([np.mean(by[k], 0) for k in keys]); P /= np.linalg.norm(P, axis=1, keepdims=True)
    signer = {}
    for c in cids:
        s = ctx.clip_signer.get(c)
        if s:
            signer.setdefault(ctx.clip_unit[c], []).append(s)
    signer = {k: max(set(v), key=v.count) for k, v in signer.items()}
    bank = PrototypeBank(keys, P, {k: ctx.unit_lemma[k] for k in keys}, signer)
    out = ART / "prototypes" / f"{a.tag}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
    bank.save(out)
    log.info("bank %s: %d prototypes", out, len(keys))


REALWORLD_REFS = {
    "ฉันปลอบเพื่อนร้องไห้.mp4": dict(sentence="ฉันปลอบเพื่อนร้องไห้", glosses=["ฉัน", "ปลอบใจ", "เพื่อน", "ร้องไห้"], question=False),
    "ไปทานข้าวด้วยกันมั้ย.mp4": dict(sentence="ไปทานข้าวด้วยกันมั้ย", glosses=["ไป", "กิน", "ข้าว", "ด้วยกัน", "ไหม"], question=True),
}
SYNONYMS = {"ทาน": "กิน", "รับประทาน": "กิน", "มั้ย": "ไหม", "ปลอบ": "ปลอบใจ", "ร่วมกัน": "ด้วยกัน"}


def canon(w):
    return SYNONYMS.get(w, w)


def synth_sentences(ctx, embed_fn, part, n=100, seed=0, speed=(1.0, 2.0), k_range=(3, 6)):
    """Pseudo-continuous test: concatenate active segments of held-out clips (query signers) — gallery enrollments
    for those signs come only from *other* signers, plus every train prototype as distractors."""
    from modules.evalkit import embed_clips, unit_prototypes
    rng = np.random.RandomState(seed)
    units, uc = ctx.eval_units(part)
    allc = sorted({c for u in units for c in uc[u]})
    emb = dict(zip(allc, embed_clips(embed_fn, ctx.store, allc, "base")))
    et = dict(zip(ctx.train_clips, embed_clips(embed_fn, ctx.store, ctx.train_clips, "base")))
    tk = sorted(ctx.unit_clips_train)
    Ptrain = unit_prototypes(et, ctx.unit_clips_train, tk)
    sents = []
    for i in range(n):
        k = rng.randint(k_range[0], k_range[1] + 1)
        chosen = list(rng.choice(units, k, replace=False))
        F, Pp, V, ref, qsig = [], [], [], [], {}
        for u in chosen:
            q = uc[u][rng.randint(len(uc[u]))]
            qsig[u] = ctx.clip_signer[q]
            f, p, v, a = ctx.store.clip(q, "base", active_only=True)
            sp = rng.uniform(*speed)
            idx = np.round(np.arange(0, len(f), sp)).astype(int).clip(0, len(f) - 1)
            gap = rng.randint(0, 4)
            f0, p0, v0, _ = ctx.store.clip(q, "base", active_only=False)
            F += [f0[:gap], f[idx]]; Pp += [p0[:gap], p[idx]]; V += [v0[:gap], v[idx]]
            ref.append(u)
        keys, G = [], []
        for u in units:
            cl = [c for c in uc[u] if ctx.clip_signer[c] != qsig.get(u)]
            g = np.mean([emb[c] for c in cl], 0)
            keys.append(u); G.append(g / np.linalg.norm(g))
        P = np.concatenate([np.stack(G), Ptrain])
        sents.append(dict(feats=np.concatenate(F), pose=np.concatenate(Pp), valid=np.concatenate(V), ref=ref,
                          keys=keys + tk, P=P.astype(np.float32)))
    return sents


def decode_sentences(model, sents, decoder, lengths, **kw):
    from modules.articulators import active_mask
    from modules.evalkit import gloss_wer
    from modules.spotting import dp_relative, dp_segment, greedy_nms, window_scores
    wers, n_ref, n_hyp, tp = [], 0, 0, 0
    for s in sents:
        wins = window_scores(model.embed, s["feats"], s["pose"], s["valid"], lengths, 2, s["P"])
        seq = (dp_segment(wins, len(s["feats"]), **kw) if decoder == "dp" else
               dp_relative(wins, len(s["feats"]), **kw) if decoder == "dp_rel" else greedy_nms(wins, **kw))
        hyp = [s["keys"][w["idx"][0]] for w in seq]
        wers.append(gloss_wer(s["ref"], hyp)); n_ref += len(s["ref"]); n_hyp += len(hyp)
        tp += len(set(hyp) & set(s["ref"]))
    prec, rec = tp / max(n_hyp, 1), tp / max(n_ref, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(WER=float(np.mean(wers)), hyp_per_ref=n_hyp / max(n_ref, 1), gloss_precision=prec, gloss_recall=rec, gloss_F1=f1)


def st_continuous(a):
    from modules.pipeline import load_islr
    ctx = Ctx()
    model = load_islr(ART / "checkpoints" / "islr" / f"{a.tag}.pt")
    res = {}
    val = synth_sentences(ctx, model.embed, "val", n=a.n_sent, seed=1)
    grid = []
    for lengths in [(8, 12, 16, 20, 24), (12, 16, 20, 24, 32), (8, 12, 16, 20, 24, 32, 40)]:
        for pen in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            grid.append(("dp", lengths, dict(penalty=pen)))
        for delta in [0.0, 0.03, 0.06, 0.1, 0.15]:
            grid.append(("dp_rel", lengths, dict(delta=delta)))
        for tau in [0.4, 0.5, 0.6, 0.7, 0.8]:
            grid.append(("nms", lengths, dict(tau=tau, min_gap=6)))
    best = None
    for dec, lengths, kw in grid:
        r = decode_sentences(model, val, dec, lengths, **kw)
        name = f"{dec}|{lengths}|{kw}"
        res[f"val|{name}"] = r
        log.info("val %-60s WER %.3f hyp/ref %.2f F1 %.3f P %.3f R %.3f", name, r["WER"], r["hyp_per_ref"], r["gloss_F1"], r["gloss_precision"], r["gloss_recall"])
        if best is None or r["gloss_F1"] > best[0]:  # WER alone rewards emitting nothing (empty output = WER 1.0)
            best = (r["gloss_F1"], dec, lengths, kw)
    test = synth_sentences(ctx, model.embed, "test", n=a.n_sent, seed=2)
    _, dec, lengths, kw = best
    r = decode_sentences(model, test, dec, lengths, **kw)
    res["best"] = dict(decoder=dec, lengths=list(lengths), kw=kw, val_F1=best[0], val=res[f"val|{dec}|{lengths}|{kw}"], test=r)
    for d2, kw2 in [("dp", dict(penalty=0.4)), ("nms", dict(tau=0.5, min_gap=6))]:
        res[f"test|{d2}|default"] = decode_sentences(model, test, d2, lengths, **kw2)
    write_json(REP / f"continuous_{a.tag}.json", res)
    log.info("continuous best %s", res["best"])


def tnc_vocab(bank, top_n=5000):
    from pythainlp.corpus import tnc
    wf = sorted(tnc.word_freqs(), key=lambda x: -x[1])[:top_n]
    common = {w for w, _ in wf}
    return [k for k in bank.keys if bank.lemma.get(k, "") in common]


def st_realworld(a):
    from modules.evalkit import chrf, gloss_wer
    from modules.pipeline import VideoPipeline
    best = read_json(REP / f"continuous_{a.tag}.json")["best"] if (REP / f"continuous_{a.tag}.json").exists() else \
        dict(decoder="dp", lengths=[8, 12, 16, 20, 24], kw=dict(penalty=0.4))
    from modules.heads import PrototypeBank
    bank = PrototypeBank.load(ART / "prototypes" / f"{a.tag}.npz")
    scen = {"full_vocab": None, f"tnc_top{a.tnc}": tnc_vocab(bank, a.tnc)}
    out = {"decoder": best, "vocab_sizes": {k: (len(bank.keys) if v is None else len(v)) for k, v in scen.items()}}
    vp = VideoPipeline(ART / "checkpoints" / "islr" / f"{a.tag}.pt", ART / "prototypes" / f"{a.tag}.npz",
                       decoder=best["decoder"], decoder_kw=best["kw"], window_lengths=tuple(best["lengths"]),
                       composer="openai", tts="local" if a.audio else None, affect="auto")
    configs = [(s, t, c) for s in scen for t in ([False, True] if a.tta else [False]) for c in (False, True)]
    for sname, tta, cen in configs:
        vocab = scen[sname]
        vp.vocab_idx = None if vocab is None else np.array([i for i, k in enumerate(vp.bank.keys) if k in set(vocab)])
        vp.tta_flip, vp.center = tta, cen
        tag = f"{sname}{'_tta' if tta else ''}{'_center' if cen else ''}"
        if True:
            for vid, ref in REALWORLD_REFS.items():
                od = ROOT / "result_reporting" / "inference" / a.tag / tag / Path(vid).stem
                r = vp.run(ROOT / "data_test" / vid, od)
                hyp = [canon(g["lemma"]) for g in r["gloss_timeline"]]
                refg = [canon(g) for g in ref["glosses"]]
                inv = [g for g in refg if any(canon(vp.lemma_of(k)) == g for k in (vp.bank.keys if vocab is None else vocab))]
                # best rank each in-vocab reference sign reaches in any window (diagnostic: does the model ever consider it?)
                wins = vp.last["wins"]
                keys = vp.bank.keys if vocab is None else vocab
                best_rank = {}
                for g in inv:
                    ranks = []
                    for w in wins:
                        lem = [canon(vp.lemma_of(keys[i])) for i in w["idx"]]
                        if g in lem:
                            ranks.append(lem.index(g) + 1)
                    best_rank[g] = min(ranks) if ranks else ">10"
                rec = dict(hyp=hyp, ref=refg, in_vocab_refs=inv, WER=gloss_wer(refg, hyp),
                           bag_recall_in_vocab=float(np.mean([g in hyp for g in inv])) if inv else None,
                           best_window_rank=best_rank, sentence=r["text"], chrF=chrf(ref["sentence"], r["text"]),
                           timings_ms=r["timings_ms"], active_frac=r["active_frac"], valid_rate=r["valid_rate"],
                           timeline=[(g["lemma"], g["conf"], g["t0"], g["t1"], g["alt"][:3]) for g in r["gloss_timeline"]])
                out[f"{tag}|{vid}"] = rec
                log.info("%s %s: hyp=%s WER=%.2f bag=%s ranks=%s", tag, vid, hyp, rec["WER"],
                         rec["bag_recall_in_vocab"], best_rank)
    write_json(REP / f"realworld_{a.tag}.json", out)


def st_latency(a):
    """T11 proxy: frame-by-frame stream simulation (p50/p95 ms per frame, RSS) + offline .mp4 stage timings, per device."""
    from modules.pipeline import StreamPipeline, VideoPipeline
    out = {}
    for dev in a.devices.split(","):
        vp = VideoPipeline(ART / "checkpoints" / "islr" / f"{a.tag}.pt", ART / "prototypes" / f"{a.tag}.npz",
                           device=dev, composer="rules", tts="local", affect="auto")
        for vid in REALWORLD_REFS:
            p = ROOT / "data_test" / vid
            vp.run(p, None)  # warm-up
            r = vp.run(p, ROOT / "result_reporting" / "inference" / "latency" / dev / Path(vid).stem)
            sim = StreamPipeline(vp).simulate(p)
            out[f"{dev}|{vid}"] = dict(offline_timings_ms=r["timings_ms"], n_frames=r["n_frames"], rss_mb=r.get("rss_mb"),
                                       offline_ms_per_frame=r["timings_ms"]["total_ms"] / r["n_frames"], stream=sim)
            log.info("%s %s offline %.1f ms/frame · stream p50 %.1f p95 %.1f ms · rss %.0f MB", dev, vid,
                     out[f"{dev}|{vid}"]["offline_ms_per_frame"], sim["p50_ms"], sim["p95_ms"], sim["rss_end_mb"])
    write_json(REP / f"latency_{a.tag}.json", out)


def st_face(a):
    """Distil the FER teacher into AffectHead on cached face-stream features. Crops are rebuilt from cached
    track geometry (no pose re-run). Held-out = clips of val/test signs + YouTube channels held out by hash."""
    import torch.nn.functional as F
    from modules.articulators import Articulators, mask_regions, parse_regions
    from modules.data import decode_frames
    from modules.encoder import FeatureStore, cache_path, profile_name, track_path
    from modules.heads import AffectHead, AffectTeacher
    from modules.utils import load_cfg
    seed_all(0)
    cfg = load_cfg("data")
    clips = pd.read_parquet(MAN / "clips_signers.parquet")
    sp = read_json(MAN / "splits.json")
    rng = np.random.RandomState(0)
    tt = [c for c in sp["train_islr"]]
    held_t = sp["val_clips"] + sp["test_clips"] + sp["seen_val_queries"] + sp["seen_test_queries"]
    yt = [c for c in clips[clips.source == "youtube"].clip_id if track_path(c).exists() and cache_path(c, "base").exists()]
    rng.shuffle(yt)
    sel = [(c, "train") for c in rng.choice(tt, a.n_ttrs, replace=False)] + [(c, "held") for c in rng.choice(held_t, min(120, len(held_t)), replace=False)]
    sel += [(c, "train") for c in yt[:int(len(yt) * 0.8)]] + [(c, "held") for c in yt[int(len(yt) * 0.8):]]
    art = Articulators.__new__(Articulators); art.crop_size = cfg["crop_size"]
    dev = a.device
    torch.set_num_threads(8)
    teacher = AffectTeacher(device=dev)
    info = clips.set_index("clip_id")
    data = []
    for cid, part in sel:
        r = info.loc[cid]
        tr = np.load(track_path(cid))
        feats = np.load(cache_path(cid, "base"))[:, 2].astype(np.float32)
        max_s = 30.0 if r.source == "youtube" else None
        frames = list(decode_frames(ROOT / r.media_path, start_s=float(tr["start_s"]), max_s=max_s))
        if not frames:
            continue
        frames = np.concatenate(frames)
        if r.source == "ttrs":
            frames = mask_regions(frames, parse_regions(r.mask_regions), tuple(cfg["ttrs"]["fill_bgr"]))
        T = min(len(frames), len(feats), len(tr["geom"]))
        idx = np.arange(0, T, 2)
        crops = np.stack([Articulators.crops(art, frames[i], tr["geom"][i])[2] for i in idx])
        probs = teacher(crops)
        ok = tr["valid"][idx, 2].astype(bool)
        data.append(dict(cid=cid, part=part, feats=feats[:T], idx=idx[ok], probs=probs[ok], src=r.source))
    log.info("face data: %d clips, %d labelled frames", len(data), sum(len(d["idx"]) for d in data))
    head = AffectHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    train = [d for d in data if d["part"] == "train" and len(d["idx"])]
    W = 48
    for it in range(a.steps):
        xb, pb, mb = [], [], []
        for _ in range(64):
            d = train[rng.randint(len(train))]
            T = len(d["feats"]); s = rng.randint(0, max(1, T - W + 1)); e = min(T, s + W)
            x = np.zeros((W, 768), np.float32); x[:e - s] = d["feats"][s:e]
            p = np.zeros((W, 7), np.float32); m = np.zeros(W, bool)
            sel_i = (d["idx"] >= s) & (d["idx"] < e)
            p[d["idx"][sel_i] - s] = d["probs"][sel_i]; m[d["idx"][sel_i] - s] = True
            xb.append(x); pb.append(p); mb.append(m)
        x, p, m = (torch.from_numpy(np.stack(v)).to(dev) for v in (xb, pb, mb))
        logp = head(x).log_softmax(-1)
        loss = -(p * logp).sum(-1)[m].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            log.info("affect %d loss %.3f", it, float(loss))
    head.eval()
    res = {}
    for part in ("train", "held"):
        for src in ("ttrs", "youtube"):
            agree = tot = 0; dist_t = np.zeros(7); dist_s = np.zeros(7)
            for d in data:
                if d["part"] != part or d["src"] != src or not len(d["idx"]):
                    continue
                with torch.no_grad():
                    ps = head(torch.from_numpy(d["feats"]).to(dev)[None])[0].softmax(-1).cpu().numpy()[d["idx"]]
                agree += int((ps.argmax(1) == d["probs"].argmax(1)).sum()); tot += len(ps)
                dist_t += np.bincount(d["probs"].argmax(1), minlength=7); dist_s += np.bincount(ps.argmax(1), minlength=7)
            res[f"{part}|{src}"] = dict(frames=tot, top1_agreement_with_teacher=agree / max(tot, 1),
                                        teacher_dist=(dist_t / max(tot, 1)).round(3).tolist(), student_dist=(dist_s / max(tot, 1)).round(3).tolist())
    out = ART / "checkpoints" / "face" / "affect_head.pt"; out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out)
    write_json(REP / "face_affect.json", res)
    log.info("affect: %s", res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack"); p.add_argument("--set", default="ttrs")
    s = sub.add_parser("signers"); s.add_argument("--min_prob", type=float, default=0.95); s.add_argument("--cluster_thr", type=float, default=0.25)
    sub.add_parser("baselines")
    for name in ("ssl", "islr"):
        q = sub.add_parser(name)
        q.add_argument("--tag", required=True); q.add_argument("--steps", type=int, default=4000)
        q.add_argument("--seed", type=int, default=0); q.add_argument("--d", type=int, default=256)
        q.add_argument("--layers", type=int, default=4); q.add_argument("--heads", type=int, default=4)
        q.add_argument("--nap", type=int, default=1); q.add_argument("--eval_every", type=int, default=1000)
        if name == "ssl":
            q.add_argument("--batch", type=int, default=48); q.add_argument("--lr", type=float, default=5e-4)
            q.add_argument("--K", type=int, default=4096); q.add_argument("--no_youtube", action="store_true")
        else:
            q.add_argument("--init", default="scratch"); q.add_argument("--lora", type=int, default=0)
            q.add_argument("--freeze", action="store_true"); q.add_argument("--P", type=int, default=96)
            q.add_argument("--lr", type=float, default=3e-4); q.add_argument("--enc_lr", type=float, default=1e-4)
            q.add_argument("--w_adv", type=float, default=0.1); q.add_argument("--w_arc", type=float, default=1.0)
            q.add_argument("--pool", default="cls"); q.add_argument("--streams", default="hand_l,hand_r,face,pose")
            q.add_argument("--versions", default="base,aug1,aug2,flip"); q.add_argument("--speed_max", type=float, default=1.25); q.add_argument("--center_train", action="store_true")
    b = sub.add_parser("bank"); b.add_argument("--tag", required=True); b.add_argument("--with_flip", action="store_true")
    c = sub.add_parser("continuous"); c.add_argument("--tag", required=True); c.add_argument("--n_sent", type=int, default=100)
    r = sub.add_parser("realworld"); r.add_argument("--tag", required=True); r.add_argument("--tnc", type=int, default=5000)
    r.add_argument("--tta", action="store_true"); r.add_argument("--audio", action="store_true")
    lt = sub.add_parser("latency"); lt.add_argument("--tag", required=True); lt.add_argument("--devices", default="cuda,cpu")
    ec = sub.add_parser("evalckpt"); ec.add_argument("--tags", required=True); ec.add_argument("--out", default="main")
    fc = sub.add_parser("face"); fc.add_argument("--device", default="cuda"); fc.add_argument("--n_ttrs", type=int, default=400); fc.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    dict(face=st_face, latency=st_latency, evalckpt=st_evalckpt, pack=st_pack, signers=st_signers, baselines=st_baselines, ssl=st_ssl, islr=st_islr, bank=st_bank,
         continuous=st_continuous, realworld=st_realworld)[a.cmd](a)


if __name__ == "__main__":
    main()
