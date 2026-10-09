"""v8 sign encoder training (local GPU; the RTX 2050 4 GB is enough): compact landmark encoder from scratch on Thai data only.

  python main.py train-encoder --run enc-v8 [--steps 12000] [--eval_every 1000]
  python main.py train-encoder --embed_ref v7|zeroshot  # the SAME evaluation clips through v7 / the Uni-Sign zero-shot base
  python main.py train-encoder --report <run>      # every protocol of a finished run (+ v7 on the same clips) → artifacts/reports/isolated_v8.json

Objective, batches and protocols are the ones v7 used (GCP/script/job_entry.py), so the comparison isolates the encoder:
  CosFace (s 30, m 0.25) over concepts + 0.5 · SupCon; P concepts × K instances, a different SIGNER as the positive when one exists,
  hard negatives re-mined from the proxies at every evaluation; real transitions / rest / straddling windows = the "not a sign" class;
  augmentation = signer retargeting, speed 0.7–1.6×, affine, finger jitter, hand drop-outs, webcam frame-rate (modules/encoder.py).
Model selection = mean MRR of the held-out-SIGNER val sets (TTRS · TSL-ONE-S · th_sl · YouTube people) — never a test set, never data_test.

Data: cloud_s3/data_prep/shards (scripts/data_prep.py shards: v8 = YouTube quality audit + concept normalisation). The pose store is
converted once to memory-mapped .npy files (cache/shards_mm/) so the loader workers share it instead of copying 1.2 GB each (Windows).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from modules.utils import ART, PREP, get_logger, read_json, seed_all, write_json

log = get_logger("train_enc")
SHARDS = Path(os.environ.get("THAISLM_SHARDS", PREP / "shards"))
MM = ROOT / "cache" / "shards_mm"
REF = ROOT / "cache" / "reference_models"                             # v7 models/ restored from git (comparisons only, see embed_ref)
NULL = "<null>"


# ----------------------------------------------------------------------------- data
class PoseStoreMM:
    """modules.encoder.PoseStore over memory-mapped arrays (opened lazily: picklable for spawn-based loader workers)."""

    def __init__(self, root=MM):
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.clip_id = meta["clip_id"]
        self.row = {c: i for i, c in enumerate(self.clip_id)}
        self.off = np.load(self.root / "off.npy"); self.hw = np.load(self.root / "hw.npy")
        self._kp = self._sc = None

    def __getstate__(self):
        d = dict(self.__dict__); d["_kp"] = d["_sc"] = None
        return d

    def _open(self):
        if self._kp is None:                      # _sc first: a thread that sees _kp set also sees _sc (eval uses threads)
            self._sc = np.load(self.root / "SC.npy", mmap_mode="r"); self._kp = np.load(self.root / "KP.npy", mmap_mode="r")

    def get(self, i, a=None, b=None):
        from modules.encoder import SLOTS
        self._open()
        s, e = int(self.off[i]), int(self.off[i + 1])
        a = 0 if a is None else a; b = e - s if b is None else b
        kp = np.zeros((b - a, 133, 2), np.float32); sc = np.zeros((b - a, 133), np.float32)
        kp[:, SLOTS] = self._kp[s + a:s + b]; sc[:, SLOTS] = self._sc[s + a:s + b]
        return kp, sc, tuple(int(x) for x in self.hw[i])

    def length(self, i):
        return int(self.off[i + 1] - self.off[i])


def ensure_mm():
    src = SHARDS / "pose_store.npz"
    stamp = MM / "meta.json"
    if stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")).get("mtime") == src.stat().st_mtime:
        return
    MM.mkdir(parents=True, exist_ok=True)
    z = np.load(src, allow_pickle=True)
    for k in ("KP", "SC", "off", "hw"):
        np.save(MM / f"{k}.npy", z[k])
    stamp.write_text(json.dumps(dict(mtime=src.stat().st_mtime, clip_id=[str(c) for c in z["clip_id"]])), encoding="utf-8")
    log.info("memory-mapped pose store → %s", MM)


def load_tables(store):
    """Isolated / null clip table exactly as v7's train_encoder stage builds it (spans padded by 2 frames)."""
    clips = pd.read_parquet(SHARDS / "clips.parquet")
    clips = clips[clips.clip_id.isin(store.row)].reset_index(drop=True)
    clips["row"] = clips.clip_id.map(store.row)
    clips["aux"] = clips.aux_clip.map(lambda c: store.row.get(c, -1) if isinstance(c, str) else -1).astype(int)
    iso = clips[clips.kind.isin(["isolated", "null"]) & (clips.view == "primary")].copy()
    concepts = sorted(set(iso.concept.dropna()))
    if NULL not in concepts:
        concepts.append(NULL)
    cidx = {c: i for i, c in enumerate(concepts)}
    iso["c"] = iso.concept.map(cidx)
    pad = 2
    iso["f0"] = np.where(iso.kind == "null", 0, (iso.span_f0 - pad).clip(lower=0)).astype(int)
    iso["f1"] = np.where(iso.kind == "null", iso.n_frames, np.minimum(iso.span_f1 + pad, iso.n_frames)).astype(int)
    iso = iso[(iso.f1 - iso.f0) >= 4].reset_index(drop=True)
    ref = ART / "eval" / "youtube_roles_reference.json"                # v7's roles of the YouTube clips: people held out by BOTH models
    if ref.exists():
        r7 = read_json(ref)
        iso["agent_common"] = (iso.dataset != "agent") | (iso.clip_id.map(r7) == iso.role)
    daily = []
    if (SHARDS / "daily_concepts.json").exists():
        daily = [cidx[c] for c in read_json(SHARDS / "daily_concepts.json") if c in cidx]
    return clips, iso, concepts, cidx, daily


# ----------------------------------------------------------------------------- evaluation (v7 protocols, same definitions)
def protocols(Z, ev, concepts, cidx, daily):
    from modules.encoder import concept_scores, retrieval_metrics
    res = {}
    C = len(concepts)
    tr = (ev.role == "train").values & (ev.concept != NULL).values
    nm = (ev.role == "train").values & (ev.concept == NULL).values
    null_proto = Z[nm].mean(0) if nm.any() else -Z[tr].mean(0)
    null_proto = null_proto / (np.linalg.norm(null_proto) + 1e-9)
    one_cols = sorted(set(ev[ev.dataset == "tslone"].c))
    t51_cols = sorted(set(ev[(ev.dataset == "tsl51") & (ev.concept != NULL)].c))
    daily_cols = sorted(set(daily))

    def q(mask, name, allowed=None):
        if mask.sum() == 0:
            return
        S = concept_scores(Z[mask], Z[tr], ev.c.values[tr], C)
        S[:, cidx[NULL]] = -2
        res[name] = retrieval_metrics(S, ev.c.values[mask], allowed)
    for part in ("val", "test"):
        for ds, tag in (("ttrs", "E1_ttrs"), ("tslone", "ONE"), ("th_sl", "THSL"), ("agent", "AGENT")):
            m = (ev.dataset == ds).values & (ev.role == part).values & (ev.concept != NULL).values
            if ds == "agent" and "agent_common" in ev:
                m &= ev.agent_common.values                     # YouTube people held out by BOTH v7 and v8 (fair comparison)
            q(m, f"{tag}_{part}|open_vocab")
            if daily_cols:
                q(m & np.isin(ev.c.values, daily_cols), f"{tag}_{part}|daily_vocab", allowed=daily_cols)
        m = (ev.dataset == "tslone").values & (ev.role == part).values
        q(m, f"ONE_{part}|closed_184", allowed=one_cols)
    sub = (ev.subset == "tsl51_user_sign").values
    m = sub & (ev.concept != NULL).values
    q(m, "T51U|open_vocab"); q(m, "T51U|closed_51", allowed=t51_cols)
    if daily_cols:
        q(m & np.isin(ev.c.values, daily_cols), "T51U|daily_vocab", allowed=daily_cols)
    if sub.any():
        s_null = Z[sub] @ null_proto
        is_null = (ev.concept[sub] == NULL).values
        if is_null.any() and (~is_null).any():
            pos, neg = s_null[is_null], s_null[~is_null]
            res["T51U|null_auc"] = dict(auc=float((pos[:, None] > neg[None, :]).mean()), n_null=int(is_null.sum()), n_sign=int((~is_null).sum()))
    return res


def val_score(r):
    ks = [k for k in ("E1_ttrs_val|open_vocab", "ONE_val|open_vocab", "THSL_val|open_vocab", "AGENT_val|open_vocab") if r.get(k, {}).get("n")]
    return float(np.mean([r[k]["MRR"] for k in ks])) if ks else 0.0


def eval_feats(store, ev, prep_fn, workers=6):
    from concurrent.futures import ThreadPoolExecutor
    from modules.parts import to_iso

    def one(r):
        kp, sc, hw = store.get(int(r[0]), int(r[1]), int(r[2]))
        return prep_fn(to_iso(kp, hw), sc).astype(np.float16)
    rows = list(zip(ev.row, ev.f0, ev.f1))
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows, chunksize=64))


# ----------------------------------------------------------------------------- training
def collate_feats(batch):
    from modules.lmenc import collate_np
    feats, labels = batch
    x, m = collate_np(feats)
    return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(labels.astype(np.int64))


def train(a):
    from modules import encoder as E
    from modules.lmenc import build_model, embed, prep as lm_prep
    seed_all(a.seed)
    ensure_mm()
    store = PoseStoreMM()
    clips, iso, concepts, cidx, daily = load_tables(store)
    train_df = iso[iso.role == "train"]
    table = [dict(row=int(r.row), aux=int(r.aux) if a.aux_p > 0 else -1, c=int(r.c), signer=r.signer, dataset=r.dataset, f0=int(r.f0), f1=int(r.f1))
             for r in train_df.itertuples()]
    null_sources = [t for t, r in zip(table, train_df.itertuples()) if r.kind == "isolated" and r.n_frames - (r.f1 - r.f0) >= 8 and r.dataset != "tslone"]
    transitions = []
    if (SHARDS / "transitions.parquet").exists():
        trn = pd.read_parquet(SHARDS / "transitions.parquet")
        transitions = [(store.row[c], int(f0), int(f1)) for c, f0, f1 in zip(trn.clip_id, trn.f0, trn.f1) if c in store.row]
    sig = train_df.groupby("c").signer.nunique()
    multi = [int(c) for c in sig[sig >= 2].index if c != cidx[NULL]]
    allc = [int(c) for c in train_df.c.unique() if c != cidx[NULL]]
    log.info("train %d clips / %d concepts / %d signers; ≥2 signers %d; daily %d; transitions %d", len(train_df), train_df.c.nunique(),
             train_df.signer.nunique(), len(multi), len(daily), len(transitions))
    out = ART / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    E.SIGNER_AUG.update(p=a.signer_aug, mirror=a.mirror)
    ds = E.ClipDataset(store, table, cidx, null_sources, seed=a.seed, transitions=transitions, daily=daily, aux_p=a.aux_p, hard_frac=a.hard_frac,
                       prep_fn=lm_prep)
    ds.jitter, ds.context_p, ds.framing_p = a.jitter, a.context_p, a.framing_p
    nb_file = out / "neighbours.npy"
    ds.neighbours_file = str(nb_file)
    it = E.BatchIter(ds, a.P, a.K, multi, allc, a.null_frac, a.seed)
    dl = torch.utils.data.DataLoader(it, batch_size=None, num_workers=a.workers, collate_fn=collate_feats, persistent_workers=a.workers > 0,
                                     prefetch_factor=4 if a.workers else None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(d=a.d, emb=a.emb, depth=a.depth, heads=a.heads, drop=a.drop).to(dev)
    log.info("model %.2f M parameters", sum(p.numel() for p in model.parameters()) / 1e6)
    cos = E.CosFace(torch.randn(len(concepts), a.emb), s=a.s, m=a.margin).to(dev)
    opt = torch.optim.AdamW([dict(params=model.parameters(), lr=a.lr), dict(params=cos.parameters(), lr=a.lr * 5, weight_decay=0)],
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    ev = iso.reset_index(drop=True)
    t0 = time.time()
    feats = eval_feats(store, ev, lm_prep)
    log.info("eval features for %d clips (%.0fs)", len(feats), time.time() - t0)
    hist, best, best_step = [], -1.0, 0
    step = 0
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if a.ema else None
    last = out / "ckpt_last.pt"                     # resume after an interruption (power loss): model, proxies, optimiser, schedule
    if a.init and not (last.exists() and not a.fresh):   # fine-tune a finished run (its weights + class proxies; new optimiser / schedule)
        ck = torch.load(a.init, map_location=dev)
        model.load_state_dict(ck["model"]); cos.load_state_dict(ck["cos"])
        if ema is not None:
            ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log.info("initialised from %s (step %d)", a.init, ck.get("step", -1))
    if last.exists() and not a.fresh:
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); cos.load_state_dict(ck["cos"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        step, best, best_step, hist = ck["step"], ck["best"], ck["best_step"], ck["hist"]
        if ema is not None and ck.get("ema") is not None:
            ema = ck["ema"]
        it.seed = a.seed + step                     # loader workers start on the first batch: new batches, not a replay
        log.info("resumed from step %d (best val %.4f at %d)", step, best, best_step)
    model.train()
    t0 = time.time()
    for x, m, y in dl:
        x, m, y = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            z = model(x, m)
        z = z.float()
        loss = cos(z, y) + a.supcon * E.supcon(z, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        if ema is not None:                         # exponential moving average of the weights (v7 used 0.998)
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(a.ema).add_(v.detach(), alpha=1 - a.ema)
                    else:
                        ema[k].copy_(v)
        if step % 100 == 0:
            log.info("step %d loss %.3f (%.1f it/s)", step, loss.item(), 100 / (time.time() - t0)); t0 = time.time()
            hist.append(dict(step=step, loss=float(loss.item())))
        if step % a.eval_every == 0 or step == a.steps:
            live = None
            if ema is not None:                     # evaluate (and keep) the averaged weights
                live = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            Z = embed(model, [f.astype(np.float32) for f in feats], device=dev)
            r = protocols(Z, ev, concepts, cidx, daily)
            sc_ = val_score(r)
            hist.append(dict(step=step, val=sc_, **{k: round(v.get("R1", v.get("auc", 0)), 4) for k, v in r.items()}))
            log.info("eval step %d: val MRR %.4f | %s", step, sc_, {k: round(v.get("R1", v.get("auc", 0)), 3) for k, v in r.items() if "val" in k or "T51U" in k})
            if sc_ > best:
                best, best_step = sc_, step
                torch.save(dict(cfg=model.cfg, state=model.state_dict(), step=step, val=sc_), out / "encoder_best.pt")
                np.save(out / "emb.npy", Z.astype(np.float16))
                write_json(out / "metrics_best.json", r)
            # hard negatives: nearest concepts by proxy direction
            W = torch.nn.functional.normalize(cos.W.detach().float(), dim=-1)
            nbs = np.zeros((len(W), 10), np.int32)
            for i in range(0, len(W), 2048):
                S = W[i:i + 2048] @ W.T
                S[torch.arange(len(S)), torch.arange(i, i + len(S))] = -2
                S[:, cidx[NULL]] = -2
                nbs[i:i + 2048] = torch.topk(S, 10, dim=1).indices.cpu().numpy()
            np.save(nb_file, nbs)
            if live is not None:
                model.load_state_dict(live)
            write_json(out / "train_history.json", hist)
            torch.save(dict(model=model.state_dict(), cos=cos.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                            step=step, best=best, best_step=best_step, hist=hist, ema=ema), last)
            model.train()
        if step >= a.steps:
            break
    ev[["clip_id", "dataset", "subset", "kind", "role", "signer", "concept", "c", "f0", "f1"]].to_parquet(out / "emb_index.parquet", index=False)
    write_json(out / "concept_index.json", concepts)
    write_json(out / "summary.json", dict(best_step=best_step, best_val=best, args=vars(a), concepts=len(concepts), train_clips=len(train_df)))
    log.info("done: best val MRR %.4f at step %d → %s", best, best_step, out)


# ----------------------------------------------------------------------------- baseline + report on the same clips
def embed_ref(a):
    """Reference encoders on exactly the v8 evaluation clips: v7 (restored from git into cache/reference_models:
    `git archive 86098cb models | tar -x -C cache/reference_models --strip-components=1`) or the Uni-Sign zero-shot base (ASL/CSL prior only)
    → artifacts/runs/{v7-on-v8, zeroshot-on-v8}/emb.npy."""
    from modules.encoder import embed_parts, load_encoder, prep
    from modules.parts import to_iso
    ensure_mm()
    store = PoseStoreMM()
    _, iso, concepts, cidx, daily = load_tables(store)
    ev = iso.reset_index(drop=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.embed_ref == "v7":
        enc, out = load_encoder(REF, REF / "encoder.pt", device=dev), ART / "runs" / "v7-on-v8"
        zs = False
    else:
        enc, out = load_encoder(ART / "pretrained", None, device=dev), ART / "runs" / "zeroshot-on-v8"
        zs = True
    out.mkdir(parents=True, exist_ok=True)
    Z = np.zeros((len(ev), 768), np.float32)
    B = 2000
    for i in range(0, len(ev), B):
        parts = []
        for r in ev.iloc[i:i + B].itertuples():
            kp, sc, hw = store.get(int(r.row), int(r.f0), int(r.f1))
            parts.append(prep(to_iso(kp, hw), sc))
        Z[i:i + B] = embed_parts(enc, parts, bs=64, device=dev, zero_shot=zs)
        log.info("%s embeddings %d/%d", a.embed_ref, min(i + B, len(ev)), len(ev))
    np.save(out / "emb.npy", Z.astype(np.float16))
    ev[["clip_id", "dataset", "subset", "kind", "role", "signer", "concept", "c", "f0", "f1"]].to_parquet(out / "emb_index.parquet", index=False)
    r = protocols(Z, ev, concepts, cidx, daily)
    write_json(out / "metrics.json", r)
    log.info("%s on the v8 clips: val %.4f | %s", a.embed_ref, val_score(r), {k: round(v.get("R1", v.get("auc", 0)), 3) for k, v in r.items()})


def report(a):
    """Every protocol for each model on the SAME v8 evaluation clips + recall by bank depth + the ASL/CSL 'inherited error' rate
    (share of a model's test errors that are the zero-shot Uni-Sign model's same wrong answer — computed exactly as v7 did)
    → artifacts/reports/isolated_v8.json."""
    from modules.encoder import NULL as _N
    from modules.runtime import Bank
    ensure_mm()
    store = PoseStoreMM()
    _, iso, concepts, cidx, daily = load_tables(store)
    ev = iso.reset_index(drop=True)
    C = len(concepts)
    names = ["zeroshot-on-v8", "v7-on-v8"] + [r for r in a.report.split(",") if r]
    embs = {}
    for n in names:
        f = ART / "runs" / n / "emb.npy"
        if f.exists():
            Z = np.load(f).astype(np.float32)
            if len(Z) == len(ev):
                embs[n] = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    out = dict(models=list(embs), protocols={}, recall_by_depth={}, prior_bias={}, latent={})
    for n, Z in embs.items():
        out["protocols"][n] = protocols(Z, ev, concepts, cidx, daily)
        out["protocols"][n]["val_score"] = val_score(out["protocols"][n])
    tr = (ev.role == "train").values & (ev.concept != _N).values
    q = (ev.role == "test").values & (ev.concept != _N).values & (ev.get("agent_common", True) if "agent_common" in ev else True)
    t = ev.c.values[q]
    dep = ev[tr].groupby("c").signer.nunique()
    d = np.array([dep.get(c, 0) for c in t])
    preds = {}
    for n, Z in embs.items():
        S = Bank(Z[tr], ev.c.values[tr], C).scores(Z[q]); S[:, cidx[_N]] = -2
        preds[n] = S.argmax(1)
        r1 = preds[n] == t
        out["recall_by_depth"][n] = [dict(depth=lab, n=int(m.sum()), R1=float(r1[m].mean()) if m.any() else None)
                                     for lab, m in (("1", d == 1), ("2", d == 2), ("3-4", (d >= 3) & (d <= 4)), ("5-9", (d >= 5) & (d <= 9)), ("10+", d >= 10))]
    if "zeroshot-on-v8" in preds:
        p0 = preds["zeroshot-on-v8"]
        for n, p1 in preds.items():
            if n == "zeroshot-on-v8":
                continue
            err1 = p1 != t
            out["prior_bias"][n] = dict(errors=int(err1.sum()), inherited_confusion_rate=float(((p0 == p1) & err1).sum() / max(err1.sum(), 1)),
                                        inherited_errors_per_query=float(((p0 == p1) & err1).mean()),
                                        zero_shot_errors_fixed=float(((p0 != t) & (p1 == t)).sum() / max((p0 != t).sum(), 1)))
    from sklearn.neighbors import NearestNeighbors
    qm = np.flatnonzero((ev.concept != _N).values)
    qm = np.random.RandomState(0).choice(qm, min(4000, len(qm)), replace=False)
    for n, Z in embs.items():
        _, nb = NearestNeighbors(n_neighbors=6, metric="cosine").fit(Z).kneighbors(Z[qm]); nb = nb[:, 1:]
        out["latent"][n] = {f"knn5_same_{k}": float((ev[k].values[nb] == ev[k].values[qm][:, None]).mean()) for k in ("concept", "signer", "dataset")}
    write_json(ART / "reports" / "isolated_v8.json", out)
    keys = [k for k in next(iter(out["protocols"].values())) if k != "val_score"]
    for k in keys:
        log.info("%-28s %s", k, " | ".join(f"{n[:14]} {out['protocols'][n].get(k, {}).get('R1', out['protocols'][n].get(k, {}).get('auc', float('nan'))):.3f}"
                                            for n in embs))
    log.info("val score: %s", {n: round(out["protocols"][n]["val_score"], 4) for n in embs})
    log.info("depth: %s", {n: [(r["depth"], r["n"], round(r["R1"] or 0, 3)) for r in v] for n, v in out["recall_by_depth"].items()})
    log.info("prior bias: %s", out["prior_bias"]); log.info("latent: %s", out["latent"])


def cont_proxy(a):
    """Continuous-window recognition (the v8 failure found in tuning): every aligned word window of the TSL51 TUNE sentences (decisions;
    --proxy_role sentence_test only for the final report)
    (an unseen signer, real co-articulation; alignment artifacts/runs/seq-v8/tsl51_alignment.json, same windows for every encoder),
    embedded on its own and ranked against that encoder's train-only bank — as given, and with the boundaries moved ±25 %.
    R@1 / R@5 in the conversation vocabulary and over all concepts. → artifacts/reports/continuous_windows.json"""
    from modules.encoder import NULL as _N
    from modules.lmenc import EncoderAPI
    from modules.parts import to_iso
    from modules.runtime import Bank
    ensure_mm()
    store = PoseStoreMM()
    _, iso, concepts, cidx, daily = load_tables(store)
    ev = iso.reset_index(drop=True)
    C = len(concepts)
    tr = (ev.role == "train").values & (ev.concept != _N).values
    conv = sorted(set(daily) | set(ev[(ev.dataset == "tsl51") & (ev.concept != _N)].c))
    al = read_json(ART / "runs" / "seq-v8" / "tsl51_alignment.json")
    wins = []
    for cid, v in al.items():
        if v["role"] != a.proxy_role:
            continue
        key = "t51s:" + cid.split(":", 1)[1] if not cid.startswith("t51s:") else cid
        if key not in store.row:
            continue
        for sgm in v["segments"]:
            if sgm["concept"] in cidx:
                wins.append((store.row[key], int(sgm["f0"]), int(sgm["f1"]), cidx[sgm["concept"]]))
    rng = np.random.RandomState(0)
    variants = {"aligned": lambda f0, f1, n: (f0, f1)}
    def moved(f0, f1, n):
        L = f1 - f0; d0, d1 = rng.randint(-int(0.25 * L), int(0.25 * L) + 1, 2)
        return int(np.clip(f0 + d0, 0, n - 4)), int(np.clip(f1 + d1, f0 + 4, n))
    variants["boundaries_moved"] = moved
    out = dict(n_windows=len(wins), models={})
    runs = {"v7": (REF / "encoder.pt", REF, ART / "runs" / "v7-on-v8")} if (REF / "encoder.pt").exists() else {}
    names = [x for x in a.cont_proxy.split(",") if x]
    for r in sorted({p for x in names for p in x.split("+")}):
        runs[r] = (ART / "runs" / r / "encoder_best.pt", ART / "pretrained", ART / "runs" / r)
    scores, tgt_of = {}, {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, (ck, md, rd) in runs.items():
        enc = EncoderAPI(ck, md, device=dev)
        Zb = np.load(rd / "emb.npy").astype(np.float32)
        bank = Bank(Zb[tr], ev.c.values[tr], C)
        res = {}
        for vn, fn in variants.items():
            rng.seed(0)
            parts, tgt = [], []
            for row, f0, f1, c in wins:
                n = store.length(row)
                a0, b0 = fn(f0, f1, n)
                kp, sc, hw = store.get(row, a0, b0)
                parts.append(enc.prep(to_iso(kp, hw), sc)); tgt.append(c)
            Z = enc.embed(parts)
            S = bank.scores(Z); S[:, cidx[_N]] = -2
            t = np.array(tgt)
            scores[(name, vn)] = S; tgt_of[vn] = t
            for vocab, cols in (("conversation", conv), ("all", None)):
                Sx = S.copy()
                if cols is not None:
                    m = np.ones(C, bool); m[cols] = False; Sx[:, m] = -9
                rank = (Sx > Sx[np.arange(len(t)), t][:, None]).sum(1)
                res[f"{vn}|{vocab}"] = dict(R1=round(float((rank < 1).mean()), 4), R5=round(float((rank < 5).mean()), 4))
        out["models"][name] = res
        log.info("%-8s %s", name, res)
        del enc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    for combo in [x for x in names if "+" in x]:          # score-level ensemble (mean of the members' concept scores)
        res = {}
        for vn in variants:
            S = np.mean([scores[(m, vn)] for m in combo.split("+")], 0); t = tgt_of[vn]
            for vocab, cols in (("conversation", conv), ("all", None)):
                Sx = S.copy()
                if cols is not None:
                    m = np.ones(C, bool); m[cols] = False; Sx[:, m] = -9
                rank = (Sx > Sx[np.arange(len(t)), t][:, None]).sum(1)
                res[f"{vn}|{vocab}"] = dict(R1=round(float((rank < 1).mean()), 4), R5=round(float((rank < 5).mean()), 4))
        out["models"][combo] = res
        log.info("%-8s %s", combo, res)
    write_json(ART / "reports" / f"continuous_windows_{a.proxy_role}.json", out)


def framing_test(a):
    """Robustness to tight webcam framing: held-out queries (val / test) re-embedded with the frame ending c shoulder widths below the
    shoulders (hands below it lost, modules.encoder.frame_crop), bank unchanged → artifacts/reports/framing_test.json.
    c = 0.5 / 0.7 = what the laptop webcam sessions show (median 0.46–0.69); c = ∞ = the normal protocols."""
    from modules.encoder import frame_crop
    from modules.lmenc import embed, load_model, prep as lm_prep
    ensure_mm()
    store = PoseStoreMM()
    _, iso, concepts, cidx, daily = load_tables(store)
    ev = iso.reset_index(drop=True)
    qm = ev.role.isin(["val", "test"]).values | (ev.subset == "tsl51_user_sign").values
    qi = np.where(qm)[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_f = ART / "reports" / "framing_test.json"
    out = read_json(out_f) if out_f.exists() else {}
    keys = ["T51U|open_vocab", "T51U|daily_vocab", "ONE_test|open_vocab", "THSL_test|open_vocab", "AGENT_test|open_vocab", "E1_ttrs_test|open_vocab",
            "ONE_val|open_vocab", "THSL_val|open_vocab", "AGENT_val|open_vocab", "E1_ttrs_val|open_vocab"]
    from modules.parts import to_iso
    for run in [r for r in a.framing_test.split(",") if r]:
        model = load_model(ART / "runs" / run / "encoder_best.pt", device=dev)
        Z0 = np.load(ART / "runs" / run / "emb.npy").astype(np.float32)
        res = {}
        for c in (0.5, 0.7, None):
            Z = Z0.copy()
            if c is not None:
                feats = []
                for i in qi:
                    r = ev.iloc[i]
                    kp, sc, hw = store.get(int(r.row), int(r.f0), int(r.f1))
                    kp = to_iso(kp, hw)
                    kp, sc = frame_crop(kp, sc, c=c)
                    feats.append(lm_prep(kp, sc))
                Z[qi] = embed(model, feats, device=dev)
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
            r = protocols(Z, ev, concepts, cidx, daily)
            res["full" if c is None else f"c{c}"] = {k: round(r[k]["R1"], 4) for k in keys if k in r}
            log.info("%s framing %s: %s", run, c, res["full" if c is None else f"c{c}"])
        out[run] = res
        write_json(out_f, out)


def ensemble(a):
    """artifacts/runs/<run>: one ensemble encoder from finished runs (+ its embeddings of the evaluation clips: members' embeddings
    concatenated / √n, so every protocol, the sequence stage and the bank use it like a single encoder)."""
    from modules.lmenc import make_ensemble
    members = a.ensemble.split(",")
    out = ART / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    make_ensemble([ART / "runs" / m / "encoder_best.pt" for m in members], out / "encoder_best.pt")
    Z = np.concatenate([np.load(ART / "runs" / m / "emb.npy").astype(np.float32) for m in members], 1) / np.sqrt(len(members))
    np.save(out / "emb.npy", Z.astype(np.float16))
    for f in ("emb_index.parquet", "concept_index.json"):
        src = ART / "runs" / members[0] / f
        if src.exists():
            (out / f).write_bytes(src.read_bytes())
    write_json(out / "summary.json", dict(ensemble=members))
    log.info("ensemble %s → %s (emb %s)", members, out, Z.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="enc-v8")
    ap.add_argument("--embed_ref", default="", help="v7 | zeroshot")
    ap.add_argument("--report", default="", help="comma-separated runs to compare with v7 and zero-shot")
    ap.add_argument("--ensemble", default="", help="comma-separated finished runs → one ensemble encoder in --run")
    ap.add_argument("--cont_proxy", default="", help="comma-separated runs: continuous-window recognition vs v7")
    ap.add_argument("--proxy_role", default="tagger_train", help="tagger_train = tune sentences (decisions) · sentence_test (report only)")
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--P", type=int, default=64); ap.add_argument("--K", type=int, default=2); ap.add_argument("--null_frac", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--emb", type=int, default=256); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--s", type=float, default=30.0); ap.add_argument("--margin", type=float, default=0.25); ap.add_argument("--supcon", type=float, default=0.5)
    ap.add_argument("--aux_p", type=float, default=0.25); ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--signer_aug", type=float, default=0.3); ap.add_argument("--mirror", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore ckpt_last.pt and start from step 0")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off; v7 used 0.998)")
    ap.add_argument("--jitter", type=float, default=0.15, help="span-boundary jitter, share of the sign length")
    ap.add_argument("--context_p", type=float, default=0.0, help="probability of co-articulation context (other signs glued around)")
    ap.add_argument("--framing_p", type=float, default=0.0, help="probability of tight webcam framing (hands lost below the frame)")
    ap.add_argument("--init", default="", help="ckpt_last.pt of a finished run to fine-tune from")
    ap.add_argument("--framing_test", default="", help="comma-separated runs: recognition when the frame ends close below the shoulders")
    a = ap.parse_args()
    if a.ensemble:
        return ensemble(a)
    if a.report:
        return report(a)
    if a.cont_proxy:
        return cont_proxy(a)
    if a.framing_test:
        return framing_test(a)
    embed_ref(a) if a.embed_ref else train(a)


if __name__ == "__main__":
    main()
