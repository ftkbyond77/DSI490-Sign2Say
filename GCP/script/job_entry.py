"""ThaiSLM — entry point inside the Vertex AI container (also runs locally with --data/--out).

  --stage train_encoder   multi-signer fine-tuning of the Uni-Sign pose encoder on the v7 data (MediaPipe everywhere + RTMW views of
                          the same videos as extra positives), evaluation on held-out signers every `eval_every` steps, embeddings of
                          every isolated clip for the vocabulary bank.
  --stage sequence        frame tagger + forced alignment + spotting tables (modules/sequence.py) — v7 runs this stage locally

Inputs  <GCS>/data/{pose_store.npz, clips.parquet, concepts.json, transitions.parquet, daily_concepts.json}
        <GCS>/models/{unisign_wlasl_enc.pt, mt5_config.json}
Outputs <GCS>/runs/<run_id>/…   (checkpoint every few evals → a Spot preemption resumes instead of restarting)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from modules.utils import get_logger, seed_all, write_json

log = get_logger("job")
A = None


def paths(a):
    gcs = Path(a.gcs or os.environ.get("THAISLM_GCS", ROOT / "cloud_s3" / "data_prep"))
    data = Path(a.data) if a.data else (gcs / "data" if (gcs / "data").exists() else gcs / "shards")
    models = Path(a.models) if a.models else (gcs / "models" if (gcs / "models").exists() else ROOT / "artifacts" / "pretrained")
    out = Path(a.out) if a.out else gcs / "runs" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    return data, models, out


def load_model(models, device, init="wlasl"):
    """Uni-Sign pose branch (CSL-News pre-training → WLASL fine-tune) + mT5 encoder, top `mt5_layers` trainable; ckpt:<path> = any
    full SignEncoder state dict (e.g. to continue from the previous release)."""
    from modules.encoder import SignEncoder
    cfg = json.loads((models / "mt5_config.json").read_text())
    layers = A.mt5_layers if A is not None else 8
    if init.startswith("ckpt:"):
        sd = torch.load(init[5:], map_location="cpu")
        m = SignEncoder(cfg, n_train_layers=layers, prefix_len=sd["prefix"].shape[0])
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        return m.to(device)
    base = torch.load(models / "unisign_wlasl_enc.pt", map_location="cpu")
    m = SignEncoder(cfg, n_train_layers=layers, prefix_len=base["prefix"].shape[0])
    m.load_base(base)
    return m.to(device)


def _prep_rows(args):
    """worker: (store path, list of (row, f0, f1)) → encoder input parts (CPU pool, so the GPU does not wait for numpy)."""
    from modules.encoder import prep, to_iso
    store, rows = args
    return [prep(to_iso(kp, hw), sc) for kp, sc, hw in (store.get(r, f0, f1) for r, f0, f1 in rows)]


def stage_train_encoder(a):
    from modules.encoder import NULL, BatchIter, ClipDataset, CosFace, PoseStore, collate, concept_scores, embed_parts, prep, retrieval_metrics, supcon, to_iso
    from modules import encoder as _enc
    seed_all(0)
    data, models, out = paths(a)
    dev = a.device
    t0 = time.time()
    store = PoseStore(data / "pose_store.npz")
    clips = pd.read_parquet(data / "clips.parquet")
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
    iso = iso[(iso.f1 - iso.f0) >= 4]
    train = iso[iso.role == "train"]
    log.info("store %d rows (%.1fs); isolated/null %d; train %d clips / %d concepts / %d signers; RTMW views %d", len(store.clip_id), time.time() - t0,
             len(iso), len(train), train.c.nunique(), train.signer.nunique(), int((train.aux >= 0).sum()))
    table = [dict(row=int(r.row), aux=int(r.aux), c=int(r.c), signer=r.signer, dataset=r.dataset, f0=int(r.f0), f1=int(r.f1)) for r in train.itertuples()]
    null_sources = [t for t, r in zip(table, train.itertuples()) if r.kind == "isolated" and r.n_frames - (r.f1 - r.f0) >= 8 and r.dataset != "tslone"]
    transitions = []
    if (data / "transitions.parquet").exists():
        tr = pd.read_parquet(data / "transitions.parquet")
        transitions = [(store.row[c], int(f0), int(f1)) for c, f0, f1 in zip(tr.clip_id, tr.f0, tr.f1) if c in store.row]
    daily = []
    if (data / "daily_concepts.json").exists():
        daily = [cidx[c] for c in json.loads((data / "daily_concepts.json").read_text(encoding="utf-8")) if c in cidx]
    sig_per_c = train.groupby("c").signer.nunique()
    multi = [int(c) for c in sig_per_c[sig_per_c >= 2].index if c != cidx[NULL]]
    allc = [int(c) for c in train.c.unique() if c != cidx[NULL]]
    log.info("concepts with >=2 signers: %d; daily concepts %d; null clips %d; real transitions %d", len(multi), len(daily),
             int((train.c == cidx[NULL]).sum()), len(transitions))

    model = load_model(models, dev, a.init)
    _enc.SIGNER_AUG.update(p=a.signer_aug, mirror=a.mirror)

    eval_df = iso.reset_index(drop=True)
    if a.limit_eval:
        eval_df = eval_df.groupby(["dataset", "role"], group_keys=False).apply(lambda g: g.sample(min(len(g), a.limit_eval), random_state=0)).reset_index(drop=True)
    from concurrent.futures import ThreadPoolExecutor
    rows = [(int(r.row), int(r.f0), int(r.f1)) for r in eval_df.itertuples()]
    with ThreadPoolExecutor(8) as ex:
        chunks = list(ex.map(_prep_rows, [(store, rows[i:i + 500]) for i in range(0, len(rows), 500)]))
    eval_parts = [p for c in chunks for p in c]
    # RTMW views of the evaluation clips: extractor-consistency check (same video, two extractors → same embedding?)
    pair = eval_df[eval_df.aux >= 0].sample(min(800, int((eval_df.aux >= 0).sum())), random_state=0) if (eval_df.aux >= 0).any() else eval_df.head(0)
    pair_parts = [prep(to_iso(kp, hw), sc) for kp, sc, hw in (store.get(int(r.aux), int(r.f0), int(r.f1)) for r in pair.itertuples())]
    pair_idx = pair.index.values
    log.info("prepared %d eval clips + %d RTMW pairs (%.0fs)", len(eval_parts), len(pair_parts), time.time() - t0)
    daily_set = set(daily)

    def protocols(Z, Zpair=None):
        res = {}
        tr = (eval_df.role == "train").values & (eval_df.concept != NULL).values
        C = len(concepts)
        nm = (eval_df.role == "train").values & (eval_df.concept == NULL).values
        null_proto = Z[nm].mean(0) if nm.any() else -Z[tr].mean(0)
        null_proto = null_proto / (np.linalg.norm(null_proto) + 1e-9)
        one_cols = sorted(set(eval_df[eval_df.dataset == "tslone"].c))
        t51_cols = sorted(set(eval_df[(eval_df.dataset == "tsl51") & (eval_df.concept != NULL)].c))
        daily_cols = sorted(daily_set)

        def q(mask, name, allowed=None):
            if mask.sum() == 0:
                return
            S = concept_scores(Z[mask], Z[tr], eval_df.c.values[tr], C)
            S[:, cidx[NULL]] = -2
            res[name] = retrieval_metrics(S, eval_df.c.values[mask], allowed)
        for part in ("val", "test"):
            for ds, tag in (("ttrs", "E1_ttrs"), ("tslone", "ONE"), ("th_sl", "THSL"), ("agent", "AGENT")):
                m = (eval_df.dataset == ds).values & (eval_df.role == part).values & (eval_df.concept != NULL).values
                q(m, f"{tag}_{part}|open_vocab")
                if daily_cols:
                    md = m & np.isin(eval_df.c.values, daily_cols)
                    q(md, f"{tag}_{part}|daily_vocab", allowed=daily_cols)
            m = (eval_df.dataset == "tslone").values & (eval_df.role == part).values
            q(m, f"ONE_{part}|closed_184", allowed=one_cols)
        m = (eval_df.dataset == "youtube_v4").values & (eval_df.role == "test").values
        q(m, "YT_title|open_vocab")
        for sub in ("tsl51_user_sign",):            # the TSL51 researcher (1 unseen signer), MediaPipe from video
            m = (eval_df.subset == sub).values & (eval_df.concept != NULL).values
            q(m, f"T51U|open_vocab"); q(m, f"T51U|closed_51", allowed=t51_cols)
            if daily_cols:
                q(m & np.isin(eval_df.c.values, daily_cols), "T51U|daily_vocab", allowed=daily_cols)
            mm = (eval_df.subset == sub).values
            if mm.any():
                s_null = Z[mm] @ null_proto
                is_null = (eval_df.concept[mm] == NULL).values
                if is_null.any() and (~is_null).any():
                    pos, neg = s_null[is_null], s_null[~is_null]
                    res["T51U|null_auc"] = dict(auc=float((pos[:, None] > neg[None, :]).mean()), n_null=int(is_null.sum()), n_sign=int((~is_null).sum()))
        if Zpair is not None and len(Zpair):
            same = float(np.mean(np.sum(Z[pair_idx] * Zpair, 1)))
            rnd = float(np.mean(np.sum(Z[pair_idx] * Zpair[np.random.RandomState(0).permutation(len(Zpair))], 1)))
            res["extractor_pair_cos"] = dict(same_clip=same, random_clip=rnd, n=len(Zpair))
        return res

    def val_score(r):
        ks = [k for k in ("E1_ttrs_val|open_vocab", "ONE_val|open_vocab", "THSL_val|open_vocab", "AGENT_val|open_vocab") if r.get(k, {}).get("n")]
        return float(np.mean([r[k]["MRR"] for k in ks])) if ks else 0.0

    Z0 = embed_parts(model, eval_parts, zero_shot=True, device=dev, bs=a.embed_bs)
    base = protocols(Z0, embed_parts(model, pair_parts, zero_shot=True, device=dev, bs=a.embed_bs) if pair_parts else None)
    np.save(out / "emb_zero_shot.npy", Z0.astype(np.float16))
    write_json(out / "metrics_zero_shot.json", base)
    log.info("zero-shot: %s", json.dumps({k: (round(v.get("R1", v.get("auc", v.get("same_clip", 0))), 3) if isinstance(v, dict) else v) for k, v in base.items()}, ensure_ascii=False))
    protos = np.zeros((len(concepts), 768), np.float32)
    trm = (eval_df.role == "train").values
    for c in range(len(concepts)):
        m = trm & (eval_df.c.values == c)
        protos[c] = Z0[m].mean(0) if m.any() else np.random.RandomState(c).normal(size=768)
    cos = CosFace(torch.from_numpy(protos), s=a.s, m=a.margin).to(dev)

    def neighbours_from(W, k=10):
        """Nearest concepts of every concept by proxy direction (hard negatives for the batch sampler)."""
        Wn = torch.nn.functional.normalize(torch.as_tensor(W, dtype=torch.float32), dim=-1)
        nb = {}
        for i in range(0, len(Wn), 2048):
            S = Wn[i:i + 2048] @ Wn.T
            S[torch.arange(len(S)), torch.arange(i, i + len(S))] = -2
            S[:, cidx[NULL]] = -2
            idx = torch.topk(S, k, dim=1).indices.numpy()
            for j, row in enumerate(idx):
                nb[i + j] = [int(x) for x in row]
        return nb

    groups = model.trainable_groups(lr_gcn=a.lr_gcn, lr_mt5=a.lr_mt5, lr_head=a.lr_head)
    groups.append(dict(params=list(cos.parameters()), lr=a.lr_head))
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.is_floating_point()} if a.ema > 0 else None
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=a.steps, pct_start=0.06, anneal_strategy="cos")
    LOCAL = Path("/tmp/thaislm_run") if out.as_posix().startswith("/gcs/") else out
    LOCAL.mkdir(parents=True, exist_ok=True)

    def to_gcs(names):
        if LOCAL == out:
            return None
        t = threading.Thread(target=lambda: [shutil.copy(LOCAL / n, out / n) for n in names], daemon=True)
        t.start()
        return t
    pending_upload, since_best = None, 0
    ck = out / "ckpt_last.pt"
    step0, best, hist = 0, -1.0, []
    if ck.exists():
        st = torch.load(ck, map_location="cpu")
        model.load_state_dict(st["model"]); cos.load_state_dict(st["cos"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step0, best, hist = st["step"], st["best"], st["hist"]
        if st.get("ema") is not None:
            ema = st["ema"]
        log.info("resumed from step %d (best %.4f)", step0, best)

    ds = ClipDataset(store, table, cidx, null_sources, seed=step0 + 1, transitions=transitions, daily=daily, aux_p=a.aux_p, hard_frac=a.hard_frac)
    multi_mask = torch.zeros(len(concepts), dtype=torch.bool, device=dev)
    multi_mask[torch.as_tensor(multi + [cidx[NULL]], dtype=torch.long)] = True

    nb_file = LOCAL / "neighbours.npy"

    def write_neighbours():
        nb = neighbours_from(cos.W.detach().cpu())
        arr = np.array([nb[i] for i in range(len(nb))], np.int32)
        tmp = LOCAL / "neighbours.tmp.npy"
        np.save(tmp, arr); os.replace(tmp, nb_file)
    write_neighbours()
    ds.neighbours_file = str(nb_file)

    def make_loader(step):
        return iter(torch.utils.data.DataLoader(BatchIter(ds, a.P, a.K, multi, allc, a.null_frac, 1000 + step * 7), batch_size=None,
                                                num_workers=a.workers, prefetch_factor=4 if a.workers else None, persistent_workers=a.workers > 0))
    loader = make_loader(step0)
    model.train()
    tl = time.time()
    oom_keep, n_oom = 1.0, 0
    for step in range(step0, a.steps):
        parts, y = next(loader)
        if oom_keep < 1.0:
            keep_i = np.sort(np.random.RandomState(step).choice(len(parts), max(8, int(len(parts) * oom_keep)), replace=False))
            parts, y = [parts[i] for i in keep_i], y[keep_i]
        try:
            P_, M_ = collate(parts, dev)
            y = torch.as_tensor(y, dtype=torch.long, device=dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                z = model(P_, M_)
            # proxy loss only for concepts seen from ≥ 2 signers (+ null); single-signer words are shaped by SupCon on augmented views
            cm = multi_mask[y]
            l_cos = cos(z[cm], y[cm]) if cm.any() else z.sum() * 0
            l_sup = supcon(z, y, a.temp)
            loss = l_cos + a.w_supcon * l_sup
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            z = loss = P_ = M_ = None  # noqa: F841
            torch.cuda.empty_cache()
            oom_keep *= 0.75; n_oom += 1
            log.warning("CUDA OOM at step %d → batch scaled to %.0f %% (OOM events %d)", step + 1, 100 * oom_keep, n_oom)
            continue
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for k_, v_ in model.state_dict().items():
                    if k_ in ema:
                        ema[k_].mul_(a.ema).add_(v_.float(), alpha=1 - a.ema)
        if (step + 1) % 50 == 0:
            log.info("step %d loss %.3f (cos %.3f supcon %.3f) %.2fs/step", step + 1, loss.item(), l_cos.item(), l_sup.item(), (time.time() - tl) / 50)
            tl = time.time()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            live = None
            if ema is not None:
                live = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
                model.load_state_dict({**live, **{k_: v_.to(live[k_].dtype) for k_, v_ in ema.items()}})
            Z = embed_parts(model, eval_parts, device=dev, bs=a.embed_bs)
            Zp = embed_parts(model, pair_parts, device=dev, bs=a.embed_bs) if pair_parts else None
            r = protocols(Z, Zp)
            sc = val_score(r)
            hist.append(dict(step=step + 1, loss=float(loss.item()), val_score=float(sc), **{k: v for k, v in r.items()}))
            log.info("eval step %d val %.4f | TTRS val R1 %.3f | ONE val %.3f | THSL val %.3f | AGENT val %.3f | T51U open %.3f daily %.3f | pair cos %.3f", step + 1, sc,
                     r.get("E1_ttrs_val|open_vocab", {}).get("R1", 0), r.get("ONE_val|open_vocab", {}).get("R1", 0), r.get("THSL_val|open_vocab", {}).get("R1", 0),
                     r.get("AGENT_val|open_vocab", {}).get("R1", 0), r.get("T51U|open_vocab", {}).get("R1", 0), r.get("T51U|daily_vocab", {}).get("R1", 0),
                     r.get("extractor_pair_cos", {}).get("same_clip", 0))
            if sc > best:
                best, since_best = sc, 0
                torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, LOCAL / "encoder_best.pt")
                np.save(LOCAL / "emb.npy", Z.astype(np.float16))
                write_json(LOCAL / "metrics_best.json", dict(step=step + 1, **r))
                to_gcs(["encoder_best.pt", "emb.npy", "metrics_best.json"])
            else:
                since_best += 1
            if live is not None:
                model.load_state_dict(live)
            write_json(out / "train_history.json", hist)
            if (len(hist) % a.ckpt_every == 0) and not (pending_upload and pending_upload.is_alive()):
                torch.save(dict(model=model.state_dict(), cos=cos.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, best=best,
                                hist=hist, ema=ema), LOCAL / "ckpt_last.pt")
                pending_upload = to_gcs(["ckpt_last.pt"])
            model.train()
            if a.patience and since_best >= a.patience:
                log.info("early stop at step %d: no val improvement in %d evals (best %.4f)", step + 1, since_best, best)
                break
            write_neighbours()                       # hard negatives re-mined from the current proxies (workers reload the file)
    if pending_upload is not None:
        pending_upload.join()
    for n in ("encoder_best.pt", "emb.npy", "metrics_best.json"):
        if LOCAL != out and (LOCAL / n).exists():
            shutil.copy(LOCAL / n, out / n)
    eval_df[["clip_id", "dataset", "subset", "kind", "role", "signer", "concept", "c", "f0", "f1", "extractor"]].to_parquet(out / "emb_index.parquet", index=False)
    write_json(out / "concept_index.json", concepts)
    write_json(out / "summary.json", dict(zero_shot=base, best=json.loads((out / "metrics_best.json").read_text()), best_val_score=best, args=vars(a),
                                          minutes=round((time.time() - t0) / 60, 1)))
    ck.unlink(missing_ok=True)
    log.info("done in %.1f min, best val score %.4f", (time.time() - t0) / 60, best)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--run_id", default="local")
    ap.add_argument("--gcs"); ap.add_argument("--data"); ap.add_argument("--models"); ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=3600); ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--P", type=int, default=48); ap.add_argument("--K", type=int, default=2); ap.add_argument("--null_frac", type=float, default=0.08)
    ap.add_argument("--lr_gcn", type=float, default=1e-4); ap.add_argument("--lr_mt5", type=float, default=2e-5); ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--mt5_layers", type=int, default=8); ap.add_argument("--s", type=float, default=30.0); ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--temp", type=float, default=0.07); ap.add_argument("--w_supcon", type=float, default=0.5); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit_eval", type=int, default=0); ap.add_argument("--device", default="cuda"); ap.add_argument("--embed_bs", type=int, default=384)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--ckpt_every", type=int, default=2); ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--init", default="wlasl", help="wlasl | ckpt:<path>")
    ap.add_argument("--signer_aug", type=float, default=0.3); ap.add_argument("--mirror", type=float, default=0.1)
    ap.add_argument("--aux_p", type=float, default=0.25); ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--encoder_run"); ap.add_argument("--n_synth", type=int, default=2500); ap.add_argument("--n_synth_heldout", type=int, default=240)
    ap.add_argument("--n_distill", type=int, default=1200); ap.add_argument("--span_epochs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=18)
    A = ap.parse_args()
    if A.stage == "train_encoder":
        stage_train_encoder(A)
    elif A.stage == "sequence":
        from modules.sequence import stage_sequence
        stage_sequence(A, paths(A), load_model)
    else:
        raise SystemExit(f"unknown stage {A.stage}")


if __name__ == "__main__":
    main()
