"""v11 landmark encoder training (prompt/phase3/prompt-opt-v11.md TPH 1 §1D, TPH 2 §2A; shared settings C8).

  python main.py train-v11 --run v11-R0 --preset R0 [--clips 10e6] [--workers 5]      # local GPU
  modal run modal_v11.py::encoder --run v11-R2 --preset R2 --clips 10e6               # Modal (same script, cache/shards_mm on a Volume)
  python main.py train-v11 --run v11-R2 --cont_proxy                                   # in-sentence gold-span R@1 of a finished run

Presets (cumulative ladder):
  R0  old recipe (v8 sampler, all concepts are classes, v10 augmentation) + the shared C8 settings, to convergence
  R1  + tiered label space: classes = Core + Extended + every concept with ≥ 2 TRAIN signers; the rest = negatives only
  R2  + cross-signer sampler: (word, signer) cells, positive = same word + same form + other signer (other source when one exists),
        hard negatives = another word of the same signer + confusion-group members (25–50 % of word slots), same word / other form
        masked out of SupCon
  R3  + augmentation / co-articulation pack (coherent virtual signer, structured hand drop-out, 10–25 fps, context 0.5 with on/off-set
        truncation 10–35 %, jitter 0.25)
  R4  + form-aware sub-centres (3 per word, logit = max, form-initialised) + confusion-group CosFace (weight 0.3, margin 0.35)
  R5  R4 + input v11 (modules/lmenc.py features_v11: motion stream, hand-to-face distances, visibility + learned "not observed")
  R6  R5 + phonology-factorised embedding (hs | or | loc | mov, pseudo-labels from modules/phonology.py, weight 0.2 each,
        phonology-only for the first 20 %, minimal-pair margin +0.1)
  R7  R6 at size M (d 384 × depth 8 × 6 heads, drop-path 0.1)
  R8  R5 + signer / source adversary (gradient reversal, weight 0 → 0.3 over 30 %, adversary lr × 5)

Shared (C8): AdamW β (0.9, 0.98), lr 1e-3 (class centres × 5, no wd), wd 0.05, warm-up 5 %, cosine to 1 %, clip 2.0, AMP, EMA 0.999 (evaluated
and exported), batch 512 = 128 words × 2 signers × 2 views, 8 % not-a-sign windows, CosFace s 30 / m 0.25 (margin ramped over 10 %),
SupCon 0.5 τ 0.07, evaluation every 2,000 steps on the TUNE signer sets (val roles: TTRS · TSL-ONE-S · th_sl · YouTube).
Selection: mean MRR of the TUNE signer sets (+ lowest TUNE signer R@1, confusion-group margins and --cont_proxy reported with it).
BLIND-v11 (modules/sealed.py) is removed from every table; the loader raises if a sealed clip reaches training or evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import ART, PREP, get_logger, read_json, seed_all, write_json

log = get_logger("train_v11")
MAN = PREP / "manifest"
NULL = "<null>"
PRESETS = {
    "R0": dict(labels="all", sampler="old", aug="v10"),
    "R1": dict(labels="tiered", sampler="old", aug="v10"),
    "R2": dict(labels="tiered", sampler="cross", aug="v10"),
    "R3": dict(labels="tiered", sampler="cross", aug="v11", context_p=0.5, jitter=0.25),
    "R4": dict(labels="tiered", sampler="cross", aug="v11", context_p=0.5, jitter=0.25, subc=3, group_w=0.3),
    "R5": dict(labels="tiered", sampler="cross", aug="v11", context_p=0.5, jitter=0.25, subc=3, group_w=0.3, fv=11),
    "R6": dict(labels="tiered", sampler="cross", aug="v11", context_p=0.5, jitter=0.25, subc=3, group_w=0.3, fv=11, phon=4),
    "R7": dict(labels="tiered", sampler="cross", aug="v11", context_p=0.5, jitter=0.25, subc=3, group_w=0.3, fv=11, phon=4,
               d=384, depth=8, heads=6, drop_path=0.1),
    "R8": dict(labels="tiered", sampler="cross", aug="v11", context_p=0.5, jitter=0.25, subc=3, group_w=0.3, fv=11, adv=0.3),
}


# ----------------------------------------------------------------------------- data tables
def tables():
    import train_encoder as TE
    from modules.sealed import assert_clean, sealed_signers
    TE.ensure_mm()
    store = TE.PoseStoreMM()
    if os.environ.get("THAISLM_STORE_RAM") == "1":       # whole pose store in RAM (Linux workers share it after fork) — 3.7× faster loader
        store._open()
    clips, iso, concepts, cidx, daily = TE.load_tables(store)
    ss = sealed_signers()
    from modules.sealed import is_sealed
    keep = ~iso.signer.isin(ss) & ~iso.clip_id.map(is_sealed)
    log.info("BLIND-v11: %d clips of %d sealed people removed from the training / evaluation table", int((~keep).sum()), len(ss))
    iso = iso[keep].reset_index(drop=True)
    assert_clean(iso.clip_id.tolist(), "train_v11 tables", signers=iso.signer.dropna().tolist())
    return TE, store, clips, iso, concepts, cidx, daily


def label_space(preset, iso, concepts, cidx):
    """concept index → class index (or -1 = negatives only) + the class names"""
    if preset["labels"] == "all":
        return np.arange(len(concepts)), list(concepts)
    tiers = read_json(MAN / "tiers.json")
    vc = read_json(ROOT / "models" / "model-v10" / "vocab_conversation.json")
    variants = vc.get("variants", {})
    keep = set(tiers["core"]) | set(tiers["extended"]) | set(tiers["label_concepts_ge2"])
    names, lut = [], np.full(len(concepts), -1)
    cls = {}
    for i, c in enumerate(concepts):
        base = variants.get(c, c)                       # a politeness / school variant concept is a FORM of its word (R4 sub-centres)
        if c == NULL:
            continue
        if base in keep or c in keep:
            if base not in cls:
                cls[base] = len(names); names.append(base)
            lut[i] = cls[base]
    cls[NULL] = len(names); names.append(NULL); lut[cidx[NULL]] = cls[NULL]
    return lut, names


# ----------------------------------------------------------------------------- dataset
def make_dataset(TE, store, iso, cidx, lut, preset, a, daily):
    from modules import encoder as E
    from modules.lmenc import prep, prep_v11

    from modules.v11data import V11Data

    prep_fn = prep_v11 if preset.get("fv", 10) == 11 else prep
    forms = read_json(MAN / "forms.json") if (MAN / "forms.json").exists() else {"clip_form": {}}
    cf = forms["clip_form"]
    tr = iso[iso.role == "train"]
    sid = {s: i for i, s in enumerate(sorted(tr.signer.fillna(tr.dataset + ":unknown").unique()))}
    def _row(r):
        return (int(r.row), int(r.aux) if a.aux_p > 0 else -1)
    table = []
    for r in tr.itertuples():
        row, aux = _row(r)
        f = cf.get(r.clip_id, [None, "f0"])[1]
        sg = r.signer if isinstance(r.signer, str) else f"{r.dataset}:unknown"
        table.append(dict(row=row, aux=aux, c=int(r.c), y=int(lut[int(r.c)]), form=int(f[1:]) if isinstance(f, str) else 0, signer=sg,
                          sid=sid[sg], dataset=r.dataset, f0=int(r.f0), f1=int(r.f1), clip=r.clip_id))
    null_sources = [t for t, r in zip(table, tr.itertuples()) if r.kind == "isolated" and r.n_frames - (r.f1 - r.f0) >= 8 and r.dataset != "tslone"]
    transitions = []
    if (TE.SHARDS / "transitions.parquet").exists():
        trn = pd.read_parquet(TE.SHARDS / "transitions.parquet")
        from modules.sealed import is_sealed
        transitions = [(store.row[c], int(f0), int(f1)) for c, f0, f1 in zip(trn.clip_id, trn.f0, trn.f1) if c in store.row and not is_sealed(c)]
    from modules import encoder as E
    E.SIGNER_AUG.update(p=a.signer_aug, mirror=a.mirror)
    ds = V11Data(store, table, cidx, null_sources, seed=a.seed, transitions=transitions, daily=daily, aux_p=a.aux_p, hard_frac=a.hard_frac,
                 prep_fn=prep_fn)
    ds.jitter, ds.context_p, ds.framing_p = preset.get("jitter", 0.15), preset.get("context_p", 0.0), a.framing_p
    ds.aug_v11 = preset.get("aug") == "v11"
    ds.lut = lut; ds.null_y = int(lut[cidx[NULL]])
    sig = tr.groupby("c").signer.nunique()
    multi = [int(c) for c in sig[sig >= 2].index if c != cidx[NULL]]
    allc = [int(c) for c in tr.c.unique() if c != cidx[NULL]]
    if preset["sampler"] == "cross":
        names = ds.class_names
        cls = {n: i for i, n in enumerate(names)}
        from v11 import GROUPS
        groups = [[cls[w] for w in g if w in cls] for g in GROUPS + extra_groups()]
        ys = pd.Series([t["y"] for t in table]); sgs = pd.Series([t["signer"] for t in table])
        nsig = sgs.groupby(ys).nunique()
        words_multi = [int(y) for y, n in nsig.items() if n >= 2 and y >= 0 and y != ds.null_y]
        daily_words = {lut[c] for c in daily}
        tail_rows = [i for i, t in enumerate(table) if t["y"] < 0] or [i for i, t in enumerate(table) if nsig.get(t["y"], 0) < 2]
        ds.setup_cross(groups, words_multi, daily_words, tail_rows)
        log.info("cross sampler: %d words with ≥ 2 signers, %d confusion groups, %d tail rows", len(ds.words_multi), len(ds.groups), len(tail_rows))
    return ds, multi, allc, table


def extra_groups():
    f = MAN / "confusion_groups_tune.json"          # groups extended from the TUNE confusion matrix (diagnose v11), when present
    return read_json(f)["groups"] if f.exists() else []


from modules.v11data import Iter, collate  # noqa: E402


# ----------------------------------------------------------------------------- losses
class SubCosFace(nn.Module):
    """CosFace with K sub-centres per class (logit = max over centres; the target centre is the clip's form when it has one)."""

    def __init__(self, n, emb, k=1, s=30.0, m=0.25):
        super().__init__()
        self.W = nn.Parameter(F.normalize(torch.randn(n * k, emb), dim=-1))
        self.n, self.k, self.s, self.m = n, k, s, m

    def cos(self, z):
        c = z @ F.normalize(self.W, dim=-1).t()
        return c.view(len(z), self.n, self.k)

    def forward(self, z, y, form=None, m=None, cols=None):
        m = self.m if m is None else m
        c3 = self.cos(z)
        c = c3.max(-1).values
        if self.k > 1 and form is not None:
            fk = form.clamp(0, self.k - 1)
            tgt = c3[torch.arange(len(z)), y].gather(1, fk[:, None]).squeeze(1)
            known = form < self.k
            tgt = torch.where(known, tgt, c[torch.arange(len(z)), y])
            c = c.scatter(1, y[:, None], tgt[:, None])
        if cols is not None:
            c = c[:, cols]
        logits = self.s * (c - m * F.one_hot(y, c.shape[1]).to(c.dtype))
        return F.cross_entropy(logits, y)


def supcon_v11(z, y, form, temp=0.07, extra_neg_margin=None):
    """multi-positive SupCon: positives = same label AND same form (other sample); same label + other form = excluded from the
    denominator; y < 0 (Tail) = negatives only (unique, never anchors)."""
    sim = z @ z.t() / temp
    n = len(z)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    same = (y[:, None] == y[None, :]) & (y[:, None] >= 0)
    pos = same & (form[:, None] == form[None, :]) & ~eye
    excl = same & ~pos & ~eye
    if extra_neg_margin is not None:
        sim = sim + extra_neg_margin / temp
    sim = sim.masked_fill(eye | excl, -1e4)
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    lp = sim.log_softmax(1)
    return -((lp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


# ----------------------------------------------------------------------------- evaluation
def tune_eval(Z, ev, concepts, cidx, daily, TE, groups_c):
    """protocols of train_encoder (+ per-signer R@1 on the TUNE signer sets and confusion-group margins)."""
    from modules.encoder import concept_scores
    r = TE.protocols(Z, ev, concepts, cidx, daily)
    C = len(concepts)
    tr = (ev.role == "train").values & (ev.concept != NULL).values
    val = (ev.role == "val").values & (ev.concept != NULL).values & ev.dataset.isin(["ttrs", "tslone", "th_sl", "agent"]).values
    S = concept_scores(Z[val], Z[tr], ev.c.values[tr], C); S[:, cidx[NULL]] = -2
    t = ev.c.values[val]
    ok = S[np.arange(len(t)), t] > -1.5
    hit = (S.argmax(1) == t) & ok
    sg = ev.signer.values[val]
    per = {}
    for s in np.unique(sg[ok].astype(str)):
        mm = (sg.astype(str) == s) & ok
        if mm.sum() >= 8:
            per[s] = float(hit[mm].mean())
    r["tune_per_signer"] = per
    r["tune_min_signer_R1"] = min(per.values()) if per else None
    # confusion-group margin: cos(query, nearest bank clip of its word, other signer) − max over other group members
    marg = []
    Zt, ct, st = Z[tr], ev.c.values[tr], ev.signer.values[tr]
    for g in groups_c:
        for qi in np.flatnonzero(val & np.isin(ev.c.values, g)):
            w = ev.c.values[qi]
            s_ = Zt @ Z[qi]
            own = (ct == w) & (st != ev.signer.values[qi])
            oth = np.isin(ct, [x for x in g if x != w])
            if own.any() and oth.any():
                marg.append(float(s_[own].max() - s_[oth].max()))
    r["confusion_margin"] = dict(mean=float(np.mean(marg)) if marg else None, n=len(marg), pos_share=float(np.mean(np.array(marg) > 0)) if marg else None)
    return r


def run_hash(a, preset):
    d = dict(preset=preset, clips=a.clips, P=a.P, K=a.K, lr=a.lr, seed=a.seed, ema=a.ema)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------- training
def train(a):
    from modules.lmenc import build_model, embed, N_FEAT, N_FEAT_V11, prep, prep_v11
    preset = dict(PRESETS[a.preset], **json.loads(a.override or "{}"))
    seed_all(a.seed)
    TE, store, clips, iso, concepts, cidx, daily = tables()
    lut, names = label_space(preset, iso, concepts, cidx)
    from modules.lmenc import build_model as _bm  # noqa: F401
    out = Path(a.out) if a.out else ART / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    import types
    import modules.encoder as E
    E.ClipDataset.class_names = names
    ds, multi, allc, table = make_dataset(TE, store, iso, cidx, lut, preset, a, daily)
    ds.neighbours_file = str(out / "neighbours.npy")
    P = a.P; K = a.K
    batch = int(P * K * (1 + a.null_frac + (0.06 if preset["sampler"] == "cross" else 0)))
    steps = a.steps or int(math.ceil(a.clips / (P * K)))
    it = Iter(ds, P, K, multi, allc, a.null_frac, a.seed, preset["sampler"])
    dl = torch.utils.data.DataLoader(it, batch_size=None, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0,
                                     prefetch_factor=4 if a.workers else None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fv = preset.get("fv", 10)
    model = build_model(n_feat=N_FEAT_V11 if fv == 11 else N_FEAT, d=preset.get("d", 256), emb=a.emb, depth=preset.get("depth", 4),
                        heads=preset.get("heads", 4), drop=a.drop, fv=fv, drop_path=preset.get("drop_path", 0.0), phon=preset.get("phon", 0)).to(dev)
    n_cls = len(names)
    cos = SubCosFace(n_cls, a.emb, k=preset.get("subc", 1), s=a.s, m=a.margin).to(dev)
    extra = nn.ModuleDict()
    phon = preset.get("phon", 0)
    if phon:
        pl = np.load(MAN / "phonology_labels.npz", allow_pickle=True)
        pmap = {str(c): i for i, c in enumerate(pl["clip"])}
        PL = pl["labels"].astype(np.int64)                              # [N, 4] (hs, or, loc, mov), -1 unknown
        ncls = [int(PL[:, k].max()) + 1 for k in range(4)]
        for k in range(4):
            extra[f"ph{k}"] = nn.Linear(a.emb // phon, ncls[k])
        # pseudo-labels per table row (by clip) — passed through the sampler via form/sid? use a lookup tensor on the row's clip
        ds_ph = np.full((len(table), 4), -1, np.int64)
        for i, t in enumerate(table):
            j = pmap.get(t["clip"])
            if j is not None:
                ds_ph[i] = PL[j]
        # the sampler returns signer ids in sg; for phonology we need the row → swap sg for the row index (adversary is off in R6/R7)
        ds.phon_rows = True
        minimal = minimal_pairs(table, ds_ph, names)
        log.info("phonology: classes %s, minimal pairs %d", ncls, len(minimal))
    if preset.get("adv"):
        n_sg = max(t["sid"] for t in table) + 1
        extra["adv"] = nn.Sequential(nn.Linear(a.emb, 256), nn.GELU(), nn.Linear(256, n_sg))
    extra = extra.to(dev)
    log.info("run %s preset %s %s | model %.2f M | classes %d | batch %d (P %d × K %d) | %d steps = %.1f M clips", a.run, a.preset, preset,
             sum(p.numel() for p in model.parameters()) / 1e6, n_cls, batch, P, K, steps, steps * P * K / 1e6)
    groups = [dict(params=model.parameters(), lr=a.lr), dict(params=cos.parameters(), lr=a.lr * 5, weight_decay=0)]
    if len(extra):
        groups.append(dict(params=extra.parameters(), lr=a.lr * (5 if preset.get("adv") else 1), weight_decay=0))
    opt = torch.optim.AdamW(groups, weight_decay=a.wd, betas=(0.9, 0.98))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    ev = iso.reset_index(drop=True)
    t0 = time.time()
    prep_fn = prep_v11 if fv == 11 else prep
    feats = TE.eval_feats(store, ev, prep_fn)
    log.info("eval features for %d clips (%.0fs)", len(feats), time.time() - t0)
    from v11 import GROUPS
    groups_c = [[cidx[w] for w in g if w in cidx] for g in GROUPS]
    hist, best, best_step, step, clips_seen = [], -1.0, 0, 0, 0
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    last = out / "ckpt_last.pt"
    if last.exists() and not a.fresh:
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); cos.load_state_dict(ck["cos"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); ema = ck["ema"]
        if len(extra) and ck.get("extra"):
            extra.load_state_dict(ck["extra"])
        step, best, best_step, hist, clips_seen = ck["step"], ck["best"], ck["best_step"], ck["hist"], ck.get("clips_seen", 0)
        it.seed = a.seed + step
        log.info("resumed from step %d (best %.4f at %d)", step, best, best_step)
        if a.extend:
            steps = int(ck.get("steps", steps) * (1 + a.extend))
            log.info("extended to %d steps", steps)
    t0 = time.time(); tick = time.time(); wall0 = time.time()
    model.train()
    ramp = max(1, int(0.1 * steps))
    for x, m, y, fm, sg in dl:
        x, m, y, fm, sg = (t_.to(dev, non_blocking=True) for t_ in (x, m, y, fm, sg))
        frac = step / steps
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            h = model.frames(x, m)
            zr = model.pool_raw(h, m)
        zr = zr.float()
        if phon:
            parts = model.parts(zr); z = torch.cat(parts, -1) / math.sqrt(phon)
        else:
            z = F.normalize(zr, dim=-1)
        lab = y >= 0
        mcur = a.margin * min(1.0, (step + 1) / ramp)
        loss_id = cos(z[lab], y[lab], fm[lab], m=mcur) if lab.any() else z.sum() * 0
        yy = torch.where(lab, y, -(torch.arange(len(y), device=dev) + 1))           # Tail: unique negative labels
        loss_sc = supcon_v11(z, yy, fm)
        loss = loss_id + a.supcon * loss_sc
        if preset.get("group_w") and lab.any():                                      # confusion-group CosFace (margin 0.35)
            gl, gn = 0.0, 0
            for g in ds.groups if hasattr(ds, "groups") else []:
                gm = lab & torch.isin(y, torch.tensor(g, device=dev))
                if gm.sum() >= 2:
                    cols = torch.tensor(g, device=dev)
                    yi = (y[gm][:, None] == cols[None]).float().argmax(1)
                    c3 = cos.cos(z[gm]).max(-1).values[:, cols]
                    gl = gl + F.cross_entropy(a.s * (c3 - 0.35 * F.one_hot(yi, len(g)).float()), yi); gn += 1
            if gn:
                loss = loss + preset["group_w"] * gl / gn
        if phon:
            rows = sg                                                             # in phonology runs the sampler's 4th output = table row
            PLb = torch.from_numpy(ds_ph).to(dev)[rows.clamp(min=0)]
            PLb[rows < 0] = -1
            ph_only = frac < 0.2
            lp = 0.0
            for k in range(4):
                lk = PLb[:, k]
                okk = lk >= 0
                if okk.sum() > 4:
                    lp = lp + 0.2 * (F.cross_entropy(extra[f"ph{k}"](parts[k][okk]), lk[okk]) + supcon_v11(parts[k][okk], lk[okk], torch.zeros_like(lk[okk])))
            if minimal and lab.any():                                             # minimal pairs: +0.1 margin on the differing sub-space
                for k in range(4):
                    pk = minimal_mask(y, minimal, k)
                    if pk is not None and pk.any():
                        s_ = parts[k] @ parts[k].t()
                        lp = lp + 0.2 * F.relu(s_[pk] - (1 - 0.1)).mean()
            loss = (lp if ph_only else loss + lp)
        if preset.get("adv"):
            lam = preset["adv"] * min(1.0, frac / 0.3)
            okk = sg >= 0
            if okk.any():
                loss = loss + F.cross_entropy(extra["adv"](GradRev.apply(z[okk], lam)), sg[okk])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cos.parameters()) + list(extra.parameters()), 2.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1; clips_seen += len(y)
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(a.ema).add_(v.detach(), alpha=1 - a.ema)
                else:
                    ema[k].copy_(v)
        if step % 100 == 0:
            el = time.time() - tick; tick = time.time()
            log.info("step %d/%d loss %.3f (id %.3f sc %.3f) %.2f it/s %.0f clips/s | %.2f M clips", step, steps, loss.item(), float(loss_id),
                     float(loss_sc), 100 / el, 100 * len(y) / el, clips_seen / 1e6)
            hist.append(dict(step=step, loss=float(loss.item()), clips=clips_seen, it_s=100 / el))
        if step % a.eval_every == 0 or step >= steps or (a.smoke and step >= a.smoke):
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
            Z = embed(model, [f.astype(np.float32) for f in feats], device=dev)
            r = tune_eval(Z, ev, concepts, cidx, daily, TE, groups_c)
            sc_ = TE.val_score(r)
            hist.append(dict(step=step, clips=clips_seen, val=sc_, min_signer=r["tune_min_signer_R1"], margin=r["confusion_margin"]["mean"],
                             **{k: round(v.get("R1", v.get("auc", 0)), 4) for k, v in r.items() if isinstance(v, dict) and ("R1" in v or "auc" in v)}))
            log.info("eval step %d: TUNE MRR %.4f | min signer R1 %s | margin %s | %s", step, sc_, r["tune_min_signer_R1"], r["confusion_margin"],
                     {k: round(v.get("R1", 0), 3) for k, v in r.items() if isinstance(v, dict) and "val" in k and "R1" in v})
            if sc_ > best:
                best, best_step = sc_, step
                torch.save(dict(cfg=model.cfg, state=model.state_dict(), step=step, val=sc_, preset=preset), out / "encoder_best.pt")
                np.save(out / "emb.npy", Z.astype(np.float16))
                write_json(out / "metrics_best.json", r)
            if preset["sampler"] == "old" or True:              # hard negatives from the class proxies (both samplers use them)
                W = F.normalize(cos.W.detach().float(), dim=-1).view(n_cls, cos.k, -1).mean(1)
                nbs = np.zeros((n_cls, 10), np.int32)
                for i in range(0, n_cls, 2048):
                    S = W[i:i + 2048] @ W.T
                    S[torch.arange(len(S)), torch.arange(i, i + len(S))] = -2
                    nbs[i:i + 2048] = torch.topk(S, 10, dim=1).indices.cpu().numpy()
                if preset["sampler"] == "old":       # the old sampler works in concept space: map class neighbours back to concepts
                    inv = {}
                    for ci, yi in enumerate(lut):
                        if yi >= 0:
                            inv.setdefault(int(yi), ci)
                    cn = np.zeros((len(concepts), 10), np.int32)
                    for ci, yi in enumerate(lut):
                        if yi >= 0:
                            cn[ci] = [inv.get(int(j), ci) for j in nbs[yi]]
                    np.save(out / "neighbours.npy", cn)
                else:
                    np.save(out / "neighbours.npy", nbs)
            model.load_state_dict(live)
            write_json(out / "train_history.json", hist)
            torch.save(dict(model=model.state_dict(), cos=cos.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                            step=step, best=best, best_step=best_step, hist=hist, ema=ema, extra=extra.state_dict() if len(extra) else None,
                            clips_seen=clips_seen, steps=steps), last)
            model.train()
            if a.smoke and step >= a.smoke:
                break
        elif step % a.ckpt_every == 0:
            torch.save(dict(model=model.state_dict(), cos=cos.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                            step=step, best=best, best_step=best_step, hist=hist, ema=ema, extra=extra.state_dict() if len(extra) else None,
                            clips_seen=clips_seen, steps=steps), last)
        if a.patience and clips_seen >= a.min_clips and len([h for h in hist if "val" in h]) and                 step - best_step >= a.patience * a.eval_every and step % a.eval_every == 0:
            log.info("converged: no TUNE gain for %d evaluations after %.1f M clips → stop at step %d (best %d)", a.patience, clips_seen / 1e6, step, best_step)
            break
        if a.budget_s and time.time() - wall0 > a.budget_s:
            log.info("time box reached at step %d → closing at the current best (time-capped)", step)
            break
        if step >= steps:
            break
    ev[["clip_id", "dataset", "subset", "kind", "role", "signer", "concept", "c", "f0", "f1"]].to_parquet(out / "emb_index.parquet", index=False)
    write_json(out / "concept_index.json", concepts)
    improving = best_step >= 0.85 * step
    write_json(out / "summary.json", dict(run=a.run, preset_name=a.preset, preset=preset, config_hash=run_hash(a, preset), best_step=best_step, best_val=best,
                                         steps_done=step, steps_planned=steps, clips_seen=clips_seen, still_improving=bool(improving),
                                         time_capped=bool(step < steps), wall_s=time.time() - wall0, args=vars(a), classes=n_cls,
                                         train_clips=len(table), gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"))
    log.info("done: best TUNE MRR %.4f at step %d / %d (%.1f M clips)%s → %s", best, best_step, step, clips_seen / 1e6,
             " — still improving" if improving else "", out)


def minimal_pairs(table, ph, names):
    """word pairs (class ids) that agree in three phonology sub-spaces and differ in the fourth (majority label per word)."""
    import collections
    maj = {}
    by = collections.defaultdict(list)
    for t, p in zip(table, ph):
        if t["y"] >= 0 and (p >= 0).all():
            by[t["y"]].append(tuple(p))
    for y, ps in by.items():
        if len(ps) >= 2:
            maj[y] = np.array([collections.Counter(x[k] for x in ps).most_common(1)[0][0] for k in range(4)])
    ys = sorted(maj)
    out = []
    for i, a_ in enumerate(ys):
        for b_ in ys[i + 1:]:
            d = maj[a_] != maj[b_]
            if d.sum() == 1:
                out.append((a_, b_, int(np.flatnonzero(d)[0])))
    return out


_MP_CACHE = {}


def minimal_mask(y, minimal, k):
    """[B,B] bool: sample pairs whose words are a minimal pair differing in sub-space k (one [C,C] table per k, built once)"""
    key = (k, len(minimal), str(y.device))
    if key not in _MP_CACHE:
        pairs = [(a_, b_) for a_, b_, kk in minimal if kk == k]
        if not pairs:
            _MP_CACHE[key] = None
        else:
            n = max(max(a_, b_) for a_, b_ in pairs) + 1
            M = torch.zeros(n, n, dtype=torch.bool, device=y.device)
            for a_, b_ in pairs:
                M[a_, b_] = M[b_, a_] = True
            _MP_CACHE[key] = M
    M = _MP_CACHE[key]
    if M is None:
        return None
    ok = (y >= 0) & (y < len(M))
    yc = y.clamp(0, len(M) - 1)
    return M[yc][:, yc] & ok[:, None] & ok[None, :]


# ----------------------------------------------------------------------------- reports on a finished run
def cont_proxy(a):
    """in-sentence gold-span R@1 (TSL51 TUNE sentences, aligned windows, bank = TRAIN clips of the same run)"""
    import train_encoder as TE
    ns = argparse.Namespace(cont_proxy=a.run, proxy_role=a.proxy_role)
    TE.cont_proxy(ns)


def eval_runs(a):
    """every run of --eval_runs (artifacts/runs/<name>/encoder_best.pt, v10's = enc-v8) on the SAME BLIND-free table with the C8
    selection metrics → artifacts/reports/v11_encoder_tune.json (+ in-sentence gold-span R@1 with --cont)"""
    from modules.lmenc import embed, load_model, prep_for
    TE, store, clips, iso, concepts, cidx, daily = tables()
    ev = iso.reset_index(drop=True)
    from v11 import GROUPS
    groups_c = [[cidx[w] for w in g if w in cidx] for g in GROUPS]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = ART / "reports" / "v11_encoder_tune.json"
    out = read_json(f) if f.exists() else {}
    feats_cache = {}
    for run in [x for x in a.eval_runs.split(",") if x]:
        ck = ART / "runs" / run / "encoder_best.pt"
        model = load_model(ck, dev)
        pf = prep_for(model.cfg)
        if pf not in feats_cache:
            feats_cache[pf] = TE.eval_feats(store, ev, pf)
        Z = embed(model, [x.astype(np.float32) for x in feats_cache[pf]], device=dev)
        r = tune_eval(Z, ev, concepts, cidx, daily, TE, groups_c)
        sets = {k: dict(R1=round(v["R1"], 4), MRR=round(v["MRR"], 4), n=v["n"]) for k, v in r.items()
                if isinstance(v, dict) and "MRR" in v and "|open_vocab" in k and "_val" in k}
        sm = read_json(ART / "runs" / run / "summary.json") if (ART / "runs" / run / "summary.json").exists() else {}
        out[run] = dict(tune_MRR=TE.val_score(r), sets=sets, min_signer_R1=r["tune_min_signer_R1"], per_signer=r["tune_per_signer"],
                        confusion_margin=r["confusion_margin"], daily={k: round(v["R1"], 4) for k, v in r.items() if isinstance(v, dict) and "daily_vocab" in k and "R1" in v},
                        clips_seen=sm.get("clips_seen"), best_step=sm.get("best_step"), steps=sm.get("steps_done"), still_improving=sm.get("still_improving"))
        log.info("%-14s TUNE MRR %.4f | min signer %.3f | margin %.3f | %s", run, out[run]["tune_MRR"], out[run]["min_signer_R1"] or -1,
                 out[run]["confusion_margin"]["mean"] or 0, {k.split("|")[0]: v["MRR"] for k, v in sets.items()})
        np.save(ART / "runs" / run / "emb_v11table.npy", Z.astype(np.float16))
        nm = (ev.role == "train").values & (ev.concept == NULL).values          # "not a sign" prototype in this encoder's space (Path B)
        if nm.any():
            npr = Z[nm].mean(0); np.save(ART / "runs" / run / "null_proto.npy", (npr / (np.linalg.norm(npr) + 1e-9)).astype(np.float32))
        if a.probe:                                   # R8 rule: a freshly trained linear signer probe on TRAIN-clip embeddings (5-fold CV)
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            tm = (ev.role == "train").values & (ev.concept != NULL).values & ev.signer.notna().values
            sg = ev.signer.values[tm]
            keep_s = {x for x, n in zip(*np.unique(sg, return_counts=True)) if n >= 50}
            mm = np.isin(sg, list(keep_s))
            rng = np.random.RandomState(0); idx = rng.choice(np.flatnonzero(mm), min(6000, mm.sum()), replace=False)
            acc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), Z[tm][idx], sg[idx], cv=5).mean()
            out[run]["signer_probe_acc"] = float(acc)
            log.info("%-14s signer probe accuracy %.4f (%d signers)", run, acc, len(keep_s))
        if a.cont:
            out[run]["in_sentence"] = cont_v11(model, pf, Z, ev, store, cidx, dev)
            log.info("%-14s in-sentence gold-span R@1 %s", run, out[run]["in_sentence"])
        write_json(f, out)


def cont_v11(model, pf, Z, ev, store, cidx, dev):
    """in-sentence gold-span R@1 (--cont_proxy of C8) on the BLIND-free table: every aligned word of the TSL51 TUNE sentences
    (seq-v7 alignment, the one the decision sets use, role tagger_train), embedded alone, ranked against this run's TRAIN-clip bank; conversation vocabulary;
    as aligned and with both boundaries moved by up to ±25 %"""
    from modules.lmenc import embed
    from modules.parts import to_iso
    from modules.runtime import Bank
    tr = (ev.role == "train").values & (ev.concept != NULL).values
    bank = Bank(Z[tr], ev.c.values[tr], len(cidx))
    vocab = read_json(ROOT / "models" / "model-v10" / "vocab_conversation.json")["concepts"]
    conv = np.array(sorted({cidx[w] for w in vocab if w in cidx}))
    al = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")
    wins = []
    for cid, v in al.items():
        if v["role"] != "tagger_train":
            continue
        key = "t51s:" + cid.split(":", 1)[1]
        if key in store.row:
            for sg in v["segments"]:
                if sg["concept"] in cidx:
                    wins.append((store.row[key], int(sg["f0"]), int(sg["f1"]), cidx[sg["concept"]]))
    rng = np.random.RandomState(0)
    res = {}
    for vn in ("aligned", "moved"):
        F_, t = [], []
        for row, f0, f1, c in wins:
            n = store.length(row)
            if vn == "moved":
                L = f1 - f0; d0, d1 = rng.randint(-int(0.25 * L), int(0.25 * L) + 1, 2)
                f0, f1 = int(np.clip(f0 + d0, 0, n - 4)), int(np.clip(f1 + d1, f0 + 4, n))
            kp, sc, hw = store.get(row, f0, f1)
            F_.append(pf(to_iso(kp, hw), sc)); t.append(c)
        S = bank.scores(embed(model, F_, device=dev))
        m = np.ones(S.shape[1], bool); m[conv] = False; S[:, m] = -9
        t = np.array(t)
        rank = (S > S[np.arange(len(t)), t][:, None]).sum(1)
        res[vn] = dict(n=len(t), R1=round(float((rank < 1).mean()), 4), R5=round(float((rank < 5).mean()), 4))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=""); ap.add_argument("--preset", default="R0", choices=list(PRESETS))
    ap.add_argument("--eval_runs", default="", help="comma-separated finished runs (+ enc-v8 = v10's landmark encoder) on the v11 TUNE protocol")
    ap.add_argument("--override", default="", help="JSON dict merged into the preset")
    ap.add_argument("--out", default="")
    ap.add_argument("--clips", type=float, default=10e6); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--P", type=int, default=128); ap.add_argument("--K", type=int, default=4); ap.add_argument("--null_frac", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--emb", type=int, default=256); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--s", type=float, default=30.0); ap.add_argument("--margin", type=float, default=0.25); ap.add_argument("--supcon", type=float, default=0.5)
    ap.add_argument("--aux_p", type=float, default=0.25); ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--signer_aug", type=float, default=0.3); ap.add_argument("--mirror", type=float, default=0.1)
    ap.add_argument("--framing_p", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--fresh", action="store_true"); ap.add_argument("--extend", type=float, default=0.0, help="resume and extend the schedule by this share")
    ap.add_argument("--budget_s", type=float, default=0.0, help="time box (s): close at the current best")
    ap.add_argument("--smoke", type=int, default=0, help="stop after this many steps (throughput / plumbing check)")
    ap.add_argument("--patience", type=int, default=0, help="early stop after this many evaluations without a TUNE gain (once --min_clips are seen)")
    ap.add_argument("--min_clips", type=float, default=6e6)
    ap.add_argument("--cont_proxy", action="store_true"); ap.add_argument("--proxy_role", default="tagger_train")
    ap.add_argument("--cont", action="store_true", help="with --eval_runs: + in-sentence gold-span R@1 (TSL51 TUNE sentences)")
    ap.add_argument("--probe", action="store_true", help="with --eval_runs: + signer-probe accuracy (R8 keep rule)")
    a = ap.parse_args()
    if a.eval_runs:
        return eval_runs(a)
    if a.cont_proxy:
        return cont_proxy(a)
    train(a)


if __name__ == "__main__":
    main()
