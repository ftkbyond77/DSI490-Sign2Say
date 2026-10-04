"""L3: continuous signing — where are the words, and where is NOT signing (rest, transitions, title cards).

v4's tagger was trained only on synthetic TTRS concatenations and on real fluent signing found no internal boundaries.
v5 adds the two things that were missing:
  (1) REAL continuous signing with word order: TSL51's 252 sentence videos (76 templates, 3–6 signs, "no_space" and
      "with_space" recordings). Their gloss sequence is known but not the timing → forced alignment with the fine-tuned
      encoder (semi-Markov DP that must place every gloss, in order, with free null gaps). Half of the templates train the
      tagger, the other half are a real continuous test set (exact gloss sequence).
  (2) an explicit NON-SIGN class in every training sentence: TSL51 null_act clips (standing still, fidgeting, touching
      clothes, transitions), rest before/after isolated signs, pauses, and "no signer" frames (pose missing → title cards,
      cut-aways). The tagger therefore learns what not-signing looks like, not only what boundaries look like.
Sentence synthesis also stitches word-level clips of *different* datasets (TSL-ONE-S nouns + TSL51 verbs/pronouns) after
retargeting each clip to a common shoulder frame — this is how word-level data becomes continuous sentences.

Frame tagger  [fine-tuned encoder frame state 768 ‖ kinematics 13] → BiGRU → {0 non-sign, 1 sign-begin, 2 sign-inside}
Spotting      windows × concept bank (+ null prototype) → tables saved for decoding/tuning on CPU (scripts/data_pipeline.py)
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import SH_L, SH_R
from .segment import FPS, N_KIN, boundary_f1, decode_tagger, gt_segments, kin_features, resample, window_list  # noqa: F401
from .utils import get_logger, write_json

log = get_logger("sequence")


class Tagger(nn.Module):
    """Frame tagger: non-sign / sign-begin / sign-inside — 2-layer BiGRU over [encoder frame state ‖ 13 kinematic features]
    (chosen over a dilated TCN and a Transformer in the v6 benchmark)."""

    def __init__(self, d_in=768 + N_KIN, d=192, arch="gru"):
        super().__init__()
        assert arch == "gru", arch
        self.arch = arch
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * d, 3)

    def forward(self, x):
        return self.out(self.gru(self.inp(x))[0])


# ----------------------------------------------------------------------------- retargeting + synthesis
def body_frame(kp, sc, thr=0.3):
    ok = (sc[:, SH_L] > thr) & (sc[:, SH_R] > thr)
    if not ok.any():
        return np.array([0.5, 0.45]), 0.2
    mid = ((kp[ok, SH_L] + kp[ok, SH_R]) / 2).mean(0)
    sw = float(np.linalg.norm(kp[ok, SH_L] - kp[ok, SH_R], axis=1).mean())
    return mid, max(sw, 1e-3)


def retarget(kp, sc, mid_to=(0.75, 0.45), sw_to=0.22):
    mid, sw = body_frame(kp, sc)
    return ((kp - mid) * (sw_to / sw) + np.array(mid_to)).astype(np.float32)


class Synth:
    """Continuous sentences from isolated clips. pools: dict signer → list of clip dict(row, f0, f1, concept); null_rows: list."""

    def __init__(self, store, pools, null_rows, rest_rows, seed=0, cross_dataset=0.35, max_len=900, speed=(0.9, 1.7)):
        self.store, self.pools, self.null_rows, self.rest_rows = store, pools, null_rows, rest_rows
        self.speed = speed
        self.signers = [s for s, v in pools.items() if len(v) >= 6]
        self.all_clips = [c for v in pools.values() for c in v]
        self.rng = np.random.RandomState(seed)
        self.cross = cross_dataset
        self.max_len = max_len

    def _clip(self, c, a=None, b=None):
        kp, sc, hw = self.store.get(c["row"], c["f0"] if a is None else a, c["f1"] if b is None else b)
        H, W = hw
        return retarget(kp * np.array([W / H, 1.0], np.float32), sc), sc

    def _rest(self, n):
        """n frames of real non-signing: a TSL51 null clip or the rest part of an isolated clip, or a still pose."""
        rng = self.rng
        r = rng.rand()
        if r < 0.5 and self.null_rows:
            c = self.null_rows[rng.randint(len(self.null_rows))]
            L = self.store.length(c["row"])
            a = rng.randint(0, max(1, L - n)); kp, sc = self._clip(c, a, min(L, a + n))
        elif self.rest_rows:
            c = self.rest_rows[rng.randint(len(self.rest_rows))]
            L = self.store.length(c["row"])
            a, b = (0, c["f0"]) if c["f0"] >= L - c["f1"] else (c["f1"], L)
            if b - a < 2:
                a, b = 0, max(2, c["f0"])
            kp, sc = self._clip(c, a, b)
        else:
            return None
        if len(kp) < n:
            reps = int(np.ceil(n / max(len(kp), 1)))
            kp = np.concatenate([kp, kp[::-1]] * reps)[:n]; sc = np.concatenate([sc, sc[::-1]] * reps)[:n]
        return kp[:n], sc[:n]

    def sample(self):
        rng = self.rng
        k = rng.randint(2, 7)
        if rng.rand() < self.cross:
            picks = [self.all_clips[i] for i in rng.choice(len(self.all_clips), k, replace=False)]
        else:
            pool = self.pools[self.signers[rng.randint(len(self.signers))]]
            picks = [pool[i] for i in rng.choice(len(pool), min(k, len(pool)), replace=False)]
        KP, SC, Y, words = [], [], [], []

        def put(kp, sc, y):
            KP.append(kp); SC.append(sc.astype(np.float32)); Y.append(np.full(len(kp), y, np.int64) if np.isscalar(y) else y)

        if rng.rand() < 0.15:  # title card / no signer before the sentence
            n = rng.randint(10, 40); put(np.zeros((n, 133, 2), np.float32), np.zeros((n, 133), np.float32), 0)
        r = self._rest(rng.randint(5, 25))
        if r:
            put(*r, 0)
        for j, c in enumerate(picks):
            kp, sc = self._clip(c)
            kp, sc = resample(kp, sc, rng.uniform(*self.speed))
            if KP:
                u = rng.rand()
                if u < 0.3:
                    r = self._rest(rng.randint(6, 30))
                    if r:
                        rk, rs = r
                        put(*_bridge(KP[-1][-1], rk[0], SC[-1][-1], rs[0], rng.randint(2, 5)), 0)
                        put(rk, rs, 0)
                elif u < 0.45:  # hold / pause (with_space style)
                    n = rng.randint(4, 15)
                    put(np.repeat(KP[-1][-1:], n, 0) + rng.normal(0, 0.002, (n, 133, 2)).astype(np.float32), np.repeat(SC[-1][-1:], n, 0), 0)
                put(*_bridge(KP[-1][-1], kp[0], SC[-1][-1], sc[0], rng.randint(1, 6)), 0)
            y = np.full(len(kp), 2, np.int64); y[0] = 1
            put(kp, sc, y); words.append(c["concept"])
        r = self._rest(rng.randint(5, 25))
        if r:
            put(*_bridge(KP[-1][-1], r[0][0], SC[-1][-1], r[1][0], 3), 0); put(*r, 0)
        if rng.rand() < 0.1:
            n = rng.randint(10, 40); put(np.zeros((n, 133, 2), np.float32), np.zeros((n, 133), np.float32), 0)
        kp = np.concatenate(KP)[:self.max_len]; sc = np.concatenate(SC)[:self.max_len]; y = np.concatenate(Y)[:self.max_len]
        ang = np.deg2rad(rng.uniform(-6, 6)); s = rng.uniform(0.85, 1.15)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
        live = sc.sum(1) > 0
        kp[live] = ((kp[live] - [0.75, 0.45]) @ R.T + [0.75, 0.45] + rng.uniform(-0.05, 0.05, 2)).astype(np.float32)
        kp = kp + rng.normal(0, 0.002, kp.shape).astype(np.float32)
        return dict(kp=kp, sc=sc, y=y, words=words)


def _bridge(k0, k1, s0, s1, n):
    w = np.linspace(0, 1, n + 2)[1:-1, None, None]
    return (k0[None] * (1 - w) + k1[None] * w).astype(np.float32), np.minimum(s0, s1)[None].repeat(n, 0)


# ----------------------------------------------------------------------------- windows × bank
def align_sentence(table, gloss_c, T, len_prior=0.08, L0=24, null_gap=0.0):
    """Place every gloss of a known sequence, in order, on non-overlapping windows maximising Σ score(window, gloss)
    − len_prior·|log(L/L0)|. table must contain full score rows for the sentence glosses: table['S'] [W, n_gloss]."""
    wins, S = table["wins"], table["S"]
    n = len(gloss_c)
    ends = {}
    for w, (a, b) in enumerate(wins):
        ends.setdefault(b, []).append((a, w))
    NEG = -1e9
    f = np.full((T + 1, n + 1), NEG); f[0, 0] = 0.0
    back = {}
    for t in range(1, T + 1):
        for i in range(n + 1):
            best, arg = f[t - 1, i] - null_gap, ("gap",)
            if i > 0:
                for a, w in ends.get(t, []):
                    L = t - a
                    v = f[a, i - 1] + S[w, i - 1] - len_prior * abs(np.log(L / L0))
                    if v > best:
                        best, arg = v, ("win", a, w)
            f[t, i] = best; back[(t, i)] = arg
    t, i, segs = T, n, []
    if f[T, n] <= NEG / 2:
        return []
    while t > 0:
        arg = back[(t, i)]
        if arg[0] == "gap":
            t -= 1
        else:
            _, a, w = arg
            segs.append((a, t, gloss_c[i - 1], float(S[w, i - 1]))); t, i = a, i - 1
    return segs[::-1]


# ----------------------------------------------------------------------------- job stage (v7, local)
def stage_sequence(a, paths_, load_model):
    """v7 continuous-signing stage (runs locally on a small GPU; everything MediaPipe).

    1. synthetic sentences from isolated clips (train signers → tagger; held-out signers → decoder tuning / report), signing speed
       0.8–2.4× (phone signing is fast), real rest and transition blocks
    2. span head (one-pass spotting): distilled from the encoder's own isolated-window embeddings on synthetic + real sentences
    3. forced alignment of the TSL51 sentence videos (span-head scores restricted to the known glosses, in order) → pseudo boundaries
    4. tagger (BiGRU) trained on synthetic + ALL tune templates (+ 1.5× / 2× copies) — prompt_9 P1; epoch chosen on the synthetic
       held-out set (v5 rule). Two fold taggers (each without half of the tune templates) give OUT-OF-FOLD probabilities for the
       tune sentences, so the decoder and the calibration are tuned on tagger outputs the tagger never trained on.
    5. evaluation items: poses (for exact re-scoring at tuning time), tagger probabilities, span-head window tables against the full
       and the conversation train-only banks, at 1× / 1.5× / 2× speed and after tempo normalisation; per-window teacher tables for
       the 1× test sentences (reference: what the slow exact spotting would give)."""
    from .encoder import NULL, PoseStore, embed_parts, frame_features, load_encoder, prep, to_iso
    from .runtime import Bank, table_from_Z
    from .schema import activity
    from .spanhead import SpanHeadNP, train_span_head
    data, models, out = paths_
    dev = a.device
    t0 = time.time()
    rng = np.random.RandomState(0)
    enc_dir = Path(a.encoder_run)
    model = load_encoder(models, enc_dir / "encoder_best.pt", device=dev)
    store = PoseStore(data / "pose_store.npz")
    clips = pd.read_parquet(data / "clips.parquet")
    clips = clips[clips.clip_id.isin(store.row) & (clips.view == "primary")].reset_index(drop=True)
    clips["row"] = clips.clip_id.map(store.row)
    idx = pd.read_parquet(enc_dir / "emb_index.parquet")
    Zall = np.load(enc_dir / "emb.npy").astype(np.float32)
    concepts = json.loads((enc_dir / "concept_index.json").read_text(encoding="utf-8"))
    cidx = {c: i for i, c in enumerate(concepts)}
    C = len(concepts)
    daily = set(json.loads((data / "daily_concepts.json").read_text(encoding="utf-8"))) if (data / "daily_concepts.json").exists() else set()
    daily |= set(clips[(clips.dataset == "tsl51") & (clips.kind == "isolated")].concept.dropna())
    trm = (idx.role == "train").values
    sign = (idx.concept != NULL).values
    bank_full = Bank(Zall[trm & sign], idx.c.values[trm & sign], C)
    conv_ids = {cidx[c] for c in daily if c in cidx}
    bank_conv = Bank(Zall[trm & sign], idx.c.values[trm & sign], C, allowed=conv_ids)
    banks = {"full": bank_full, "conversation": bank_conv}
    nm = trm & (idx.concept == NULL).values
    null_proto = Zall[nm].mean(0) if nm.any() else -Zall[trm & sign].mean(0)      # no null clips (debug subsets): 'away from all signs'
    null_proto /= np.linalg.norm(null_proto) + 1e-9
    np.save(out / "null_proto.npy", null_proto)
    log.info("sequence stage: bank %d clips / %d concepts (conversation %d concepts) (%.0fs)", len(bank_full.c), len(bank_full.concepts_present),
             len(bank_conv.concepts_present), time.time() - t0)
    lens = list(range(8, 50, 4))

    def embed(parts_list):
        return embed_parts(model, parts_list, bs=a.embed_bs, device=dev)

    def seq_feats(kp, sc):
        return np.concatenate([frame_features(model, kp, sc, device=dev), kin_features(kp, sc)], 1).astype(np.float16)

    def active_mask(kp, sc):
        """tagger-independent superset of 'signing' (a hand up or moving) — where candidate windows are placed"""
        act_k = activity(kp, sc, (1, 1))
        on = (act_k["height"] > -1.4) | (act_k["speed"] > 1.0)
        return np.convolve(on.astype(float), np.ones(7) / 7, "same") > 0.2

    def tables(kp, sc, H, head, teacher=False):
        wins = window_list(len(kp), lens, 2, act=active_mask(kp, sc), min_active=0.5)
        if not wins:
            return {}
        res = {}
        Zs = {"student": head(H.astype(np.float32), wins)}
        if teacher:
            Zs["teacher"] = embed([prep(kp[x:y], sc[x:y]) for x, y in wins])
        for src, Z in Zs.items():
            for name, bank in banks.items():
                t = table_from_Z(Z, wins, bank, null_proto)
                t.pop("Z"); t["top_sc"] = t["top_sc"].astype(np.float16)
                res[f"{src}/{name}"] = t
        return res

    # ---------------- 1. synthetic sentences (+ distillation windows)
    iso = idx[(idx.kind == "isolated") & sign].merge(clips[["clip_id", "row", "n_frames"]], on="clip_id")

    def pools_for(mask):
        pools = {}
        for r in iso[mask].itertuples():
            pools.setdefault(r.signer, []).append(dict(row=int(r.row), f0=int(r.f0), f1=int(r.f1), concept=r.concept))
        return pools
    train_pools = pools_for((iso.role == "train").values)
    heldout_pools = pools_for((iso.role.isin(["val", "test"]) & iso.dataset.isin(["tslone", "th_sl", "agent", "ttrs"])).values)
    nulls = idx[(idx.concept == NULL) & (idx.role == "train")].merge(clips[["clip_id", "row"]], on="clip_id")
    null_rows = [dict(row=int(r.row), f0=0, f1=0, concept=NULL) for r in nulls.itertuples()]
    rest_rows = [dict(row=int(r.row), f0=int(r.f0), f1=int(r.f1)) for r in iso[(iso.role == "train") & (iso.dataset != "tslone") & ((iso.n_frames - (iso.f1 - iso.f0)) >= 10)].itertuples()]
    synth_tr = Synth(store, train_pools, null_rows, rest_rows, seed=1, speed=(0.8, 2.4))
    synth_ho = Synth(store, heldout_pools, null_rows, rest_rows, seed=2, cross_dataset=0.0, speed=(0.8, 2.4))
    syn_train, distill = [], []

    def distill_windows(T, y=None, n=24):
        ws = {(int(a0), int(a0) + int(L)) for a0, L in zip(rng.randint(0, max(1, T - 8), n), rng.choice(lens, n)) if a0 + L <= T}
        if y is not None:
            for g0, g1 in gt_segments(y):
                for _ in range(2):
                    a0 = int(np.clip(g0 + rng.randint(-4, 5), 0, max(0, T - 8))); b0 = int(np.clip(g1 + rng.randint(-4, 5), a0 + 8, T))
                    ws.add((a0, b0))
        return sorted(w for w in ws if 8 <= w[1] - w[0] and w[1] <= T)
    for i in range(a.n_synth):
        s = synth_tr.sample()
        x = seq_feats(s["kp"], s["sc"])
        syn_train.append(dict(y=s["y"], x=x))
        if i < a.n_distill:
            ws = distill_windows(len(s["kp"]), s["y"])
            distill.append((x[:, :768], ws, embed([prep(s["kp"][x0:y0], s["sc"][x0:y0]) for x0, y0 in ws]).astype(np.float16)))
        if (i + 1) % 500 == 0:
            log.info("synth train %d/%d (%.0fs)", i + 1, a.n_synth, time.time() - t0)
    syn_ho = []
    for i in range(a.n_synth_heldout):
        s = synth_ho.sample()
        syn_ho.append(dict(y=s["y"], words=s["words"], x=seq_feats(s["kp"], s["sc"]), kp=s["kp"].astype(np.float16), sc=s["sc"].astype(np.float16),
                           split="tune" if i % 2 == 0 else "test"))
    # real sentences: poses + frame features (TSL51 researcher, MediaPipe from video)
    sents = clips[clips.kind == "sentence"].copy()
    if a.limit_eval:
        sents = sents.groupby("role", group_keys=False).head(a.limit_eval)
    real = []
    for r in sents.itertuples():
        kp, sc, hw = store.get(int(r.row))
        kp = to_iso(kp, hw)
        real.append(dict(clip_id=r.clip_id, role=r.role, variation=r.recording_variation, glosses=r.concept.split("|"), kp=kp, sc=sc, x=seq_feats(kp, sc)))
    for s_ in [s for s in real if s["role"] == "tagger_train"]:
        ws = distill_windows(len(s_["kp"]), None, 40)
        distill.append((s_["x"][:, :768], ws, embed([prep(s_["kp"][x0:y0], s_["sc"][x0:y0]) for x0, y0 in ws]).astype(np.float16)))
    log.info("synthetic: train %d, held-out %d; real sentences %d; distillation %d sequences / %d windows (%.0fs)", len(syn_train), len(syn_ho), len(real),
             len(distill), sum(len(d[1]) for d in distill), time.time() - t0)

    # ---------------- 2. span head
    perm = rng.permutation(len(distill))
    n_val = max(20, len(distill) // 20)
    w_head = train_span_head([distill[i] for i in perm[n_val:]], epochs=a.span_epochs, device=dev, log=log, val=[distill[i] for i in perm[:n_val]])
    np.savez(out / "span_head.npz", **w_head)
    head = SpanHeadNP(out / "span_head.npz")
    del distill

    # ---------------- 3. forced alignment of the TSL51 sentences (span-head scores of the known glosses, in order)
    align = {}
    for s in real:
        T = len(s["kp"])
        wins = window_list(T, lens, 2)
        S_full = bank_full.scores(head(s["x"][:, :768].astype(np.float32), wins))
        gc = [cidx.get(g, -1) for g in s["glosses"]]
        S = np.stack([S_full[:, g] if g >= 0 else np.full(len(wins), -1.0) for g in gc], 1)
        segs = align_sentence(dict(wins=wins, S=S), s["glosses"], T)
        y = np.zeros(T, np.int64)
        for s0, s1, _, _ in segs:
            y[s0] = 1; y[s0 + 1:s1] = 2
        s["y"] = y
        align[s["clip_id"]] = dict(segments=[dict(f0=int(s0), f1=int(s1), concept=g, score=round(v, 4)) for s0, s1, g, v in segs], T=T,
                                   role=s["role"], variation=s["variation"], glosses=s["glosses"])
    write_json(out / "tsl51_alignment.json", align)
    log.info("aligned %d sentences (%.0fs)", len(real), time.time() - t0)

    # ---------------- 4. taggers: final (all tune templates) + 2 out-of-fold
    tune = [s for s in real if s["role"] == "tagger_train"]
    tpl = sorted({tuple(s["glosses"]) for s in tune})
    order = np.random.RandomState(11).permutation(len(tpl))
    fold_of = {tpl[j]: k % 2 for k, j in enumerate(order)}

    def fast_copies(s_):
        outl = []
        for f in (1.5, 2.0):
            k2, c2 = resample(s_["kp"], s_["sc"], f)
            ii = np.clip(np.round(np.linspace(0, len(s_["kp"]) - 1, len(k2))).astype(int), 0, len(s_["kp"]) - 1)
            y2 = s_["y"][ii].copy()
            y2[(y2 == 1) & (np.r_[-1, y2[:-1]] == 1)] = 2
            outl.append(dict(y=y2, x=seq_feats(k2, c2)))
        return outl
    fast = {id(s): fast_copies(s) for s in tune}
    wts = torch.tensor([1.0, 4.0, 1.0], device=dev)

    def evaluate(tagger, items):
        tagger.eval()
        f1s, cerr, sign_f1 = [], [], []
        with torch.no_grad():
            for it in items:
                p = tagger(torch.from_numpy(it["x"].astype(np.float32))[None].to(dev)).softmax(-1)[0].cpu().numpy()
                pred, gt = decode_tagger(p), gt_segments(it["y"])
                f1s.append(boundary_f1(pred, gt, 3)); cerr.append(abs(len(pred) - len(gt)))
                ps, gs = p.argmax(1) > 0, it["y"] > 0
                tp = (ps & gs).sum(); sign_f1.append(2 * tp / max(ps.sum() + gs.sum(), 1))
        tagger.train()
        return dict(boundary_f1=float(np.mean(f1s)), sign_frame_f1=float(np.mean(sign_f1)), count_abs_err=float(np.mean(cerr)), n=len(items))

    def train_tagger(train_set, tag):
        tagger = Tagger().to(dev)
        opt = torch.optim.AdamW(tagger.parameters(), lr=1e-3, weight_decay=0.01)
        bs = min(32, len(train_set))
        steps = max(1, a.epochs * (len(train_set) // bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
        best, best_state, hist = -1e9, None, []
        sel_items = [s for s in syn_ho if s["split"] == "tune"]
        for ep in range(a.epochs):
            perm_ = rng.permutation(len(train_set))
            for b in range(0, len(perm_) - bs + 1, bs):
                batch = [train_set[j] for j in perm_[b:b + bs]]
                L = max(len(s["x"]) for s in batch)
                X = torch.zeros(bs, L, 768 + N_KIN); Y = torch.full((bs, L), -100, dtype=torch.long)
                for i, s in enumerate(batch):
                    X[i, :len(s["x"])] = torch.from_numpy(s["x"].astype(np.float32)); Y[i, :len(s["y"])] = torch.from_numpy(s["y"])
                X, Y = X.to(dev), Y.to(dev)
                loss = F.cross_entropy(tagger(X).reshape(-1, 3), Y.reshape(-1), weight=wts, ignore_index=-100)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0); opt.step(); sched.step()
            r = evaluate(tagger, sel_items)
            sel = r["boundary_f1"] + r["sign_frame_f1"] - 0.25 * r["count_abs_err"]
            hist.append(dict(epoch=ep, loss=float(loss), synth_heldout=r, sel=sel))
            log.info("tagger[%s] ep %d loss %.3f | synth held-out bF1 %.3f signF1 %.3f cnt %.2f | sel %.3f", tag, ep, float(loss), r["boundary_f1"],
                     r["sign_frame_f1"], r["count_abs_err"], sel)
            if sel > best:
                best, best_state = sel, {k: v.detach().clone() for k, v in tagger.state_dict().items()}
        tagger.load_state_dict(best_state); tagger.eval()
        return tagger, dict(best_sel=best, hist=hist)

    def set_for(items):
        return syn_train + items * 4 + [f for s in items for f in fast[id(s)]] * 2
    tagger, info_final = train_tagger(set_for(tune), "final")
    torch.save(tagger.state_dict(), out / "tagger.pt")
    folds = {f: train_tagger(set_for([s for s in tune if fold_of[tuple(s["glosses"])] != f]), f"fold{f}")[0] for f in (0, 1)}

    def prob_of(tg, x):
        with torch.no_grad():
            return tg(torch.from_numpy(x.astype(np.float32))[None].to(dev)).softmax(-1)[0].cpu().numpy().astype(np.float16)

    # ---------------- 5. evaluation items
    items = []
    t2 = time.time()
    for k, s in enumerate(real):
        split = "tune" if s["role"] == "tagger_train" else "test"
        tg = folds[fold_of[tuple(s["glosses"])]] if split == "tune" else tagger
        for speed in (1.0, 1.5, 2.0):
            kp, sc = (resample(s["kp"], s["sc"], speed) if speed != 1.0 else (s["kp"], s["sc"]))
            x = s["x"] if speed == 1.0 else seq_feats(kp, sc)
            T = len(kp)
            y = np.zeros(T, np.int64)
            for g in align[s["clip_id"]]["segments"]:
                f0, f1 = int(round(g["f0"] / speed)), max(int(round(g["f1"] / speed)), int(round(g["f0"] / speed)) + 1)
                y[min(f0, T - 1)] = 1; y[f0 + 1:min(f1, T)] = 2
            it = dict(kind="real", id=f"{s['clip_id']}@{speed}", clip_id=s["clip_id"], split=split, speed=speed, variation=s["variation"], glosses=s["glosses"],
                      y=y, prob=prob_of(tg, x), tables=tables(kp, sc, x[:, :768], head, teacher=(split == "test" and speed == 1.0)),
                      kp=kp.astype(np.float16), sc=sc.astype(np.float16), stretched={})
            for fct in (1.5, 2.0):               # what the decoder sees after tempo normalisation by fct
                k2, s2 = resample(kp, sc, 1.0 / fct)
                x2 = seq_feats(k2, s2)
                it["stretched"][fct] = dict(prob=prob_of(tg, x2), tables=tables(k2, s2, x2[:, :768], head), kp=k2.astype(np.float16), sc=s2.astype(np.float16))
            items.append(it)
        if (k + 1) % 50 == 0:
            log.info("items %d/%d real sentences (%.0fs)", k + 1, len(real), time.time() - t2)
    for j, s in enumerate(syn_ho):
        items.append(dict(kind="synth", id=f"synth{j}", split=s["split"], speed=1.0, glosses=s["words"], y=s["y"], prob=prob_of(tagger, s["x"]),
                          tables=tables(s["kp"].astype(np.float32), s["sc"].astype(np.float32), s["x"][:, :768], head, teacher=(s["split"] == "test")),
                          kp=s["kp"], sc=s["sc"], stretched={}))
    pickle.dump(dict(items=items, concepts=concepts, conversation=sorted(conv_ids)), open(out / "seq_tables.pkl", "wb"))
    metrics = dict(final=info_final, synth_heldout_test=evaluate(tagger, [s for s in syn_ho if s["split"] == "test"]),
                   tsl51_test=evaluate(tagger, [s for s in real if s["role"] == "sentence_test"]),
                   tsl51_tune_out_of_fold=float(np.mean([evaluate(folds[fold_of[tuple(s["glosses"])]], [s])["boundary_f1"] for s in tune])) if tune else None)
    write_json(out / "sequence_metrics.json", metrics)
    write_json(out / "tagger_arch.json", dict(arch="gru"))
    log.info("sequence stage done in %.1f min: %s", (time.time() - t0) / 60, {k: v for k, v in metrics.items() if k != "final"})
