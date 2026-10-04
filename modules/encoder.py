"""L2: signer-invariant sign encoder — Uni-Sign (WLASL ISLR) fine-tuned on multi-signer Thai data.

Why v4's adapter failed and what changes here
  v4 trained a residual MLP on *frozen* embeddings with SupCon, positives mostly from 3 TTRS signers → nothing to learn
  signer invariance from. v5 has 29 TSL-ONE-S signers × 184 glosses, 2 TSL51 interpreters + 2 web dictionaries × 51 signs,
  and the same concepts across datasets (e.g. พ่อ in TTRS and TSL-ONE-S). So the whole pose encoder is fine-tuned:
    ST-GCN (all) + mT5 encoder (top layers) + residual embedding head (initialised to identity → training starts at zero-shot).
  Loss   CosFace over concepts (proxies initialised from zero-shot class means) + multi-positive SupCon in the batch.
  Batch  P concepts × K instances; K picks a *different signer / dataset / extractor* when the concept has one, otherwise an
         augmented view of the same clip (single-signer TTRS words still get a well-formed neighbourhood).
  Null   "<null>" is a class: TSL51 null_act clips, rest frames outside isolated signs, and windows straddling two signs
         (so a spotting window between words — e.g. hands returning to rest — has its own place in the space).
  Augmentation models what differs between datasets/extractors instead of what differs between signs:
         speed 0.75–1.35×, start/end trimming, small affine, RTMW-like jitter, MediaPipe-like hand dropouts and binary
         confidences.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import USED
from .parts import MAX_LEN, MODES, prepare_parts, resample_len as _resample_len, to_iso  # noqa: F401
from .unisign import UniSignPose

NULL = "<null>"
SLOTS = np.array(USED)


# ----------------------------------------------------------------------------- model
class SignEncoder(nn.Module):
    def __init__(self, mt5_config: dict, n_train_layers=8, prefix_len=13):
        super().__init__()
        from transformers import MT5Config, MT5EncoderModel
        self.pose = UniSignPose()
        self.mt5 = MT5EncoderModel(MT5Config(**mt5_config))
        self.mt5.shared = nn.Embedding(1, 1)
        self.mt5.encoder.embed_tokens = self.mt5.shared
        self.register_buffer("prefix", torch.zeros(prefix_len, 768))
        self.head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 768))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.n_train_layers = n_train_layers

    def load_base(self, sd):
        """sd = state dict of modules.unisign.UniSignEncoder (pose.*, mt5.*, prefix)."""
        self.prefix.data = sd["prefix"].float().clone()
        missing, unexpected = self.load_state_dict({k: v for k, v in sd.items() if k != "prefix"}, strict=False)
        bad = [k for k in missing if not k.startswith("head.") and "left" not in k and k != "prefix"]
        assert not bad, bad[:10]

    def trainable_groups(self, lr_gcn=1e-4, lr_mt5=2e-5, lr_head=5e-4):
        for p in self.mt5.parameters():
            p.requires_grad = False
        blocks = self.mt5.encoder.block
        for b in blocks[len(blocks) - self.n_train_layers:]:
            for p in b.parameters():
                p.requires_grad = True
        for p in self.mt5.encoder.final_layer_norm.parameters():
            p.requires_grad = True
        return [dict(params=[p for p in self.pose.parameters() if p.requires_grad], lr=lr_gcn),
                dict(params=[p for p in self.mt5.parameters() if p.requires_grad], lr=lr_mt5),
                dict(params=list(self.head.parameters()), lr=lr_head)]

    def frames(self, parts, mask):
        g = self.pose(parts)
        B = g.shape[0]
        n_pre = self.prefix.shape[0]
        pre = self.prefix[None].expand(B, -1, -1).to(g.dtype)
        am = torch.cat([torch.ones(B, n_pre, dtype=torch.long, device=g.device), mask.long()], 1)
        h = self.mt5(inputs_embeds=torch.cat([pre, g], 1), attention_mask=am).last_hidden_state
        return h[:, n_pre:]

    def forward(self, parts, mask, zero_shot=False):
        ctx = self.frames(parts, mask)
        m = mask[..., None].to(ctx.dtype)
        z = (ctx * m).sum(1) / m.sum(1).clamp(min=1)
        if not zero_shot:
            z = z + self.head(z.float()).to(z.dtype)
        return F.normalize(z.float(), dim=-1)


def collate(parts_list, device="cuda"):
    L = max(len(p["body"]) for p in parts_list)
    B = len(parts_list)
    out = {m: np.zeros((B, L, parts_list[0][m].shape[1], 3), np.float32) for m in MODES}
    mask = np.zeros((B, L), bool)
    for b, p in enumerate(parts_list):
        n = len(p["body"])
        for m in MODES:
            out[m][b, :n] = p[m]
            if n < L:
                out[m][b, n:] = p[m][-1]
        mask[b, :n] = True
    return {m: torch.from_numpy(v).to(device, non_blocking=True) for m, v in out.items()}, torch.from_numpy(mask).to(device)


# ----------------------------------------------------------------------------- data
class PoseStore:
    """All CS5 clips in RAM: KP [F,69,2] float16, SC [F,69] float16, offsets."""

    def __init__(self, npz):
        z = np.load(npz, allow_pickle=True)
        self.KP, self.SC, self.off = z["KP"], z["SC"], z["off"]
        self.hw = z["hw"]; self.clip_id = list(z["clip_id"])
        self.row = {c: i for i, c in enumerate(self.clip_id)}

    def get(self, i, a=None, b=None):
        s, e = int(self.off[i]), int(self.off[i + 1])
        a = 0 if a is None else a; b = e - s if b is None else b
        kp = np.zeros((b - a, 133, 2), np.float32); sc = np.zeros((b - a, 133), np.float32)
        kp[:, SLOTS] = self.KP[s + a:s + b]; sc[:, SLOTS] = self.SC[s + a:s + b]
        return kp, sc, tuple(int(x) for x in self.hw[i])

    def length(self, i):
        return int(self.off[i + 1] - self.off[i])


SIGNER_AUG = dict(p=0.3, mirror=0.1)    # v7 defaults (the training job may override): body-proportion change, left-handed signer
_BODY_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
_FACE_PAIRS = [(23 + i, 23 + 16 - i) for i in range(8)] + [(83, 87), (84, 86), (88, 90)]
_MIRROR = np.arange(133)
for _a, _b in _BODY_PAIRS + _FACE_PAIRS + [(91 + i, 112 + i) for i in range(21)]:
    _MIRROR[_a], _MIRROR[_b] = _b, _a


def signer_augment(kp, sc, rng, p=0.5, mirror=0.0):
    """Simulate a different signer on isotropic keypoints [T,133,2]: shoulder width, upper-arm / forearm lengths and hand size
    are rescaled per sample (the hand follows its wrist), and with probability `mirror` the signer becomes left-handed
    (x mirrored around the body centre, left/right slots swapped)."""
    kp = kp.copy(); sc = sc.copy()
    if rng.rand() < p:
        ok = (sc[:, 5] > 0) & (sc[:, 6] > 0)
        if ok.any():
            mid = ((kp[ok, 5] + kp[ok, 6]) / 2).mean(0)
            ws = rng.uniform(0.92, 1.08)
            for sh, el, wr, hand in ((5, 7, 9, slice(91, 112)), (6, 8, 10, slice(112, 133))):
                sh0, el0, wr0 = kp[:, sh].copy(), kp[:, el].copy(), kp[:, wr].copy()
                sh1 = mid + (sh0 - mid) * ws
                el1 = sh1 + (el0 - sh0) * rng.uniform(0.88, 1.12)
                wr1 = el1 + (wr0 - el0) * rng.uniform(0.88, 1.12)
                root = kp[:, hand][:, :1]
                hs = rng.uniform(0.9, 1.1)
                kp[:, hand] = wr1[:, None] + (kp[:, hand] - root) * hs + (root - wr0[:, None])
                kp[:, sh], kp[:, el], kp[:, wr] = sh1, el1, wr1
    if mirror and rng.rand() < mirror:
        ok = (sc[:, 5] > 0) & (sc[:, 6] > 0)
        cx = float(((kp[ok, 5, 0] + kp[ok, 6, 0]) / 2).mean()) if ok.any() else float(kp[sc > 0, 0].mean()) if (sc > 0).any() else 0.0
        kp[..., 0] = np.where(sc > 0, 2 * cx - kp[..., 0], kp[..., 0])
        kp, sc = kp[:, _MIRROR], sc[:, _MIRROR]
    return kp.astype(np.float32), sc.astype(np.float32)


def webcam_augment(kp, sc, rng):
    """What a phone / webcam + in-browser MediaPipe does to a sequence (v7): an uneven, lower frame rate (frames dropped and
    duplicated after resampling to 25 fps), hand landmarks that flicker for a few frames, and per-joint jitter that is larger on the
    fingers than on the body."""
    n = len(kp)
    if n > 8 and rng.rand() < 0.35:                        # 12–24 fps capture → 25 fps (nearest-frame duplicates)
        f = rng.uniform(12, 24) / 25.0
        m = max(4, int(round(n * f)))
        src = np.round(np.linspace(0, n - 1, m)).astype(int)
        back = src[np.clip(np.round(np.linspace(0, m - 1, n)).astype(int), 0, m - 1)]
        kp, sc = kp[back], sc[back]
    if rng.rand() < 0.4:                                   # finger jitter (MediaPipe hand landmarks at low resolution)
        kp = kp.copy()
        for hand in (slice(91, 112), slice(112, 133)):
            kp[:, hand] += rng.normal(0, rng.uniform(0.002, 0.006), kp[:, hand].shape).astype(np.float32)
    if rng.rand() < 0.3 and n > 6:                         # short hand flicker (≤ 4 frames, like a missed detection)
        sc = sc.copy()
        hand = slice(91, 112) if rng.rand() < 0.5 else slice(112, 133)
        for _ in range(rng.randint(1, 4)):
            a = rng.randint(0, n - 2); sc[a:a + rng.randint(1, 5), hand] = 0.0
    return kp, sc


def augment(kp, sc, hw, rng, strength=1.0):
    kp = to_iso(kp, hw)
    if SIGNER_AUG["p"] or SIGNER_AUG["mirror"]:
        kp, sc = signer_augment(kp, sc, rng, SIGNER_AUG["p"], SIGNER_AUG["mirror"])
    n = len(kp)
    f = rng.uniform(0.7, 1.6) if rng.rand() < 0.8 * strength else 1.0      # v7: up to 1.6× (fast phone signing)
    kp, sc = _resample_len(kp, sc, int(round(n / f)))
    ang = np.deg2rad(rng.uniform(-10, 10) * strength); s = rng.uniform(0.9, 1.1)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    R = R @ np.array([[1, rng.uniform(-0.08, 0.08) * strength], [0, 1]])
    ok = sc > 0
    c = kp[ok].mean(0) if ok.any() else np.zeros(2)
    kp = ((kp - c) @ R.T + c + rng.uniform(-0.05, 0.05, 2)).astype(np.float32)
    if rng.rand() < 0.5:
        kp = kp + rng.normal(0, 0.004, kp.shape).astype(np.float32)
    sc = sc.copy()
    if rng.rand() < 0.5:
        sc = np.where(sc > 0.3, 1.0, 0.0).astype(np.float32)
    if rng.rand() < 0.3 * strength and len(kp) > 12:
        hand = slice(91, 112) if rng.rand() < 0.5 else slice(112, 133)
        a = rng.randint(0, len(kp) - 6); b = min(len(kp), a + rng.randint(6, 16))
        sc[a:b, hand] = 0.0
    if strength >= 1.0:
        kp, sc = webcam_augment(kp, sc, rng)
    return kp, sc


def prep(kp_iso, sc, max_len=MAX_LEN):
    if len(kp_iso) > max_len:
        kp_iso, sc = _resample_len(kp_iso, sc, max_len)
    return prepare_parts(kp_iso, sc)


class ClipDataset:
    """Training batches (v7). Clip table rows: row (store row of the MediaPipe view), aux (store row of the RTMW view of the same
    video, -1 if none), c (concept), signer, dataset, f0, f1.
    Batch = P concepts × K instances:
      * seed concepts: 65 % from concepts seen by ≥ 2 signers, daily-conversation concepts drawn 3× as often;
      * hard negatives: each seed brings (p = hard_frac) one of its nearest concepts (`neighbours`, from the current proxies — at the
        start these are the ASL/CSL prior's own confusions), so the loss must separate exactly the pairs the prior mixes up;
      * K instances: a different signer when one exists (85 %), else the other extractor's view of the same video (RTMW ↔
        MediaPipe — a free cross-extractor positive), else an augmented view; any instance uses its RTMW view with p = aux_p."""

    def __init__(self, store: PoseStore, table, concept_index, null_sources, seed=0, transitions=None, daily=None, aux_p=0.25, hard_frac=0.5):
        self.store, self.t = store, table
        self.cidx = concept_index
        self.by_c = {}
        for i, r in enumerate(table):
            self.by_c.setdefault(r["c"], []).append(i)
        self.null_c = concept_index[NULL]
        self.null_sources = null_sources          # rows usable for rest/straddle null windows
        self.transitions = transitions or []      # real between-sign movements (TSL51 sentence alignment gaps): (row, f0, f1)
        self.daily = set(daily or [])
        self.neighbours = {}
        self.neighbours_file, self._nb_mtime, self._n_batches = None, None, 0
        self.aux_p, self.hard_frac = aux_p, hard_frac
        self.rng = np.random.RandomState(seed)

    def view(self, i, rng, aug=True, use_aux=None):
        r = self.t[i]
        row = r["row"]
        if r.get("aux", -1) >= 0 and (use_aux if use_aux is not None else (aug and rng.rand() < self.aux_p)):
            row = r["aux"]
        n = self.store.length(row)
        f0, f1 = min(r["f0"], max(0, n - 4)), min(r["f1"], n)
        L = f1 - f0
        if aug:
            f0 = int(np.clip(f0 + rng.randint(-int(0.15 * L) - 1, int(0.15 * L) + 2), 0, max(0, f1 - 4)))
            f1 = int(np.clip(f1 + rng.randint(-int(0.15 * L) - 1, int(0.15 * L) + 2), f0 + 4, n))
        kp, sc, hw = self.store.get(row, f0, f1)
        if aug:
            kp, sc = augment(kp, sc, hw, rng)
        else:
            kp = to_iso(kp, hw)
        return prep(kp, sc)

    def null_view(self, rng):
        """Not a sign: rest before/after an isolated sign, a real between-sign movement from continuous signing, or a synthetic
        window straddling the end of one sign and the start of another."""
        u = rng.rand()
        if self.transitions and u < 0.35:
            row, a, b = self.transitions[rng.randint(len(self.transitions))]
            n = self.store.length(row)
            a = max(0, a - rng.randint(0, 3)); b = min(n, max(b + rng.randint(0, 3), a + 6))
            kp, sc, hw = self.store.get(row, a, b)
            kp, sc = augment(kp, sc, hw, rng, 0.5)
            return prep(kp, sc)
        r = self.null_sources[rng.randint(len(self.null_sources))]
        n = self.store.length(r["row"])
        if u < 0.65:
            pre, post = r["f0"], n - r["f1"]
            if max(pre, post) >= 8:
                a, b = (0, r["f0"]) if pre >= post else (r["f1"], n)
                kp, sc, hw = self.store.get(r["row"], a, b)
                kp, sc = augment(kp, sc, hw, rng, 0.5)
                return prep(kp, sc)
        o = self.null_sources[rng.randint(len(self.null_sources))]
        L1, L2 = r["f1"] - r["f0"], o["f1"] - o["f0"]
        ka, sa, hwa = self.store.get(r["row"], r["f1"] - max(3, int(L1 * rng.uniform(0.3, 0.5))), r["f1"])
        kb, sb, hwb = self.store.get(o["row"], o["f0"], o["f0"] + max(3, int(L2 * rng.uniform(0.3, 0.5))))
        ka, kb = to_iso(ka, hwa), to_iso(kb, hwb)

        def anchor(k, s):
            ok = (s[:, 5] > 0) & (s[:, 6] > 0)
            if not ok.any():
                return np.array([0.0, 0.0]), 1.0
            mid = ((k[ok, 5] + k[ok, 6]) / 2).mean(0); sw = np.linalg.norm(k[ok, 5] - k[ok, 6], axis=1).mean()
            return mid, max(sw, 1e-3)
        (ma, swa), (mb, swb) = anchor(ka, sa), anchor(kb, sb)
        kb = (kb - mb) * (swa / swb) + ma
        ntr = rng.randint(2, 6)
        w = np.linspace(0, 1, ntr + 2)[1:-1, None, None]
        tr = ka[-1][None] * (1 - w) + kb[0][None] * w
        kp = np.concatenate([ka, tr, kb]).astype(np.float32)
        sc = np.concatenate([sa, np.minimum(sa[-1], sb[0])[None].repeat(ntr, 0), sb]).astype(np.float32)
        kp = (kp + rng.normal(0, 0.003, kp.shape)).astype(np.float32)
        return prep(kp, sc)

    def _concepts(self, P, multi, allc, rng):
        n_multi = min(len(multi), int(P * 0.65))
        wm = np.array([3.0 if c in self.daily else 1.0 for c in multi]); wm /= wm.sum()
        seeds = list(rng.choice(multi, n_multi, replace=False, p=wm)) + list(rng.choice(allc, max(0, P - n_multi), replace=False))
        out, seen = [], set()
        for c in seeds:
            if len(out) >= P:
                break
            if c in seen:
                continue
            out.append(c); seen.add(c)
            nb = self.neighbours.get(int(c))
            if nb is not None and len(out) < P and rng.rand() < self.hard_frac:
                cand = [x for x in nb if x not in seen and x in self.by_c]
                if cand:
                    x = cand[rng.randint(min(len(cand), 5))]
                    out.append(x); seen.add(x)
        while len(out) < P:
            c = int(rng.choice(allc))
            if c not in seen:
                out.append(c); seen.add(c)
        return out

    def _refresh_neighbours(self):
        """Hard negatives are re-mined by the trainer at every evaluation and written to `neighbours_file` ([C,k] int32);
        long-lived loader workers pick the new table up here (no worker restart, no re-pickling of the pose store)."""
        import os
        f = self.neighbours_file
        if f is None or not os.path.exists(f):
            return
        mt = os.path.getmtime(f)
        if mt != self._nb_mtime:
            try:
                nb = np.load(f)
                self.neighbours = {i: [int(x) for x in row] for i, row in enumerate(nb)}
                self._nb_mtime = mt
            except Exception:  # noqa: BLE001  (file being replaced — try again next time)
                pass

    def batch(self, P, K, multi_concepts, all_concepts, null_frac=0.08):
        if self._n_batches % 20 == 0:
            self._refresh_neighbours()
        self._n_batches += 1
        rng = self.rng
        n_null = int(round(P * K * null_frac))
        parts, labels = [], []
        for c in self._concepts(P, multi_concepts, all_concepts, rng):
            idx = self.by_c[c]
            first = idx[rng.randint(len(idx))]
            picks = [(first, None)]
            others = [j for j in idx if self.t[j]["signer"] != self.t[first]["signer"]]
            for _ in range(K - 1):
                if others and rng.rand() < 0.85:
                    picks.append((others[rng.randint(len(others))], None))
                elif self.t[first].get("aux", -1) >= 0 and rng.rand() < 0.7:
                    picks.append((first, True))                    # same video, the other extractor
                else:
                    picks.append((idx[rng.randint(len(idx))], None))
            for j, ua in picks:
                parts.append(self.view(j, rng, use_aux=ua)); labels.append(c)
        for _ in range(n_null):
            parts.append(self.null_view(rng)); labels.append(self.null_c)
        return parts, np.array(labels)


class BatchIter(torch.utils.data.IterableDataset):
    """Endless stream of ready batches (each DataLoader worker gets its own RNG)."""

    def __init__(self, ds, P, K, multi, allc, null_frac, seed):
        super().__init__()
        self.ds, self.P, self.K, self.multi, self.allc, self.null_frac, self.seed = ds, P, K, multi, allc, null_frac, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        self.ds.rng = np.random.RandomState(self.seed + (wi.id if wi else 0))
        while True:
            yield self.ds.batch(self.P, self.K, self.multi, self.allc, self.null_frac)


# ----------------------------------------------------------------------------- losses
class CosFace(nn.Module):
    def __init__(self, protos: torch.Tensor, s=30.0, m=0.25):
        super().__init__()
        self.W = nn.Parameter(F.normalize(protos.float(), dim=-1).clone())
        self.s, self.m = s, m

    def forward(self, z, y):
        cos = z @ F.normalize(self.W, dim=-1).t()
        logits = self.s * (cos - self.m * F.one_hot(y, cos.shape[1]).to(cos.dtype))
        return F.cross_entropy(logits, y)


def supcon(z, y, temp=0.07):
    sim = z @ z.t() / temp
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y[:, None] == y[None, :]) & ~eye
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    lp = sim.log_softmax(1)
    return -((lp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


# ----------------------------------------------------------------------------- inference helpers
@torch.no_grad()
def embed_parts(model, parts_list, bs=256, zero_shot=False, device="cuda"):
    """Embeddings of many clips, longest-similar batches. GPU OOM-safe: a batch that does not fit is halved and retried."""
    model.eval()
    order = np.argsort([len(p["body"]) for p in parts_list])
    out = np.zeros((len(parts_list), 768), np.float32)
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        try:
            P, M = collate([parts_list[j] for j in idx], device)
            with torch.autocast("cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
                z = model(P, M, zero_shot=zero_shot)
            out[idx] = z.float().cpu().numpy()
            i += len(idx)
        except torch.cuda.OutOfMemoryError:
            del P, M
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
    return out


@torch.no_grad()
def frame_features(model, kp_iso, sc, device="cuda", chunk=400):
    """Contextual frame states [T,768] for a long sequence (chunks of `chunk` frames with 50-frame overlap)."""
    model.eval()
    T = len(kp_iso)
    out = np.zeros((T, 768), np.float32); wsum = np.zeros(T, np.float32)
    step = chunk - 50
    for a in range(0, max(1, T - 50), step):
        b = min(T, a + chunk)
        P, M = collate([prepare_parts(kp_iso[a:b], sc[a:b])], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = model.frames(P, M)[0].float().cpu().numpy()
        w = np.minimum(np.arange(1, b - a + 1), np.arange(b - a, 0, -1)).clip(max=25).astype(np.float32)
        out[a:b] += h * w[:, None]; wsum[a:b] += w
        if b == T:
            break
    return out / np.maximum(wsum, 1e-6)[:, None]


def concept_scores(Q, Z, inst_c, n_concepts, topm=2):
    """[Q, C] = mean of the top-`topm` instance similarities per concept (v4 rule), -2 where a concept has no instance."""
    S = torch.as_tensor(Q) @ torch.as_tensor(Z).T
    idx = torch.as_tensor(inst_c, dtype=torch.long).expand_as(S)
    top1 = torch.full((len(Q), n_concepts), -2.0).scatter_reduce(1, idx, S, "amax")
    if topm == 1:
        return top1.numpy()
    is_max = S >= top1.gather(1, idx) - 1e-7
    top2 = torch.full((len(Q), n_concepts), -2.0).scatter_reduce(1, idx, S.masked_fill(is_max, -2.0), "amax")
    return torch.where(top2 > -1.5, (top1 + top2) / 2, top1).numpy()


def hub_vector(Q, Z, inst_c, n_concepts, k=10, bs=1024):
    """Hubness of every concept (CSLS, Conneau et al. 2018): r_c = mean of the k highest concept-c scores over a reference set of
    UNLABELED query windows Q (continuous video). A concept that is close to everything (a hub) gets a large r_c.
    Score correction: S'(q, c) = S(q, c) − α·r_c (the query term of CSLS is constant per query and does not change rankings)."""
    top = None
    for i in range(0, len(Q), bs):
        S = torch.as_tensor(concept_scores(Q[i:i + bs], Z, inst_c, n_concepts))
        S = torch.where(S > -1.5, S, torch.full_like(S, -1.0))
        top = S if top is None else torch.cat([top, S])
        top = torch.topk(top, min(k, len(top)), dim=0).values
    return top.mean(0).numpy().astype(np.float32)


def csls(S, hub, alpha):
    """Apply the hubness correction to a score matrix [Q, C] (missing concepts stay at -2)."""
    if hub is None or not alpha:
        return S
    return np.where(S > -1.5, S - alpha * hub[None, :], S)


def retrieval_metrics(S, target, allowed=None):
    """S [Q,C] scores, target [Q] concept idx (-1 = not in bank), allowed = optional column subset (closed set)."""
    S = S.copy()
    if allowed is not None:
        mask = np.full(S.shape[1], True); mask[allowed] = False
        S[:, mask] = -9
    ok = target >= 0
    ok &= S[np.arange(len(S)), np.maximum(target, 0)] > -1.5
    S, t = S[ok], target[ok]
    if len(t) == 0:
        return dict(n=0)
    rank = (S > S[np.arange(len(t)), t][:, None]).sum(1)
    n = len(t)
    r1 = float((rank < 1).mean())
    return dict(n=n, R1=r1, R1_ci95=float(1.96 * math.sqrt(r1 * (1 - r1) / n)), R5=float((rank < 5).mean()), R10=float((rank < 10).mean()),
                MRR=float((1 / (rank + 1)).mean()), median_rank=float(np.median(rank) + 1))


def load_encoder(models_dir, weights=None, device="cuda"):
    """Fine-tuned encoder (weights = models/encoder.pt: full state dict, needs only mt5_config.json next to it) or the zero-shot
    Uni-Sign encoder (weights None: artifacts/pretrained/unisign_wlasl_enc.pt)."""
    import json
    from pathlib import Path
    models_dir = Path(models_dir)
    cfg = json.loads((models_dir / "mt5_config.json").read_text())
    if weights is not None:
        sd = torch.load(weights, map_location="cpu")
        m = SignEncoder(cfg, prefix_len=sd["prefix"].shape[0])
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        return m.to(device).eval()
    base = torch.load(models_dir / "unisign_wlasl_enc.pt", map_location="cpu")
    m = SignEncoder(cfg, prefix_len=base["prefix"].shape[0])
    m.load_base(base)
    return m.to(device).eval()
