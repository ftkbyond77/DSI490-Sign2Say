"""L4 heads: ISLR prototype head (metric space, one-shot enrollment, unknown rejection),
face affect head, NMM (non-manual marker) rules."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssl import STREAMS, SignEncoder
from .utils import get_logger

log = get_logger("heads")


# ----------------------------------------------------------------------------- batching helpers
def collate(items, max_len=64):
    """items: list of (feats [T,3,768], pose [T,44], valid [T,3]) → padded tensors."""
    L = min(max_len, max(len(f) for f, _, _ in items))
    B = len(items)
    F_ = np.zeros((B, L, 3, 768), np.float32); P = np.zeros((B, L, 44), np.float32)
    V = np.zeros((B, L, 3), bool); M = np.zeros((B, L), bool)
    for b, (f, p, v) in enumerate(items):
        if len(f) > L:  # uniform temporal resample to L
            idx = np.round(np.linspace(0, len(f) - 1, L)).astype(int)
            f, p, v = f[idx], p[idx], v[idx]
        n = len(f)
        F_[b, :n], P[b, :n], V[b, :n], M[b, :n] = f, p, v.astype(bool), True
    return [torch.from_numpy(x) for x in (F_, P, V, M)]


def pooled(out: dict, pad: torch.Tensor, streams=STREAMS, mode="cls"):
    parts = []
    for s in streams:
        cls, tok = out[s]
        if mode in ("cls", "cls+mean"):
            parts.append(cls.float())
        if mode in ("mean", "cls+mean"):
            m = pad[..., None].float()
            parts.append((tok.float() * m).sum(1) / m.sum(1).clamp(min=1))
    return torch.cat(parts, -1)


# ----------------------------------------------------------------------------- ISLR
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class ISLRHead(nn.Module):
    def __init__(self, in_dim, emb=512, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, emb))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ISLRModel(nn.Module):
    def __init__(self, encoder: SignEncoder, n_classes: int, n_signers: int, emb=512, pool="cls",
                 streams=STREAMS, arc_m=0.2, arc_s=30.0):
        super().__init__()
        self.encoder, self.pool, self.streams = encoder, pool, tuple(streams)
        dims = encoder.out_dims()
        mult = 2 if pool == "cls+mean" else 1
        self.head = ISLRHead(sum(dims[s] for s in self.streams) * mult, emb)
        self.arc_W = nn.Parameter(torch.randn(n_classes, emb) * 0.01)
        self.adv = nn.Linear(emb, max(n_signers, 1))
        self.arc_m, self.arc_s = arc_m, arc_s

    def embed(self, feats, pose, valid, pad):
        out = self.encoder(feats, pose, valid, pad, streams=self.streams)
        return self.head(pooled(out, pad, self.streams, self.pool))

    def arcface(self, z, y):
        W = F.normalize(self.arc_W, dim=-1)
        cos = (z @ W.t()).clamp(-1 + 1e-6, 1 - 1e-6)
        th = torch.acos(cos.gather(1, y[:, None]))
        target = torch.cos(th + self.arc_m)
        logits = cos.scatter(1, y[:, None], target)
        return F.cross_entropy(logits * self.arc_s, y)


def supcon(z, y, temp=0.07):
    sim = z @ z.t() / temp
    B = len(z)
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


class ISLRSampler:
    """P classes × 2 views. A view = random version (base/aug*/flip) · random temporal crop · speed."""

    def __init__(self, store, unit_clips: dict, unit_index: dict, clip_signer: dict, signer_index: dict,
                 versions=("base", "aug1", "aug2", "flip"), max_len=48, seed=0, cross_clip_p=0.5, speed=(0.8, 1.25)):
        self.store, self.unit_clips, self.uidx = store, unit_clips, unit_index
        self.units = sorted(unit_clips)
        self.clip_signer, self.sidx = clip_signer, signer_index
        self.versions = [v for v in versions if v in store.feat]
        self.max_len, self.rng, self.cross_p, self.speed = max_len, np.random.RandomState(seed), cross_clip_p, speed
        self.multi = [u for u in self.units if len(unit_clips[u]) >= 2]

    def view(self, cid):
        ver = self.versions[self.rng.randint(len(self.versions))]
        f, p, v, _ = self.store.clip(cid, ver, active_only=True)
        n = len(f)
        keep = self.rng.uniform(0.8, 1.0)
        span = max(4, int(n * keep))
        o = self.rng.randint(0, n - span + 1)
        L = int(np.clip(round(span / self.rng.uniform(*self.speed)), 4, self.max_len))
        idx = o + np.round(np.linspace(0, span - 1, L)).astype(int)
        v = v[idx].copy()
        if self.rng.rand() < 0.1:
            v[:, self.rng.randint(3)] = 0
        return f[idx], p[idx], v

    def batch(self, P):
        n_multi = min(len(self.multi), P // 4)
        units = list(self.rng.choice(self.multi, n_multi, replace=False)) if n_multi else []
        rest = self.rng.choice(self.units, P - len(units), replace=False)
        units += [u for u in rest]
        items, y, s = [], [], []
        for u in units:
            cl = self.unit_clips[u]
            if len(cl) >= 2 and self.rng.rand() < self.cross_p:
                a, b = self.rng.choice(cl, 2, replace=False)
            else:
                a = b = cl[self.rng.randint(len(cl))]
            for c in (a, b):
                items.append(self.view(c)); y.append(self.uidx[u]); s.append(self.sidx.get(self.clip_signer.get(c), -1))
        return collate(items, self.max_len), torch.tensor(y), torch.tensor(s)


def train_islr(model: ISLRModel, sampler: ISLRSampler, steps=3000, P=96, lr=3e-4, enc_lr=5e-5, wd=0.05,
               w_sup=1.0, w_arc=1.0, w_adv=0.1, eval_fn=None, eval_every=500, device="cuda", center_train=False):
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": other, "lr": lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    hist, best, best_state = [], -1.0, None
    t0 = time.time()
    for it in range(steps):
        model.train()
        (f, p, v, m), y, s = sampler.batch(P)
        f, p, v, m, y, s = (t.to(device, non_blocking=True) for t in (f, p, v, m, y, s))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.embed(f, p, v, m).float()
        if center_train:  # remove each signer's batch mean: learn the within-signer residual (matches session centering)
            zc = z.clone()
            for sid in s.unique():
                g = s == sid
                if g.sum() >= 4:
                    zc[g] = z[g] - z[g].mean(0, keepdim=True).detach()
            z = F.normalize(zc, dim=-1)
        loss_s = supcon(z, y)
        loss_a = model.arcface(z, y)
        ok = s >= 0
        loss_adv = F.cross_entropy(model.adv(GradReverse.apply(z[ok], 1.0)), s[ok]) if ok.any() else z.sum() * 0
        loss = w_sup * loss_s + w_arc * loss_a + w_adv * loss_adv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_ for g in groups for p_ in g["params"]], 3.0)
        opt.step(); sched.step()
        if it % 100 == 0:
            rec = dict(step=it, loss=float(loss), supcon=float(loss_s), arc=float(loss_a), adv=float(loss_adv), sec=time.time() - t0)
            hist.append(rec)
            log.info("islr %d loss %.3f sup %.3f arc %.3f adv %.3f (%.0fs)", it, rec["loss"], rec["supcon"], rec["arc"], rec["adv"], rec["sec"])
        if eval_fn is not None and ((it + 1) % eval_every == 0 or it == steps - 1):
            r = eval_fn(model)
            hist.append(dict(step=it, eval=r))
            log.info("islr %d eval %s", it, {k: round(v_, 3) for k, v_ in r.items() if isinstance(v_, float)})
            key = r.get("MRR", 0) + r.get("C_MRR", 0) + 0.05 * r.get("E2_R@5_small", 0)  # MRR: less noisy than R@k on ~58 queries
            if key > best:
                best = key
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# ----------------------------------------------------------------------------- prototypes
class PrototypeBank:
    def __init__(self, keys: list[str], protos: np.ndarray, lemma: dict | None = None, signer: dict | None = None):
        self.keys, self.P = list(keys), protos.astype(np.float32)
        self.lemma = lemma or {}
        self.signer = signer or {}

    def centered(self) -> np.ndarray:
        """Session centering on the bank side: each prototype minus the mean prototype of its (majority) signer."""
        sig = np.array([self.signer.get(k, "unk") for k in self.keys])
        P = self.P.copy()
        for s in set(sig):
            if (sig == s).sum() >= 3:
                P[sig == s] -= self.P[sig == s].mean(0)
        return (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-6)).astype(np.float32)

    def enroll(self, key: str, z: np.ndarray, lemma: str | None = None):
        z = z / (np.linalg.norm(z) + 1e-9)
        if key in self.keys:
            i = self.keys.index(key)
            p = self.P[i] + z
            self.P[i] = p / np.linalg.norm(p)
        else:
            self.keys.append(key)
            self.P = np.vstack([self.P, z[None]])
        if lemma:
            self.lemma[key] = lemma

    def topk(self, z: np.ndarray, k=5):
        sims = z @ self.P.T
        idx = np.argsort(-sims, axis=-1)[..., :k]
        return idx, np.take_along_axis(sims, idx, -1)

    def save(self, path):
        np.savez(path, keys=np.array(self.keys), P=self.P, lemma_keys=np.array(list(self.lemma)),
                 lemma_vals=np.array(list(self.lemma.values())), signer_keys=np.array(list(self.signer)),
                 signer_vals=np.array(list(self.signer.values())))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        signer = dict(zip(z["signer_keys"].tolist(), z["signer_vals"].tolist())) if "signer_keys" in z else {}
        return cls(z["keys"].tolist(), z["P"], dict(zip(z["lemma_keys"].tolist(), z["lemma_vals"].tolist())), signer)


# ----------------------------------------------------------------------------- face affect
AFFECT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class AffectTeacher:
    """Open FER teacher (trpakov/vit-face-expression, ViT-B/16, 7 classes) on the centre of the face crop."""

    def __init__(self, name="trpakov/vit-face-expression", device="cuda"):
        from transformers import AutoModelForImageClassification
        self.m = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.m = self.m.to(self.dtype)
        self.device = device

    @torch.no_grad()
    def __call__(self, crops: np.ndarray, bs=128) -> np.ndarray:
        """crops [N,S,S,3] BGR uint8 face-stream crops → probs [N,7]."""
        out = []
        S = crops.shape[1]
        a, b = int(S * 0.18), int(S * 0.82)  # face crop includes neck/shoulders; FER wants the face
        for i in range(0, len(crops), bs):
            x = torch.from_numpy(crops[i:i + bs, a:b, a:b, ::-1].copy()).to(self.device).permute(0, 3, 1, 2).to(self.dtype) / 255
            x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
            x = (x - 0.5) / 0.5
            out.append(self.m(pixel_values=x).logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 7), np.float32)


class AffectHead(nn.Module):
    """Distilled student: temporal conv over frozen face-stream features (768-d) → 7-class affect per frame."""

    def __init__(self, in_dim=768, d=192, n=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU())
        self.tconv = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        self.out = nn.Linear(d, n)

    def forward(self, x):  # [B,T,768] → logits [B,T,7]
        h = self.net(x)
        h = self.tconv(h.transpose(1, 2)).transpose(1, 2) + h
        return self.out(h)

    @torch.no_grad()
    def predict(self, face_feats: np.ndarray, valid: np.ndarray | None = None, ema_s=0.8, fps=12.5) -> dict:
        dev = next(self.parameters()).device
        if len(face_feats) == 0:
            return dict(emotion="neutral", intensity=0.0, probs={})
        p = self(torch.as_tensor(face_feats, dtype=torch.float32, device=dev)[None])[0].softmax(-1).cpu().numpy()
        if valid is not None and valid.sum():
            p = p[valid.astype(bool)]
        a = 1 - np.exp(-1 / (ema_s * fps))
        sm = p[0].copy()
        for q in p[1:]:
            sm = (1 - a) * sm + a * q
        mean = p.mean(0)
        k = int(mean.argmax())
        emo = AFFECT_CLASSES[k]
        intensity = float(np.clip((mean[k] - mean[AFFECT_CLASSES.index("neutral")]) if emo != "neutral" else 0.0, 0, 1))
        return dict(emotion="surprised" if emo == "surprise" else emo, intensity=round(intensity, 3),
                    probs={c: round(float(v), 3) for c, v in zip(AFFECT_CLASSES, mean)})


# ----------------------------------------------------------------------------- face / NMM
def nmm_flags(kpts: np.ndarray, fps=12.5, min_conf=0.3, shake_min_cycles=2, window_s=1.0) -> dict:
    """Head shake (negation) / nod (affirmation) from nose oscillation relative to the shoulder midpoint."""
    from .articulators import LSHO, NOSE, RSHO
    ok = (kpts[:, NOSE, 2] >= min_conf) & (kpts[:, LSHO, 2] >= min_conf) & (kpts[:, RSHO, 2] >= min_conf)
    if ok.sum() < int(fps):
        return dict(negation=False, affirmation=False, shake_cycles=0, nod_cycles=0)
    sw = np.linalg.norm(kpts[:, LSHO, :2] - kpts[:, RSHO, :2], axis=1)
    mid = (kpts[:, LSHO, :2] + kpts[:, RSHO, :2]) / 2
    rel = (kpts[:, NOSE, :2] - mid) / np.maximum(sw, 1)[:, None]
    rel = rel[ok]

    def cycles(sig, amp):
        sig = sig - np.convolve(sig, np.ones(7) / 7, "same")  # remove slow drift
        best = 0
        W = int(window_s * fps * 1.5)
        for s in range(0, max(1, len(sig) - W + 1)):
            w = sig[s:s + W]
            big = np.abs(w) > amp
            sgn = np.sign(w[big])
            best = max(best, int((np.diff(sgn) != 0).sum() // 2) if len(sgn) > 1 else 0)
        return best

    sc, nc = cycles(rel[:, 0], 0.03), cycles(rel[:, 1], 0.025)
    return dict(negation=sc >= shake_min_cycles, affirmation=nc >= shake_min_cycles and sc < shake_min_cycles,
                shake_cycles=sc, nod_cycles=nc)
