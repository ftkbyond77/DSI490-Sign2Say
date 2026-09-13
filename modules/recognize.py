"""v4 L3/L4: segment embeddings from the pose encoder, vocabulary bank (multi-instance, multi-signer),
open-set recognition with calibrated rejection ("there is a sign here — closest known word is ___")."""
from __future__ import annotations

import numpy as np
import torch

from .articulators import active_mask
from .unisign import collate_parts, prepare_parts, upsample
from .utils import get_logger

log = get_logger("recognize")


def load_pose(path):
    d = np.load(path)
    return d["kp"].astype(np.float32), d["sc"].astype(np.float32), tuple(int(x) for x in d["hw"]), float(d["fps"])


def body17_px(kp, sc, hw):
    H, W = hw
    return np.concatenate([kp[:, :17] * np.array([W, H], np.float32), sc[:, :17, None]], -1)


def active_span(kp, sc, hw, mode="clip", pad=1):
    act = active_mask(body17_px(kp, sc, hw), mode=mode)
    if act.sum() < 4:
        return 0, len(kp)
    i0, i1 = np.flatnonzero(act)[[0, -1]]
    return max(0, i0 - pad), min(len(kp), i1 + 1 + pad)


def segment_parts(kp, sc, t0, t1, up=2, scale_ref=None):
    k, s = upsample(kp[t0:t1], sc[t0:t1], up)
    return prepare_parts(k, s, scale_ref=scale_ref)


@torch.no_grad()
def embed(enc, parts_list, level="ctx", bs=64, pool="mean", max_len=256):
    """Mean-pool frame features over valid frames → L2-normalised [N,768]."""
    order = np.argsort([len(p["body"]) for p in parts_list])
    out = [None] * len(parts_list)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        parts, mask = collate_parts([parts_list[j] for j in idx], max_len)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enc.device == "cuda"):
            f = enc.frames(parts, mask, level)
        x = f[level].float()
        m = mask.to(x.device)[..., None].float()
        if pool == "mean":
            z = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:  # mean ‖ max
            z = torch.cat([(x * m).sum(1) / m.sum(1).clamp(min=1), (x + (m - 1) * 1e4).amax(1)], -1)
        z = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
        for j, k in enumerate(idx):
            out[k] = z[j]
    return np.stack(out)


# ----------------------------------------------------------------------------- vocabulary bank
class VocabBank:
    """Instance-level bank. score(word) = mean of the top-2 instance similarities of that word (robust to a single
    odd instance, rewards words seen from several signers)."""

    def __init__(self, Z: np.ndarray, words: list[str], meta: list[dict] | None = None):
        self.Z = Z.astype(np.float32)
        self.words = list(words)
        self.meta = meta or [{} for _ in words]
        self.vocab = sorted(set(words))
        self.widx = {w: i for i, w in enumerate(self.vocab)}
        self.inst_word = np.array([self.widx[w] for w in self.words])

    def word_scores(self, q: np.ndarray, topm=2):
        """[Q, W] word scores = mean of the top-`topm` instance similarities (fewer if the word has fewer instances)."""
        S = torch.as_tensor(q @ self.Z.T, dtype=torch.float32)  # [Q, N]
        W = len(self.vocab)
        idx = torch.as_tensor(self.inst_word, dtype=torch.long).expand_as(S)
        top1 = torch.full((len(q), W), -2.0).scatter_reduce(1, idx, S, "amax")
        if topm == 1:
            return top1.numpy()
        is_max = S >= top1.gather(1, idx) - 1e-7
        S2 = S.masked_fill(is_max, -2.0)
        top2 = torch.full((len(q), W), -2.0).scatter_reduce(1, idx, S2, "amax")
        both = torch.where(top2 > -1.5, (top1 + top2) / 2, top1)
        return both.numpy()

    def topk(self, q: np.ndarray, k=5, topm=2):
        s = self.word_scores(q, topm)
        idx = np.argsort(-s, axis=1)[:, :k]
        return [[(self.vocab[i], float(s[r, i])) for i in row] for r, row in enumerate(idx)]


class Adapter(torch.nn.Module):
    """Residual metric adapter on frozen Uni-Sign embeddings: z' = norm(z + g·MLP(z))."""

    def __init__(self, d=768, h=1024, drop=0.1):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(h, d))
        self.gate = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, z):
        return torch.nn.functional.normalize(z + self.gate * self.mlp(z), dim=-1)

    @torch.no_grad()
    def apply_np(self, Z, device="cuda"):
        self.eval()
        return self(torch.as_tensor(Z, dtype=torch.float32, device=device)).cpu().numpy()


def train_adapter(Z, words, signers, aug_Z=None, aug_owner=None, steps=3000, P=128, lr=1e-3, temp=0.07, eval_fn=None,
                  eval_every=250, device="cuda", seed=0):
    """SupCon with P words × 2 views; a view is either another instance of the word from a DIFFERENT signer (preferred)
    or an augmented view of the same clip."""
    rng = np.random.RandomState(seed)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=device)
    Za = torch.as_tensor(aug_Z, dtype=torch.float32, device=device) if aug_Z is not None else None
    by_word = {}
    for i, w in enumerate(words):
        by_word.setdefault(w, []).append(i)
    augs = {}
    if aug_owner is not None:
        for j, o in enumerate(aug_owner):
            augs.setdefault(int(o), []).append(j)
    wlist = list(by_word)
    multi = [w for w in wlist if len({signers[i] for i in by_word[w]}) >= 2]
    model = Adapter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    best, best_state, hist = -1, None, []
    for it in range(steps):
        n_multi = min(len(multi), P // 2)
        ws = list(rng.choice(multi, n_multi, replace=False)) + list(rng.choice(wlist, P - n_multi, replace=False))
        A, B = [], []
        for w in ws:
            idx = by_word[w]
            i = idx[rng.randint(len(idx))]
            others = [j for j in idx if signers[j] != signers[i]]
            if others and rng.rand() < 0.8:
                A.append(Zt[i]); B.append(Zt[others[rng.randint(len(others))]])
            elif Za is not None and augs.get(i):
                A.append(Zt[i]); B.append(Za[augs[i][rng.randint(len(augs[i]))]])
            else:
                j = idx[rng.randint(len(idx))]
                A.append(Zt[i]); B.append(Zt[j] + 0.02 * torch.randn_like(Zt[j]))
        x = torch.stack(A + B)
        x = x + 0.01 * torch.randn_like(x)
        model.train()
        z = model(x)
        y = torch.arange(P, device=device).repeat(2)
        sim = z @ z.t() / temp
        eye = torch.eye(2 * P, device=device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        pos = (y[:, None] == y[None, :]) & ~eye
        loss = -(sim.log_softmax(1) * pos).sum(1).div(pos.sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if eval_fn is not None and ((it + 1) % eval_every == 0 or it == 0):
            r = eval_fn(model)
            hist.append(dict(step=it, loss=float(loss), **r))
            log.info("adapter %d loss %.3f val MRR %.3f R1 %.3f R5 %.3f", it, float(loss), r["MRR"], r["R1"], r["R5"])
            if r["MRR"] > best:
                best, best_state = r["MRR"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def calibrate_rejection(top1, margin, correct, target_precision=0.7):
    """Pick (tau_score, tau_margin) maximising accepted recall s.t. precision ≥ target on labelled queries."""
    best = (1.0, 1.0, 0.0, 0.0)
    for ts in np.quantile(top1, np.linspace(0, 0.95, 40)):
        for tm in np.quantile(margin, np.linspace(0, 0.95, 20)):
            acc = (top1 >= ts) & (margin >= tm)
            if acc.sum() < 3:
                continue
            prec = correct[acc].mean()
            rec = (correct & acc).sum() / max(correct.sum(), 1)
            if prec >= target_precision and acc.mean() > best[3]:
                best = (float(ts), float(tm), float(prec), float(acc.mean()))
    return dict(tau_score=best[0], tau_margin=best[1], precision=best[2], accept_rate=best[3])
