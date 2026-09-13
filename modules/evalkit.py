"""Metrics & benchmark helpers: cross-signer retrieval (R@k, MRR), gloss WER, chrF, latency/RAM probes."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .heads import collate
from .utils import get_logger

log = get_logger("evalkit")


# ----------------------------------------------------------------------------- embedding
@torch.no_grad()
def embed_clips(embed_fn, store, clip_ids, version="base", active_only=True, bs=128, max_len=64, device="cuda"):
    order = sorted(range(len(clip_ids)), key=lambda i: store.row[clip_ids[i]][1])
    out = [None] * len(clip_ids)
    for s in range(0, len(order), bs):
        ids = order[s:s + bs]
        items = [store.clip(clip_ids[i], version, active_only)[:3] for i in ids]
        f, p, v, m = (t.to(device) for t in collate(items, max_len))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = embed_fn(f, p, v, m).float()
        z = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
        for j, i in enumerate(ids):
            out[i] = z[j]
    return np.stack(out)


def frozen_embed_fn(nap_N=None, streams=(0, 1, 2), use_pose=False, part="both"):
    """Baseline: masked mean-pool of frozen ViT features (optionally NAP-projected), per-stream L2, concat."""
    N = torch.as_tensor(nap_N).cuda() if nap_N is not None and np.asarray(nap_N).shape[1] > 0 else None

    def fn(f, p, v, m):
        f = f.float()
        if part == "cls":
            f = torch.cat([f[..., :384], torch.zeros_like(f[..., 384:])], -1)
        if N is not None:
            proj = torch.einsum("btkd,kjd->btkj", f, N)
            f = f - torch.einsum("btkj,kjd->btkd", proj, N)
        w = (m[..., None] & v).float()[..., None]
        pooled = (f * w).sum(1) / w.sum(1).clamp(min=1)
        parts = [torch.nn.functional.normalize(pooled[:, k], dim=-1) for k in streams]
        if use_pose:
            mp = m[..., None].float()
            parts.append(torch.nn.functional.normalize((p * mp).sum(1) / mp.sum(1).clamp(min=1), dim=-1))
        return torch.cat(parts, -1)
    return fn


# ----------------------------------------------------------------------------- retrieval
def knn_mean_sim(A: np.ndarray, B: np.ndarray, k: int, self_mask: bool = False) -> np.ndarray:
    """Mean similarity of each row of A to its k nearest rows of B (hubness score used by CSLS)."""
    S = A @ B.T
    if self_mask:
        np.fill_diagonal(S, -np.inf)
    k = min(k, S.shape[1] - 1)
    return np.partition(S, -k, axis=1)[:, -k:].mean(1)


def retrieval_eval(emb: dict, eval_units: list, unit_clips: dict, clip_signer: dict,
                   distractor_keys: list | None = None, distractor_P: np.ndarray | None = None, ks=(1, 5, 10),
                   csls_k: int = 0):
    """Cross-signer one-shot retrieval. For each query clip q of unit u, the gallery holds, for every eval unit,
    the mean embedding of its clips performed by signers ≠ signer(q) (enrollment), plus distractor prototypes.
    csls_k > 0: bank-side hubness correction s'(q,p) = s(q,p) − r_p, r_p = mean sim of p to its k nearest prototypes
    (label-free, computed from the bank only → usable at deployment)."""
    hits = {k: 0 for k in ks}
    hits_small = {k: 0 for k in ks}
    rr, n = 0.0, 0
    rP = None
    if csls_k and distractor_P is not None and len(distractor_P):
        rP = knn_mean_sim(distractor_P, distractor_P, csls_k, self_mask=True)
    for u in eval_units:
        for q in unit_clips[u]:
            sq = clip_signer.get(q)
            keys, G = [], []
            for u2 in eval_units:
                cl = [c for c in unit_clips[u2] if clip_signer.get(c) != sq and c != q]
                if not cl:
                    continue
                g = np.mean([emb[c] for c in cl], 0)
                keys.append(u2); G.append(g / (np.linalg.norm(g) + 1e-9))
            if u not in keys:
                continue
            G = np.stack(G)
            zq = emb[q]
            s_small = G @ zq
            tgt = keys.index(u)
            rank_small = int((s_small > s_small[tgt]).sum())
            s_all = s_small
            if distractor_P is not None and len(distractor_P):
                s_all = np.concatenate([s_small, distractor_P @ zq])
                if rP is not None:
                    rG = knn_mean_sim(G, np.concatenate([G, distractor_P]), csls_k + 1)
                    s_all = s_all - np.concatenate([rG, rP])
            rank = int((s_all > s_all[tgt]).sum())
            for k in ks:
                hits[k] += rank < k
                hits_small[k] += rank_small < k
            rr += 1.0 / (rank + 1)
            n += 1
    res = {f"R@{k}": hits[k] / max(n, 1) for k in ks}
    res.update({f"R@{k}_small": hits_small[k] / max(n, 1) for k in ks})
    for k in (1, 5):  # 95% normal-approximation half-width: small query sets are noisy
        for suf in ("", "_small"):
            p = res[f"R@{k}{suf}"]
            res[f"R@{k}{suf}_ci95"] = 1.96 * float(np.sqrt(p * (1 - p) / max(n, 1)))
    res["MRR"] = rr / max(n, 1)
    res["n_queries"] = n
    res["gallery_size"] = len(eval_units) + (0 if distractor_P is None else len(distractor_P))
    return res


def unit_prototypes(emb: dict, unit_clips: dict, units: list):
    P = []
    for u in units:
        g = np.mean([emb[c] for c in unit_clips[u]], 0)
        P.append(g / (np.linalg.norm(g) + 1e-9))
    return np.stack(P) if P else np.zeros((0, 1), np.float32)


# ----------------------------------------------------------------------------- sequence metrics
def gloss_wer(ref: list[str], hyp: list[str]) -> float:
    import jiwer
    if not ref:
        return float(len(hyp) > 0)
    r = " ".join(t.replace(" ", "_") for t in ref)
    h = " ".join(t.replace(" ", "_") for t in hyp) if hyp else ""
    return jiwer.wer(r, h) if h else 1.0


def chrf(ref: str, hyp: str) -> float:
    import sacrebleu
    return sacrebleu.sentence_chrf(hyp, [ref]).score


# ----------------------------------------------------------------------------- runtime
def rss_mb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")
