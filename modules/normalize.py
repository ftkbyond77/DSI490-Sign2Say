"""L2.5 Signer-Nuisance Projection (NAP) — fixed projection fitted on TRAIN signers only.

P = I − NᵀN where N = top-d right singular vectors of (per-signer mean − global mean).
Applied identically in SSL, head training and inference: no per-session calibration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import ART


class NAP:
    def __init__(self, N: np.ndarray | None = None, mu: np.ndarray | None = None):
        self.N = N          # [K_streams, d, D]
        self.mu = mu        # [K_streams, D] (global mean, kept for diagnostics)

    @classmethod
    def fit(cls, feats_by_signer: dict[str, np.ndarray], d: int) -> "NAP":
        """feats_by_signer: signer → [n, K, D] active-frame features."""
        means = np.stack([f.astype(np.float64).mean(0) for f in feats_by_signer.values()])  # [S,K,D]
        mu = means.mean(0)
        K = means.shape[1]
        Ns = []
        for k in range(K):
            M = means[:, k] - mu[k]
            _, _, vt = np.linalg.svd(M, full_matrices=False)
            Ns.append(vt[:d])
        return cls(np.stack(Ns).astype(np.float32), mu.astype(np.float32))

    def apply_np(self, x: np.ndarray) -> np.ndarray:
        """x [..., K, D]"""
        if self.N is None or self.N.shape[1] == 0:
            return x
        xf = x.astype(np.float32)
        proj = np.einsum("...kd,kjd->...kj", xf, self.N)
        return xf - np.einsum("...kj,kjd->...kd", proj, self.N)

    def apply_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.N is None or self.N.shape[1] == 0:
            return x
        N = torch.as_tensor(self.N, device=x.device, dtype=x.dtype)
        proj = torch.einsum("...kd,kjd->...kj", x, N)
        return x - torch.einsum("...kj,kjd->...kd", proj, N)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, N=self.N if self.N is not None else np.zeros((3, 0, 768), np.float32), mu=self.mu)

    @classmethod
    def load(cls, path: Path | None = None) -> "NAP":
        path = path or ART / "nap" / "nap.npz"
        z = np.load(path)
        return cls(z["N"], z["mu"])
