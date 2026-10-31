"""Baseline sign recogniser — the original Uni-Sign pose encoder + nearest-neighbour retrieval against a Thai clip bank.

  from baseline import Baseline
  b = Baseline("zeroshot")            # frozen Uni-Sign (WLASL pose-only checkpoint) — "the baseline as published"
  b = Baseline("adapted")             # + light residual adapter trained on our train signers — "the baseline fitted to our data"
  b.recognise_landmarks(raw)          # MediaPipe landmarks of a video → [{t0, t1, top5: [(word, score)], ...}]

Uni-Sign (Li et al., ICLR 2025) has no Thai output: its checkpoint is an mT5 *text* generator for ASL/Chinese. What it does give is a
pretrained sign encoder, so the baseline = that encoder, untouched, as a feature extractor + a bank of embedded Thai clips.
No tagger, no spotting model, no calibration, no language model: those are what v8–v10 add on top.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from modules.encoder import SignEncoder, collate, concept_scores
from modules.parts import prep, to_iso
from modules.utils import device as resolve_device

FPS = 25
SEG_MIN = 6            # frames: shorter active runs are noise
BRIDGE = 8             # frames: a hand drop-out / pause shorter than this does not split a run


def load_encoder(path=HERE, adapter=True, device=None):
    """Frozen Uni-Sign encoder (+ the trained residual adapter when adapter=True)."""
    path = Path(path)
    cfg = json.loads((path / "mt5_config.json").read_text())
    base = torch.load(path / "unisign_wlasl_enc.pt", map_location="cpu")
    m = SignEncoder(cfg, prefix_len=base["prefix"].shape[0])
    m.load_base(base)
    if adapter:
        m.head.load_state_dict(torch.load(path / "adapter.pt", map_location="cpu"))
    return m.to(resolve_device(device)).eval()


@torch.no_grad()
def pooled(enc, feats):
    """List of prepared part dicts → [B,768] mean-pooled mT5 states (before the adapter), float32 numpy."""
    parts, mask = collate(feats, device=next(enc.parameters()).device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=next(enc.parameters()).is_cuda):
        ctx = enc.frames(parts, mask)
    m = mask[..., None].to(ctx.dtype)
    return ((ctx * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()


class Baseline:
    def __init__(self, variant="zeroshot", device=None, vocab="full"):
        assert variant in ("zeroshot", "adapted")
        self.variant, self.vocab = variant, vocab
        self.enc = load_encoder(HERE, adapter=variant == "adapted", device=device)
        z = np.load(HERE / "bank.npz", allow_pickle=True)
        self.concepts = [str(c) for c in z["concepts"]]
        self.Z = z["Z1" if variant == "adapted" else "Z0"].astype(np.float32)
        self.inst_c = z["concept"].astype(np.int64)
        self.allowed = np.ones(len(self.concepts), bool)
        if vocab == "daily":
            self.allowed[:] = False
            self.allowed[[self.concepts.index(c) for c in z["daily"] if str(c) in self.concepts]] = True
        self.span_len = int(json.loads((HERE / "config.json").read_text())["median_sign_frames"])

    # ----------------------------------------------------------------- embedding / retrieval
    @torch.no_grad()
    def embed(self, feats):
        out = []
        for i in range(0, len(feats), 32):
            z = torch.tensor(pooled(self.enc, feats[i:i + 32]))
            if self.variant == "adapted":
                z = z + self.enc.head(z.to(next(self.enc.parameters()).device).float()).cpu()
            out.append(F.normalize(z, dim=-1).numpy())
        return np.concatenate(out)

    def scores(self, E):
        S = concept_scores(E, self.Z, self.inst_c, len(self.concepts))
        S[:, ~self.allowed] = -9
        return S

    def top(self, S, k=5):
        idx = np.argsort(-S, axis=1)[:, :k]
        return [[(self.concepts[j], float(S[i, j])) for j in row] for i, row in enumerate(idx)]

    # ----------------------------------------------------------------- continuous video (the deliberately simple part)
    def segments(self, kp, sc, hw):
        """Hands-raised runs (same activity rule as the data prep), each cut into round(length / median sign length) equal pieces."""
        from modules.schema import activity
        a = activity(kp, sc, hw)
        h = a["height"]
        base = np.percentile(h[h > -2.5], 10) if (h > -2.5).any() else -2.0
        act = (h > max(base + 0.45, -1.6)) | ((a["speed"] > 1.2) & (h > base + 0.2))
        idx = np.flatnonzero(act)
        runs = []
        for i in idx:
            if runs and i - runs[-1][1] <= BRIDGE:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        out = []
        for f0, f1 in runs:
            f1 += 1
            if f1 - f0 < SEG_MIN:
                continue
            n = max(1, int(round((f1 - f0) / self.span_len)))
            cuts = np.linspace(f0, f1, n + 1).round().astype(int)
            out += [(int(x), int(y)) for x, y in zip(cuts[:-1], cuts[1:])]
        return out

    def recognise(self, kp, sc, hw, k=5):
        segs = self.segments(kp, sc, hw)
        if not segs:
            return []
        feats = [prep(to_iso(kp[a:b], hw), sc[a:b]) for a, b in segs]
        S = self.scores(self.embed(feats))
        return [dict(f0=a, f1=b, t0=a / FPS, t1=b / FPS, top=t) for (a, b), t in zip(segs, self.top(S, k))]

    def recognise_landmarks(self, raw, k=5):
        """raw = modules.mediapipe_pose.extract(...) / load_raw(...) → segments (same landmark standardisation as v10, nothing else)."""
        from modules.mediapipe_pose import to_standard
        from modules.schema import SH_L, SH_R
        d = to_standard(raw)
        kp, sc, hw = np.asarray(d["kp"], np.float32), np.asarray(d["sc"], np.float32), tuple(int(x) for x in d["hw"])
        present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
        sc = sc.copy(); sc[~present] = 0.0; kp = kp.copy(); kp[~present] = 0.0
        return self.recognise(kp, sc, hw, k)

    def recognise_video(self, video, k=5):
        from modules.mediapipe_pose import extract
        return self.recognise_landmarks(extract(video), k)


# ----------------------------------------------------------------------------- scoring (same definitions as scripts/eval.py)
def canon_fn():
    from modules.lexicon import TEST_SURFACE
    cm = json.loads((ROOT / "cloud_s3" / "data_prep" / "manifest" / "concepts.json").read_text(encoding="utf-8"))["concept"]
    return lambda w: cm.get(TEST_SURFACE.get(w, w), TEST_SURFACE.get(w, w))


def score_video(segs, ref, canon, bank_vocab):
    """Recall@1/@5 = reference word is the top-1 / in the top-5 of SOME segment; WER on the top-1 sequence; count error."""
    from modules.decode import wer
    refc = [canon(w) for w in ref]
    near = [s["top"][0][0] for s in segs]
    top5 = [[w for w, _ in s["top"]] for s in segs]
    inv = [w for w in refc if w in bank_vocab]
    return dict(ref=refc, hyp=near, n_ref=len(refc), n_seg=len(segs), n_in_vocab=len(inv),
                r1=float(np.mean([w in near for w in refc])), r5=float(np.mean([any(w in t for t in top5) for w in refc])),
                r1_in_vocab=float(np.mean([w in near for w in inv])) if inv else None, wer=wer(refc, near), count_err=abs(len(segs) - len(refc)))
