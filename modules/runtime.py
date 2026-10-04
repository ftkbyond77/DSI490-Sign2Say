"""Inference back-ends with one NumPy interface — ONNX Runtime (production / web API, no PyTorch) or PyTorch (development, GPU).

    be = OnnxBackend(models_dir, device="cpu")      # models/onnx/encoder.onnx + tagger.onnx
    be = TorchBackend(models_dir, device="cuda")    # models/encoder.pt + tagger.pt
    Z    = be.embed(parts_list)                      # [N,768] L2-normalised sign embeddings of N windows / clips
    H    = be.frames(kp_iso, sc)                     # [T,768] contextual frame states (tagger input)
    prob = be.tag(x)                                 # [T,3]   non-sign / sign-begin / sign-inside

Vocabulary scoring (`concept_scores`, `window_table`) is NumPy and identical for both back-ends.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .parts import MODES, collate_np, prepare_parts


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _frames_chunked(run_frames, kp_iso, sc, chunk=400, overlap=50):
    """Contextual frame states for a long sequence: chunks of `chunk` frames with `overlap`, blended with triangular weights."""
    T = len(kp_iso)
    out = np.zeros((T, 768), np.float32); wsum = np.zeros(T, np.float32)
    step = chunk - overlap
    for a in range(0, max(1, T - overlap), step):
        b = min(T, a + chunk)
        h = run_frames(prepare_parts(kp_iso[a:b], sc[a:b]))
        w = np.minimum(np.arange(1, b - a + 1), np.arange(b - a, 0, -1)).clip(max=25).astype(np.float32)
        out[a:b] += h * w[:, None]; wsum[a:b] += w
        if b == T:
            break
    return out / np.maximum(wsum, 1e-6)[:, None]


class OnnxBackend:
    name = "onnx"

    def __init__(self, model_dir, device="cpu", threads=None):
        import onnxruntime as ort
        md = Path(model_dir) / "onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads or os.environ.get("ORT_THREADS"):
            so.intra_op_num_threads = int(threads or os.environ["ORT_THREADS"])
        prov = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            prov = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
        # CPU: the int8 encoder when the export's parity check kept it (models/onnx/parity.json); THAISLM_INT8=0 forces fp32
        int8 = md / "encoder.int8.onnx"
        use_int8 = prov[0] == "CPUExecutionProvider" and int8.exists() and os.environ.get("THAISLM_INT8", "1") != "0"
        self.enc = ort.InferenceSession(str(int8 if use_int8 else md / "encoder.onnx"), so, providers=prov)
        self.precision = "int8" if use_int8 else "fp32"
        self.tagger = ort.InferenceSession(str(md / "tagger.onnx"), so, providers=prov)
        self.device = "cuda" if prov[0] != "CPUExecutionProvider" else "cpu"

    def _run(self, parts_list):
        P, M = collate_np(parts_list)
        feeds = {k: P[k] for k in MODES}
        feeds["mask"] = M
        return self.enc.run(None, feeds)

    def embed(self, parts_list, bs=64):
        if not parts_list:
            return np.zeros((0, 768), np.float32)
        order = np.argsort([len(p["body"]) for p in parts_list])
        out = np.zeros((len(parts_list), 768), np.float32)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            out[idx] = self._run([parts_list[j] for j in idx])[0]
        return out

    def frames(self, kp_iso, sc):
        return _frames_chunked(lambda p: self._run([p])[1][0], kp_iso, sc)

    def tag(self, x):
        return _softmax(self.tagger.run(None, {"x": x[None].astype(np.float32)})[0][0])


class TorchBackend:
    name = "torch"

    def __init__(self, model_dir, device="cuda", encoder=None, tagger=None, tagger_arch="gru"):
        import torch
        from .encoder import load_encoder
        from .sequence import Tagger
        md = Path(model_dir)
        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model = encoder if encoder is not None else load_encoder(md, md / "encoder.pt", device=self.device)
        if tagger is None:
            tagger = Tagger(arch=tagger_arch).to(self.device)
            tagger.load_state_dict(torch.load(md / "tagger.pt", map_location=self.device))
        self.tg = tagger.eval()

    def embed(self, parts_list, bs=256):
        from .encoder import embed_parts
        return embed_parts(self.model, parts_list, bs=bs, device=self.device)

    def frames(self, kp_iso, sc):
        from .encoder import frame_features
        return frame_features(self.model, kp_iso, sc, device=self.device)

    def tag(self, x):
        with self.torch.no_grad():
            return self.tg(self.torch.from_numpy(x.astype(np.float32))[None].to(self.device)).softmax(-1)[0].float().cpu().numpy()


def make_backend(model_dir, backend="onnx", device="cpu", **kw):
    return OnnxBackend(model_dir, device, **kw) if backend == "onnx" else TorchBackend(model_dir, device, **kw)


# ----------------------------------------------------------------------------- vocabulary scoring (NumPy)
class Bank:
    """Vocabulary bank: one embedding per labelled clip, grouped by concept so per-concept top-2 means are two `reduceat`s."""

    def __init__(self, Z, c, n_concepts, allowed=None):
        Z, c = np.asarray(Z, np.float32), np.asarray(c, np.int64)
        if allowed is not None:
            keep = np.isin(c, np.asarray(sorted(allowed)))
            Z, c = Z[keep], c[keep]
        o = np.argsort(c, kind="stable")
        self.Z, self.c = Z[o], c[o]
        self.n_concepts = n_concepts
        self.concepts_present, self.starts = np.unique(self.c, return_index=True)
        self.seg = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(self.c)]))

    def scores(self, Q, topm=2):
        """[N, C] = mean of the top-`topm` instance similarities per concept (v4 rule); -2 where a concept has no instance."""
        S = np.asarray(Q, np.float32) @ self.Z.T
        top1 = np.maximum.reduceat(S, self.starts, axis=1)
        if topm >= 2:
            is_max = S >= top1[:, self.seg] - 1e-7
            S2 = np.where(is_max, -2.0, S)
            top2 = np.maximum.reduceat(S2, self.starts, axis=1)
            top = np.where(top2 > -1.5, (top1 + top2) / 2, top1)
        else:
            top = top1
        out = np.full((len(S), self.n_concepts), -2.0, np.float32)
        out[:, self.concepts_present] = top
        return out


def table_from_Z(Z, wins, bank: Bank, null_proto, topk=50):
    """Window embeddings → the decoder's table: top-k concepts, z-statistics over the vocabulary, similarity to the null prototype."""
    if not len(wins):
        return dict(wins=[], top_idx=np.zeros((0, topk), np.int32), top_sc=np.zeros((0, topk), np.float32), mu=np.zeros(0), sd=np.zeros(0),
                    null=np.zeros(0), Z=np.zeros((0, 768), np.float32))
    S = bank.scores(Z)
    valid = S > -1.5
    k = min(topk, int(valid.sum(1).min()))
    o = np.argpartition(-S, k - 1, axis=1)[:, :k]
    o = np.take_along_axis(o, np.argsort(-np.take_along_axis(S, o, 1), axis=1), 1)
    Sv = np.where(valid, S, np.nan)
    mu = np.nanmean(Sv, 1); sd = np.nanstd(Sv, 1) + 1e-6
    return dict(wins=list(wins), top_idx=o.astype(np.int32), top_sc=np.take_along_axis(S, o, 1).astype(np.float32), mu=mu.astype(np.float32),
                sd=sd.astype(np.float32), null=(Z @ null_proto).astype(np.float32), Z=Z)


def window_table(backend, kp, sc, wins, bank: Bank, null_proto, topk=50, prep_fn=None):
    """Exact (teacher) table: every window embedded on its own by the encoder (slow on CPU; training / evaluation)."""
    from .parts import prep
    prep_fn = prep_fn or prep
    Z = backend.embed([prep_fn(kp[a:b], sc[a:b]) for a, b in wins]) if len(wins) else np.zeros((0, 768), np.float32)
    return table_from_Z(Z, wins, bank, null_proto, topk)


def rescore_windows(segs):
    """Windows whose exact embeddings re-score a segment: the decoder's picked window(s) + the whole segment span."""
    out = []
    for s in segs:
        ws = list(dict.fromkeys([tuple(w) for w in s.get("wins", [])] + [(s["f0"], s["f1"])]))
        out.append(ws)
    return out


def rescore(segs, Zs, bank: Bank, null_proto, concepts, prior=None, gamma=0.0, topn=5):
    """Exact re-scoring of the chosen segments: Zs[i] = encoder embeddings of rescore_windows(segs)[i] (each window on its own).
    The segment's concept scores = mean over its exact windows; candidates, score, margin, z and null gap come from them,
    segmentation evidence (p_sign, length) stays from the decoder."""
    for s, Z in zip(segs, Zs):
        S = bank.scores(Z)
        valid = (S > -1.5).all(0)
        row = np.where(valid, S.mean(0), -2.0)
        R = row + (gamma * prior if (prior is not None and gamma) else 0.0)
        R = np.where(valid, R, -9.0)
        o = np.argsort(-R)[:topn]
        mu, sd = float(row[valid].mean()), float(row[valid].std() + 1e-6)
        ens_top = s["candidates"][0]["word"]
        s["candidates"] = [dict(word=concepts[int(c)], score=round(float(row[c]), 4), support=round(float(R[c]), 4)) for c in o]
        s.update(nearest=s["candidates"][0]["word"], score=float(row[o[0]]), margin=float(row[o[0]] - row[o[1]]), z=float((row[o[0]] - mu) / sd),
                 null_gap=float((Z @ null_proto).mean() - row[o[0]]), agree=float(ens_top == s["candidates"][0]["word"]), exact=True)
    return segs
