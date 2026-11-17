"""Inference back-ends with one NumPy interface — ONNX Runtime (production / web API, no PyTorch) or PyTorch (development, GPU).

    be = OnnxBackend(models_dir, device="cpu")      # models/onnx/encoder.onnx + tagger.onnx (device: cpu | auto | cuda | dml | coreml)
    be = TorchBackend(models_dir, device=None)      # models/encoder.pt + tagger.pt (None / auto: cuda → mps → cpu)
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


ORT_GPU = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider", "coreml": "CoreMLExecutionProvider"}
ORT_OPTS = {"CUDAExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"},
            "CoreMLExecutionProvider": {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}}   # fp32 tensors, no Neural Engine


def ort_providers(device="cpu"):
    """ONNX Runtime providers for `device`, limited to what this onnxruntime build has:
    cpu · cuda (NVIDIA, onnxruntime-gpu) · dml (any DirectX 12 GPU on Windows, onnxruntime-directml) · coreml (Apple, macOS
    onnxruntime; "mps" means the same) · auto = cuda → dml, else CPU. CoreML stays opt-in: its parity with the CPU results has not
    been measured on a Mac yet. An unavailable GPU falls back to CPU (with a message) so the same command runs on every machine."""
    import onnxruntime as ort
    d = str(device or "cpu").lower().split(":")[0]
    d = {"mps": "coreml", "directml": "dml", "gpu": "auto"}.get(d, d)
    have = ort.get_available_providers()
    if d == "auto":
        d = next((k for k in ("cuda", "dml") if ORT_GPU[k] in have), "cpu")
    p = ORT_GPU.get(d)
    if p == "CUDAExecutionProvider" and p in have and hasattr(ort, "preload_dlls"):
        try:        # Windows / pip CUDA: without this the CUDA + cuDNN DLLs (torch's / nvidia-* wheels) are not found and ORT
            ort.preload_dlls()      # silently runs on the CPU
        except Exception:  # noqa: BLE001
            pass
    if p is None or p not in have:
        if p is not None:
            print(f"onnxruntime: {p} is not in this build ({', '.join(have)}) → CPU")
        return ["CPUExecutionProvider"]
    return [(p, ORT_OPTS[p]) if p in ORT_OPTS else p, "CPUExecutionProvider"]


def ort_session(path, so, prov):
    """InferenceSession on `prov`; a GPU provider that fails to start (driver, unsupported op) → CPU instead of an error."""
    import onnxruntime as ort
    try:
        s = ort.InferenceSession(str(path), so, providers=prov)
    except Exception as e:  # noqa: BLE001
        if prov == ["CPUExecutionProvider"]:
            raise
        print(f"onnxruntime: {prov[0]} failed for {Path(path).name} ({e}) → CPU")
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    want = prov[0][0] if isinstance(prov[0], tuple) else prov[0]
    if s.get_providers()[0] != want:
        print(f"onnxruntime: {want} did not load for {Path(path).name} (drivers / CUDA libraries?) → CPU")
    return s


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
        # auto-detect: min(8, this machine's CPU count) — 8 is what was measured best on a 12-thread laptop; a 4-core machine / a Mac never gets more threads than cores
        so.intra_op_num_threads = int(threads or os.environ.get("ORT_THREADS") or min(8, os.cpu_count() or 4))
        # phase 3: worker threads sleep instead of spin-waiting between runs (ORT_SPIN=1 = the old busy-wait). The browser's MediaPipe shares this
        # machine's CPU; with spinning, the 0.5 s live-stream steps kept up to 8 cores busy and the camera dropped from ~15 to ~11 fps (measured).
        if os.environ.get("ORT_SPIN", "0") != "1":
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        prov = ort_providers(device)
        # CPU: the int8 encoder when the export's parity check kept it (models/onnx/parity.json); THAISLM_INT8=0 forces fp32
        int8 = md / "encoder.int8.onnx"
        use_int8 = prov[0] == "CPUExecutionProvider" and int8.exists() and os.environ.get("THAISLM_INT8", "1") != "0"
        self.enc = ort_session(int8 if use_int8 else md / "encoder.onnx", so, prov)
        self.precision = "int8" if use_int8 else "fp32"
        self.tagger = ort_session(md / "tagger.onnx", so, prov)
        # phase-3 signing gate (optional file): the frame tagger fine-tuned with "hands up but NOT signing" negatives; the pipeline uses it only
        # as a veto on long non-sign runs (SignPipeline._tag, spotting.json "gate"). THAISLM_GATE=0 = exactly the old tagger.
        gate = md / "gate_tagger.onnx"
        self.gate = ort_session(gate, so, prov) if gate.exists() and os.environ.get("THAISLM_GATE", "1") != "0" else None
        self.providers = self.enc.get_providers()
        self.device = {v: k for k, v in ORT_GPU.items()}.get(self.providers[0], "cpu")

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

    def gate_tag(self, x):
        """the signing gate's frame probabilities (same input / output as `tag`), or None when no gate model is shipped / it is turned off"""
        return None if self.gate is None else _softmax(self.gate.run(None, {"x": x[None].astype(np.float32)})[0][0])


class TorchBackend:
    name = "torch"

    def __init__(self, model_dir, device=None, encoder=None, tagger=None, tagger_arch="gru"):
        import torch
        from .encoder import load_encoder
        from .sequence import Tagger
        md = Path(model_dir)
        self.torch = torch
        from .utils import device as resolve_device
        self.device = resolve_device(device)
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


class FormBank:
    """v11 bank scoring (prompt-opt-v11 §1B.3), same interface as Bank (.scores → [N, C], -2 where a concept has no example):

        s(c) = max over forms f of c [ 0.5·cos(q, proto_f) + 0.5·mean top-2 cos(q, exemplars_f of DIFFERENT signers) ] − λ / √n_signers(f)

    proto_f = mean of the per-signer means of form f (L2-normalised); the exemplar term takes each signer's best clip and averages the
    two best signers (one signer: its best clip). λ shrinks forms seen from few people (the attractor rule: กลับ with 1 clip took 42/47
    "ไป"). A word with one form and λ = 0 differs from Bank only by the prototype half and the one-best-clip-per-signer rule."""

    def __init__(self, Z, c, n_concepts, signer, form=None, lam=0.0, allowed=None, w_proto=0.5):
        Z, c = np.asarray(Z, np.float32), np.asarray(c, np.int64)
        signer = np.asarray(signer).astype(str)
        form = np.zeros(len(c), np.int64) if form is None else np.asarray(form, np.int64)
        keep = np.ones(len(c), bool) if allowed is None else np.isin(c, np.asarray(sorted(allowed)))
        Z, c, signer, form = Z[keep], c[keep], signer[keep], form[keep]
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        cell_key = np.array([f"{a}|{b}|{s}" for a, b, s in zip(c, form, signer)])
        o = np.lexsort((cell_key, form, c))
        self.Z, self.c, self.form, self.signer = Z[o], c[o], form[o], signer[o]
        ck = cell_key[o]
        self.cell_starts = np.r_[0, np.flatnonzero(ck[1:] != ck[:-1]) + 1]               # (concept, form, signer) cells
        cell_c, cell_f = self.c[self.cell_starts], self.form[self.cell_starts]
        fk = cell_c * 1000 + cell_f
        self.form_starts = np.r_[0, np.flatnonzero(fk[1:] != fk[:-1]) + 1]               # forms over cells
        self.form_c = cell_c[self.form_starts]
        self.n_sign = np.diff(np.r_[self.form_starts, len(self.cell_starts)])
        self.cell_seg = np.repeat(np.arange(len(self.form_starts)), self.n_sign)
        protos = []
        for i, a in enumerate(self.form_starts):
            b = self.form_starts[i + 1] if i + 1 < len(self.form_starts) else len(self.cell_starts)
            ms = []
            for j in range(a, b):
                x0 = self.cell_starts[j]; x1 = self.cell_starts[j + 1] if j + 1 < len(self.cell_starts) else len(self.Z)
                m = self.Z[x0:x1].mean(0); ms.append(m / (np.linalg.norm(m) + 1e-8))
            p = np.mean(ms, 0); protos.append(p / (np.linalg.norm(p) + 1e-8))
        self.P = np.stack(protos).astype(np.float32)
        self.cstarts = np.r_[0, np.flatnonzero(self.form_c[1:] != self.form_c[:-1]) + 1]  # concepts over forms
        self.concepts_present = self.form_c[self.cstarts]
        self.n_concepts, self.lam, self.w_proto = n_concepts, lam, w_proto
        self.starts = self.cstarts                                                      # (Bank compatibility)

    def form_scores(self, Q):
        Q = np.asarray(Q, np.float32)
        S = Q @ self.Z.T
        cell = np.maximum.reduceat(S, self.cell_starts, axis=1)                          # each signer's best clip
        t1 = np.maximum.reduceat(cell, self.form_starts, axis=1)
        is_max = cell >= t1[:, self.cell_seg] - 1e-7
        t2 = np.maximum.reduceat(np.where(is_max, -2.0, cell), self.form_starts, axis=1)
        ex = np.where(t2 > -1.5, (t1 + t2) / 2, t1)
        return self.w_proto * (Q @ self.P.T) + (1 - self.w_proto) * ex - self.lam / np.sqrt(self.n_sign)[None]

    def scores(self, Q, topm=2):
        F = self.form_scores(Q)
        top = np.maximum.reduceat(F, self.cstarts, axis=1)
        out = np.full((len(F), self.n_concepts), -2.0, np.float32)
        out[:, self.concepts_present] = top
        return out


def variant_lut(concepts, variants):
    """[C] concept index → the index it is read as. variants {"สวัสดี (อายุเท่ากันหรือน้อยกว่า)": "สวัสดี", …}: a style / school /
    politeness variant of a word is that word (models/vocab_conversation.json "variants"; v8.1)."""
    cidx = {c: i for i, c in enumerate(concepts)}
    lut = np.arange(len(concepts))
    for v, base in variants.items():
        if v in cidx and base in cidx:
            lut[cidx[v]] = cidx[base]
    return lut


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
