"""Landmark encoder: a compact encoder trained from scratch on Thai data only (no ASL / CSL pre-training).

v9 role: the SECOND OPINION next to the Uni-Sign encoder. Each sign segment's span is embedded by this ensemble (onnx/lmenc.onnx,
`OnnxLM`) and matched against its own bank (lm_bank.npz); SignPipeline.hand_rescore fuses that row with the encoder + hand-shape
rows at weight spotting.json "read.lm_w" (chosen on tune data). Its errors differ from Uni-Sign's (it never saw ASL / CSL), which is
why the fusion helps although it is weaker alone. Built by `python main.py decision lmbank` from artifacts/runs/enc-v8.

History (v8, prompt_10):

Why (prompt_10): v5–v7 fine-tuned Uni-Sign (pre-trained on Chinese CSL-News, fine-tuned on American WLASL). 39 % of v7's remaining
errors were confusions the foreign zero-shot model already made, and hard-negative training did not remove them. A model that never
saw ASL / CSL cannot inherit their sign space. It is also ~30× smaller than Uni-Sign + mT5, so the whole utterance runs in a few
milliseconds on a CPU (v7's encoder pass was most of the 0.3–4 s server latency).

Input = signer-normalised landmark features (NumPy, shared by training, the ONNX runtime and the web API — `features()`):
  body frame   origin = shoulder centre, unit = the clip's shoulder width (a tall or a small signer, near or far: same numbers)
  per hand     hand SHAPE: 20 joints relative to the wrist ÷ hand size (handshape independent of where the hand is)
               hand POSITION: wrist + hand centre in the body frame, hand centre relative to the nose, hand size ÷ shoulder width
               (≈ distance to the camera), seen flag; a missing hand keeps its pose-wrist position
  between      right − left hand centre
  motion       frame-to-frame velocity of both wrists and hand centres, handshape change speed
  body         nose, shoulders, elbows, wrists in the body frame
Model = Conv1D (local motion) + Transformer (order, context) blocks → frame states H [T, d] (the tagger / span head read them)
        → masked mean + attention pooling → L2-normalised embedding [D].
Training objective unchanged from v7 (CosFace over concepts + SupCon, P×K batches across signers, hard negatives) — see
scripts/train_encoder.py.
"""
from __future__ import annotations

import math

import numpy as np

from .schema import LH, RH

# body slots in the 133 layout: nose, shoulders, elbows, wrists
NOSE, SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 0, 5, 6, 7, 8, 9, 10
BODY = [NOSE, SH_L, SH_R, EL_L, EL_R, WR_L, WR_R]
FPS = 25.0
N_FEAT = 14 + 2 * 48 + 2 + 10        # = 122
MAX_LEN = 128                        # frames (5.1 s at 25 fps) per isolated window; longer windows are sub-sampled


def _interp_missing(x, ok):
    """Fill rows of x [T, ...] where ok is False by linear interpolation over time (edges hold the nearest value)."""
    if ok.all() or not ok.any():
        return x
    t = np.arange(len(x)); idx = np.flatnonzero(ok)
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(t, idx, flat[idx, j]) for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape).astype(x.dtype)


def features(kp, sc, thr=0.3):
    """kp [T,133,2] in isotropic units (x·W/H, y), sc [T,133] → float32 [T, N_FEAT] (see module docstring). NumPy only."""
    kp = np.asarray(kp, np.float32); sc = np.asarray(sc, np.float32)
    T = len(kp)
    if T == 0:
        return np.zeros((0, N_FEAT), np.float32)
    sh_ok = (sc[:, SH_L] > thr) & (sc[:, SH_R] > thr)
    mid = (kp[:, SH_L] + kp[:, SH_R]) / 2
    sw = np.linalg.norm(kp[:, SH_L] - kp[:, SH_R], axis=1)
    if sh_ok.any():
        mid = _interp_missing(mid, sh_ok)
        S = float(np.median(sw[sh_ok]))
    else:
        live = sc > thr
        mid = np.repeat((kp[live].mean(0) if live.any() else np.zeros(2, np.float32))[None], T, 0)
        S = 0.25
    S = max(S, 1e-3)
    body = (kp[:, BODY] - mid[:, None]) / S
    bok = sc[:, BODY] > thr
    for j in range(len(BODY)):
        body[:, j] = _interp_missing(body[:, j], bok[:, j]) if bok[:, j].any() else 0.0
    nose = body[:, 0]
    feats = [body.reshape(T, -1)]                                                     # 14
    centers, wrists, shape_v = [], [], []
    for hand, wr in ((LH, 5), (RH, 6)):                                               # body[:, 5] = left wrist, 6 = right wrist
        h = (kp[:, hand] - mid[:, None]) / S                                          # [T,21,2] body frame
        ok = sc[:, hand].mean(1) > thr
        size = np.linalg.norm(h[:, 9] - h[:, 0], axis=1)                              # wrist → middle-finger MCP
        size_ok = ok & (size > 1e-3)
        shp = np.where(size_ok[:, None, None], (h[:, 1:] - h[:, :1]) / np.maximum(size, 1e-3)[:, None, None], 0.0)
        cen = np.where(ok[:, None], h.mean(1), body[:, wr])
        wri = np.where(ok[:, None], h[:, 0], body[:, wr])
        rel = cen - nose
        hs = np.where(size_ok, size, 0.0)
        feats += [shp.reshape(T, -1).clip(-4, 4), wri, cen, rel, hs[:, None] * 2.0, ok[:, None].astype(np.float32)]   # 40+2+2+2+1+1
        centers.append(cen); wrists.append(wri)
        dshape = np.r_[0.0, np.abs(np.diff(shp.reshape(T, -1), axis=0)).mean(1)]
        dshape = np.where(ok & np.r_[False, ok[:-1]], dshape, 0.0)
        shape_v.append(dshape * FPS / 10.0)
    feats.append(centers[1] - centers[0])                                             # 2
    vel = []
    for x in wrists + centers:
        v = np.r_[np.zeros((1, 2), np.float32), np.diff(x, axis=0)] * (FPS / 10.0)
        vel.append(v)
    feats += vel + [shape_v[0][:, None], shape_v[1][:, None]]                         # 8 + 2
    f = np.concatenate(feats, 1).astype(np.float32)
    return np.nan_to_num(f).clip(-6, 6)


def resample_feats(f, n_out):
    n = len(f)
    if n_out == n or n < 2:
        return f
    t = np.linspace(0, n - 1, max(2, n_out))
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, n - 1); w = (t - i0)[:, None]
    return (f[i0] * (1 - w) + f[i1] * w).astype(np.float32)


def prep(kp_iso, sc, max_len=MAX_LEN):
    """Isolated window → features (sub-sampled to max_len frames). The `parts` object of this encoder."""
    f = features(kp_iso, sc)
    return resample_feats(f, max_len) if len(f) > max_len else f


def collate_np(feats_list):
    L = max(len(f) for f in feats_list)
    B = len(feats_list)
    x = np.zeros((B, L, N_FEAT), np.float32); m = np.zeros((B, L), np.float32)
    for b, f in enumerate(feats_list):
        x[b, :len(f)] = f; m[b, :len(f)] = 1.0
    return x, m


# ----------------------------------------------------------------------------- model (PyTorch; exported to ONNX for production)
def build_model(**kw):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, d, k=7, drop=0.1):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.pw1 = nn.Linear(d, 2 * d)
            self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
            self.bn = nn.BatchNorm1d(d)
            self.pw2 = nn.Linear(d, d)
            self.drop = nn.Dropout(drop)

        def forward(self, x, m):
            y = self.norm(x)
            y = F.glu(self.pw1(y), dim=-1) * m[..., None]
            y = self.dw(y.transpose(1, 2))
            y = F.silu(self.bn(y)).transpose(1, 2)
            return x + self.drop(self.pw2(y)) * m[..., None]

    class Attn(nn.Module):
        """Multi-head self-attention with the same parameters (names and layout) as nn.MultiheadAttention, written out so the ONNX
        graph keeps a dynamic sequence length (the built-in module's traced reshape froze T to the export example)."""

        def __init__(self, d, heads, dropout=0.0):
            super().__init__()
            self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d)); nn.init.xavier_uniform_(self.in_proj_weight)
            self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
            self.out_proj = nn.Linear(d, d)
            self.h, self.drop = heads, dropout

        def forward(self, x, pad):
            B, T, D = x.shape[0], x.shape[1], x.shape[2]
            qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(D // self.h))
            att = att.masked_fill(pad[:, None, None, :], -1e4).softmax(-1)
            att = F.dropout(att, self.drop, self.training)
            return self.out_proj((att @ v).transpose(1, 2).reshape(B, T, D))

    class TfBlock(nn.Module):
        def __init__(self, d, heads=4, ff=2, drop=0.1):
            super().__init__()
            self.n1 = nn.LayerNorm(d); self.att = Attn(d, heads, dropout=drop)
            self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ff * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, m):
            y = self.n1(x)
            y = self.att(y, m < 0.5)
            x = x + self.drop(y)
            return x + self.drop(self.ff(self.n2(x)))

    class LandmarkEncoder(nn.Module):
        def __init__(self, n_feat=N_FEAT, d=256, emb=256, depth=4, heads=4, k=7, drop=0.1, max_len=1024):
            super().__init__()
            self.cfg = dict(n_feat=n_feat, d=d, emb=emb, depth=depth, heads=heads, k=k, drop=drop, max_len=max_len)
            self.inp = nn.Sequential(nn.Linear(n_feat, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
            pe = torch.zeros(max_len, d)
            pos = torch.arange(max_len)[:, None].float(); div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe * 0.1)
            self.blocks = nn.ModuleList()
            for _ in range(depth):
                self.blocks.append(ConvBlock(d, k, drop)); self.blocks.append(TfBlock(d, heads, 2, drop))
            self.out_norm = nn.LayerNorm(d)
            self.att_pool = nn.Linear(d, 1)
            self.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, emb))
            self.dim, self.emb_dim = d, emb

        def frames(self, x, m):
            T = x.shape[1]
            h = self.inp(x) + self.pe[:T][None]
            h = h * m[..., None]
            for b in self.blocks:
                h = b(h, m)
            return self.out_norm(h)

        def pool(self, h, m):
            mm = m[..., None]
            mean = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            a = self.att_pool(h).squeeze(-1).masked_fill(m < 0.5, -1e4).softmax(1)
            att = (h * a[..., None]).sum(1)
            return F.normalize(self.head(torch.cat([mean, att], -1)).float(), dim=-1)

        def forward(self, x, m):
            return self.pool(self.frames(x, m), m)

    class LandmarkEnsemble(nn.Module):
        """v8 release: several landmark encoders trained with different seeds / augmentation, run side by side on the same features.
        frames = their frame states concatenated; embedding = their embeddings concatenated / √n (cosine = mean of the members'
        cosines). Chosen on the TSL51 tune sentences: more robust to window boundaries than any single member (result_v8 §5)."""

        def __init__(self, members):
            super().__init__()
            self.cfg = dict(members=members)
            self.members = nn.ModuleList([LandmarkEncoder(**m) for m in members])
            self.dim = sum(m.dim for m in self.members); self.emb_dim = sum(m.emb_dim for m in self.members)

        def frames(self, x, m):
            return torch.cat([e.frames(x, m) for e in self.members], -1)

        def pool(self, h, m):
            out, a = [], 0
            for e in self.members:
                out.append(e.pool(h[..., a:a + e.dim], m)); a += e.dim
            return torch.cat(out, -1) / math.sqrt(len(out))

        def forward(self, x, m):
            return self.pool(self.frames(x, m), m)

    if "members" in kw:
        return LandmarkEnsemble(kw["members"])
    return LandmarkEncoder(**kw)


def make_ensemble(paths, out_path):
    """Combine trained encoder checkpoints (encoder_best.pt) into one ensemble checkpoint."""
    import torch
    sds = [torch.load(p, map_location="cpu") for p in paths]
    state = {f"members.{i}.{k}": v for i, sd in enumerate(sds) for k, v in sd["state"].items()}
    torch.save(dict(cfg=dict(members=[sd["cfg"] for sd in sds]), state=state, members=[str(p) for p in paths]), out_path)


def load_model(path, device="cpu"):
    import torch
    sd = torch.load(path, map_location="cpu")
    m = build_model(**sd["cfg"])
    m.load_state_dict(sd["state"])
    return m.to(device).eval()


def embed(model, feats_list, device="cuda", bs=256):
    """Embeddings of many windows (longest-similar batches)."""
    import torch
    model.eval()
    order = np.argsort([len(f) for f in feats_list])
    out = np.zeros((len(feats_list), model.emb_dim), np.float32)
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            x, m = collate_np([feats_list[j] for j in idx])
            z = model(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))
            out[idx] = z.float().cpu().numpy()
    return out


def frames_long(model, kp_iso, sc, device="cpu", chunk=400, overlap=50):
    """Contextual frame states [T, d] of a whole utterance (chunks of `chunk` frames blended over `overlap`, as the v7 runtime)."""
    import torch
    f = features(kp_iso, sc)
    T = len(f)
    out = np.zeros((T, model.dim), np.float32); wsum = np.zeros(T, np.float32)
    step = chunk - overlap
    model.eval()
    with torch.no_grad():
        for a in range(0, max(1, T - overlap), step):
            b = min(T, a + chunk)
            x = torch.from_numpy(f[a:b][None]).to(device); m = torch.ones(1, b - a, device=device)
            h = model.frames(x, m)[0].float().cpu().numpy()
            w = np.minimum(np.arange(1, b - a + 1), np.arange(b - a, 0, -1)).clip(max=25).astype(np.float32)
            out[a:b] += h * w[:, None]; wsum[a:b] += w
            if b == T:
                break
    return out / np.maximum(wsum, 1e-6)[:, None]


def export_onnx(ckpt, out_path):
    """Checkpoint → ONNX (x [B,T,N_FEAT], m [B,T] → embedding [B,D]), dynamic batch and length; returns the max |torch − onnx|."""
    import onnxruntime as ort
    import torch
    model = load_model(ckpt, "cpu")
    x = torch.randn(2, 40, N_FEAT); m = torch.ones(2, 40); m[1, 30:] = 0
    torch.onnx.export(model, (x, m), str(out_path), input_names=["x", "m"], output_names=["emb"], opset_version=17,
                      dynamic_axes={"x": {0: "B", 1: "T"}, "m": {0: "B", 1: "T"}, "emb": {0: "B"}})
    x2 = torch.randn(3, 77, N_FEAT); m2 = torch.ones(3, 77); m2[2, 50:] = 0
    with torch.no_grad():
        ref = model(x2, m2).numpy()
    got = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"]).run(None, {"x": x2.numpy(), "m": m2.numpy()})[0]
    return float(np.abs(ref - got).max())


class OnnxLM:
    """Production runtime of the landmark encoder (v9 second opinion): ONNX Runtime + NumPy features, no PyTorch."""

    def __init__(self, path, threads=None):
        import os
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads or os.environ.get("ORT_THREADS"):
            so.intra_op_num_threads = int(threads or os.environ["ORT_THREADS"])
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def embed(self, feats_list, bs=64):
        out = None
        order = np.argsort([len(f) for f in feats_list])
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            x, m = collate_np([feats_list[j] for j in idx])
            z = self.sess.run(None, {"x": x, "m": m})[0]
            if out is None:
                out = np.zeros((len(feats_list), z.shape[1]), np.float32)
            out[idx] = z
        return out if out is not None else np.zeros((0, 768), np.float32)

    def spans(self, kp_iso, sc, spans):
        return self.embed([prep(kp_iso[a:b], sc[a:b]) for a, b in spans])


class EncoderAPI:
    """One interface for the continuous-signing stage over either encoder: v8 landmark encoder (this module) or v7 Uni-Sign."""

    def __init__(self, ckpt, models_dir, device="cuda"):
        import torch
        sd = torch.load(ckpt, map_location="cpu")
        self.device = device
        if isinstance(sd, dict) and "cfg" in sd:
            self.kind = "lmenc"
            self.model = load_model(ckpt, device)
            self.dim = self.model.emb_dim
            self.prep = prep
        else:
            from .encoder import load_encoder, prep as up
            self.kind = "unisign"
            self.model = load_encoder(models_dir, ckpt, device=device)
            self.dim = 768
            self.prep = up

    def embed(self, parts_list, bs=256):
        if self.kind == "lmenc":
            return embed(self.model, parts_list, device=self.device, bs=bs)
        from .encoder import embed_parts
        return embed_parts(self.model, parts_list, bs=min(bs, 64), device=self.device)

    def frames(self, kp_iso, sc):
        if self.kind == "lmenc":
            return frames_long(self.model, kp_iso, sc, device=self.device)
        from .encoder import frame_features
        return frame_features(self.model, kp_iso, sc, device=self.device)
