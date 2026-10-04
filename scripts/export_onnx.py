"""Export the deployable model to ONNX (production / web back-end: ONNX Runtime + NumPy only, no PyTorch at serving time).

  python main.py export-onnx         # models/onnx/{encoder.onnx, encoder.int8.onnx, tagger.onnx} + parity.json

What is exported
  encoder.onnx       inputs  body [B,T,9,3] · left [B,T,21,3] · right [B,T,21,3] · face_all [B,T,18,3] · mask [B,T] (1.0 = frame)
                     outputs embedding [B,768] (L2-normalised sign embedding) · frames [B,T,768] (contextual frame states)
                     (parts come from modules.parts.prepare_parts on Standard-Schema keypoints — plain NumPy)
  encoder.int8.onnx  the same graph with int8 weights (dynamic quantisation of every MatMul) — ~4× smaller, faster on CPU; used by
                     the CPU runtime only if parity.json shows it keeps the decisions (top-1 word on real windows)
  tagger.onnx        input x [B,T,781] (= frames ‖ 13 kinematic features) → logits [B,T,3] (non-sign / begin / inside)
The span head (models/span_head.npz) and everything else (schema, windows, bank scoring, decoding, calibration) are NumPy.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from modules.utils import MODEL, get_logger, read_json, write_json

log = get_logger("onnx")
OUT = MODEL / "onnx"


class _EncoderGraph(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, body, left, right, face_all, mask):
        m = mask > 0.5
        h = self.enc.frames(dict(body=body, left=left, right=right, face_all=face_all), m)
        w = mask[..., None]
        z = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        z = z + self.enc.head(z)
        return torch.nn.functional.normalize(z, dim=-1), h


def export():
    from modules.encoder import load_encoder
    from modules.parts import prep
    from modules.sequence import N_KIN, Tagger
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.onnx"):
        f.unlink()
    enc = load_encoder(MODEL, MODEL / "encoder.pt", device="cpu").float().eval()
    g = _EncoderGraph(enc).eval()
    rng = np.random.RandomState(0)
    kp = rng.uniform(0.3, 0.7, (40, 133, 2)).astype(np.float32); sc = np.ones((40, 133), np.float32)
    p = prep(kp, sc)
    ex = tuple(torch.from_numpy(p[k][None].astype(np.float32)) for k in ("body", "left", "right", "face_all")) + (torch.ones(1, 40),)
    dyn = {n: {0: "B", 1: "T"} for n in ("body", "left", "right", "face_all", "mask")}
    dyn.update(embedding={0: "B"}, frames={0: "B", 1: "T"})
    torch.onnx.export(g, ex, str(OUT / "encoder.onnx"), input_names=["body", "left", "right", "face_all", "mask"], output_names=["embedding", "frames"],
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
    tg = Tagger(); tg.load_state_dict(torch.load(MODEL / "tagger.pt", map_location="cpu")); tg.eval()
    torch.onnx.export(tg, (torch.randn(1, 50, 768 + N_KIN),), str(OUT / "tagger.onnx"), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "B", 1: "T"}, "logits": {0: "B", 1: "T"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(OUT / "encoder.onnx"), str(OUT / "encoder.int8.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    return enc, tg


def parity(enc, tg):
    """ONNX Runtime (fp32 and int8) vs PyTorch on REAL windows (data_test / self-test landmarks if present, else random):
    embedding cosine, tagger logits, and whether the top-1 word against the bank stays the same."""
    import onnxruntime as ort
    from modules.encoder import embed_parts
    from modules.mediapipe_pose import load_raw, to_standard
    from modules.parts import prep, to_iso
    from modules.runtime import Bank
    so = ort.SessionOptions()
    sess = {k: ort.InferenceSession(str(OUT / f), so, providers=["CPUExecutionProvider"]) for k, f in (("fp32", "encoder.onnx"), ("int8", "encoder.int8.onnx"))}
    st = ort.InferenceSession(str(OUT / "tagger.onnx"), so, providers=["CPUExecutionProvider"])
    parts = []
    for f in sorted((ROOT / "cache" / "data_test_mp").glob("*.npz")):
        d = to_standard(load_raw(f))
        kp, sc = to_iso(d["kp"], tuple(int(x) for x in d["hw"])), d["sc"]
        for a in range(0, len(kp) - 24, 9):
            parts.append(prep(kp[a:a + 24], sc[a:a + 24]))
    if not parts:
        rng = np.random.RandomState(1)
        parts = [prep(rng.uniform(0.3, 0.7, (28, 133, 2)).astype(np.float32), np.ones((28, 133), np.float32)) for _ in range(32)]
    Zt = embed_parts(enc, parts, device="cpu")
    b = np.load(MODEL / "bank.npz")
    bank = Bank(b["Z"].astype(np.float32), b["c"], len(read_json(MODEL / "concepts.json")))
    top_t = bank.scores(Zt).argmax(1)
    rep = dict(n_windows=len(parts))
    for k, s in sess.items():
        Z = []
        for p in parts:
            feeds = {m: p[m][None].astype(np.float32) for m in ("body", "left", "right", "face_all")}
            feeds["mask"] = np.ones((1, len(p["body"])), np.float32)
            Z.append(s.run(None, feeds)[0][0])
        Z = np.stack(Z)
        cos = (Z * Zt).sum(1) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(Zt, axis=1))
        rep[k] = dict(cos_mean=float(cos.mean()), cos_min=float(cos.min()), top1_same_as_torch=float((bank.scores(Z).argmax(1) == top_t).mean()),
                      size_mb=round((OUT / ("encoder.onnx" if k == "fp32" else "encoder.int8.onnx")).stat().st_size / 1e6, 1))
    x = np.random.RandomState(2).normal(size=(1, 60, 781)).astype(np.float32)
    with torch.no_grad():
        lt = tg(torch.from_numpy(x)).numpy()
    rep["tagger_max_abs_diff"] = float(np.abs(lt - st.run(None, {"x": x})[0]).max())
    # whole-utterance frame states (tagger input) from fp32 vs int8 → tagger P(sign): does int8 move the segmentation evidence?
    from modules.segment import kin_features
    dp = []
    for f in sorted((ROOT / "cache" / "data_test_mp").glob("*.npz"))[:3]:
        d = to_standard(load_raw(f))
        kp, sc = to_iso(d["kp"], tuple(int(x) for x in d["hw"])), d["sc"]
        p = prep(kp, sc, max_len=10 ** 6)
        feeds = {m: p[m][None].astype(np.float32) for m in ("body", "left", "right", "face_all")}
        feeds["mask"] = np.ones((1, len(p["body"])), np.float32)
        probs = []
        for k in ("fp32", "int8"):
            H = sess[k].run(None, feeds)[1][0]
            lg = st.run(None, {"x": np.concatenate([H, kin_features(kp, sc)], 1)[None].astype(np.float32)})[0][0]
            e = np.exp(lg - lg.max(-1, keepdims=True)); probs.append((e / e.sum(-1, keepdims=True))[:, 1:].sum(1))
        dp.append(float(np.abs(probs[0] - probs[1]).max()))
    rep["int8_tagger_p_sign_max_abs_diff"] = max(dp) if dp else None
    rep["int8_ok"] = bool(rep["int8"]["top1_same_as_torch"] >= 0.97 and rep["int8"]["cos_mean"] >= 0.99 and (not dp or max(dp) < 0.1))
    if not rep["int8_ok"]:
        (OUT / "encoder.int8.onnx").unlink()
    return rep


if __name__ == "__main__":
    enc, tg = export()
    rep = parity(enc, tg)
    write_json(OUT / "parity.json", rep)
    log.info("onnx files (MB): %s", {f.name: round(f.stat().st_size / 1e6, 1) for f in OUT.glob("*.onnx")})
    log.info("parity: %s", rep)
