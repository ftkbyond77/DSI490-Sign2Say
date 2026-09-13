"""v4 end-to-end inference: video → pose → segmentation → per-segment recognition (open-set) → Face model → LLM fusion.

Every detected segment is reported, even when the word is unknown:
  {"t0","t1","status": accepted|uncertain|unknown, "candidates":[{word,score}], "nearest": word}
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .face_v4 import FaceModel, nmm
from .kinematics import frame_features, heuristic_segments
from .recognize import VocabBank, embed
from .segment import Segmenter, decode_segments, sequence_features
from .translate import compose
from .unisign import UniSignEncoder, prepare_parts, upsample
from .utils import ART, get_logger, read_json, write_json
from .wholebody import RTMWBatch, extract_video

log = get_logger("pipeline_v4")
V4 = ART / "v4"


def load_bank(path=None):
    z = np.load(path or V4 / "bank.npz", allow_pickle=False)
    meta = [dict(source=s, signer=g, clip=c) for s, g, c in zip(z["source"], z["signer"], z["clip"])]
    return VocabBank(z["Z"], z["words"].tolist(), meta)


class SignPipelineV4:
    def __init__(self, device="cuda", bank_path=None, seg_path=None, calib_path=None, use_face=True, use_llm=True, spotting=True):
        from .articulators import Articulators
        self.device = device
        self.art = Articulators(device=device, threads=4)
        self.rtmw = RTMWBatch(device)
        self.enc = UniSignEncoder("wlasl", use_mt5=True, device=device).eval()
        self.bank = load_bank(bank_path)
        self.seg = None
        sp = Path(seg_path or V4 / "segmenter.pt")
        if sp.exists():
            self.seg = Segmenter().to(device)
            self.seg.load_state_dict(torch.load(sp, map_location=device))
            self.seg.eval()
        cp = Path(calib_path or V4 / "calibration.json")
        self.calib = read_json(cp) if cp.exists() else dict(accept=dict(tau_score=0.9, tau_margin=0.05), uncertain=dict(tau_score=0.8, tau_margin=0.0))
        spp = V4 / "spotting.json"
        self.spot_cfg = read_json(spp) if (spotting and spp.exists()) else None
        self.face = FaceModel(device) if use_face else None
        self.use_llm = use_llm

    # ------------------------------------------------------------------ steps
    def segments(self, kp, sc, hw):
        feats = sequence_features(self.enc, kp, sc, hw)
        if self.seg is not None:
            with torch.no_grad():
                prob = self.seg(torch.from_numpy(feats)[None].to(self.device)).softmax(-1)[0].cpu().numpy()
            segs = decode_segments(prob)
            return segs, prob
        _, sig = frame_features(kp, sc, hw)
        return heuristic_segments(sig), None

    def recognise(self, kp, sc, segs, pad=1, topk=5):
        parts = []
        for a, b in segs:
            a0, b0 = max(0, a - pad), min(len(kp), b + pad)
            k, s = upsample(kp[a0:b0], sc[a0:b0], 2)
            parts.append(prepare_parts(k, s, scale_ref=(kp, sc)))
        if not parts:
            return []
        Z = embed(self.enc, parts, "ctx")
        S = self.bank.word_scores(Z)
        out = []
        for i, (a, b) in enumerate(segs):
            order = np.argsort(-S[i])[:max(topk, 50)]
            cands = [dict(word=self.bank.vocab[j], score=float(S[i, j])) for j in order[:topk]]
            top1, top2 = S[i, order[0]], S[i, order[1]]
            rel = float(top1 - np.median(S[i, order[:50]]))
            ca, cu = self.calib["accept"], self.calib["uncertain"]
            feats = dict(score=float(top1), margin=float(top1 - top2), rel=rel)
            if top1 >= ca["tau_score"] and top1 - top2 >= ca["tau_margin"]:
                status = "accepted"
            elif top1 >= cu["tau_score"] and top1 - top2 >= cu.get("tau_margin", 0.0):
                status = "uncertain"
            else:
                status = "unknown"
            out.append(dict(f0=int(a), f1=int(b), status=status, candidates=cands, nearest=cands[0]["word"], **feats))
        return out

    def spot(self, kp, sc, hw, prob):
        """Recognition-driven spotting (modules/spotting.py) + E1-calibrated status."""
        from .spotting import active_frames, decode, window_table
        c = self.spot_cfg
        if prob is not None:
            act = active_frames(prob, thr=c["act_thr"])
        else:
            act = active_frames(sig=frame_features(kp, sc, hw)[1])
        table = window_table(self.enc, self.bank, kp, sc, act, lens=tuple(c["lens"]), stride=c["stride"])
        segs = decode(table, act, self.bank.vocab, c["lam"], c["gap_cost"], c["stride"], min_len=c.get("min_len", 6), agg=c.get("agg"))
        ca, cu = self.calib["accept"], self.calib["uncertain"]
        for s in segs:  # same E1-calibrated rule as isolated recognition (z-tiers were not informative on held-out sentences)
            if s["score"] >= ca["tau_score"] and s["margin"] >= ca["tau_margin"] and s.get("window_top1", s["nearest"]) == s["nearest"]:
                s["status"] = "accepted"
            elif s["score"] >= cu["tau_score"]:
                s["status"] = "uncertain"
            else:
                s["status"] = "unknown"
        return segs

    # ------------------------------------------------------------------ run
    def run(self, video, out_dir=None, fps=12.5, tts=False):
        timings = {}
        t = time.perf_counter()
        d = extract_video(video, fps=fps, art=self.art, rtmw=self.rtmw)
        kp, sc, hw, frames = d["kp"], d["sc"], tuple(d["hw"]), d["frames"]
        timings["pose_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); segs, prob = self.segments(kp, sc, hw); timings["segmentation_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        recs = self.spot(kp, sc, hw, prob) if self.spot_cfg else self.recognise(kp, sc, segs)
        timings["recognition_ms"] = (time.perf_counter() - t) * 1000
        for r in recs:
            r["t0"], r["t1"] = round(r["f0"] / fps, 2), round(r["f1"] / fps, 2)
        face = {}
        if self.face is not None:
            t = time.perf_counter()
            face = self.face.emotions(frames, kp, sc)
            face["nmm"] = nmm(kp, sc, hw, fps, [(r["f0"], r["f1"]) for r in recs])
            face.pop("per_frame", None); face.pop("frames", None)
            timings["face_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        fusion = compose(recs, face) if self.use_llm else None
        timings["llm_ms"] = (time.perf_counter() - t) * 1000
        result = dict(video=str(video), fps=fps, n_frames=len(kp), segments=recs, face=face, fusion=fusion,
                      timings_ms={k: round(v, 1) for k, v in timings.items()})
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            if prob is not None:
                np.save(out / "segment_prob.npy", prob)
            write_json(out / "result.json", result)
            with open(out / "subtitle.srt", "w", encoding="utf-8") as f:
                for i, r in enumerate(recs, 1):
                    lab = r["nearest"] if r["status"] == "accepted" else (f"{r['nearest']}?" if r["status"] == "uncertain" else f"[?] ≈ {r['nearest']}")
                    f.write(f"{i}\n{_srt(r['t0'])} --> {_srt(r['t1'])}\n{lab}\n\n")
            self.contact_sheet(frames, kp, recs, out / "segments.jpg")
            if tts and fusion and fusion.get("sentence_th"):
                from .tts import LocalMMSTTS, SpeechStyle, write_wav
                emo = {"happy": "happy", "sad": "sad", "angry": "angry", "surprise": "surprised"}.get(face.get("emotion"), "neutral")
                wav, sr = LocalMMSTTS().synthesize(fusion["sentence_th"].replace("[?]", ""), SpeechStyle.from_affect(emo, face.get("intensity", 0)))
                write_wav(out / "audio.wav", wav, sr)
        return result

    @staticmethod
    def contact_sheet(frames, kp, recs, path, w=150):
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        H, W = frames.shape[1:3]
        tiles = []
        for r in recs:
            mid = (r["f0"] + r["f1"]) // 2
            ims = []
            for i in (r["f0"], mid, max(r["f1"] - 1, r["f0"])):
                cx = int(np.clip(kp[i, 0, 0] * W, H * 0.3, W - H * 0.3))
                crop = frames[i][:, max(0, cx - int(H * 0.3)):cx + int(H * 0.3)]
                ims.append(cv2.resize(crop, (w, int(w * H / max(crop.shape[1], 1)))))
            tile = np.concatenate(ims, 1)
            band = np.full((64, tile.shape[1], 3), 255, np.uint8)
            tiles.append(np.concatenate([tile, band], 0))
        if not tiles:
            return
        hmax = max(t.shape[0] for t in tiles)
        sheet = np.concatenate([np.pad(t, ((0, hmax - t.shape[0]), (0, 6), (0, 0)), constant_values=255) for t in tiles], 1)
        img = Image.fromarray(sheet[..., ::-1])
        draw = ImageDraw.Draw(img)
        font = None
        for fp in ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/LeelawUI.ttf"):
            try:
                font = ImageFont.truetype(fp, 15); break
            except Exception:
                pass
        x = 0
        for r, t in zip(recs, tiles):
            col = {"accepted": (0, 128, 0), "uncertain": (200, 120, 0), "unknown": (180, 0, 0)}[r["status"]]
            draw.text((x + 4, hmax - 62), f"{r['t0']:.1f}-{r['t1']:.1f}s {r['status']}", fill=col, font=font)
            lab = r["nearest"] if r["status"] != "unknown" else f"[?] ≈ {r['nearest']}"
            draw.text((x + 4, hmax - 42), lab, fill=col, font=font)
            draw.text((x + 4, hmax - 22), ", ".join(c["word"] for c in r["candidates"][1:4]), fill=(90, 90, 90), font=font)
            x += t.shape[1] + 6
        img.save(path, quality=90)


def _srt(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
