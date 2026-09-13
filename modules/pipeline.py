"""End-to-end inference.
VideoPipeline (.mp4): decode 12.5 fps → articulators → frozen ViT → temporal encoder + ISLR head →
spotting (DP / NMS) → NMM + affect → language layer → TTS → result.json / subtitle.srt / audio.wav
StreamPipeline: webcam-style per-frame processing with a fixed ring buffer (simulated from a file here)."""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

from .articulators import Articulators, active_mask, pose_features
from .data import decode_frames, normalize_text
from .encoder import FrozenEncoder
from .evalkit import pct, rss_mb
from .heads import ISLRModel, PrototypeBank, nmm_flags
from .language import get_composer
from .spotting import dp_relative, dp_segment, greedy_nms, pause_segments, to_timeline, window_scores
from .ssl import SignEncoder
from .tts import SpeechStyle, get_tts, write_wav
from .utils import ART, get_logger, load_cfg, write_json

log = get_logger("pipeline")


def load_islr(path: Path, device="cuda") -> ISLRModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    enc = SignEncoder(nap_N=ck["state"]["encoder.nap"].cpu().numpy(), **ck["enc_cfg"])
    m = ISLRModel(enc, ck["n_classes"], ck["n_signers"], pool=ck.get("pool", "cls"), streams=ck.get("streams"))
    if ck.get("lora"):
        enc.enable_lora(**ck["lora"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


class VideoPipeline:
    def __init__(self, islr_path=None, bank_path=None, device="cuda", decoder="dp", decoder_kw=None,
                 composer="openai", tts="local", affect=None, window_lengths=(8, 12, 16, 20, 24), tta_flip=False,
                 vocab_keys=None, center=False):
        cfg = load_cfg("data")
        self.cfg, self.device = cfg, device
        self.art = Articulators(device=device, threads=4)
        self.enc = FrozenEncoder(cfg["cache"]["backbone"], device, cfg["crop_size"])
        self.model = load_islr(islr_path or ART / "checkpoints" / "islr" / "best.pt", device)
        self.bank = PrototypeBank.load(bank_path or ART / "prototypes" / "lexeme_protos.npz")
        self.vocab_idx = None
        if vocab_keys is not None:
            keep = set(vocab_keys)
            self.vocab_idx = np.array([i for i, k in enumerate(self.bank.keys) if k in keep])
        self.decoder, self.decoder_kw = decoder, decoder_kw or {}
        self.composer = get_composer(composer)
        self.tts = get_tts(tts) if tts else None
        if affect == "auto":
            p = ART / "checkpoints" / "face" / "affect_head.pt"
            affect = None
            if p.exists():
                from .heads import AffectHead
                affect = AffectHead().to(device).eval()
                affect.load_state_dict(torch.load(p, map_location=device))
        self.affect = affect
        self.lengths, self.tta_flip, self.center = window_lengths, tta_flip, center

    def lemma_of(self, key):
        return self.bank.lemma.get(key, key)

    # ------------------------------------------------------------------ perception
    def perceive(self, path, timings):
        t = time.perf_counter()
        frames = np.concatenate(list(decode_frames(path, fps=self.cfg["fps"])))
        timings["decode_ms"] = (time.perf_counter() - t) * 1000
        H, W = frames.shape[1:3]
        self.art.reset()
        K, C, V = [], [], []
        t = time.perf_counter()
        for f in frames:
            kp, g, v = self.art.step(f)
            K.append(kp); V.append(v); C.append(self.art.crops(f, g))
        timings["articulators_ms"] = (time.perf_counter() - t) * 1000
        K, C, V = np.stack(K), np.stack(C), np.stack(V)
        t = time.perf_counter()
        x = torch.from_numpy(C).to(self.device)
        feats = self.enc(x).float().cpu().numpy()
        feats_flip = None
        if self.tta_flip:
            from .augment import flip_swap
            feats_flip = self.enc(flip_swap(x)).float().cpu().numpy()
        if self.device == "cuda":
            torch.cuda.synchronize()
        timings["encoder_ms"] = (time.perf_counter() - t) * 1000
        return dict(frames_hw=(H, W), kpts=K, crops=C, valid=V, feats=feats, feats_flip=feats_flip, n=len(frames))

    def embed_fn(self):
        return self.model.embed

    # ------------------------------------------------------------------ recognition
    def recognize(self, per, timings, active_override=None):
        from .articulators import mirror_kpts
        pose = pose_features(per["kpts"])
        act = active_override if active_override is not None else active_mask(per["kpts"], mode="stream")
        PB = self.bank.centered() if self.center else self.bank.P
        P = PB if self.vocab_idx is None else PB[self.vocab_idx]
        keys = self.bank.keys if self.vocab_idx is None else [self.bank.keys[i] for i in self.vocab_idx]
        t = time.perf_counter()
        if per.get("feats_flip") is None:
            wins = window_scores(self.model.embed, per["feats"], pose, per["valid"], self.lengths, 2, P, self.device, center=self.center)
        else:  # test-time mirror augmentation: average full similarity rows of original and mirrored views
            posef = pose_features(mirror_kpts(per["kpts"], per["frames_hw"][1]))
            wa = window_scores(self.model.embed, per["feats"], pose, per["valid"], self.lengths, 2, P, self.device, topk=len(P), center=self.center)
            wb = window_scores(self.model.embed, per["feats_flip"], posef, per["valid"][:, [1, 0, 2]], self.lengths, 2, P,
                               self.device, topk=len(P), center=self.center)
            wins = []
            for w1, w2 in zip(wa, wb):
                full = np.zeros(len(P), np.float32)
                full[w1["idx"]] += w1["sim"]; full[w2["idx"]] += w2["sim"]
                order = np.argsort(-full)[:10]
                wins.append(dict(t0=w1["t0"], t1=w1["t1"], idx=order, sim=full[order] / 2))
        timings["windows_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if self.decoder == "dp":
            seq = dp_segment(wins, per["n"], active=act, **self.decoder_kw)
        elif self.decoder == "dp_rel":
            seq = dp_relative(wins, per["n"], active=act, **self.decoder_kw)
        else:
            seq = greedy_nms(wins, active=act, **self.decoder_kw)
        timings["decode_seq_ms"] = (time.perf_counter() - t) * 1000
        tl = to_timeline(seq, keys, self.lemma_of, fps=self.cfg["fps"])
        return tl, act, wins

    # ------------------------------------------------------------------ full run
    def run(self, path, out_dir=None, write_audio=True):
        path = Path(path)
        timings = {}
        t_all = time.perf_counter()
        per = self.perceive(path, timings)
        timeline, act, wins = self.recognize(per, timings)
        self.last = dict(per=per, active=act, wins=wins)
        fps = self.cfg["fps"]
        sents = []
        for a, b in pause_segments(act, fps) or [(0, per["n"])]:
            gl = [g for g in timeline if a / fps - 0.2 <= (g["t0"] + g["t1"]) / 2 <= b / fps + 0.2]
            nmm = nmm_flags(per["kpts"][a:b], fps)
            nmm["question"] = False
            aff = self.affect.predict(per["feats"][a:b, 2], per["valid"][a:b, 2]) if self.affect else dict(emotion="neutral", intensity=0.0)
            if nmm.get("negation") or nmm.get("affirmation"):  # grammar on the face first: damp affect while NMM active
                aff = dict(aff, intensity=round(aff.get("intensity", 0) * 0.5, 3))
            t = time.perf_counter()
            comp = self.composer.compose(gl, nmm, aff)
            timings["language_ms"] = timings.get("language_ms", 0) + (time.perf_counter() - t) * 1000
            sents.append(dict(t0=round(a / fps, 2), t1=round(b / fps, 2), glosses=[g["lemma"] for g in gl], nmm=nmm,
                              affect=aff, **comp))
        text = " ".join(s["sentence_th"] for s in sents if s["sentence_th"])
        result = dict(video=str(path), n_frames=per["n"], fps=fps, active_frac=float(act.mean()),
                      valid_rate=per["valid"].mean(0).round(3).tolist(), gloss_timeline=timeline, sentences=sents,
                      text=text, timings_ms={k: round(v, 1) for k, v in timings.items()})
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            if self.tts and write_audio and text:
                t = time.perf_counter()
                emo = sents[0]["affect"] if sents else {}
                wav, sr = self.tts.synthesize(text, SpeechStyle.from_affect(emo.get("emotion", "neutral"), emo.get("intensity", 0)))
                write_wav(out / "audio.wav", wav, sr)
                result["timings_ms"]["tts_ms"] = round((time.perf_counter() - t) * 1000, 1)
            result["timings_ms"]["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
            result["rss_mb"] = round(rss_mb(), 1)
            write_json(out / "result.json", result)
            with open(out / "subtitle.srt", "w", encoding="utf-8") as f:
                for i, g in enumerate(timeline, 1):
                    f.write(f"{i}\n{_srt(g['t0'])} --> {_srt(g['t1'])}\n{g['lemma']} ({g['conf']:.2f})\n\n")
        return result


def _srt(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


class StreamPipeline:
    """Webcam-mode perception loop: per-frame pose → crops → ViT, ring buffer of 48 frames, incremental
    window scoring every `every` frames. `simulate(path)` replays a file frame by frame to measure latency."""

    def __init__(self, vp: VideoPipeline, ring=48, every=2, lengths=(12, 20)):
        self.vp, self.ring, self.every, self.lengths = vp, ring, every, lengths

    def simulate(self, path):
        vp = self.vp
        buf = collections.deque(maxlen=self.ring)
        lat, rss = [], []
        vp.art.reset()
        P = torch.as_tensor(vp.bank.P, device=vp.device)
        events = []
        i = 0
        for chunk in decode_frames(path, fps=12.5, chunk=1):
            f = chunk[0]
            t = time.perf_counter()
            kp, g, v = vp.art.step(f)
            cr = torch.from_numpy(vp.art.crops(f, g)).to(vp.device)
            feat = vp.enc(cr[None])[0].float().cpu().numpy()
            buf.append((feat, kp, v))
            if i % self.every == 0 and len(buf) >= min(self.lengths):
                feats = np.stack([b[0] for b in buf]); kps = np.stack([b[1] for b in buf]); vs = np.stack([b[2] for b in buf])
                pose = pose_features(kps)
                wins = window_scores(vp.model.embed, feats[-max(self.lengths):], pose[-max(self.lengths):],
                                     vs[-max(self.lengths):], self.lengths, 1000, vp.bank.P, vp.device, topk=1)
                if wins:
                    best = max(wins, key=lambda w: w["sim"][0])
                    events.append((i, vp.bank.keys[best["idx"][0]], float(best["sim"][0])))
            if vp.device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t) * 1000)
            rss.append(rss_mb())
            i += 1
        return dict(frames=i, p50_ms=pct(lat[5:], 50), p95_ms=pct(lat[5:], 95), max_ms=float(np.max(lat[5:])) if len(lat) > 5 else None,
                    rss_start_mb=rss[5] if len(rss) > 5 else None, rss_end_mb=rss[-1] if rss else None, n_events=len(events))
