"""v11 Path B front end (prompt-opt-v11 §2D / §3A.4 S1): the NEW landmark encoder is the only backbone.

    pb = PathB(encoder_ckpt, boundary_ckpt, null_Z)        # torch on CPU (evaluation harness) — production would export ONNX
    st = pb.front(kpi, sc, present)                          # frame states H (new encoder) + boundary-head probabilities in the v10
                                                             #   tagger's classes, so the existing semi-Markov decoder runs unchanged
    E  = pb.span(H, windows)                                 # window embeddings = the encoder's own pooling over the utterance's
                                                             #   contextual frame states (one pass, like v10's span head)
SignPipeline.back() switches to this front end when the state carries "pathb" (spotting.json "pathb"); the window table is then scored
against the landmark-encoder bank (bank.lm), the null competitor is the mean "not a sign" embedding of the encoder's training windows,
and the signing-gate veto of v10 stays on top (its frame probabilities come from the cached v10 front pass when present).
"""
from __future__ import annotations

import numpy as np


class PathB:
    """mode: "S1" boundary head → decoder · "S2" prototype-CTC alone (greedy runs = segments, CTC words) · "S3" CTC proposals ∪ boundary
    head (frame P(sign) = max of both) → decoder → span retrieval (+ verifier in the decision layer)"""

    def __init__(self, enc_ckpt, boundary_ckpt, null_proto, device="cpu", gate_veto=True, ctc_ckpt=None, mode="S1"):
        import torch
        from .boundary import build
        from .lmenc import load_model
        self.torch = torch
        self.enc = load_model(enc_ckpt, device)
        self.head = None
        if boundary_ckpt is not None and mode in ("S1", "S3"):
            ck = torch.load(boundary_ckpt, map_location=device, weights_only=False)
            self.head = build(**ck["cfg"]).to(device); self.head.load_state_dict(ck["state"]); self.head.eval()
        self.ctc = None
        if ctc_ckpt is not None and mode in ("S2", "S3"):
            from .protoctc import build as build_ctc
            ck = torch.load(ctc_ckpt, map_location=device, weights_only=False)
            enc2 = load_model(enc_ckpt, device)
            P = ck["state"]["P"].cpu().numpy()
            self.ctc = build_ctc(enc2, P, np.array(ck["form_word"]), len(ck["words"])).to(device)
            self.ctc.load_state_dict(ck["state"]); self.ctc.eval()
            self.ctc_words = ck["words"]
            if ck.get("encoder_changed"):          # prototype-CTC fine-tuned the encoder: its frame states / spans are the ones to use
                self.enc = self.ctc.enc
        self.null = np.asarray(null_proto, np.float32)
        self.dev, self.mode, self.gate_veto = device, mode, gate_veto

    def ctc_logits(self, kpi, sc):
        from .lmenc import feats_fn
        cfg = self.enc.cfg["members"][0] if "members" in self.enc.cfg else self.enc.cfg
        f = feats_fn(cfg.get("fv", 10))(kpi, sc)
        with self.torch.no_grad():
            lg, _ = self.ctc(self.torch.from_numpy(f)[None].to(self.dev), self.torch.ones(1, len(f), device=self.dev))
        return lg[0].float().cpu().numpy()

    def front(self, kpi, sc, present):
        from .boundary import as_tagger_prob
        from .lmenc import frames_long
        from .segment import kin_features
        H = frames_long(self.enc, kpi, sc, device=self.dev)
        prob = None
        if self.head is not None:
            X = np.concatenate([H, kin_features(kpi, sc)], 1).astype(np.float32)
            with self.torch.no_grad():
                p4 = self.head(self.torch.from_numpy(X)[None].to(self.dev), self.torch.ones(1, len(X), device=self.dev))[-1][0].softmax(-1).cpu().numpy()
            prob = as_tagger_prob(p4)
        if self.ctc is not None:
            lg = self.ctc_logits(kpi, sc)
            e = np.exp(lg - lg.max(1, keepdims=True)); e /= e.sum(1, keepdims=True)
            p_sign = 1.0 - e[:, 0]
            a = lg.argmax(1)
            begin = np.r_[a[0] != 0, (a[1:] != 0) & (a[1:] != a[:-1])].astype(np.float32) * p_sign
            pc = np.stack([1 - p_sign, begin, np.clip(p_sign - begin, 0, 1)], 1).astype(np.float32)
            self._ctc_cache = lg
            prob = pc if prob is None else np.where((pc[:, 0] < prob[:, 0])[:, None], pc, prob)       # S3: union of the proposals
        prob[~present] = [1.0, 0.0, 0.0]
        return H, prob

    def ctc_segments(self, concepts, min_len=3):
        """S2: greedy CTC runs → segment dicts with CTC candidates (mean frame softmax over the run)"""
        from .protoctc import greedy
        lg = self._ctc_cache
        e = np.exp(lg - lg.max(1, keepdims=True)); e /= e.sum(1, keepdims=True)
        out = []
        for f0, f1, w in greedy(lg, min_len):
            pm = e[f0:f1, 1:].mean(0)
            o = np.argsort(-pm)[:5]
            cands = [dict(word=self.ctc_words[int(k)], score=round(float(pm[k]), 4), support=round(float(pm[k]), 4)) for k in o]
            out.append(dict(f0=int(f0), f1=int(f1), nearest=cands[0]["word"], candidates=cands, score=float(pm[o[0]]),
                            margin=float(pm[o[0]] - pm[o[1]]), z=0.0, null_gap=float(e[f0:f1, 0].mean() - pm[o[0]]), p_sign=float(1 - e[f0:f1, 0].mean()),
                            agree=1.0, support_ratio=float(pm[o[0]] / max(pm[o[1]], 1e-6)), merged=1, depth=0.0))
        return out

    def span(self, H, wins):
        if not len(wins):
            return np.zeros((0, self.enc.emb_dim), np.float32)
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(wins), 512):
                ws = wins[i:i + 512]
                L = max(b - a for a, b in ws)
                x = np.zeros((len(ws), L, H.shape[1]), np.float32); m = np.zeros((len(ws), L), np.float32)
                for k, (a, b) in enumerate(ws):
                    x[k, :b - a] = H[a:b]; m[k, :b - a] = 1
                out.append(self.enc.pool(torch.from_numpy(x).to(self.dev), torch.from_numpy(m).to(self.dev)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
