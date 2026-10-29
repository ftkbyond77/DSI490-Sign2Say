"""End-to-end inference (v7): MediaPipe landmarks → Standard Schema → right-hand-dominant canonical form → tagger (sign / rest /
boundary) → tempo normalisation → window spotting against the vocabulary bank with a null competitor → semi-Markov decoding →
calibrated status + transition / repeat filters → TSL word-order re-rank → face cues → evidence-guarded Thai sentence (+ speech).

Inputs: a video file (MediaPipe runs here, CPU) or the landmarks the web client already extracted in the browser — the same
MediaPipe Holistic model, so both paths are identical from the Standard Schema on. Everything is loaded from models/
(models/README.md). Backend 'onnx' needs only NumPy + ONNX Runtime (the web API); 'torch' uses models/*.pt (development).

Every detected sign segment is reported, recognised or not:
  {t0, t1, status: accepted | uncertain | unknown | transition, p_correct, nearest, candidates: [{word, score, support}]}
Vocabulary: 'conversation' (default — the daily-conversation lexicon in models/vocab_conversation.json) or 'full' (every concept).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .decode import (LogReg, active_from_prob, apply_csls, candidate_probs, class_features, concept_prior, decide, decode, merge_split,
                     postprocess, rerank_candidates, seg_features, split_long)
from .mediapipe_pose import extract, from_browser, to_standard
from .parts import prep, to_iso
from .runtime import Bank, make_backend, variant_lut, rescore, rescore_windows, table_from_Z, window_table
from .spanhead import SpanHeadNP
from .schema import LH, RH, SH_L, SH_R, WR_L, WR_R, canonical_hands


def rest_trim_mask(kp, sc, hw, pre_s=0.5, post_s=0.4, fps=25.0, rule="live"):
    """pitty: the web live mode's utterance window applied to any landmark sequence — frames from `pre_s` before the hands first
    come up to `post_s` after they were last up (webapp/web: handsUp, LIVE_PREROLL_MS = 500, lastUp + 400 ms). "Up" = a hand's
    21 points above shoulder-line + 1.5 shoulder widths, or a body wrist above + 1.3 (the same rule as the browser). No up frame →
    everything is kept.
    rule="hands": the window is where a hand's 21 points are detected at all — a seated signer's hands never drop below the
    height line, and a hand that leaves / enters the bottom of the frame leaves the body wrist at a jumping, invalid position."""
    T = len(kp)
    if rule == "hands":
        det = (sc[:, LH].mean(1) > .3) | (sc[:, RH].mean(1) > .3)
        idx = np.flatnonzero(det)
        if not len(idx):
            return np.ones(T, bool)
        a, b = max(0, idx[0] - int(round(pre_s * fps))), min(T, idx[-1] + 1 + int(round(post_s * fps)))
        keep = np.zeros(T, bool); keep[a:b] = True
        return keep
    p = to_iso(kp, hw)
    ok_s = (sc[:, SH_L] > .3) & (sc[:, SH_R] > .3)
    if T == 0 or ok_s.sum() < 2:
        return np.ones(T, bool)
    sw = max(float(np.median(np.linalg.norm(p[ok_s, SH_L] - p[ok_s, SH_R], axis=1))), 1e-3)
    mid = (p[:, SH_L, 1] + p[:, SH_R, 1]) / 2
    mid = np.where(ok_s, mid, float(np.median(mid[ok_s])))
    up = np.zeros(T, bool)
    for hand in (LH, RH):
        up |= (sc[:, hand].mean(1) > .3) & (p[:, hand, 1].mean(1) < mid + 1.5 * sw)
    for wr in (WR_L, WR_R):
        up |= (sc[:, wr] > .5) & (p[:, wr, 1] < mid + 1.3 * sw)
    idx = np.flatnonzero(up)
    if not len(idx):
        return np.ones(T, bool)
    a, b = max(0, idx[0] - int(round(pre_s * fps))), min(T, idx[-1] + 1 + int(round(post_s * fps)))
    keep = np.zeros(T, bool); keep[a:b] = True
    return keep
from .segment import decode_tagger, kin_features, resample, window_list
from .utils import MODEL, get_logger, read_json, write_json
from .fingerspell import EN_LETTERS, NONE, TH_LETTERS, load_speller, spell_runs, window_features
from .handshape import HandBank, descriptor, fuse

log = get_logger("pipeline")
NAME_SIGNS = {"ชื่อ", "ชื่อเล่น", "นามสกุล"}       # signs after which a spelled name is expected


def merge_repeats(segs, max_gap=12):
    """Neighbouring sign segments with the same word (one long sign cut in two — live tests showed "ฉัน กิน กิน ขนมปัง") → one
    segment spanning both, with the better confidence. Transitions between them (≤ max_gap frames) are absorbed."""
    out = []
    for s in segs:
        if s["status"] == "transition":
            out.append(s); continue
        j = len(out) - 1
        while j >= 0 and out[j]["status"] == "transition" and s["f0"] - out[j]["f1"] <= max_gap:
            j -= 1
        prev = out[j] if j >= 0 else None
        if prev is not None and prev["status"] != "transition" and prev["nearest"] == s["nearest"] and s["f0"] - prev["f1"] <= max_gap:
            del out[j + 1:]                                        # drop the transitions in between
            keep = prev if prev.get("p_correct", 0) >= s.get("p_correct", 0) else s
            merged = dict(keep, f0=prev["f0"], f1=s["f1"], t0=prev.get("t0"), t1=s.get("t1"))
            out[j] = merged
        else:
            out.append(s)
    return out


def lm_bank(path, n_concepts, allowed=None, fold=None):
    """lm_bank.npz (Z fp16, c, ok) → a runtime.Bank of landmark-encoder embeddings (clips the pose store lacked are skipped)."""
    z = np.load(path)
    ok = z["ok"].astype(bool)
    c = z["c"] if fold is None else fold[z["c"]]
    return Bank(z["Z"][ok].astype(np.float32), c[ok], n_concepts, allowed)


class SignPipeline:
    def __init__(self, model_dir=MODEL, backend="onnx", device="cpu", vocab="conversation", use_llm=True, use_face=True, bank_file=None):
        """bank_file: another bank (e.g. artifacts/eval/bank_eval.npz = training signers only, for honest evaluation)."""
        md = Path(model_dir)
        self.md = md
        self.cfg = read_json(md / "config.json")
        self.be = make_backend(md, backend, device, **({} if backend == "onnx" else {"tagger_arch": self.cfg.get("tagger_arch", "gru")}))
        self.concepts = read_json(md / "concepts.json")
        b = np.load(bank_file or md / "bank.npz", allow_pickle=False)
        self.vocab = vocab
        allowed, self.fold = None, None
        if vocab == "conversation" and (md / "vocab_conversation.json").exists():
            vc = read_json(md / "vocab_conversation.json")
            cidx = {c: i for i, c in enumerate(self.concepts)}
            allowed = {cidx[c] for c in vc["concepts"] if c in cidx}
            self.fold = variant_lut(self.concepts, vc.get("variants", {}))
        lab = self.fold[b["c"]] if self.fold is not None else b["c"]
        self.bank = Bank(b["Z"], lab, len(self.concepts), allowed)
        # v8.1 hand-shape cue: descriptors of the same bank's clips (models/hand_bank.npz, or hand_<bank_file name> next to it)
        hf = Path(bank_file).with_name("hand_" + Path(bank_file).name) if bank_file else md / "hand_bank.npz"
        self.bank.hand = HandBank.load(hf, len(self.concepts), allowed, self.fold) if hf.exists() else None
        # v9 second opinion: the Thai-only landmark encoder (modules/lmenc.py) and its bank (lm_bank.npz, or lm_<bank_file name>)
        lf = Path(bank_file).with_name("lm_" + Path(bank_file).name) if bank_file else md / "lm_bank.npz"
        self.lm = None
        if lf.exists() and (md / "onnx" / "lmenc.onnx").exists():
            from .lmenc import OnnxLM
            self.lm = OnnxLM(md / "onnx" / "lmenc.onnx")
            self.bank.lm = lm_bank(lf, len(self.concepts), allowed, self.fold)
        else:
            self.bank.lm = None
        self.null_proto = np.load(md / "null_proto.npy").astype(np.float32)
        spot = read_json(md / "spotting.json")
        self.spot_all = spot
        self.spot = spot.get(vocab, spot) if isinstance(spot.get(vocab), dict) else spot
        cal = read_json(md / "segment_calibration.json")
        self.cal_all = cal
        self.cal = cal.get(vocab, cal) if isinstance(cal.get(vocab), dict) else cal
        self.lr = LogReg.from_dict(self.cal["model"])
        # a word is shown / spoken / used in the sentence only when its calibrated P(correct) ≥ accept_min (users: "80+ or don't guess")
        self.cal = dict(self.cal, tau_accept=max(float(self.cal.get("tau_accept", 0.0)), float(spot.get("accept_min", 0.8))))
        cp = md / "class_prior.npz"            # v8: per-word reliability inside the bank (scripts/decision.py), a calibration feature
        self.class_prior = dict(np.load(cp)) if cp.exists() else None
        pf = md / "freq_prior.npy"
        prior = np.load(pf).astype(np.float32) if pf.exists() else None
        self.word_prior = concept_prior(prior, None, self.spot.get("freq_gamma", 0.0), 0.0)
        hf = md / "hub.npy"
        self.hub = np.load(hf).astype(np.float32) if hf.exists() else None
        self.prior = prior
        self.span = SpanHeadNP(md / "span_head.npz") if (md / "span_head.npz").exists() else None
        self.grammar = read_json(md / "grammar.json") if (md / "grammar.json").exists() else {}
        self.roles = read_json(md / "roles.json") if (md / "roles.json").exists() else {}
        self.lang_roles = read_json(md / "roles_language.json") if (md / "roles_language.json").exists() else self.roles
        self.use_llm, self.use_face = use_llm, use_face
        # v9 face emotion model (a separate folder, models/face-model): THAISLM_FACE_MODEL, else <models>/face-model next to this one
        import os
        fdir = Path(os.environ.get("THAISLM_FACE_MODEL", md.parent / "face-model"))
        self.face_model = None
        if use_face and (fdir / "face_emotion.npz").exists():
            from .face_model import FaceEmotion
            self.face_model = FaceEmotion.load(fdir)
        self.collect_pairs = None             # tuning only: a list that receives the merge-pair features (scripts/decision.py)
        self.collect_splits = None            # tuning only: the same for the split test's candidate cuts
        # fingerspelling: letter recogniser + letter bank (second opinion through the sign encoder); settings in spotting.json "spell"
        self.speller = load_speller(md)
        lb = md / "letters_bank.npz"
        if lb.exists():
            z = np.load(lb, allow_pickle=False)
            self.lbank = (z["Z"].astype(np.float32), np.array([str(x) for x in z["letter"]]))
        else:
            self.lbank = None
        self.spell_cfg = dict(dict(thr=0.55, accept=0.8, single_min=0.85, run_gap=15, w_bank=0.5, run_min=0.5), **spot.get("spell", {}))

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def landmarks_from_video(video, progress=None):
        return extract(video, fps=25.0, progress=progress, normalize=False)   # pitty: whole-frame exposure normalisation doubled missed hands on data_test (pitty/notebooks/pitty_phase1_hands.ipynb)

    @staticmethod
    def landmarks_from_browser(frames, hw):
        return from_browser(frames, hw)

    # ------------------------------------------------------------------ steps
    def _tag(self, kpi, sc, present):
        """One encoder pass over the whole utterance → contextual frame states H (reused by the span head) → tagger."""
        H = self.be.frames(kpi, sc)
        x = np.concatenate([H, kin_features(kpi, sc)], 1).astype(np.float32)
        prob = self.be.tag(x)
        prob[~present] = [1.0, 0.0, 0.0]
        return prob, H

    def user_bank(self, entries):
        """Personal sign memory: [(word, embedding[768])] saved by one user (the web client keeps them) → a Bank for that user's
        requests only = this vocabulary's bank + those embeddings. The shared bank is never modified."""
        # one prototype per word (the mean of that word's saved examples): on held-out TSL51 test sentences of an unseen signer,
        # a 2-example prototype per word raised recall@1 0.65 → 0.80 and accepted-correct words 141 → 178 with fewer accepted
        # errors (14 → 12); separate entries per example instead let the person's own style inflate every enrolled word (59 errors).
        cidx = {c: i for i, c in enumerate(self.concepts)}
        groups = {}
        for w, z in entries or []:
            z = np.asarray(z, np.float32).reshape(-1)
            if w in cidx and z.shape == (768,) and np.isfinite(z).all():
                groups.setdefault(w, []).append(z / (np.linalg.norm(z) + 1e-8))
        if not groups:
            return self.bank
        Z, c = [], []
        for w, zs in groups.items():
            m = np.mean(zs[-3:], 0)
            Z.append(m / (np.linalg.norm(m) + 1e-8)); c.append(cidx[w])
        ub = Bank(np.r_[self.bank.Z, np.stack(Z)], np.r_[self.bank.c, np.array(c)], len(self.concepts))
        ub.hand, ub.lm, ub.personal = self.bank.hand, getattr(self.bank, "lm", None), True
        return ub

    def _spot(self, kpi, sc, prob, present, H, bank=None):
        """Windows inside signing → embeddings (span head on H: one pass; or the encoder per window when no head is shipped) →
        decoder → exact re-scoring of the chosen segments. With spotting.json "user_bank_rank_only", a personal bank (bank ≠ the
        shared one) only picks the word: where the signs are is decided on the shared bank alone."""
        T = len(kpi)
        bank = bank or self.bank
        seg_bank = self.bank if self.spot_all.get("user_bank_rank_only") else bank
        act, _ = active_from_prob(prob, self.spot["act_thr"], present, self.spot.get("act_off"))
        wins = window_list(T, self.spot["lens"], self.spot["stride"], act=act, min_active=self.spot.get("min_inside", 0.6))
        if self.span is not None:
            Zw = self.span(H, wins)
            table = table_from_Z(Zw, wins, seg_bank, self.null_proto)
            rtable = table if seg_bank is bank else table_from_Z(Zw, wins, bank, self.null_proto)
        else:
            table = window_table(self.be, kpi, sc, wins, seg_bank, self.null_proto, prep_fn=prep)
            rtable = table if seg_bank is bank else window_table(self.be, kpi, sc, wins, bank, self.null_proto, prep_fn=prep)
        seg_t = apply_csls(table, self.hub, self.spot.get("csls_alpha", 0.0))
        rank_t = apply_csls(rtable, self.hub, self.spot.get("csls_alpha", 0.0), self.word_prior, 1.0)
        segs = decode(seg_t, prob, self.concepts, self.spot, present, rank_table=rank_t)
        if segs and rtable is not table and (self.spot_all.get("user_bank_neighbour_guard") or self.spot_all.get("user_bank_topk")):
            # pitty: the personal bank may not turn a sign into its neighbour's word (live: "คุณ ฉัน" → "ฉัน ฉัน" → merged "ฉัน")
            base = decode(seg_t, prob, self.concepts, self.spot, present,
                          rank_table=apply_csls(table, self.hub, self.spot.get("csls_alpha", 0.0), self.word_prior, 1.0))
            if [(s["f0"], s["f1"]) for s in base] == [(s["f0"], s["f1"]) for s in segs]:
                words = [s["nearest"] for s in segs]
                for i, (s, b) in enumerate(zip(segs, base)):
                    near = set(words[max(0, i - 1):i] + words[i + 1:i + 2])
                    k = self.spot_all.get("user_bank_topk")      # pitty: … and may only pick a word the shared bank already had in its top-k
                    off = bool(k) and s["nearest"] not in [c["word"] for c in b["candidates"][:k]]
                    nb = bool(self.spot_all.get("user_bank_neighbour_guard")) and s["nearest"] in near
                    if s["nearest"] != b["nearest"] and (nb or off):
                        s.update(nearest=b["nearest"], candidates=b["candidates"])
        if segs and self.spot.get("merge") and self.span is not None:      # v8: one sign cut in two → one segment (decode.merge_split)
            segs = merge_split(segs, lambda w: self.span(H, w), bank, self.null_proto, self.concepts, prob, rank_t, self.spot,
                               speed=kin_features(kpi, sc)[:, 12], collect=self.collect_pairs)
        if segs and self.spot.get("split") and self.span is not None:      # v8.1: two fluent signs read as one → two segments
            segs = split_long(segs, lambda w: self.span(H, w), bank, self.null_proto, self.concepts, prob, rank_t, self.spot,
                              speed=kin_features(kpi, sc)[:, 12], collect=self.collect_splits)
        if segs and self.spot.get("exact_rescore", True) and self.span is not None:
            ws = rescore_windows(segs)
            flat = [w for x in ws for w in x]
            Z = self.be.embed([prep(kpi[a:b], sc[a:b]) for a, b in flat])
            k, Zs = 0, []
            for x in ws:
                Zs.append(Z[k:k + len(x)]); k += len(x)
            rescore(segs, Zs, bank, self.null_proto, self.concepts, self.prior, self.spot.get("freq_gamma", 0.0))
        return segs, act, len(wins)

    def tempo_factor(self, prob):
        tc = self.spot.get("tempo")
        if not tc:
            return 1.0
        segs = decode_tagger(prob)
        if len(segs) < tc.get("min_segments", 2):
            return 1.0
        med = float(np.median([b - a for a, b in segs]))
        f = float(np.clip(tc["ref_len"] / max(med, 1.0), 1.0, tc.get("max_factor", 2.5)))
        return f if f >= tc.get("min_factor", 1.2) else 1.0

    def standardise(self, raw):
        d = to_standard(raw, gap_s=float(self.spot_all.get("hand_gap_s", 0.2)))     # live webcams drop hands for longer
        kp, sc, hw = d["kp"], d["sc"], tuple(int(x) for x in d["hw"])
        present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
        sc = sc.copy(); sc[~present] = 0.0; kp = kp.copy(); kp[~present] = 0.0
        kp, sc, mirrored = canonical_hands(kp, sc, hw)
        return kp, sc, hw, present, mirrored

    def _spell(self, kpi, sc, H, segs, alphabet, user_letters=None, spell_only=False):
        """Letters → letter segments that replace the overlapping word segments a word was not confidently recognised for.
        alphabet: "en" (A–Z), "th" (Thai consonants + vowels), "both" or "off"."""
        if alphabet == "off" or self.speller is None or len(kpi) < 8:
            return segs
        cfg = self.spell_cfg
        if spell_only:            # the user said "this utterance is spelling": every confident handshape is a letter, words are dropped
            cfg = dict(cfg, thr=0.3, single_min=0.0, run_min=0.0, run_gap=10 ** 6)
        allowed = set(EN_LETTERS if alphabet == "en" else TH_LETTERS if alphabet == "th" else EN_LETTERS + TH_LETTERS)
        X, starts, keep = window_features(kpi, sc)
        if not keep.any():
            return segs
        P = np.zeros((len(X), len(self.speller.classes)), np.float32)
        P[keep] = self.speller.proba(X[keep])
        events = spell_runs(P, starts, keep, self.speller.classes, allowed, thr=cfg["thr"])
        if not events:
            return segs
        cls = self.speller.classes
        idx = [i for i, c in enumerate(cls) if c in allowed]
        names = [cls[i] for i in idx]
        if self.lbank is not None and H is not None and self.span is not None:
            E = self.span(H, [(e["f0"], min(len(kpi), e["f1"])) for e in events])
            E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
            Z, lab = self.lbank
            if user_letters:                         # this user's confirmed letters: one averaged prototype per letter
                groups = {}
                for w, z in user_letters:
                    groups.setdefault(w, []).append(np.asarray(z, np.float32) / (np.linalg.norm(z) + 1e-8))
                UZ = np.stack([np.mean(v[-3:], 0) for v in groups.values()])
                Z = np.r_[Z, UZ / (np.linalg.norm(UZ, axis=1, keepdims=True) + 1e-8)]
                lab = np.r_[lab, np.array(list(groups))]
        for k, e in enumerate(events):
            w = (starts >= e["f0"] - 1) & (starts <= e["f1"] - 8 + 1) & keep
            pc = P[w][:, idx].mean(0) if w.any() else P[keep][:, idx].mean(0)
            pc = pc / (pc.sum() + 1e-8)
            pb = None
            if self.lbank is not None and H is not None and self.span is not None:
                sim = Z @ E[k]
                top = np.array([np.sort(sim[lab == n])[-2:].mean() if (lab == n).any() else -1.0 for n in names])
                pb = np.exp((top - top.max()) / 0.03); pb /= pb.sum()
            pm = pc if pb is None else np.exp((1 - cfg["w_bank"]) * np.log(pc + 1e-6) + cfg["w_bank"] * np.log(pb + 1e-6))
            pm /= pm.sum()
            o = np.argsort(-pm)[:5]
            e.update(letter=names[o[0]], p=float(pm[o[0]]), cands=[dict(word=names[j], score=float(pm[j])) for j in o])
            if self.lbank is not None and H is not None and self.span is not None:
                e["emb"] = E[k].astype(np.float32)
        # letter runs: neighbouring letters (gap ≤ run_gap frames) belong to one spelled word
        runs, cur = [], [events[0]]
        for e in events[1:]:
            if e["f0"] - cur[-1]["f1"] <= cfg["run_gap"]:
                cur.append(e)
            else:
                runs.append(cur); cur = [e]
        runs.append(cur)
        out = list(segs)
        for run in runs:
            ps = [e["p"] for e in run]
            if len(run) == 1 and ps[0] < cfg["single_min"]:
                # v9: a lone letter right after a name sign ("ชื่อ J") is how a name is spelled — lower bar there only
                prev = [s for s in out if s["status"] != "transition" and s.get("kind") != "letter" and s["f1"] <= run[0]["f0"] + 4
                        and run[0]["f0"] - s["f1"] <= cfg.get("name_gap", 50)]
                if not (prev and prev[-1]["nearest"] in NAME_SIGNS and ps[0] >= cfg.get("single_after_name", 2.0)):
                    continue                                # one weak handshape inside word signing is not spelling
            if float(np.mean(ps)) < cfg.get("run_min", 0.0):
                continue
            a, b = run[0]["f0"], run[-1]["f1"]
            over = [s for s in out if s["status"] != "transition" and min(s["f1"], b) - max(s["f0"], a) > 0.5 * max(1, s["f1"] - s["f0"])]
            if not spell_only and any(s["status"] == "accepted" and s.get("p_correct", 0) >= float(np.mean(ps)) for s in over):
                continue                                    # a confidently recognised word wins over the letters
            out = [s for s in out if s not in over]
            for e in run:
                out.append(dict(f0=int(e["f0"]), f1=int(e["f1"]), t0=round(e["f0"] / 25.0, 2), t1=round(e["f1"] / 25.0, 2), kind="letter",
                                nearest=e["letter"], candidates=e["cands"], p_correct=e["p"],
                                # a lone letter inside word signing is never "confident": spelled names have ≥ 2 letters
                                status="accepted" if (e["p"] >= cfg["accept"] and len(run) >= 2) else "uncertain",
                                **({"emb": e["emb"]} if "emb" in e else {})))
        if spell_only:
            out = [s for s in out if s.get("kind") == "letter" or s["status"] == "transition"]
        return sorted(out, key=lambda s: s["f0"])

    def front(self, kp, sc, hw, present):
        """The expensive part (encoder frame pass + tagger, again on a time-stretched copy for fast signing) → a state dict that
        back() turns into segments. Split so tuning can cache it (scripts/decision.py); production calls both in a row."""
        kpi = to_iso(kp, hw)
        T = len(kpi)
        if T < 8:
            return dict(kpi=kpi, sc=sc, present=present, prob=np.tile([1.0, 0.0, 0.0], (T, 1)), H=None, f=1.0)
        prob, H = self._tag(kpi, sc, present)
        st = dict(kpi=kpi, sc=sc, present=present, prob=prob, H=H, f=self.tempo_factor(prob))
        if st["f"] > 1.0:       # fast signing: time-stretch to the trained pace, spot there, map frames back
            f = st["f"]
            k2, s2 = resample(kpi, sc, 1.0 / f)
            p2 = present[np.clip(np.round(np.arange(len(k2)) * f).astype(int), 0, T - 1)]
            prob2, H2 = self._tag(k2, s2, p2)
            st.update(k2=k2, s2=s2, p2=p2, prob2=prob2, H2=H2)
        return st

    def back(self, st, bank=None, raw=False):
        """front() state → sign segments with calibrated status (spotting · split-sign merge · calibration · decision · transition /
        repeat filters · grammar re-rank · repeat merge). raw=True stops after spotting (tuning re-runs decide() many times).
        v9: a personal bank (the user's own confirmed signs) runs with the "enrolled" settings of spotting.json /
        segment_calibration.json when the model ships them (v9: v8.1's — on tune sentences of an enrolled signer the v9 settings
        did not beat them, and the landmark encoder has no personal examples to add)."""
        bank = bank or self.bank
        enr = self.spot_all.get("enrolled") if getattr(bank, "personal", False) else None
        if enr:
            keep = (self.spot, self.cal, self.lr)
            self.spot = enr[self.vocab] if self.vocab in enr else enr
            cal = self.cal_all.get("enrolled") or self.cal
            self.cal = dict(cal, tau_accept=max(float(cal.get("tau_accept", 0.0)), float(self.spot_all.get("accept_min", 0.8))))
            self.lr = LogReg.from_dict(self.cal["model"])
            try:
                return self._back(st, bank, raw)
            finally:
                self.spot, self.cal, self.lr = keep
        return self._back(st, bank, raw)

    def _back(self, st, bank, raw=False):
        prob, present, T = st["prob"], st["present"], len(st["prob"])
        f = st["f"]
        if f > 1.0:
            spot0, tc = self.spot, self.spot.get("tempo") or {}
            if tc.get("max_len"):   # pitty: fast signing — shorter windows only, so one segment does not swallow two signs
                self.spot = dict(self.spot, lens=[l for l in spot0["lens"] if l <= tc["max_len"]])
            if tc.get("split"):     # pitty: fast signing — split segments that hold two signs (split_long, cfg "split")
                self.spot = dict(self.spot, split=tc["split"])
            try:
                segs, _, n_win = self._spot(st["k2"], st["s2"], st["prob2"], st["p2"], st["H2"], bank)
            finally:
                self.spot = spot0
            for s in segs:
                s["f0"], s["f1"] = int(round(s["f0"] / f)), max(int(round(s["f1"] / f)), int(round(s["f0"] / f)) + 1)
            act, _ = active_from_prob(prob, self.spot["act_thr"], present, self.spot.get("act_off"))
        elif T >= 8:
            segs, act, n_win = self._spot(st["kpi"], st["sc"], prob, present, st["H"], bank)
        else:
            segs, act, n_win = [], np.zeros(T, bool), 0
        for s in segs:
            s["t0"], s["t1"] = round(s["f0"] / 25.0, 2), round(s["f1"] / 25.0, 2)
        segs = self.hand_rescore(segs, st, bank)
        if raw:
            return segs, act, n_win
        return self.decide(segs, st["sc"]), act, n_win

    def hand_rescore(self, segs, st, bank):
        """(v8.1) Word ranking of every segment = the encoder's reading of the whole segment (span head, every vocabulary word)
        fused with the hand-shape cue (modules/handshape.py) at weight spot["hand_w"] (chosen on tune data). Segmentation is
        untouched; the decoder's reading stays in "window_top1"."""
        w = float(self.spot.get("hand_w", 0.0))
        rd = self.spot.get("read", {})              # v9 reader: multi-crop span reading + the landmark encoder's opinion
        hb = getattr(bank, "hand", None)
        if not segs or not w or hb is None or self.span is None or st.get("H") is None:
            return segs
        spans = [(s["f0"], max(s["f1"], s["f0"] + 1)) for s in segs]
        f = st.get("f", 1.0)
        if f > 1.0 and st.get("H2") is not None:     # fast signing: read the span where the decoder read it (trained pace)
            HH, T2 = st["H2"], len(st["H2"])
            sp = [(min(int(a * f), T2 - 1), min(max(int(round(b * f)), int(a * f) + 1), T2)) for a, b in spans]
        else:
            HH, sp = st["H"], spans
        enc = bank.scores(self.span(HH, sp))
        if rd.get("crops"):     # boundary-robust reading: the span ± a sixth of its length on either side (5 readings averaged)
            T_ = len(HH)
            crops = []
            for a, b in sp:
                d = max(2, (b - a) // 6)
                crops += [(max(0, a - d), min(T_, b + d)), (a + d, max(a + d + 1, b - d)), (a, max(a + 1, b - d)), (min(a + d, b - 1), b)]
            ce = bank.scores(self.span(HH, crops)).reshape(len(sp), 4, -1)
            enc = np.where(enc > -1.5, (enc + ce.sum(1)) / 5, enc)
        hand = hb.scores(np.stack([descriptor(st["kpi"], st["sc"], a, b) for a, b in spans]))
        fused = fuse(enc, hand, w)
        lw, lb = float(rd.get("lm_w", 0.0)), getattr(bank, "lm", None)
        lm_top = None
        if lw and lb is not None and self.lm is not None:
            lm = lb.scores(self.lm.spans(st["kpi"], st["sc"], spans))
            fused = fuse(fused, lm, lw)
            lm_top = lm.argmax(1)
        for k, (s, e, h, f) in enumerate(zip(segs, enc, hand, fused)):
            if lm_top is not None:
                s["lm_agree"] = float(int(lm_top[k]) == int(np.argmax(f)))
            o = np.argsort(-f)[:5]
            hv = h > -1.5
            zh = (h - h[hv].mean()) / (h[hv].std() + 1e-6) if hv.sum() > 2 else np.zeros_like(h)
            s["candidates"] = [dict(word=self.concepts[int(j)], score=round(float(e[j]), 4), support=round(float(f[j]), 4)) for j in o]
            s.update(nearest=self.concepts[int(o[0])], score=float(e[o[0]]), margin=float(f[o[0]] - f[o[1]]),
                     support_ratio=float(f[o[0]] / max(f[o[1]], 1e-6)), hand_z=float(zh[o[0]]), hand_agree=float(int(np.argmax(h)) == int(o[0])))
        return segs

    def decide(self, segs, sc=None):
        """Decoded segments → calibrated P(correct) → Accept / Uncertain / Reject (decode.decide) → transition / repeat filters →
        grammar re-rank → repeat merge. sc (landmark scores) enables spotting.json post.min_hand_frac."""
        if self.cal.get("rerank"):              # v8 learned top-5 re-ranker (only shipped when it wins on tune, scripts/decision.py)
            rerank_candidates(segs, self.cal["rerank"]["w"], self.concepts, self.class_prior)
        if self.cal.get("temperature"):
            candidate_probs(segs, self.cal["temperature"])
        class_features(segs, self.concepts, self.class_prior)
        P = self.lr.predict(seg_features(segs, getattr(self.lr, "feats", None))) if segs else np.zeros(0)
        iso = self.cal.get("isotonic")           # v9: monotone re-mapping of P(correct) fitted out-of-fold on tune (flatter ECE)
        if iso and len(P):
            P = np.interp(P, iso["x"], iso["y"])
        decide(segs, P, self.cal)
        postprocess(segs, self.spot)
        mh = self.spot.get("post", {}).get("min_hand_frac", 0.0)
        if mh and segs and sc is not None:   # pitty: live webcams lose the hands (exposure, low fps) — a sign nobody saw the hands of is a transition
            seen = (sc[:, LH].mean(1) > 0.3) | (sc[:, RH].mean(1) > 0.3)
            for s in segs:
                if s["status"] not in ("accepted", "transition") and seen[s["f0"]:max(s["f1"], s["f0"] + 1)].mean() < mh:
                    s["status"] = "transition"
        w = self.spot.get("grammar_rerank_weight", 0.0)
        live = [s for s in segs if s["status"] != "transition"]
        g = self.spot.get("grammar", {})          # v8.1 guards: never a one-sign clip, the ranking's own quantity, a tuned visual gap
        if w and self.grammar and len(live) >= g.get("min_segments", 1):
            from .grammar import rerank
            for s, c in zip(live, rerank(live, self.roles, self.grammar, weight=w, min_gap=g.get("min_gap", 0.03), key=g.get("key", "score"))):
                if c != s["nearest"]:
                    s["visual_top1"] = s["nearest"]; s["nearest"] = c; s["reranked_by_grammar"] = True
                    s["candidates"] = sorted(s["candidates"], key=lambda x: x["word"] != c)
        return merge_repeats(segs)

    def run_landmarks(self, raw, out_dir=None, tts=False, title="", progress=None, face_question=False, face_baseline=None, user_bank=None,
                      alphabet=None, spell_only=False):
        """progress(stage, frac) reports the stages for a progress bar: schema → tagger → spotting → face → language.
        face_question (experimental, off by default): brows furrowed relative to the signer's own resting face at the end of the
        utterance → yes/no question (ไหม). face_baseline = 52 blendshape scores of that resting face (the web client measures it
        while the hands are down); without it the hands-down frames of the utterance are used. user_bank: [(word, embedding)] of this
        user's confirmed signs (`user_bank`); every sign segment returns its embedding ("emb") so the client can save it.
        alphabet None = the model's default (spotting.json "alphabet"; v8.1: "en" — English letters are read on every utterance)."""
        alphabet = self.spot_all.get("alphabet", "off") if alphabet is None else alphabet
        tm = {}
        say = progress or (lambda stage, frac: None)
        say("schema", 0.0)
        t = time.perf_counter()
        kp, sc, hw, present, mirrored = self.standardise(raw)
        tr = self.spot_all.get("trim_rest")
        if tr:      # pitty: uploaded videos get the live-mode utterance window (rest before / after signing would be read as signs)
            keep = rest_trim_mask(kp, sc, hw, tr.get("pre_s", 0.5), tr.get("post_s", 0.4), rule=tr.get("rule", "live"))
            present = present & keep; sc = sc.copy(); sc[~keep] = 0.0; kp = kp.copy(); kp[~keep] = 0.0
        T = len(kp)
        tm["schema_ms"] = (time.perf_counter() - t) * 1000
        say("tagger", 0.15)
        t = time.perf_counter()
        st = self.front(kp, sc, hw, present)
        kpi, H, prob = st["kpi"], st["H"], st["prob"]
        tm["tagger_ms"] = (time.perf_counter() - t) * 1000
        say("spotting", 0.45)
        t = time.perf_counter()
        letters = set(EN_LETTERS) | set(TH_LETTERS)
        is_letter = [w in letters and w not in self.concepts for w, _ in (user_bank or [])]     # A–Z: letter bank; ก–ฮ are bank words too
        user_letters = [e for e, f in zip(user_bank or [], is_letter) if f]
        bank = self.user_bank([e for e, f in zip(user_bank or [], is_letter) if not f])
        segs, act, n_win = self.back(st, bank)
        f = st["f"]
        segs = self._spell(kpi, sc, H, segs, alphabet if not (spell_only and alphabet == "off") else "en", user_letters, spell_only)
        if H is not None and self.span is not None:      # one embedding per sign segment (span head on the frame states: cheap)
            live = [s for s in segs if s["status"] != "transition" and s.get("kind") != "letter"]
            if live:
                E = self.span(H, [(s["f0"], max(s["f1"], s["f0"] + 1)) for s in live])
                for s, e in zip(live, E):
                    s["emb"] = (e / (np.linalg.norm(e) + 1e-8)).astype(np.float32)
        tm["spotting_ms"] = (time.perf_counter() - t) * 1000
        tm["n_windows"] = n_win; tm["tempo_factor"] = round(f, 2)
        face = {}
        cues = tuple(self.spot.get("face_cues", ["negation"]))
        if self.use_face:
            from .face import analyse
            say("face", 0.7)
            t = time.perf_counter()
            face = analyse(raw, [(s["f0"], s["f1"]) for s in segs if s["status"] != "transition"], baseline_bs=face_baseline)
            if face_question:               # experimental: the furrow cue (not validated on held-out signers) marks yes/no questions
                face["question_yesno"] = bool(face.get("question_furrow"))
                cues = cues + ("question_yesno",)
            else:
                face["question_yesno"] = False
            if self.face_model is not None:     # v9: learned face emotion + fusion with the signed words and the signing energy
                from .face_model import fuse_affect, signing_energy
                fr = self.face_model.analyse(raw, [(s["f0"], s["f1"]) for s in segs if s["status"] != "transition"])
                en, en_th = signing_energy(kpi, sc, segs)
                aff = fuse_affect(fr, segs, en, en_th, grammar=face)
                face.update(rule_emotion=face.get("emotion"), emotion=aff["emotion"], emotion_th=aff["emotion_th"], intensity=aff["intensity"],
                            affect=dict(aff, face_only={k: fr.get(k) for k in ("emotion", "probs", "confidence", "evidence", "explanation_th")},
                                        per_segment=fr.get("per_segment", [])))
            tm["face_ms"] = (time.perf_counter() - t) * 1000
        from .language import compose_sentence
        say("language", 0.8)
        t = time.perf_counter()
        fusion = compose_sentence(segs, face, self.grammar, self.lang_roles, use_llm=self.use_llm, face_cues=cues)
        tm["language_ms"] = (time.perf_counter() - t) * 1000
        result = dict(title=str(title), fps=25.0, n_frames=T, duration_s=round(T / 25.0, 2), signer_visible_frac=float(present.mean()) if T else 0.0,
                      signing_frac=float(act.mean()) if T else 0.0, mirrored_to_right_hand=bool(mirrored),
                      segments=[{k: v for k, v in s.items()} for s in segs], face=face, fusion=fusion,
                      sentence=fusion.get("sentence_th") or "", timings_ms={k: round(v, 1) for k, v in tm.items()},
                      model=self.cfg.get("version", ""), vocab=self.vocab)
        if tts and result["sentence"]:
            from .tts import synthesize_wav_bytes
            result["audio_wav"] = synthesize_wav_bytes(result["sentence"], face.get("emotion", "neutral"), face.get("intensity", 0.0))
        if out_dir:
            self.save(result, prob, present, out_dir)
        return result

    def run(self, video, out_dir=None, tts=False, raw=None, progress=None, face_question=False, alphabet=None, spell_only=False):
        t = time.perf_counter()
        if raw is None:
            raw = self.landmarks_from_video(video, progress=(lambda f: progress("mediapipe", f)) if progress else None)
        ms = (time.perf_counter() - t) * 1000
        if raw is None:
            return dict(title=str(video), segments=[], sentence="", error="no frames decoded")
        r = self.run_landmarks(raw, out_dir=out_dir, tts=tts, title=video, progress=progress, face_question=face_question, alphabet=alphabet, spell_only=spell_only)
        r["timings_ms"]["mediapipe_ms"] = round(ms, 1)
        return r

    # ------------------------------------------------------------------ outputs
    def save(self, result, prob, present, out_dir):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        r = {k: v for k, v in result.items() if k != "audio_wav"}
        r["segments"] = [{k: v for k, v in s.items() if k != "emb"} for s in r.get("segments", [])]
        write_json(out / "result.json", r)
        np.save(out / "tagger_prob.npy", prob.astype(np.float16))
        with open(out / "subtitle.srt", "w", encoding="utf-8") as f:
            for i, s in enumerate([s for s in result["segments"] if s["status"] != "transition"], 1):
                lab = s["nearest"] if s["status"] == "accepted" else (f"{s['nearest']}?" if s["status"] == "uncertain" else f"[?] ≈ {s['nearest']}")
                f.write(f"{i}\n{_srt(s['t0'])} --> {_srt(s['t1'])}\n{lab}\n\n")
        timeline(prob, present, result["segments"], out / "timeline.png", result.get("title", ""))
        if result.get("audio_wav"):
            (out / "speech.wav").write_bytes(result["audio_wav"])


def timeline(prob, present, segs, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "Noto Sans Thai", "DejaVu Sans"]
    T = len(prob); t = np.arange(T) / 25.0
    fig, ax = plt.subplots(figsize=(max(8, T / 40), 2.8))
    ax.fill_between(t, 0, (~present).astype(float), color="#bbbbbb", alpha=0.5, step="mid", label="no signer")
    ax.plot(t, prob[:, 1] + prob[:, 2], color="#1f77b4", lw=1.2, label="P(sign)")
    ax.plot(t, prob[:, 1], color="#ff7f0e", lw=0.8, label="P(begin)")
    col = {"accepted": "#2ca02c", "uncertain": "#e3a008", "unknown": "#d62728", "transition": "#9e9e9e"}
    for s in segs:
        ax.axvspan(s["t0"], s["t1"], ymin=0.82, ymax=0.98, color=col[s["status"]], alpha=0.8)
        lab = {"accepted": s["nearest"], "uncertain": f"{s['nearest']}?", "unknown": f"?≈{s['nearest']}", "transition": "·"}[s["status"]]
        ax.text((s["t0"] + s["t1"]) / 2, 1.08, lab, ha="center", fontsize=8, rotation=20)
    ax.set_ylim(0, 1.25); ax.set_xlim(0, max(T, 1) / 25.0); ax.set_xlabel("seconds"); ax.legend(loc="lower right", fontsize=7, ncol=3)
    ax.set_title(Path(str(title)).stem, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _srt(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
