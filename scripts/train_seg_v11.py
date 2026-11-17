"""v11 Path B training (prompt-opt-v11 §3A): stage-3 sentences → boundary head (MS-TCN) and prototype-CTC on the final encoder.

  python main.py train-seg-v11 --enc v11-final --head boundary --run v11-bnd [--minutes 60]
  python main.py train-seg-v11 --enc v11-final --head ctc --run v11-ctc [--minutes 105]
  python main.py train-seg-v11 --enc v11-final --head boundary --smoke 20      # tiny-batch unit run (TPH 2 §2C)

Stage-3 data, re-synthesised every batch (real : synthetic = 1 : 1):
  real       TSL51 sentence clips of the TRAIN role ("tagger_train" — the v10 tagger trained on the same sentences, so S0 and Path B
             share that bias on the TSL51 TUNE sets; the fluent and isolated TUNE sets are unseen people) with their aligned words;
  synthetic  pitty FluentSynth sentences from TRAIN signers (2–7 signs, speed 0.8–2×, trim 0.1–0.35, overlap 0–6 frames, holds
             removed), + hands-up idle (TSL51 null clips, TRAIN role) as non-sign, 30 % of the sequences at 12 fps;
  sequences 8–20 s (several sentences joined by real rest frames).
Selection: boundary F1 (± 3 frames), segment F1 (IoU ≥ 0.3), short-sign (≤ 12 frames) recall, at 25 and 12 fps, on synthetic
sentences of the TUNE (val-role) signers — never on dev-test / BLIND.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "pitty" / "scripts"))

import numpy as np
import pandas as pd
import torch

from modules.utils import ART, PREP, get_logger, read_json, write_json

log = get_logger("seg_v11")


class StoreAdapter:
    """PoseStoreMM with the PoseStore interface FluentSynth expects (row / length / get)"""

    def __init__(self, S):
        self.S, self.row = S, S.row

    def length(self, i):
        return self.S.length(i)

    def get(self, i, a=None, b=None):
        return self.S.get(i, a, b)


class SentenceGen:
    def __init__(self, role="train", seed=0, vocab=None):
        from fluent_synth import FluentSynth
        from train_encoder import PoseStoreMM, ensure_mm
        from modules.sealed import is_sealed, sealed_signers
        ensure_mm()
        self.S = PoseStoreMM()
        if os.environ.get("THAISLM_STORE_RAM") == "1":
            self.S._open()
        st = StoreAdapter(self.S)
        C = pd.read_parquet(PREP / "shards" / "clips.parquet")
        ss = sealed_signers()
        q = C[(C.kind == "isolated") & (C.view == "primary") & (C.role == role) & C.has_span & C.clip_id.isin(self.S.row)
              & C.concept.notna() & ~C.signer.isin(ss) & (C.dataset != "tsl51")]
        q = q[~q.clip_id.map(is_sealed)]
        if vocab is not None:
            q = q[q.concept.isin(vocab)]
        self.groups = [g.drop_duplicates("concept") for _, g in q.groupby("signer")]
        self.groups = [g for g in self.groups if len(g) >= 3]
        self.fs = FluentSynth(st, q, seed=seed)
        self.rng = np.random.RandomState(seed)
        self.real = []
        if role == "train":
            A = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")
            for cid, v in A.items():
                key = "t51s:" + cid.split(":", 1)[1]
                if v["role"] == "tagger_train" and key in self.S.row and not is_sealed(key):
                    self.real.append((self.S.row[key], [(s["f0"], s["f1"]) for s in v["segments"]], [s["concept"] for s in v["segments"]]))
        nul = C[(C.kind == "null") & (C.view == "primary") & (C.role == role) & C.clip_id.isin(self.S.row)]
        self.idle = [self.S.row[c] for c in nul.clip_id]
        log.info("sentence generator (%s): %d signer groups, %d real sentences, %d idle clips", role, len(self.groups), len(self.real), len(self.idle))

    def _iso(self, kp, sc, hw):
        from modules.parts import to_iso
        from modules.schema import canonical_hands
        kp, sc, _ = canonical_hands(kp, sc, hw)
        return to_iso(kp, hw), sc

    def one(self, real_p=0.5):
        """→ kp iso [T,133,2], sc, bounds [(f0,f1)], words"""
        from modules.segment import resample
        rng = self.rng
        if self.real and rng.rand() < real_p:
            row, bounds, words = self.real[rng.randint(len(self.real))]
            kp, sc, hw = self.S.get(row)
            kp, sc = self._iso(kp, sc, hw)
        else:
            g = self.groups[rng.randint(len(self.groups))]
            picks = g.sample(min(len(g), rng.randint(2, 8)), random_state=rng.randint(1e9))
            kp, sc, bounds, words = self.fs.sentence(picks, mode="fluent+blend", trim=rng.uniform(0.1, 0.35), overlap=(0, 6), hold=0.5)
            hw = tuple(int(x) for x in self.S.hw[self.S.row[picks.clip_id.iloc[0]]])
            kp, sc = self._iso(kp / np.array([hw[1] / hw[0], 1.0], np.float32), sc, hw)      # FluentSynth returns iso x: back to image units first
        f = rng.uniform(0.8, 2.0)
        n0 = len(kp)
        kp, sc = resample(kp, sc, f)
        r = (len(kp) - 1) / max(n0 - 1, 1)
        bounds = [(int(round(a * r)), int(round(b * r))) for a, b in bounds]
        return kp, sc, bounds, list(words)

    def sequence(self, lo=200, hi=500, idle_p=0.3, fps12_p=0.3):
        from modules.segment import resample
        K, S_, B, W = [], [], [], []
        T = 0
        while T < lo:
            if self.idle and self.rng.rand() < idle_p:
                kp, sc, hw = self.S.get(self.idle[self.rng.randint(len(self.idle))])
                kp, sc = self._iso(kp, sc, hw); b, w = [], []
            else:
                kp, sc, b, w = self.one()
            if T + len(kp) > hi and T >= lo // 2:
                break
            K.append(kp); S_.append(sc); B += [(a + T, c + T) for a, c in b]; W += w; T += len(kp)
        kp, sc = np.concatenate(K), np.concatenate(S_)
        if self.rng.rand() < fps12_p:                                   # 12 fps capture, played back at 25 (nearest-frame duplicates)
            n = len(kp); m = max(8, int(round(n * 12 / 25)))
            src = np.round(np.linspace(0, n - 1, m)).astype(int)
            back = src[np.clip(np.round(np.linspace(0, m - 1, n)).astype(int), 0, m - 1)]
            kp, sc = kp[back], sc[back]
        return kp[:hi], sc[:hi], [(a, min(b, hi)) for a, b in B if a < hi - 2], W


def featurize(model, kp, sc, dev):
    """frozen encoder frame states ‖ kinematic features → [T, d + 13]"""
    from modules.lmenc import frames_long
    from modules.segment import kin_features
    H = frames_long(model, kp, sc, device=dev)
    return np.concatenate([H, kin_features(kp, sc)], 1).astype(np.float32)


def evaluate(head, model, gen_val, dev, n=60, seed=123):
    from modules.boundary import boundary_f1, segment_f1, segments_from_prob
    if not hasattr(gen_val, "_fixed"):                 # the SAME validation sentences at every evaluation (both RNGs fixed once)
        gen_val.rng = np.random.RandomState(seed); gen_val.fs.rng = np.random.RandomState(seed + 1)
        gen_val._fixed = [gen_val.sequence(idle_p=0.0, fps12_p=0.0) for _ in range(n)]
    out = {}
    head.eval()
    items = gen_val._fixed
    for fps in (25, 12):
        bf, sf, sh, shn = [], [], 0, 0
        for kp, sc, B, _ in items:
            if fps == 12:
                m = max(8, int(round(len(kp) * 12 / 25)))
                src = np.round(np.linspace(0, len(kp) - 1, m)).astype(int)
                back = src[np.clip(np.round(np.linspace(0, m - 1, len(kp))).astype(int), 0, m - 1)]
                kp, sc = kp[back], sc[back]
            X = featurize(model, kp, sc, dev)
            with torch.no_grad():
                o = head(torch.from_numpy(X)[None].to(dev), torch.ones(1, len(X), device=dev))[-1][0].softmax(-1).cpu().numpy()
            P = segments_from_prob(o)
            bf.append(boundary_f1(P, B)); sf.append(segment_f1(P, B))
            for a, b in B:
                if b - a <= 12:
                    shn += 1; sh += any(max(0, min(b, d) - max(a, c)) / max(max(b, d) - min(a, c), 1) >= 0.3 for c, d in P)
        out[f"fps{fps}"] = dict(boundary_F1=float(np.mean(bf)), segment_F1=float(np.mean(sf)), short_recall=sh / max(shn, 1), short_n=shn)
    head.train()
    return out


def train_boundary(a):
    from modules.boundary import N_CLS, build, labels_from_bounds, loss
    from modules.lmenc import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_model(ART / "runs" / a.enc / "encoder_best.pt", dev)
    for p in enc.parameters():
        p.requires_grad = False
    gen, gval = SentenceGen("train", a.seed), SentenceGen("val", a.seed + 1)
    d_in = enc.dim + 13
    head = build(d_in).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    out = ART / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    cls_cnt = np.ones(N_CLS)
    best, hist, step, t0 = -1, [], 0, time.time()
    while True:
        batch = [gen.sequence() for _ in range(a.batch)]
        Xs = [featurize(enc, kp, sc, dev) for kp, sc, _, _ in batch]
        L = max(len(x) for x in Xs)
        X = np.zeros((len(Xs), L, d_in), np.float32); M = np.zeros((len(Xs), L), np.float32)
        Y = np.zeros((len(Xs), L), np.int64); Wt = np.zeros((len(Xs), L), np.float32)
        for i, (x, (kp, sc, B, _)) in enumerate(zip(Xs, batch)):
            y, w = labels_from_bounds(len(x), B)
            X[i, :len(x)] = x; M[i, :len(x)] = 1; Y[i, :len(x)] = y; Wt[i, :len(x)] = w
            cls_cnt += np.bincount(y, minlength=N_CLS)
        cw = torch.tensor((cls_cnt.sum() / cls_cnt) / (cls_cnt.sum() / cls_cnt).mean(), dtype=torch.float32, device=dev).clamp(max=10)
        outs = head(torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev))
        l = loss(outs, torch.from_numpy(Y).to(dev), torch.from_numpy(Wt).to(dev), torch.from_numpy(M).to(dev), cw)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0); opt.step()
        step += 1
        if step % 20 == 0:
            log.info("step %d loss %.4f (%.1f s/step)", step, l.item(), (time.time() - t0) / step)
        done = (a.smoke and step >= a.smoke) or (time.time() - t0 > a.minutes * 60)
        if step % a.eval_every == 0 or done:
            ev = evaluate(head, enc, gval, dev, n=20 if a.smoke else 60)
            sc_ = ev["fps25"]["boundary_F1"] + ev["fps25"]["segment_F1"] + 0.5 * (ev["fps12"]["boundary_F1"] + ev["fps12"]["segment_F1"])
            hist.append(dict(step=step, loss=float(l.item()), **{f"{k}_{m}": v for k, d in ev.items() for m, v in d.items()}))
            log.info("eval step %d: %s", step, ev)
            if sc_ > best:
                best = sc_
                torch.save(dict(cfg=head.cfg, state=head.state_dict(), step=step, eval=ev, enc=a.enc), out / "boundary_best.pt")
            write_json(out / "history.json", hist)
        if done:
            break
    write_json(out / "summary.json", dict(head="boundary", enc=a.enc, steps=step, best=best, minutes=(time.time() - t0) / 60,
                                          still_improving=bool(hist and hist[-1].get("fps25_boundary_F1", 0) >= max(h.get("fps25_boundary_F1", 0) for h in hist) - 1e-9)))


def form_prototypes(enc, dev, vocab):
    """form prototypes of the vocabulary words from TRAIN-signer clips (mean of per-signer means, forms.json)"""
    from train_encoder import PoseStoreMM, ensure_mm
    from modules.lmenc import embed, prep_for
    from modules.parts import to_iso
    from modules.schema import canonical_hands
    ensure_mm(); S = PoseStoreMM()
    C = pd.read_parquet(PREP / "shards" / "clips.parquet")
    q = C[(C.kind == "isolated") & (C.view == "primary") & (C.role == "train") & C.has_span & C.concept.isin(vocab) & C.clip_id.isin(S.row)]
    cf = read_json(PREP / "manifest" / "forms.json")["clip_form"]
    prep = prep_for(enc.cfg)
    feats = []
    for r in q.itertuples():
        kp, sc, hw = S.get(S.row[r.clip_id], int(r.span_f0), int(r.span_f1))
        kp, sc, _ = canonical_hands(kp, sc, hw)
        feats.append(prep(to_iso(kp, hw), sc))
    Z = embed(enc, feats, device=dev)
    q = q.assign(form=[cf.get(c, [None, "f0"])[1] for c in q.clip_id], signer=q.signer.fillna(q.dataset + ":unknown"))
    P, fw, words = [], [], sorted(set(q.concept))
    widx = {w: i for i, w in enumerate(words)}
    q = q.reset_index(drop=True)
    for (w, f), g in q.groupby(["concept", "form"]):
        ms = [Z[gg.index].mean(0) for _, gg in g.groupby("signer")]
        m = np.mean([x / (np.linalg.norm(x) + 1e-8) for x in ms], 0)
        P.append(m / (np.linalg.norm(m) + 1e-8)); fw.append(widx[w])
    # isolated TUNE clips for the drift check
    return np.stack(P).astype(np.float32), np.array(fw), words, (Z, q.concept.map(widx).values)


def train_ctc(a):
    from modules.lmenc import collate_np, embed, load_model, prep_for
    from modules.protoctc import build, ctc_loss
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_model(ART / "runs" / a.enc / "encoder_best.pt", dev)
    vocab = set(read_json(ROOT / "models" / "model-v10" / "vocab_conversation.json")["concepts"])
    P, fw, words, _ = form_prototypes(enc, dev, vocab)
    widx = {w: i for i, w in enumerate(words)}
    model = build(enc, P, fw, len(words)).to(dev)
    members = list(enc.members) if hasattr(enc, "members") else [enc]              # an ensemble: the last 2 blocks of EVERY member
    for p in enc.parameters():
        p.requires_grad = False
    tune_enc = [] if a.freeze_enc else [p for m in members for b in list(m.blocks)[-4:] for p in b.parameters()]   # last 2 (conv + transformer) blocks
    for p in tune_enc:
        p.requires_grad = True
    opt = torch.optim.AdamW([dict(params=[model.proj.weight, model.proj.bias, model.blank, model.log_g], lr=3e-4),
                             dict(params=tune_enc, lr=3e-5)], weight_decay=1e-4)
    gen, gval = SentenceGen("train", a.seed, vocab=vocab), SentenceGen("val", a.seed + 1, vocab=vocab)
    prep_feats = prep_for(enc.cfg)
    from modules.lmenc import feats_fn
    ff = feats_fn((enc.cfg.get("members") or [enc.cfg])[0].get("fv", 10) if "members" in enc.cfg else enc.cfg.get("fv", 10))
    out = ART / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    # drift reference: isolated R@1 of TUNE clips against TRAIN form prototypes, before training
    def iso_r1():
        from train_encoder import PoseStoreMM
        from modules.parts import to_iso
        from modules.schema import canonical_hands
        S = gval.S
        C = pd.read_parquet(PREP / "shards" / "clips.parquet")
        q = C[(C.kind == "isolated") & (C.view == "primary") & (C.role == "val") & C.has_span & C.concept.isin(words) & C.clip_id.isin(S.row)]
        q = q.sample(min(400, len(q)), random_state=0)
        fe = []
        for r in q.itertuples():
            kp, sc, hw = S.get(S.row[r.clip_id], int(r.span_f0), int(r.span_f1)); kp, sc, _ = canonical_hands(kp, sc, hw)
            fe.append(prep_feats(to_iso(kp, hw), sc))
        Z = embed(enc, fe, device=dev)
        Sw = np.full((len(Z), len(words)), -2.0)
        sims = Z @ P.T
        for j, w in enumerate(fw):
            Sw[:, w] = np.maximum(Sw[:, w], sims[:, j])
        return float((Sw.argmax(1) == q.concept.map(widx).values).mean())
    r0 = iso_r1(); log.info("drift reference: isolated TUNE R@1 against form prototypes %.4f", r0)
    hist, step, t0, best = [], 0, time.time(), 1e9
    frozen = bool(a.freeze_enc)
    while True:
        batch = [gen.sequence(lo=150, hi=375, idle_p=0.2) for _ in range(a.batch)]
        F_ = [ff(kp, sc) for kp, sc, _, _ in batch]
        x, m = collate_np(F_)
        tg = [[widx[w] for w in W if w in widx] for _, _, _, W in batch]
        x, m = torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logits, h = model(x, m)
        l_ctc = ctc_loss(logits, m, tg)
        # retrieval loss: pooled gold spans vs the prototypes (CosFace-like, s = γ)
        spans, lab = [], []
        for i, (_, _, B, W) in enumerate(batch):
            for (a0, b0), w in zip(B, W):
                if w in widx and b0 - a0 >= 3:
                    spans.append((i, a0, b0)); lab.append(widx[w])
        l_ret = torch.zeros((), device=dev)
        if spans:
            e = torch.stack([F.normalize(model.proj(h[i, a0:b0].float().mean(0)), dim=-1) for i, a0, b0 in spans])
            sf = e @ model.P.T
            sw = torch.full((len(e), len(words)), -1e4, device=dev).scatter_reduce(1, model.fw.expand(len(e), -1), sf, "amax")
            l_ret = F.cross_entropy(model.log_g.exp() * (sw - 0.2 * F.one_hot(torch.tensor(lab, device=dev), len(words))), torch.tensor(lab, device=dev))
        l = l_ctc + 0.5 * l_ret
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 2.0); opt.step()
        step += 1
        if step % 20 == 0:
            log.info("step %d ctc %.4f ret %.4f (%.2f s/step)", step, l_ctc.item(), l_ret.item(), (time.time() - t0) / step)
        done = (a.smoke and step >= a.smoke) or (time.time() - t0 > a.minutes * 60)
        if step % a.eval_every == 0 or done:
            r1 = iso_r1()
            drift = r0 - r1
            if drift > 0.005 and not frozen:                      # drift check: freeze the encoder, keep training the projection only
                for p in tune_enc:
                    p.requires_grad = False
                frozen = True
                log.info("drift %.4f > 0.005 at step %d → encoder frozen (projection / γ / blank only from here)", drift, step)
            vl = []
            if not hasattr(gval, "_fixed_ctc"):
                gval.rng = np.random.RandomState(7); gval.fs.rng = np.random.RandomState(8)
                gval._fixed_ctc = [gval.sequence(lo=150, hi=375, idle_p=0.0) for _ in range(16)]
            with torch.no_grad():
                vb = gval._fixed_ctc
                xv, mv = collate_np([ff(kp, sc) for kp, sc, _, _ in vb])
                lv, _ = model(torch.from_numpy(xv).to(dev), torch.from_numpy(mv).to(dev))
                vl = float(ctc_loss(lv, torch.from_numpy(mv).to(dev), [[widx[w] for w in W if w in widx] for _, _, _, W in vb]).item())
            hist.append(dict(step=step, ctc=float(l_ctc.item()), ret=float(l_ret.item()), val_ctc=vl, iso_r1=r1, drift=drift, frozen=frozen))
            log.info("eval step %d: val CTC %.4f · isolated R@1 %.4f (drift %.4f)", step, vl, r1, drift)
            if vl < best:
                best = vl
                torch.save(dict(enc_cfg=enc.cfg, state=model.state_dict(), words=words, form_word=fw.tolist(), step=step, iso_r1=r1, drift=drift,
                                encoder_changed=bool(tune_enc) and not (frozen and step <= a.eval_every)), out / "protoctc_best.pt")
            write_json(out / "history.json", hist)
        if done:
            break
    write_json(out / "summary.json", dict(head="protoctc", enc=a.enc, steps=step, best_val_ctc=best, drift_ref=r0, frozen=frozen,
                                          minutes=(time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True); ap.add_argument("--head", choices=["boundary", "ctc"], required=True)
    ap.add_argument("--run", required=True); ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--freeze_enc", action="store_true")
    a = ap.parse_args()
    if a.head == "ctc" and a.batch == 32:
        a.batch = 48
    (train_boundary if a.head == "boundary" else train_ctc)(a)


if __name__ == "__main__":
    main()
