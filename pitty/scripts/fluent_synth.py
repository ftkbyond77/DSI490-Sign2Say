"""pitty: more fluent synthetic sentences from isolated clips (words → sentences without sentence data).

The existing Synth (modules/sequence.py) glues whole sign spans with a 1–5 frame straight line. Each span still holds the
hand-raise / hand-lower of an isolated recording, and the straight bridge has a speed jump at both ends, so the tagger finds
every seam (≈1 run per word) — real fluent signing has no such cue (data_test: 1 run per sentence).

Here, per joint:
  core   keep only the frames where the signing hand is up (no raise / lower), then trim `trim` of the core on each side
  bridge cubic Hermite from the end pose of A to the start pose of B, with the end/start velocities of the two signs and a
         duration proportional to the distance the hands travel (no speed jump, no fixed length)
  blend  the last `overlap` frames of A cross-fade into the first frames of B (coarticulation)

Realism check without any sentence data: the CURRENT tagger is a seam detector. Synthetic sentences it can still cut
word-by-word are not fluent; the closer runs/word gets to real fluent signing, the more the synthesis looks like it.
    python pitty/scripts/fluent_synth.py check [--n 60]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.encoder import PoseStore  # noqa: E402
from modules.sequence import _bridge, retarget  # noqa: E402

WR_L, WR_R = 9, 10                 # COCO-WholeBody wrists
HAND_L, HAND_R = slice(91, 112), slice(112, 133)


def hands_up(kp, sc, mid_y=0.45, sw=0.22, k=1.3):
    """after retarget: shoulders at y = 0.45, shoulder width 0.22 (y grows downwards) → a wrist above the lower chest."""
    up = np.zeros(len(kp), bool)
    for w in (WR_L, WR_R):
        up |= (sc[:, w] > 0.3) & (kp[:, w, 1] < mid_y + k * sw)
    return up


def hand_speed(kp, sc):
    """per-frame speed of the wrists + hand joints (0 for the first frame)."""
    idx = np.r_[WR_L, WR_R, np.arange(91, 133)]
    v = np.linalg.norm(np.diff(kp[:, idx], axis=0), axis=2)
    ok = (sc[1:, idx] > 0.3) & (sc[:-1, idx] > 0.3)
    s = np.where(ok, v, np.nan)
    with np.errstate(all="ignore"):
        s = np.nanmean(s, 1)
    return np.r_[0.0, np.nan_to_num(s)]


def core(kp, sc, trim=0.0, hold=None, raise_k=1.3):
    """trim: fixed fraction of the hands-up core on each side · hold: also drop the still lead-in / lead-out (frames slower than
    `hold` × the clip's median hand speed), i.e. the pause an isolated recording has before and after the sign."""
    up = np.flatnonzero(hands_up(kp, sc, k=raise_k))
    if len(up) < 6 and raise_k < 1.3:                 # sign made low (chest level): fall back to the usual rule
        up = np.flatnonzero(hands_up(kp, sc))
    a, b = (up[0], up[-1] + 1) if len(up) >= 4 else (0, len(kp))
    if hold:
        sp = np.convolve(hand_speed(kp, sc), np.ones(3) / 3, "same")
        thr = hold * np.median(sp[a:b]) if b > a else 0
        mv = np.flatnonzero(sp[a:b] > thr)
        if len(mv) >= 4:
            a, b = a + mv[0], a + mv[-1] + 1
    t = int(round((b - a) * trim))
    if b - a - 2 * t >= 4:
        a, b = a + t, b - t
    return kp[a:b], sc[a:b]


def hermite(k0, v0, k1, v1, n):
    """n inner frames of a cubic Hermite curve from k0 (velocity v0 per frame) to k1 (velocity v1)."""
    s = np.linspace(0, 1, n + 2)[1:-1, None, None]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    m = n + 1
    return (h00 * k0[None] + h10 * (v0[None] * m) + h01 * k1[None] + h11 * (v1[None] * m)).astype(np.float32)


def fill_hands(kp, sc, max_gap=6):
    """fill short hand dropouts (≤ max_gap frames, hand seen on both sides) by linear interpolation — isolated recordings lose
    hands more often than fluent signing (left hand seen 82 % vs 91 % of signing frames)."""
    kp, sc = kp.copy(), sc.copy()
    for sl in (HAND_L, HAND_R):
        seen = sc[:, sl].mean(1) > 0.3
        i = 0
        while i < len(seen):
            if seen[i]:
                i += 1; continue
            j = i
            while j < len(seen) and not seen[j]:
                j += 1
            if 0 < i and j < len(seen) and j - i <= max_gap:
                w = np.linspace(0, 1, j - i + 2)[1:-1, None, None]
                kp[i:j, sl] = kp[i - 1, sl][None] * (1 - w) + kp[j, sl][None] * w
                sc[i:j, sl] = np.minimum(sc[i - 1, sl], sc[j, sl])[None]
            i = j
    return kp, sc


def join(A, B, mode, rng, overlap=(0, 0), speed=0.02, max_n=8, sc_join="min", lift=False, smooth_w=False, shape_rel=False):
    (ka, sa), (kb, sb) = A, B
    if mode == "plain":
        bk, bs = _bridge(ka[-1], kb[0], sa[-1], sb[0], 3)
        return np.concatenate([ka, bk, kb]), np.concatenate([sa, bs, sb])
    va = ka[-1] - ka[-2] if len(ka) > 1 else np.zeros_like(ka[-1])
    vb = kb[1] - kb[0] if len(kb) > 1 else np.zeros_like(kb[0])
    d = float(np.linalg.norm(ka[-1, [WR_L, WR_R]] - kb[0, [WR_L, WR_R]], axis=1).max())
    n = int(np.clip(round(d / speed), 1, max_n))
    bk = hermite(ka[-1], va, kb[0], vb, n)
    if lift:                                           # stay in signing space: no dip below the lower end height (y grows down)
        for w, hs in ((WR_L, HAND_L), (WR_R, HAND_R)):
            floor = max(ka[-1, w, 1], kb[0, w, 1])
            dy = np.maximum(bk[:, w, 1] - floor, 0)
            bk[:, w, 1] -= dy; bk[:, hs, 1] -= dy[:, None]
    bs = (np.minimum if sc_join == "min" else np.maximum)(sa[-1], sb[0])[None].repeat(n, 0)
    o = rng.randint(overlap[0], overlap[1] + 1) if overlap[1] else 0
    o = min(o, len(ka) // 3, len(kb) // 3)
    if o:
        w = np.linspace(0, 1, o + 2)[1:-1, None, None]
        if smooth_w:                                   # ease in / out instead of a straight line
            w = w * w * (3 - 2 * w)
        mix = (ka[-o:] * (1 - w) + kb[:o] * w).astype(np.float32)
        if shape_rel:                                  # fingers: blend the shape around the hand root, carried by the blended root
            for hs in (HAND_L, HAND_R):
                ra, rb = ka[-o:, hs][:, :1], kb[:o, hs][:, :1]
                shp = (ka[-o:, hs] - ra) * (1 - w) + (kb[:o, hs] - rb) * w
                mix[:, hs] = (ra * (1 - w) + rb * w + shp).astype(np.float32)
        ms = np.minimum(sa[-o:], sb[:o])
        return np.concatenate([ka[:-o], mix, bk[: max(0, n - o)], kb[o:]]), np.concatenate([sa[:-o], ms, bs[: max(0, n - o)], sb[o:]])
    return np.concatenate([ka, bk, kb]), np.concatenate([sa, bs, sb])


class FluentSynth:
    def __init__(self, store, clips, seed=0):
        self.store, self.clips, self.rng = store, clips, np.random.RandomState(seed)

    def get(self, r, a=None, b=None):
        L = self.store.length(self.store.row[r.clip_id])
        kp, sc, hw = self.store.get(self.store.row[r.clip_id], 0 if a is None else a, L if b is None else b)
        H, W = hw
        return retarget(kp * np.array([W / H, 1.0], np.float32), sc), sc.astype(np.float32)

    def sentence(self, picks, mode="fluent", trim=0.1, overlap=(1, 4), hold=None, speed=0.02, max_n=8, sc_join="min", fill=False, raise_k=1.3, lift=False,
                 smooth_w=False, shape_rel=False):
        """picks: rows of clips (same signer). → kp, sc, word boundaries [(f0, f1)], words"""
        rng = self.rng
        first, last = picks.iloc[0], picks.iloc[-1]
        parts = []
        for r in picks.itertuples():
            kp, sc = self.get(r, int(r.span_f0), int(r.span_f1))
            if fill:
                kp, sc = fill_hands(kp, sc)
            parts.append(core(kp, sc, trim, hold, raise_k) if mode != "plain" else (kp, sc))
        kp, sc = parts[0]; bounds = [(0, len(kp))]
        for p in parts[1:]:
            n0 = len(kp)
            kp, sc = join((kp, sc), p, mode, rng, overlap if mode == "fluent+blend" else (0, 0), speed=speed, max_n=max_n, sc_join=sc_join, lift=lift, smooth_w=smooth_w, shape_rel=shape_rel)
            bounds.append((len(kp) - len(p[0]), len(kp)))
        pre = self.get(first, 0, max(2, int(first.span_f0)))      # real rest before / after (hands down)
        post = self.get(last, int(last.span_f1), None)
        out_k, out_s, off = [pre[0], kp], [pre[1], sc], len(pre[0])
        if len(post[0]) >= 2:
            out_k.append(post[0]); out_s.append(post[1])
        return np.concatenate(out_k), np.concatenate(out_s), [(a + off, b + off) for a, b in bounds], list(picks.concept)


def recog(a):
    """is a trimmed sign still the same sign? isolated R@1 of held-out clips, full span vs trimmed core, bank = training signers only."""
    import logging
    logging.disable(logging.WARNING)
    from modules.encoder import prep, to_iso
    from modules.pipeline import SignPipeline
    pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False, bank_file=ROOT / "artifacts/eval/bank_eval.npz")
    cidx = {c: i for i, c in enumerate(pipe.concepts)}
    store, q = heldout_pool()
    q = q[q.concept.map(lambda c: c in cidx)].sample(min(a.n_recog, len(q)), random_state=0)
    fs = FluentSynth(store, q)
    cfgs = {"full span": None, "trim10%": dict(trim=0.1), "trim20%": dict(trim=0.2), "trim30%": dict(trim=0.3),
            "hold0.5": dict(trim=0.0, hold=0.5), "hold0.5+trim10%": dict(trim=0.1, hold=0.5)}
    rows = []
    for name, kw in cfgs.items():
        parts, tgt, lens = [], [], []
        for r in q.itertuples():
            kp, sc = fs.get(r, int(r.span_f0), int(r.span_f1))
            if kw:
                kp, sc = core(kp, sc, **kw)
            parts.append(prep(to_iso(kp, (1, 1)), sc)); tgt.append(cidx[r.concept]); lens.append(len(kp))
        Z = pipe.be.embed(parts)
        S = pipe.bank.scores(Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8))
        t = np.array(tgt); ok = S[np.arange(len(t)), t] > -1.5
        rank = (S > S[np.arange(len(t)), t][:, None]).sum(1)
        rows.append(dict(cfg=name, n=int(ok.sum()), R1=float((rank[ok] == 0).mean()), R5=float((rank[ok] < 5).mean()), median_frames=float(np.median(lens))))
    R = pd.DataFrame(rows).set_index("cfg"); print(R.round(3).to_string())
    R.to_csv(ROOT / "cache/analysis/fluent_synth_recog.csv", encoding="utf-8-sig")


def heldout_pool():
    store = PoseStore(ROOT / "cloud_s3/data_prep/shards/pose_store.npz")
    C = pd.read_parquet(ROOT / "cloud_s3/data_prep/shards/clips.parquet")
    # pinned to the v8.1 vocabulary (Jack-model) so the tune / test sets stay identical across versions
    V = set(json.loads((ROOT / "models/Jack-model/vocab_conversation.json").read_text(encoding="utf-8"))["concepts"])
    q = C[(C.kind == "isolated") & (C.view != "rtmw") & C.role.isin(["val", "test"]) & C.concept.isin(V) & C.has_span & C.clip_id.isin(store.row)]
    return store, q


def check(a):
    import logging
    logging.disable(logging.WARNING)
    from modules.encoder import to_iso
    from modules.pipeline import SignPipeline
    pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)
    # the seam detector is ALWAYS the v7 tagger (models/ may hold a retrained one, which finds fluent seams by design)
    import onnxruntime as ort
    v7 = ROOT / "cache/backup_models_v7/tagger.onnx"
    if v7.exists():
        pipe.be.tagger = ort.InferenceSession(str(v7), providers=["CPUExecutionProvider"])
    store, q = heldout_pool()
    groups = [g for _, g in q.groupby("signer") if g.concept.nunique() >= 4]
    rng = np.random.RandomState(1)
    picks_list = []
    for i in range(a.n):
        g = groups[i % len(groups)]
        picks_list.append(g.drop_duplicates("concept").sample(rng.randint(3, 6), random_state=rng.randint(1e9)))
    variants = {"plain (เดิม)": dict(mode="plain"), "core (ตัดยก/ลดมือ)": dict(mode="core", trim=0.0), "core+trim10%": dict(mode="core", trim=0.1),
                "fluent (Hermite)": dict(mode="fluent", trim=0.1), "fluent+blend": dict(mode="fluent+blend", trim=0.1, overlap=(1, 4)),
                "fluent+blend trim20%": dict(mode="fluent+blend", trim=0.2, overlap=(2, 6)),
                "hold0.5+blend": dict(mode="fluent+blend", trim=0.0, hold=0.5, overlap=(2, 6)),
                "hold0.5+trim10%+blend": dict(mode="fluent+blend", trim=0.1, hold=0.5, overlap=(2, 6)),
                "fluent+blend trim30%": dict(mode="fluent+blend", trim=0.3, overlap=(3, 8)),
                # round 2 (2026-10-11): seams were 2–3× faster than signing and dropped hands → slower joins, keep hands
                "r2 slow join": dict(mode="fluent", trim=0.3, speed=0.012, max_n=15),
                "r2 slow join+hands": dict(mode="fluent", trim=0.3, speed=0.012, max_n=15, sc_join="max", fill=True),
                "r2 slower join+hands": dict(mode="fluent", trim=0.3, speed=0.009, max_n=20, sc_join="max", fill=True),
                "r2 slow+hands trim20%": dict(mode="fluent", trim=0.2, speed=0.012, max_n=15, sc_join="max", fill=True),
                "r2 slow+hands+blend": dict(mode="fluent+blend", trim=0.3, overlap=(1, 3), speed=0.012, max_n=15, sc_join="max", fill=True),
                # round 3: dips sit at seams where the hands drop (isolated signs start/end low) → keep only raised frames / lift joins
                "r3 high core k1.0": dict(mode="fluent+blend", trim=0.15, overlap=(3, 8), raise_k=1.0),
                "r3 high core k0.8": dict(mode="fluent+blend", trim=0.15, overlap=(3, 8), raise_k=0.8),
                "r3 lift join trim30%": dict(mode="fluent", trim=0.3, lift=True, sc_join="max", fill=True),
                "r3 high k1.0 + lift": dict(mode="fluent", trim=0.15, raise_k=1.0, lift=True, sc_join="max", fill=True),
                "r3 high k0.8 + lift": dict(mode="fluent", trim=0.15, raise_k=0.8, lift=True, sc_join="max", fill=True),
                # round 4: the cue is in the arms/hands, not head/torso → how the hand changes across the seam
                "r4 long blend 6-12": dict(mode="fluent+blend", trim=0.3, overlap=(6, 12)),
                "r4 eased blend": dict(mode="fluent+blend", trim=0.3, overlap=(3, 8), smooth_w=True),
                "r4 shape-relative blend": dict(mode="fluent+blend", trim=0.3, overlap=(3, 8), shape_rel=True),
                "r4 long eased shape blend": dict(mode="fluent+blend", trim=0.3, overlap=(6, 12), smooth_w=True, shape_rel=True),
                "r4 long eased shape trim20%": dict(mode="fluent+blend", trim=0.2, overlap=(6, 12), smooth_w=True, shape_rel=True),
                # round 5: trimming is the only lever that moved — the tagger is trained on these, recognition is not, so trim harder
                "r5 trim35% long blend": dict(mode="fluent+blend", trim=0.35, overlap=(6, 12)),
                "r5 trim40% long blend": dict(mode="fluent+blend", trim=0.4, overlap=(6, 12)),
                "r5 trim45% long blend": dict(mode="fluent+blend", trim=0.45, overlap=(4, 10))}
    if a.only:
        variants = {k: v for k, v in variants.items() if k in a.only.split(",")}
    rows = []
    for name, kw in variants.items():
        fs = FluentSynth(store, q, seed=0)
        if kw["mode"] == "core":
            kw = dict(kw, mode="fluent"); kw["_core_plain_bridge"] = True
        for picks in picks_list:
            mode = kw["mode"]
            if kw.get("_core_plain_bridge"):            # core frames, old straight 3-frame bridge
                parts = [core(*fs.get(r, int(r.span_f0), int(r.span_f1)), kw["trim"]) for r in picks.itertuples()]
                kp, sc = parts[0]
                for p in parts[1:]:
                    kp, sc = join((kp, sc), p, "plain", fs.rng)
                pre = fs.get(picks.iloc[0], 0, max(2, int(picks.iloc[0].span_f0))); post = fs.get(picks.iloc[-1], int(picks.iloc[-1].span_f1), None)
                kp = np.concatenate([pre[0], kp] + ([post[0]] if len(post[0]) >= 2 else [])); sc = np.concatenate([pre[1], sc] + ([post[1]] if len(post[1]) >= 2 else []))
            else:
                kp, sc, _, _ = fs.sentence(picks, mode=mode, trim=kw.get("trim", 0.1), overlap=kw.get("overlap", (1, 4)), hold=kw.get("hold"),
                                           speed=kw.get("speed", 0.02), max_n=kw.get("max_n", 8), sc_join=kw.get("sc_join", "min"), fill=kw.get("fill", False),
                                           raise_k=kw.get("raise_k", 1.3), lift=kw.get("lift", False),
                                           smooth_w=kw.get("smooth_w", False), shape_rel=kw.get("shape_rel", False))
            present = sc[:, [0, 5, 6]].mean(1) > 0.3
            prob, _ = pipe._tag(to_iso(kp, (1, 1)), sc, present)
            ps = prob[:, 1] + prob[:, 2]
            act = ps > 0.5
            q1 = np.convolve(prob[:, 1], np.ones(3) / 3, "same") > 0.3
            rows.append(dict(variant=name, words=len(picks), runs=int((act[1:] & ~act[:-1]).sum() + act[0]), begins=int((q1[1:] & ~q1[:-1]).sum() + q1[0])))
    R = pd.DataFrame(rows)
    out = R.groupby("variant", sort=False).apply(lambda d: pd.Series(dict(sentences=len(d), runs_per_word=d.runs.sum() / d.words.sum(),
                                                                          begins_per_word=d.begins.sum() / d.words.sum(), single_run=(d.runs == 1).mean())))
    ref = json.loads((ROOT / "cache/analysis/fluent_diagnostics.json").read_text(encoding="utf-8")) if (ROOT / "cache/analysis/fluent_diagnostics.json").exists() else None
    print(out.round(2).to_string())
    print("\nเป้า (วิดีโอจริงที่ทำท่าต่อเนื่อง, v7 tagger): runs/word ≈ 0.25 · begins/word ≈ 0.25 · TSL51 (มีพักทุกคำ) ≈ 0.97")
    (ROOT / "cache/analysis").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "cache/analysis/fluent_synth_check.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "recog"]); ap.add_argument("--n_recog", type=int, default=600); ap.add_argument("--n", type=int, default=60); ap.add_argument("--only", default="")
    a = ap.parse_args()
    check(a) if a.cmd == "check" else recog(a)
