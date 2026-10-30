"""Extractor study (v10, prompt_14 §4 / W2): MediaPipe Holistic (tier B, X0) vs RTMW whole-body (tier A, X1–X3) vs hybrids — the
Pose Quality Gate and the extractor latency, every number written to artifacts/reports/pose_gate_v10.json.

    python main.py extractor parity [--n 6]        # modules/wholebody.py vs the corpus RTMW extraction (pose_raw/) on the same videos
    python main.py extractor latency [--n 3]       # per-frame cost: MediaPipe (CPU) · RTMW-x / -l, FP32 / FP16, GPU / CPU, tracking on/off
    python main.py extractor proxies               # G-P3 on the paired clips (same video, both extractors): presence inside the signing
                                                   #   span, gaps, jitter, bone stability, L/R swaps, wrist agreement, cross-extractor distance
    python main.py extractor sheets [--n 96]       # G-P2 (visual): crops where the two extractors disagree most, both skeletons drawn →
                                                   #   result_reporting/figures_v10/handjudge_*.jpg + a judgement file to fill in
    python main.py extractor chain                 # Recognition with each extractor's landmarks through the full v9 chain (tune sets)

Paired data (no new extraction needed): cloud_s3/data_prep/shards/pose_store.npz holds th_sl / TTRS / TSL51-researcher / YouTube
clips twice — MediaPipe (`<clip>`) and RTMW-X (`<clip>#rtmw`), same video, same 25 fps time axis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from modules.utils import ART, PREP, read_json, write_json  # noqa: E402

REP = ART / "reports" / "pose_gate_v10.json"
CACHE = ROOT / "cache" / "v10" / "extractor"


def _p(fmt, *a):
    print(fmt % a, flush=True)


def _update(section, value):
    rep = read_json(REP) if REP.exists() else {}
    rep[section] = value
    write_json(REP, rep)


def _manifest():
    return pd.read_parquet(PREP / "manifest" / "clips.parquet", columns=["clip_id", "dataset", "source_file", "pose_raw_file", "role", "signer",
                                                                          "concept", "span_f0", "span_f1", "has_span", "width", "height", "n_frames"])


def _video(row):
    p = ROOT / "cloud_s3" / row.source_file
    return p if p.is_file() else None


# ----------------------------------------------------------------------------- parity
def cmd_parity(a):
    """The live extractor must produce the corpus' numbers: RTMW-X on the same videos, compared with pose_raw/ (the v6 corpus
    extraction, which used per-frame detection + the v1 signer track). Reported: median keypoint distance (px of a 256-px-high
    crop ≈ the model's own resolution) for body / hands, score difference, and hand presence at score ≥ 0.3."""
    from modules import wholebody
    M = _manifest()
    rng = np.random.RandomState(0)
    rows = []
    wb = wholebody.WholeBody("x", "auto", fp16=False, redetect=1)
    wb16 = wholebody.WholeBody("x", "auto", fp16=True, redetect=10) if wb.prov[0] != "CPUExecutionProvider" else None
    for ds in ("th_sl", "ttrs"):
        sub = M[(M.dataset == ds) & M.pose_raw_file.notna()]
        for r in sub.iloc[rng.choice(len(sub), a.n, replace=False)].itertuples():
            v = _video(r)
            if v is None:
                continue
            z = np.load(PREP / r.pose_raw_file)
            ref_kp, ref_sc = z["kp"].astype(np.float32), z["sc"].astype(np.float32)
            for name, w in (("fp32_det_every_frame", wb), ("fp16_tracked", wb16)):
                if w is None:
                    continue
                raw = wholebody.extract(v, fps=float(z["fps"]), wb=w)
                T = min(len(raw["kp133"]), len(ref_kp))
                H, W = (int(x) for x in raw["hw"])
                k, s = raw["kp133"][:T] * [W, H], raw["sc133"][:T]
                rk = ref_kp[:T] * [W, H]
                unit = 256.0 / max(H, 1)                  # distances in "model pixels"
                ok = (s > 0.3) & (ref_sc[:T] > 0.3)
                d = np.linalg.norm(k - rk, axis=2) * unit
                part = {"body": slice(0, 17), "hands": slice(91, 133), "face": slice(23, 91)}
                rows.append(dict(clip=r.clip_id, dataset=ds, variant=name, frames=T,
                                 **{f"dist_{p}": float(np.median(d[:, sl][ok[:, sl]])) if ok[:, sl].any() else None for p, sl in part.items()},
                                 score_diff_hands=float(np.mean(np.abs(s[:, 91:] - ref_sc[:T, 91:]))),
                                 hands_seen_live=float((s[:, 91:].reshape(T, 2, 21).mean(2) > 0.3).mean()),
                                 hands_seen_corpus=float((ref_sc[:T, 91:].reshape(T, 2, 21).mean(2) > 0.3).mean())))
                _p("%s %-22s %s", r.clip_id, name, {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in rows[-1].items() if k_.startswith(("dist", "score", "hands"))})
    D = pd.DataFrame(rows)
    summ = {v: D[D.variant == v].drop(columns=["clip", "dataset", "variant"]).median(numeric_only=True).to_dict() for v in D.variant.unique()}
    _update("parity", dict(per_clip=rows, median=summ))
    _p("parity medians %s", json.dumps(summ, indent=1))


# ----------------------------------------------------------------------------- latency
def cmd_latency(a):
    """Per-frame cost on real videos (data_test + team-drive, native resolution): MediaPipe Holistic CPU (tier B, the browser runs
    the same model), RTMW variants. Steady state: the first 10 frames (session warm-up) are not counted."""
    import cv2
    from modules import wholebody
    from modules.mediapipe_pose import Holistic
    vids = sorted((ROOT / "data_test").glob("*.mp4"))[: a.n] + sorted((ROOT / "data_test" / "team-drive").rglob("*.mp4"))[: a.n]
    frames = []
    for v in vids:
        cap = cv2.VideoCapture(str(v)); k = 0
        while k < a.frames:
            ok, f = cap.read()
            if not ok:
                break
            if k % 2 == 0:
                frames.append(f)
            k += 1
        cap.release()
    _p("%d frames from %d videos (%s)", len(frames), len(vids), sorted({f.shape[:2] for f in frames}))
    res = {}

    def run(name, fn, warm=10):
        ts = []
        for i, f in enumerate(frames):
            t = time.perf_counter(); fn(f, i); dt = (time.perf_counter() - t) * 1000
            if i >= warm:
                ts.append(dt)
        res[name] = dict(ms_p50=float(np.percentile(ts, 50)), ms_p95=float(np.percentile(ts, 95)), fps_sustained=float(1000 / np.mean(ts)), n=len(ts))
        _p("%-28s p50 %6.1f ms  p95 %6.1f ms  %5.1f fps", name, res[name]["ms_p50"], res[name]["ms_p95"], res[name]["fps_sustained"])

    hol = {}                               # VIDEO mode needs one landmarker per frame size (segmentation smoothing keeps the last mask)

    def mp_frame(f, i):
        key = f.shape[:2]
        if key not in hol:
            hol[key] = Holistic()
        hol[key].frame(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), i * 40)
    run("mediapipe_holistic_cpu", mp_frame)
    for h_ in hol.values():
        h_.close()
    for model in ("x", "l"):
        for dev, fp16 in (("auto", False), ("auto", True), ("cpu", False)):
            if dev == "cpu" and model == "x" and not a.cpu_x:
                continue
            for redet in (1, 10):
                try:
                    wb = wholebody.WholeBody(model, dev, fp16=fp16, redetect=redet)
                except Exception as e:                      # noqa: BLE001
                    _p("skip %s %s fp16=%s: %s", model, dev, fp16, e); continue
                tag = f"rtmw_{model}_{'gpu' if wb.prov[0] != 'CPUExecutionProvider' else 'cpu'}_{'fp16' if wb.fp16 else 'fp32'}_det{'every' if redet == 1 else redet}"
                if tag in res:
                    continue
                run(tag, lambda f, i: wb.frame(None, i * 40, bgr=f))
                res[tag].update(det_ms=float(np.median(wb.timing["det_ms"])) if wb.timing["det_ms"] else None,
                                pose_ms=float(np.median(wb.timing["pose_ms"])) if wb.timing["pose_ms"] else None, providers=wb.prov[:1])
                del wb
    import onnxruntime as ort
    import torch
    _update("latency", dict(per_frame=res, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", ort=ort.__version__,
                            frames=len(frames), resolutions=sorted({f"{f.shape[1]}x{f.shape[0]}" for f in frames})))


# ----------------------------------------------------------------------------- G-P3 proxies on paired clips
def paired_store():
    """Memory-mapped pose store + the paired clips table (MediaPipe clip, its RTMW twin) with signing spans."""
    from decision import _npz_memmap
    S = _npz_memmap(PREP / "shards" / "pose_store.npz")
    row = {str(c): i for i, c in enumerate(S["clip_id"])}
    C = pd.read_parquet(PREP / "shards" / "clips.parquet")
    P = C[(C.view == "primary") & C.aux_clip.notna()].copy()
    P = P[P.aux_clip.isin(row) & P.clip_id.isin(row)]
    return S, row, P


def load_pair(S, row, cid, aux):
    """Both views of one clip, canonicalised with ONE mirroring decision (MediaPipe's), isotropic units."""
    from modules.encoder import SLOTS
    from modules.parts import to_iso
    from modules.schema import mirror, dominant_hand
    out = []
    for c in (cid, aux):
        i = row[c]; s, e = int(S["off"][i]), int(S["off"][i + 1])
        kp = np.zeros((e - s, 133, 2), np.float32); sc = np.zeros((e - s, 133), np.float32)
        kp[:, SLOTS] = S["KP"][s:e]; sc[:, SLOTS] = S["SC"][s:e]
        out.append((kp, sc, tuple(int(x) for x in S["hw"][row[c]])))
    side, share = dominant_hand(out[0][0], out[0][1], out[0][2])
    res = []
    for kp, sc, hw in out:
        if side == "L" and share >= 0.62:
            kp, sc = mirror(kp, sc)
        res.append((to_iso(kp, hw), sc, hw))
    return res


def _conditions(kp, sc, f0, f1, hw):
    """Span conditions E4 points at: fast (dominant wrist p90 speed ≥ 3 shoulder widths/s), near the face (median hand–nose distance
    ≤ 0.6 sw), small hands (median hand size < 18 px), hands close together (median distance < 1.2 hand sizes)."""
    from modules.schema import LH, RH, SH_L, SH_R
    k, s = kp[f0:f1], sc[f0:f1]
    okS = (s[:, SH_L] > .3) & (s[:, SH_R] > .3)
    sw = float(np.median(np.linalg.norm(k[okS, SH_L] - k[okS, SH_R], axis=1))) if okS.any() else 0.2
    R = s[:, RH].mean(1) > .3
    if R.sum() < 3:
        return {}
    c = k[R][:, RH].mean(1)
    v = np.linalg.norm(np.diff(c, axis=0), axis=1) * 25 / sw if len(c) > 1 else np.zeros(1)
    size = np.linalg.norm(k[R][:, RH[9]] - k[R][:, RH[0]], axis=1)
    nose = k[R][:, 0]
    both = R & (s[:, LH].mean(1) > .3)
    close = float(np.median(np.linalg.norm(k[both][:, RH].mean(1) - k[both][:, LH].mean(1), axis=1) / (np.linalg.norm(k[both][:, RH[9]] - k[both][:, RH[0]], axis=1) + 1e-6))) if both.sum() >= 3 else 99.0
    return dict(fast=bool(np.percentile(v, 90) >= 3.0), near_face=bool(np.median(np.linalg.norm(c - nose, axis=1)) / sw <= 0.6),
                small=bool(np.median(size) * hw[0] < 18), close_hands=bool(close < 1.2))


def cmd_proxies(a):
    from modules.posequality import agreement, segment_quality
    S, row, P = paired_store()
    rng = np.random.RandomState(0)
    rows = []
    t = time.time()
    for ds, g in P.groupby("dataset"):
        g = g[g.has_span | (g.kind != "isolated")]
        g = g.iloc[rng.choice(len(g), min(a.n, len(g)), replace=False)]
        for r in g.itertuples():
            (km, sm, hw), (kr, sr, _) = load_pair(S, row, r.clip_id, r.aux_clip)
            T = min(len(km), len(kr))
            f0, f1 = (int(r.span_f0), int(r.span_f1)) if r.has_span else (0, T)
            f1 = min(f1, T)
            if f1 - f0 < 6:
                continue
            qm, qr = segment_quality(km, sm, f0, f1), segment_quality(kr, sr, f0, f1)
            ag = agreement(km, sm, kr, sr, f0, f1)
            cond = _conditions(km, sm, f0, f1, hw)
            for ext, q in (("mp", qm), ("rtmw", qr)):
                rows.append(dict(dataset=ds, clip=r.clip_id, extractor=ext, frames=f1 - f0, H=hw[0], **cond,
                                 dom_seen=q["dom_seen"], nd_seen=q["L"]["seen"], gap_s=q["R"]["gap_s"], jitter=q["R"]["jitter"],
                                 bone_cv=q["R"]["bone_cv"], wrist_gap=q["R"]["wrist_gap"], swap=q["R"]["swap"],
                                 dist_R=ag["R"]["dist"], only_this_R=ag["R"]["only_a" if ext == "mp" else "only_b"]))
        _p("%s: %d pairs (%.0f s)", ds, len(g), time.time() - t)
    D = pd.DataFrame(rows)
    D.to_parquet(CACHE / "proxies.parquet") if CACHE.mkdir(parents=True, exist_ok=True) is None else None
    keys = ["dom_seen", "nd_seen", "gap_s", "jitter", "bone_cv", "wrist_gap", "swap", "only_this_R"]

    def summ(g):
        return {e: {k: float(x[k].mean()) for k in keys} | {"dom_seen_lt_80": float((x.dom_seen < .8).mean()), "n": int(len(x))}
                for e, x in g.groupby("extractor")}
    out = dict(per_source={ds: summ(g) for ds, g in D.groupby("dataset")}, all=summ(D),
               conditions={c: {str(v): summ(g) for v, g in D.groupby(c)} for c in ("fast", "near_face", "small", "close_hands") if c in D},
               cross_distance_R=dict(median=float(D[D.extractor == "mp"].dist_R.median()),
                                     per_source={ds: float(g[g.extractor == "mp"].dist_R.median()) for ds, g in D.groupby("dataset")}))
    # paired differences with a bootstrap CI over clips (rtmw − mp, per proxy, per source)
    boot = {}
    for ds, g in list(D.groupby("dataset")) + [("all", D)]:
        w = g.pivot_table(index="clip", columns="extractor", values=keys)
        boot[ds] = {}
        for k in keys:
            d = (w[(k, "rtmw")] - w[(k, "mp")]).dropna().values
            if len(d) < 10:
                continue
            bs = d[rng.randint(0, len(d), (2000, len(d)))].mean(1)
            boot[ds][k] = dict(diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=len(d))
    out["paired_rtmw_minus_mp"] = boot
    _update("proxies", out)
    for ds, b in boot.items():
        _p("%-14s %s", ds, {k: f"{v['diff']:+.3f}[{v['lo']:+.3f},{v['hi']:+.3f}]" for k, v in b.items()})


# ----------------------------------------------------------------------------- RTMW at a higher input resolution (384×288)
def cmd_hires(a):
    """Does more input resolution fix RTMW's hands? RTMW-X 384×288 (vs the corpus' 256×192) re-extracted from the videos of paired
    clips of the sources where RTMW's fingers were least stable (th_sl 600×600, TSL51 researcher 640×480) → the same G-P3 proxies
    against MediaPipe on the same frames (MediaPipe's mirroring decision), and against the corpus RTMW-X 256×192."""
    from modules import wholebody
    from modules.encoder import SLOTS
    from modules.parts import to_iso
    from modules.posequality import agreement, segment_quality
    from modules.schema import dominant_hand, from_rtmw, mirror
    S, row, P = paired_store()
    M = _manifest().set_index("clip_id")
    wb = wholebody.WholeBody("x384", "auto", fp16=False, redetect=1)
    rng = np.random.RandomState(0)
    rows = []
    for ds in ("th_sl", "tsl51"):
        g = P[(P.dataset == ds) & P.has_span]
        g = g.iloc[rng.choice(len(g), min(a.n, len(g)), replace=False)]
        for r in g.itertuples():
            if ds == "tsl51":
                vid = ROOT / "cloud_s3" / "raw_data" / "TSL51" / "videos" / "user_sign" / (r.clip_id.split(":", 1)[1] + ".mp4")
            else:
                vid = _video(M.loc[r.clip_id]) if r.clip_id in M.index else None
            if vid is None or not Path(vid).is_file():
                continue
            raw = wholebody.extract(vid, fps=25.0, wb=wb)
            if raw is None:
                continue
            hw = tuple(int(x) for x in raw["hw"])
            d = from_rtmw(raw["kp133"], raw["sc133"], hw, t=raw["t"])
            views = {}
            for name, c in (("mp", r.clip_id), ("rtmw256", r.aux_clip)):
                i = row[c]; s0, e0 = int(S["off"][i]), int(S["off"][i + 1])
                kp = np.zeros((e0 - s0, 133, 2), np.float32); sc = np.zeros((e0 - s0, 133), np.float32)
                kp[:, SLOTS] = S["KP"][s0:e0]; sc[:, SLOTS] = S["SC"][s0:e0]
                views[name] = (kp, sc, tuple(int(x) for x in S["hw"][i]))
            views["rtmw384"] = (d["kp"], d["sc"], hw)
            side, share = dominant_hand(*views["mp"])
            T = min(len(v[0]) for v in views.values())
            iso_v = {}
            for k, (kp, sc, h_) in views.items():
                kp, sc = kp[:T], sc[:T]
                if side == "L" and share >= 0.62:
                    kp, sc = mirror(kp, sc)
                iso_v[k] = (to_iso(kp, h_), sc)
            f0, f1 = int(r.span_f0), min(int(r.span_f1), T)
            if f1 - f0 < 6:
                continue
            for k, (kp, sc) in iso_v.items():
                q = segment_quality(kp, sc, f0, f1)
                ag = agreement(iso_v["mp"][0], iso_v["mp"][1], kp, sc, f0, f1) if k != "mp" else None
                rows.append(dict(dataset=ds, clip=r.clip_id, extractor=k, dom_seen=q["dom_seen"], bone_cv=q["R"]["bone_cv"], jitter=q["R"]["jitter"],
                                 wrist_gap=q["R"]["wrist_gap"], swap=q["R"]["swap"], dist_to_mp=ag["R"]["dist"] if ag else 0.0))
        _p("%s done (%d rows)", ds, len(rows))
    D = pd.DataFrame(rows)
    D.to_parquet(CACHE / "hires.parquet") if CACHE.mkdir(parents=True, exist_ok=True) is None else None
    out = {ds: {e: x.drop(columns=["dataset", "clip", "extractor"]).median(numeric_only=True).round(4).to_dict() for e, x in g.groupby("extractor")}
           for ds, g in D.groupby("dataset")}
    lat = wb.timing["pose_ms"]
    out["pose_ms_p50_gpu_fp32"] = float(np.median(lat)) if lat else None
    _update("hires_384", out)
    _p("%s", json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- G-P2 (visual): which skeleton matches the hand?
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
              (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def _frame_at(video, t_s):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def cmd_sheets(a):
    """Frames inside signing spans where MediaPipe and RTMW disagree about the dominant hand (median point distance ≥ 0.35 hand sizes,
    or one of them sees it and the other does not) → crops with each skeleton drawn on its own copy, LEFT / RIGHT order randomised
    (blind), 12 pairs per sheet → result_reporting/figures_v10/handjudge_<k>.jpg + cache/v10/extractor/handjudge_key.json.
    Fill cache/v10/extractor/handjudge_answers.json {item: "L" | "R" | "both" | "neither" | "unclear"}; `sheets --score` scores it."""
    import cv2
    from modules.schema import LH, RH
    key_f, ans_f = CACHE / "handjudge_key.json", CACHE / "handjudge_answers.json"
    if a.score:
        key, ans = read_json(key_f), read_json(ans_f)
        res = {"mp": 0, "rtmw": 0, "both": 0, "neither": 0, "unclear": 0}
        per = {}
        for item, v in ans.items():
            k = key[item]
            won = {"L": k["left"], "R": k["right"]}.get(v, v)
            res[won] += 1
            per.setdefault(k["dataset"], {}).setdefault(won, 0); per[k["dataset"]][won] += 1
        n = sum(res.values())
        out = dict(n=n, counts=res, per_source=per, mp_correct=(res["mp"] + res["both"]) / max(n, 1), rtmw_correct=(res["rtmw"] + res["both"]) / max(n, 1))
        _update("visual_judgement", out)
        _p("%s", json.dumps(out, ensure_ascii=False))
        return
    S, row, P = paired_store()
    M = _manifest().set_index("clip_id")
    rng = np.random.RandomState(1)
    cand = []
    for ds in ("th_sl", "ttrs", "tsl51"):
        g = P[(P.dataset == ds) & P.has_span]
        g = g.iloc[rng.choice(len(g), min(150, len(g)), replace=False)]
        for r in g.itertuples():
            cid = r.clip_id if ds != "tsl51" else r.clip_id
            src = M.loc[cid] if cid in M.index else None
            if src is None:
                continue
            vid = ROOT / "cloud_s3" / src.source_file
            if ds == "tsl51":
                vid = ROOT / "cloud_s3" / "raw_data" / "TSL51" / "videos" / "user_sign" / (cid.split(":", 1)[1] + ".mp4")
            if not vid.is_file():
                continue
            from modules.encoder import SLOTS
            raw = []
            for c in (r.clip_id, r.aux_clip):
                i = row[c]; s0, e0 = int(S["off"][i]), int(S["off"][i + 1])
                kp = np.zeros((e0 - s0, 133, 2), np.float32); sc = np.zeros((e0 - s0, 133), np.float32)
                kp[:, SLOTS] = S["KP"][s0:e0]; sc[:, SLOTS] = S["SC"][s0:e0]
                raw.append((kp, sc))
            hw = tuple(int(x) for x in S["hw"][row[r.clip_id]])
            (km, sm), (kr, sr) = raw
            T = min(len(km), len(kr))
            for hand in (RH, LH):
                for f in range(int(r.span_f0), min(int(r.span_f1), T), 3):
                    a_, b_ = sm[f, hand].mean() > .3, sr[f, hand].mean() > .3
                    if a_ and b_:
                        size = np.linalg.norm(km[f, hand[9]] - km[f, hand[0]]) + 1e-6
                        d = np.median(np.linalg.norm(km[f, hand] - kr[f, hand], axis=1)) / size
                        if d >= 0.35:
                            cand.append((d, ds, cid, str(vid), f, hand == RH, hw, km[f, hand].copy(), kr[f, hand].copy()))
                    elif a_ != b_:
                        cand.append((0.3, ds, cid, str(vid), f, hand == RH, hw, km[f, hand].copy() if a_ else None, kr[f, hand].copy() if b_ else None))
    _p("%d disagreement frames", len(cand))
    # balanced sample per source, one frame per clip
    pick, seen = [], set()
    for ds in ("th_sl", "ttrs", "tsl51"):
        cs = [c for c in cand if c[1] == ds]
        rng.shuffle(cs)
        k = 0
        for c in cs:
            if c[2] in seen:
                continue
            seen.add(c[2]); pick.append(c); k += 1
            if k >= a.n // 3:
                break
    out_dir = ROOT / "result_reporting" / "figures_v10"; out_dir.mkdir(parents=True, exist_ok=True)
    key, tiles = {}, []
    for j, (d, ds, cid, vid, f, is_r, hw, pm, pr) in enumerate(pick):
        img = _frame_at(vid, f / 25.0)
        if img is None:
            continue
        H, W = img.shape[:2]
        pts = [p for p in (pm, pr) if p is not None]
        allp = np.concatenate(pts) * [W, H]
        cx, cy = allp.mean(0)
        half = max(40, 1.1 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])))
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half)); x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        sz = 220
        sc_ = sz / max(crop.shape[:2])
        panels = {}
        for name, p in (("mp", pm), ("rtmw", pr)):
            c = cv2.resize(crop, (int(crop.shape[1] * sc_), int(crop.shape[0] * sc_)))
            c = cv2.copyMakeBorder(c, 0, sz - c.shape[0], 0, sz - c.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
            if p is not None:
                q = (p * [W, H] - [x0, y0]) * sc_
                for u, v in HAND_LINKS:
                    cv2.line(c, tuple(int(x) for x in q[u]), tuple(int(x) for x in q[v]), (0, 255, 255), 1, cv2.LINE_AA)
                for x in q:
                    cv2.circle(c, tuple(int(v) for v in x), 2, (0, 0, 255), -1)
            else:
                cv2.putText(c, "no hand", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            panels[name] = c
        left = "mp" if rng.rand() < 0.5 else "rtmw"; right = "rtmw" if left == "mp" else "mp"
        item = f"{len(key):03d}"
        tile = np.concatenate([panels[left], np.full((sz, 4, 3), 255, np.uint8), panels[right]], 1)
        tile = cv2.copyMakeBorder(tile, 22, 4, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, f"#{item}  L | R", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
        key[item] = dict(left=left, right=right, dataset=ds, clip=cid, frame=int(f), hand="dominant-side R" if is_r else "L", dist=float(d))
    per = 12
    for s in range(0, len(tiles), per):
        grid = tiles[s:s + per]
        while len(grid) % 3:
            grid.append(np.full_like(tiles[0], 255))
        rows_ = [np.concatenate(grid[i:i + 3], 1) for i in range(0, len(grid), 3)]
        cv2.imwrite(str(out_dir / f"handjudge_{s // per:02d}.jpg"), np.concatenate(rows_, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    CACHE.mkdir(parents=True, exist_ok=True)
    write_json(key_f, key)
    _p("%d items on %d sheets → %s", len(key), (len(tiles) + per - 1) // per, out_dir)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("sheets"); sh.add_argument("--n", type=int, default=96); sh.add_argument("--score", action="store_true")
    hr = sub.add_parser("hires"); hr.add_argument("--n", type=int, default=30)
    p = sub.add_parser("parity"); p.add_argument("--n", type=int, default=6)
    lt = sub.add_parser("latency"); lt.add_argument("--n", type=int, default=3); lt.add_argument("--frames", type=int, default=160)
    lt.add_argument("--cpu_x", action="store_true")
    px = sub.add_parser("proxies"); px.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    dict(parity=cmd_parity, latency=cmd_latency, proxies=cmd_proxies, sheets=cmd_sheets, hires=cmd_hires)[a.cmd](a)


if __name__ == "__main__":
    main()
