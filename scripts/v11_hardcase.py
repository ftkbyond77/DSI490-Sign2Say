"""v11 hard-case extractor set + Pose Gate (prompt-opt-v11 §1C, §3D).

  python main.py v11-hardcase mine [--per 22]       # mine 7 hard categories on cached landmarks of owned video → 1.2 s cuts
                                                     #   → artifacts/eval/hardcase_v11/{clips/*.mp4, manifest.json, sheet_*.jpg}
  python main.py v11-hardcase label --file labels.json   # by-eye visibility labels (middle frame: which hands are truly visible)
  python main.py v11-hardcase gate --arm X0 [--arm X2]   # Pose Gate metrics of an extractor arm on the hard-case set (+ --tune N TUNE videos)
                                                     #   → artifacts/reports/pose_gate_v11.json

Sources (owned video only): TSL51 researcher recordings (cloud_s3/raw_data/TSL51/videos/user_sign|user_sentence, landmarks in the pose
store) and the team's own recordings (data_test/team-drive, data_test/iphone_test, landmarks cached by scripts/decision.py). The owner's
saved web requests are landmarks only (no pixels) — they cannot be labelled by eye and are left out of the visual set.

Categories (per frame, MediaPipe landmarks of the v10 extractor): touching = both hands seen and centres < 1.2 hand sizes apart ·
crossing = the hands' horizontal order is swapped · over_face = a hand centre < 0.5 shoulder widths from the nose · side_on = a
seen hand with |palm area| < 0.12 (edge-on) · fast = wrist speed in the top 3 % (cut also re-timed to 12 fps) · close = shoulder width
> 0.42 of the image width · bottom = a pose wrist below 88 % of the image height · lost = a raised pose wrist whose hand is missing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from modules.utils import ART, PREP, read_json, write_json

OUT = ART / "eval" / "hardcase_v11"
REP = ART / "reports" / "pose_gate_v11.json"
CATS = ["touching", "crossing", "over_face", "side_on", "fast", "close", "bottom", "lost"]
LH = list(range(91, 112)); RH = list(range(112, 133))


def _p(fmt, *a):
    print(fmt % a, flush=True)


def frame_flags(kp, sc, hw, thr=0.3):
    """kp [T,133,2] normalised image coords (x, y in [0,1]), sc [T,133] → dict category → bool [T]"""
    H, W = hw
    T = len(kp)
    ls, rs = sc[:, LH].mean(1) > thr, sc[:, RH].mean(1) > thr
    lc, rc = kp[:, LH].mean(1), kp[:, RH].mean(1)
    px = np.array([W, H], np.float32)
    sh_ok = (sc[:, 5] > thr) & (sc[:, 6] > thr)
    sw = np.where(sh_ok, np.linalg.norm((kp[:, 5] - kp[:, 6]) * px, axis=1), np.nan)
    swm = np.nanmedian(sw) if np.isfinite(sw).any() else 0.2 * W
    hs_l = np.linalg.norm((kp[:, LH[9]] - kp[:, LH[0]]) * px, axis=1); hs_r = np.linalg.norm((kp[:, RH[9]] - kp[:, RH[0]]) * px, axis=1)
    hs = np.nanmedian(np.r_[hs_l[ls], hs_r[rs]]) if (ls | rs).any() else 0.1 * W
    f = {}
    both = ls & rs
    f["touching"] = both & (np.linalg.norm((lc - rc) * px, axis=1) < 1.2 * hs)
    f["crossing"] = both & ((rc[:, 0] - lc[:, 0]) * np.sign(np.nanmedian(kp[sh_ok, 6, 0] - kp[sh_ok, 5, 0]) if sh_ok.any() else 1) > 0.02) & \
        (np.linalg.norm((lc - rc) * px, axis=1) > 0.6 * hs)
    nose = kp[:, 0]
    f["over_face"] = (sc[:, 0] > thr) & ((ls & (np.linalg.norm((lc - nose) * px, axis=1) < 0.5 * swm)) | (rs & (np.linalg.norm((rc - nose) * px, axis=1) < 0.5 * swm)))
    def palm(hand):
        h = kp[:, hand] * px
        v1, v2 = h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]
        sz = np.linalg.norm(h[:, 9] - h[:, 0], axis=1) + 1e-6
        return np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / sz ** 2
    f["side_on"] = (ls & (palm(LH) < 0.12)) | (rs & (palm(RH) < 0.12))
    v = np.zeros(T)
    if T > 1:
        wr = kp[:, [9, 10]] * px
        v[1:] = np.max(np.linalg.norm(np.diff(wr, axis=0), axis=2), 1) / max(swm, 1)
    f["fast"] = v > max(np.percentile(v, 97), 0.12)
    f["close"] = np.nan_to_num(sw / W) > 0.42
    f["bottom"] = ((sc[:, 9] > thr) & (kp[:, 9, 1] > 0.88)) | ((sc[:, 10] > thr) & (kp[:, 10, 1] > 0.88))
    raised = lambda w: (sc[:, w] > thr) & (kp[:, w, 1] < np.nanmedian(kp[sh_ok, 5:7, 1]) + 1.0 * swm / H if sh_ok.any() else False)  # noqa: E731
    f["lost"] = (raised(9) & ~ls) | (raised(10) & ~rs)
    return f


def _sources():
    """(id, video path, kp normalised [T,133,2], sc, hw) of owned videos with cached MediaPipe landmarks"""
    from modules.encoder import SLOTS
    from train_encoder import PoseStoreMM, ensure_mm
    ensure_mm(); S = PoseStoreMM()
    c = pd.read_parquet(PREP / "shards" / "clips.parquet")
    c = c[c.subset.isin(["tsl51_user_sign", "tsl51_user_sentence"]) & (c.view == "primary")]
    vdir = {"tsl51_user_sign": ROOT / "cloud_s3/raw_data/TSL51/videos/user_sign", "tsl51_user_sentence": ROOT / "cloud_s3/raw_data/TSL51/videos/user_sentence"}
    for r in c.itertuples():
        name = r.clip_id.split(":", 1)[1]
        v = vdir[r.subset] / f"{name}.mp4"
        if v.exists() and r.clip_id in S.row:
            kp, sc, hw = S.get(S.row[r.clip_id])
            yield r.clip_id, v, kp, sc, hw
    from modules.mediapipe_pose import load_raw, to_standard
    for f in sorted((ROOT / "cache" / "modeling_compare").glob("*.npz")) + sorted((ROOT / "cache" / "iphone_test_mp").glob("*.npz")):
        items = read_json(ROOT / "cache" / "modeling_compare" / "team_items.json") if "modeling_compare" in str(f) else None
        video = None
        if items:
            video = next((it["video"] for it in items if Path(it["mp"]).name == f.name), None)
        else:
            kind, stem = f.stem.split("_", 1)
            video = ROOT / "cache" / "iphone_test_mp4" / f"{kind}_{stem}.mp4"
        if video is None or not Path(video).exists():
            continue
        d = to_standard(load_raw(f), gap_s=0.0)
        yield f"own:{f.stem}", Path(video), d["kp"], d["sc"], tuple(int(x) for x in d["hw"])


def cmd_mine(a):
    import cv2
    rng = np.random.RandomState(0)
    cand = {k: [] for k in CATS}
    n_src = 0
    for cid, video, kp, sc, hw in _sources():
        n_src += 1
        fl = frame_flags(kp, sc, hw)
        T = len(kp)
        for k, m in fl.items():
            idx = np.flatnonzero(m)
            if len(idx) < 3:
                continue
            # densest 30-frame window of this flag
            cs = np.convolve(m.astype(float), np.ones(30), "valid") if T >= 30 else np.array([m.sum()])
            t0 = int(np.argmax(cs)); dens = float(cs.max()) / min(30, T)
            cand[k].append(dict(src=cid, video=str(video), t0=t0, T=T, density=dens, own=cid.startswith("own:")))
    _p("%d source videos scanned; candidates %s", n_src, {k: len(v) for k, v in cand.items()})
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "clips").mkdir(exist_ok=True)
    picked, used = [], {}
    for k in CATS:
        L = sorted(cand[k], key=lambda d: (-d["own"], -d["density"]))       # the team's own webcam / phone video first, then the densest
        n = 0
        for d in L:
            if n >= a.per or used.get(d["src"], 0) >= 2:
                continue
            used[d["src"]] = used.get(d["src"], 0) + 1
            picked.append(dict(d, cat=k, id=f"{k}_{n:02d}")); n += 1
    for d in picked:
        out = OUT / "clips" / f"{d['id']}.mp4"
        ss = max(0.0, d["t0"] / 25.0 - 0.1)
        vf = "fps=12" if d["cat"] == "fast" else "fps=25"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.2f}", "-i", d["video"], "-t", "1.3", "-vf", vf, "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        d["clip"] = str(out.relative_to(ROOT)).replace("\\", "/")
    write_json(OUT / "manifest.json", dict(rule=__doc__.split("Categories")[1], clips=picked))
    # contact sheets: middle frame of each clip, numbered (for the by-eye labels)
    for s0 in range(0, len(picked), 30):
        tiles = []
        for d in picked[s0:s0 + 30]:
            cap = cv2.VideoCapture(str(ROOT / d["clip"])); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2)); ok, fr = cap.read(); cap.release()
            fr = cv2.resize(fr, (240, int(240 * fr.shape[0] / fr.shape[1]))) if ok else np.zeros((240, 240, 3), np.uint8)
            fr = cv2.copyMakeBorder(fr[:240], 0, max(0, 240 - fr[:240].shape[0]), 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(fr, d["id"], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(fr)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
        cv2.imwrite(str(OUT / f"sheet_{s0 // 30}.jpg"), np.vstack(rows))
    _p("%d clips → %s (%s)", len(picked), OUT, {k: sum(d["cat"] == k for d in picked) for k in CATS})


def cmd_label(a):
    lab = read_json(Path(a.file))
    m = read_json(OUT / "manifest.json")
    ids = {d["id"] for d in m["clips"]}
    bad = [k for k in lab if k not in ids]
    assert not bad, bad[:5]
    m["labels"] = lab
    m["label_rule"] = "by eye on the middle frame: which of the signer's hands are truly visible in the image (L / R = the signer's own sides)"
    write_json(OUT / "manifest.json", m)
    _p("%d / %d clips labelled", len(lab), len(ids))


# ----------------------------------------------------------------------------- Pose Gate
def extract_arm(video, arm):
    """X0 = v10 production extractor (Holistic + ROI hand recovery); X2 = X0 + adaptive full-frame Hand Landmarker re-acquisition with
    box tracking while a raised hand is lost; X1 = hand detector crop → hand landmarks (TPH 3, needs its weights)."""
    import os
    from modules.mediapipe_pose import extract
    os.environ["THAISLM_HAND_RECOVERY"] = "1"
    os.environ["THAISLM_HAND_REACQUIRE"] = "1" if arm == "X2" else "0"
    os.environ["THAISLM_HAND_DETECTOR"] = "1" if arm == "X1" else "0"
    t = time.perf_counter()
    raw = extract(video, fps=25.0, normalize=False)
    ms = (time.perf_counter() - t) * 1000 / max(len(raw["t"]), 1)
    return raw, ms


def gate_metrics(raw, label=None, fps=25.0):
    from modules.mediapipe_pose import to_standard
    from modules.posequality import hand_quality
    d = to_standard(raw, gap_s=0.0)
    kp, sc = d["kp"], d["sc"]
    ls, rs = sc[:, LH].mean(1) > 0.3, sc[:, RH].mean(1) > 0.3
    T = len(kp)
    def flick(s):
        return int(((~s[1:-1]) & s[:-2] & s[2:]).sum() + (s[1:-1] & ~s[:-2] & ~s[2:]).sum()) if T > 2 else 0
    q = {h: hand_quality(kp, sc, h) for h in ("L", "R")}
    out = dict(T=T, hand_found=float(np.mean(np.r_[ls, rs])), both_found=float((ls & rs).mean()), flicker_per_10s=10 * (flick(ls) + flick(rs)) / max(T / fps, 1e-6),
               bone_cv=float(np.nanmean([q["L"]["bone_cv"], q["R"]["bone_cv"]])), wrist_gap=float(np.nanmean([q["L"]["wrist_gap"], q["R"]["wrist_gap"]])))
    if label is not None:                                  # recall / false presence of the truly visible hands at the middle frames (±3)
        mid = slice(max(0, T // 2 - 3), T // 2 + 4)
        vis = {"L": "L" in label, "R": "R" in label}
        hits, n, fp, nn = 0, 0, 0, 0
        for h, s in (("L", ls), ("R", rs)):
            if vis[h]:
                n += 1; hits += float(s[mid].mean())
            else:
                nn += 1; fp += float(s[mid].mean())
        out.update(visible_found=hits / max(n, 1) if n else None, invisible_found=fp / max(nn, 1) if nn else None)
    return out


def cmd_gate(a):
    m = read_json(OUT / "manifest.json")
    labels = m.get("labels", {})
    rep = read_json(REP) if REP.exists() else {}
    for arm in a.arm:
        per, ms = [], []
        for d in m["clips"]:
            raw, t = extract_arm(ROOT / d["clip"], arm)
            g = gate_metrics(raw, labels.get(d["id"]))
            per.append(dict(id=d["id"], cat=d["cat"], **g)); ms.append(t)
        df = pd.DataFrame(per)
        summ = {"all": df.drop(columns=["id", "cat"]).mean(numeric_only=True).to_dict()}
        for k, g in df.groupby("cat"):
            summ[k] = g.drop(columns=["id", "cat"]).mean(numeric_only=True).to_dict()
        summ["ms_per_frame"] = float(np.mean(ms))
        rep.setdefault("hardcase", {})[arm] = dict(summary=summ, clips=per)
        _p("%s hard-case: %s | ms/frame %.1f", arm, {k: round(v, 3) for k, v in summ["all"].items() if v == v}, summ["ms_per_frame"])
        if a.tune:
            vids = tune_videos(a.tune)
            pt = []
            for v in vids:
                raw, t = extract_arm(v, arm)
                pt.append(dict(video=str(v), ms=t, **gate_metrics(raw)))
            dt = pd.DataFrame(pt)
            rep.setdefault("tune_videos", {})[arm] = dict(summary=dt.drop(columns=["video"]).mean(numeric_only=True).to_dict(), n=len(pt))
            _p("%s TUNE videos (%d): %s", arm, len(pt), {k: round(v, 3) for k, v in rep["tune_videos"][arm]["summary"].items()})
        write_json(REP, rep)


def tune_videos(n, seed=0):
    """source videos of TUNE (val-role) isolated clips that exist locally (th_sl / TTRS dictionaries)"""
    man = pd.read_parquet(PREP / "manifest" / "clips.parquet", columns=["clip_id", "source_file", "role", "dataset"])
    v = man[(man.role == "val") & man.dataset.isin(["th_sl", "ttrs"])]
    paths = [ROOT / "cloud_s3" / p for p in v.source_file if isinstance(p, str)]
    paths = [p for p in paths if p.exists()]
    rng = np.random.RandomState(seed)
    return [paths[i] for i in rng.choice(len(paths), min(n, len(paths)), replace=False)] if paths else []


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mi = sub.add_parser("mine"); mi.add_argument("--per", type=int, default=20)
    lb = sub.add_parser("label"); lb.add_argument("--file", required=True)
    g = sub.add_parser("gate"); g.add_argument("--arm", action="append", default=[]); g.add_argument("--tune", type=int, default=0)
    a = ap.parse_args()
    dict(mine=cmd_mine, label=cmd_label, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
