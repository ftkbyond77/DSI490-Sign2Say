"""v8 data-quality audit of the YouTube material (the user's concern: edited / cut / busy YouTube videos are noise, not signal).

  python main.py audit-youtube measure        # per clip: shot cuts, people in frame, signer size, hand visibility, label check
  python main.py audit-youtube sheets         # contact sheets (clip frames next to a dictionary reference of the same word) for eyeballing
  python main.py audit-youtube decide         # rules + eye-check verdicts → qc/quality_v8.csv (keep / drop + reason)

What "good" means here = what data_test looks like: ONE signer, facing the camera, upper body in frame, one continuous shot, the
hands visible while signing, the word shown once (or repeated), label = the title word and consistent with other people's signing
of that word. A clip is dropped when the video itself shows hard evidence against that (a cut or zoom inside the sign, a second
person / picture-in-picture, a signer too small or partly out of frame, hands mostly lost), or when the eye check finds the sign
does not match the word. Only clips that pass enter v8 training and the bank (scripts/data_prep.py shards reads quality_v8.csv).

Output: cloud_s3/agent_data/youtube_words/qc/{quality_v8.csv, sheets_v8/*.jpg, eyecheck_v8.csv}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import DATA, PREP, get_logger

log = get_logger("audit_yt")
YT = DATA / "agent_data" / "youtube_words"
QC = YT / "qc"
FACE_TASK = Path.home() / ".cache" / "mediapipe_tasks" / "face_landmarker.task"
FPS = 25.0


# ----------------------------------------------------------------------------- per-video measurements
def _video_stats(args):
    """One video → cut / zoom / face-count measurements over the whole video and inside the sign span [f0, f1) (25 fps frames)."""
    import cv2
    path, f0, f1, clip_id = args
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(src) or src <= 1 or src > 240:
        src = 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    small, hists, times, keep = [], [], [], []
    t_span = (f0 / FPS, f1 / FPS)
    sample_t = np.linspace(t_span[0], max(t_span[0], t_span[1] - 0.04), 8) if f1 > f0 else []
    si, i = 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / src; i += 1
        g = cv2.cvtColor(cv2.resize(fr, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        small.append(g.astype(np.float32) / 255.0)
        hsv = cv2.cvtColor(cv2.resize(fr, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
        hists.append(h / (h.sum() + 1e-9)); times.append(t)
        while si < len(sample_t) and t >= sample_t[si]:
            s = 360 / max(fr.shape[:2])
            keep.append(cv2.cvtColor(cv2.resize(fr, (int(fr.shape[1] * s), int(fr.shape[0] * s))), cv2.COLOR_BGR2RGB)); si += 1
    cap.release()
    out = dict(clip_id=clip_id, width=W, height=H, src_fps=float(src), n_src=len(times), dur_s=round(len(times) / src, 2))
    if len(small) < 3:
        return dict(out, error="unreadable")
    S = np.stack(small); Hh = np.stack(hists); T = np.array(times)
    d = np.abs(np.diff(S, axis=0)).mean((1, 2))                         # mean absolute frame difference
    hc = np.array([np.minimum(Hh[k], Hh[k + 1]).sum() for k in range(len(Hh) - 1)])   # histogram intersection (1 = same colours)
    med = float(np.median(d)) + 1e-4
    cut = (d > 0.08) & (d > 6 * med) & (hc < 0.7)                         # hard cut: big change of the picture AND of its colours
    tc = T[1:][cut]
    pad = 0.5
    in_span = (tc >= t_span[0] - pad) & (tc <= t_span[1] + pad)
    out.update(cuts_total=int(cut.sum()), cuts_in_span=int(in_span.sum()), cut_times=";".join(f"{x:.2f}" for x in tc[:20]),
               motion_med=round(med, 4))
    # faces in the span: a second person, picture-in-picture or an interpreter box shows up as > 1 face
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_TASK)), running_mode=vision.RunningMode.IMAGE, num_faces=4,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5))
        nf, sizes = [], []
        for rgb in keep:
            r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            nf.append(len(r.face_landmarks))
            for f in r.face_landmarks:
                xs = [p.x for p in f]; sizes.append(max(xs) - min(xs))
        fl.close()
        out.update(faces_max=int(max(nf) if nf else 0), faces_multi_frac=round(float(np.mean([n > 1 for n in nf])) if nf else 0.0, 2),
                   faces_zero_frac=round(float(np.mean([n == 0 for n in nf])) if nf else 1.0, 2))
    except Exception as e:  # noqa: BLE001
        out.update(error=f"faces:{e}")
    return out


def _pose_stats(raw_path, f0, f1):
    """From the MediaPipe raw landmarks (the training extractor): signer size, framing, hand visibility, zoom inside the span."""
    from modules.mediapipe_pose import load_raw
    r = load_raw(raw_path)
    t = r["t"]
    m = (t >= f0 / FPS) & (t < f1 / FPS)
    if m.sum() < 3:
        return dict(pose_frames=int(m.sum()))
    p = r["pose"][m]
    ok = (p[:, 11, 3] > 0.5) & (p[:, 12, 3] > 0.5)
    H, W = (int(x) for x in r["hw"])
    sw = np.linalg.norm((p[:, 11, :2] - p[:, 12, :2]) * [W, H], axis=1) / max(W, H)   # shoulder width / long side of the frame
    lh = np.abs(r["lh"][m]).sum((1, 2)) > 0; rh = np.abs(r["rh"][m]).sum((1, 2)) > 0
    wr = p[:, [15, 16], :2]
    # the SIGNING hand = the wrist that moves / rises most in the span (the other one may rest below the frame: normal)
    act = [np.nansum(np.linalg.norm(np.diff(wr[:, k], axis=0), axis=1)) - 0.5 * np.median(wr[:, k, 1]) for k in (0, 1)]
    k = int(np.argmax(act))
    out_of_frame = ((wr[:, k] < -0.02) | (wr[:, k] > 1.02)).any(-1)
    head_cut = (p[:, 0, 1] < 0.02)
    face_vis = (p[:, 0, 3] > 0.5) & (p[:, 2, 3] > 0.5) & (p[:, 5, 3] > 0.5)
    sws = sw[ok]
    return dict(pose_frames=int(m.sum()), body_frac_span=round(float(ok.mean()), 2),
                shoulder_frac=round(float(np.median(sws)), 3) if len(sws) else 0.0,
                zoom_ratio=round(float(np.percentile(sws, 95) / max(np.percentile(sws, 5), 1e-3)), 2) if len(sws) > 4 else 1.0,
                hand_frac_span=round(float((lh | rh).mean()), 2), both_hands_frac=round(float((lh & rh).mean()), 2),
                wrist_out_frac=round(float(out_of_frame.mean()), 2), head_cut_frac=round(float(head_cut.mean()), 2),
                face_vis_frac=round(float(face_vis.mean()), 2))


def label_check(items):
    """Is the clip's sign close to OTHER people's signing of its word? Bank embeddings of the v7 encoder; the clip's own signer is
    left out of the reference set. Returns the rank of the labelled concept (1 = nearest) over all concepts that have a reference."""
    b = np.load(ROOT / "models" / "bank.npz", allow_pickle=False)
    Z = b["Z"].astype(np.float32); C = b["c"]; clip = b["clip"]; signer = b["signer"]; ds = b["dataset"]
    concepts = json.loads((ROOT / "models" / "concepts.json").read_text(encoding="utf-8"))
    cidx = {c: i for i, c in enumerate(concepts)}
    row = {c: i for i, c in enumerate(clip)}
    ref = ~np.char.startswith(ds.astype(str), "tsl51_sentence")            # auto-aligned sentence words are not a reference
    out = []
    for r in items.itertuples():
        i = row.get(r.clip_id)
        c = cidx.get(r.concept, -1)
        if i is None or c < 0:
            out.append(dict(clip_id=r.clip_id, ref_signers=0, label_rank=np.nan, label_sim=np.nan, best_other=None, best_other_sim=np.nan))
            continue
        m = ref & (signer != signer[i])
        s = Z[m] @ Z[i]; cc = C[m]
        # top-2 mean per concept (what spotting does)
        order = np.argsort(-s)
        best = {}
        for k in order[:4000]:
            best.setdefault(int(cc[k]), []).append(float(s[k]))
        score = {k: float(np.mean(v[:2])) for k, v in best.items()}
        n_ref = int(len(set(signer[m & (C == c)])))
        if c in score:
            rank = 1 + sum(v > score[c] for v in score.values())
        else:
            rank = np.nan
        bo = max(((k, v) for k, v in score.items() if k != c), key=lambda x: x[1], default=(None, np.nan))
        out.append(dict(clip_id=r.clip_id, ref_signers=n_ref, label_rank=rank, label_sim=round(score.get(c, np.nan), 3),
                        best_other=concepts[bo[0]] if bo[0] is not None else None, best_other_sim=round(bo[1], 3)))
    return pd.DataFrame(out)


def cmd_measure(a):
    it = pd.read_csv(YT / "items.csv")
    it = it[it.status == "accepted"].copy()
    jobs = [(YT / r.video, int(r.f0), int(r.f1), r.clip_id) for r in it.itertuples()]
    log.info("measuring %d accepted harvested clips", len(jobs))
    old = QC / "measure_v8.csv"
    vcols = ["clip_id", "width", "height", "src_fps", "n_src", "dur_s", "cuts_total", "cuts_in_span", "cut_times", "motion_med", "faces_max",
             "faces_multi_frac", "faces_zero_frac"]
    if a.reuse_video and old.exists():
        V = pd.read_csv(old)[vcols]
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            V = pd.DataFrame(list(ex.map(_video_stats, jobs, chunksize=4)))
    P = []
    for r in it.itertuples():
        f = PREP / "pose_mp_raw" / "agent" / f"{r.clip_id.replace(':', '_')}.npz"
        P.append(dict(clip_id=r.clip_id, **(_pose_stats(f, int(r.f0), int(r.f1)) if f.exists() else {})))
    L = label_check(it)
    M = it[["clip_id", "video", "word", "concept", "channel", "signer", "role", "title", "f0", "f1", "span_s", "n_bursts", "consistency"]] \
        .merge(V, on="clip_id", how="left").merge(pd.DataFrame(P), on="clip_id", how="left").merge(L, on="clip_id", how="left")
    QC.mkdir(parents=True, exist_ok=True)
    M.to_csv(QC / "measure_v8.csv", index=False, encoding="utf-8")
    log.info("measure → %s\n%s", QC / "measure_v8.csv", M.describe().T[["mean", "50%", "min", "max"]].to_string())


# ----------------------------------------------------------------------------- rules
RULES = [   # (name, test) — hard evidence that the clip is not a clean single-signer demonstration
    ("cut_in_sign", lambda r: r.cuts_in_span >= 1),
    ("edited_video", lambda r: r.cuts_total >= 4),
    ("second_person", lambda r: r.faces_multi_frac >= 0.25),
    ("face_hidden", lambda r: r.face_vis_frac < 0.5),
    ("zoom_in_sign", lambda r: r.zoom_ratio >= 1.35),
    ("signer_too_small", lambda r: r.shoulder_frac < 0.10),
    ("hands_lost", lambda r: r.hand_frac_span < 0.6),
    ("out_of_frame", lambda r: r.wrist_out_frac >= 0.3 or r.body_frac_span < 0.8),
    ("low_resolution", lambda r: min(r.width, r.height) < 240),
]


def rule_flags(M):
    flags = []
    for r in M.itertuples():
        f = []
        for name, test in RULES:
            try:
                if test(r):
                    f.append(name)
            except TypeError:
                pass
        flags.append("|".join(f))
    return flags


# ----------------------------------------------------------------------------- contact sheets for the eye check
def _frames_at(path, times, h=150):
    import cv2
    cap = cv2.VideoCapture(str(path)); src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * src)))
        ok, fr = cap.read()
        if not ok:
            out.append(np.zeros((h, int(h * 16 / 9), 3), np.uint8)); continue
        s = h / fr.shape[0]
        out.append(cv2.resize(fr, (int(fr.shape[1] * s), h)))
    cap.release()
    return out


def _reference_video(concept, exclude_signer):
    """A dictionary video (TTRS / th_sl) of the same concept by another signer, with its sign span."""
    m = pd.read_parquet(PREP / "manifest" / "clips.parquet", columns=["clip_id", "dataset", "concept", "signer", "source_file", "span_f0", "span_f1", "use", "kind"])
    m = m[(m.concept == concept) & m.dataset.isin(["ttrs", "th_sl"]) & m.use & (m.kind == "isolated") & (m.signer != exclude_signer)]
    for r in m.itertuples():
        p = DATA / r.source_file
        if p.exists():
            return p, r.span_f0, r.span_f1, r.dataset
    return None


def cmd_sheets(a):
    import cv2
    M = pd.read_csv(QC / "measure_v8.csv")
    M["flag"] = rule_flags(M)
    sel = M if a.all else M[(M.flag == "") & ((M.label_rank > a.rank) | M.label_rank.isna())]
    if a.only:
        sel = M[M.clip_id.isin(a.only.split(","))]
    out = QC / "sheets_v8" / a.name; out.mkdir(parents=True, exist_ok=True)
    rows, k = [], 0
    for r in sel.itertuples():
        t0, t1 = r.f0 / FPS, r.f1 / FPS
        fr = _frames_at(YT / r.video, np.linspace(t0, t1, 6))
        strip = np.concatenate(fr, 1)
        ref = _reference_video(r.concept, r.signer)
        if ref:
            p, a0, a1, dsn = ref
            rf = _frames_at(p, np.linspace((a0 or 0) / FPS, (a1 or 50) / FPS, 6))
            rstrip = np.concatenate(rf, 1)
        else:
            rstrip = np.zeros((150, 400, 3), np.uint8)
        wmax = max(strip.shape[1], rstrip.shape[1])
        pad = lambda x: np.pad(x, ((0, 0), (0, wmax - x.shape[1]), (0, 0)))
        lab = np.zeros((26, wmax, 3), np.uint8)
        cv2.putText(lab, f"#{k} {r.clip_id.split(':')[-1]}  rank={r.label_rank}  ref={'yes' if ref else 'none'}", (5, 19), 0, 0.6, (255, 255, 255), 1)
        rows.append(np.concatenate([lab, pad(strip), pad(rstrip)], 0)); k += 1
        if len(rows) == 6:
            cv2.imwrite(str(out / f"sheet_{k // 6:03d}.jpg"), np.concatenate([pad2(x, rows) for x in rows], 0)); rows = []
    if rows:
        cv2.imwrite(str(out / f"sheet_{k // 6 + 1:03d}.jpg"), np.concatenate([pad2(x, rows) for x in rows], 0))
    sel.assign(sheet_index=range(len(sel)))[["clip_id", "word", "concept", "label_rank", "best_other", "flag"]].to_csv(out / "index.csv", index=False, encoding="utf-8")
    log.info("%d clips → %s", len(sel), out)


def pad2(x, rows):
    w = max(r.shape[1] for r in rows)
    return np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0)))


def cmd_decide(a):
    M = pd.read_csv(QC / "measure_v8.csv")
    M["flag"] = rule_flags(M)
    eye = QC / "eyecheck_v8.csv"
    E = pd.read_csv(eye) if eye.exists() else pd.DataFrame(columns=["clip_id", "eye_verdict", "eye_note"])
    M = M.merge(E, on="clip_id", how="left")
    eye_v = M.eye_verdict.fillna("")
    keep = ((M.flag == "") | eye_v.eq("keep")) & ~eye_v.eq("drop")          # the eye check overrides a rule flag both ways
    M["keep"] = keep
    M["reason"] = np.where(eye_v.eq("drop"), "eye:" + M.eye_note.fillna(""), np.where(keep, "", M.flag))
    M.to_csv(QC / "quality_v8.csv", index=False, encoding="utf-8")
    log.info("quality_v8: keep %d / %d (%d concepts, %d signers); drops by reason: %s", int(keep.sum()), len(M), M[keep].concept.nunique(),
             M[keep].signer.nunique(), M[~keep].reason.str.split("|").explode().value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure"); m.add_argument("--workers", type=int, default=5); m.add_argument("--reuse_video", action="store_true")
    s = sub.add_parser("sheets"); s.add_argument("--all", action="store_true"); s.add_argument("--rank", type=int, default=50); s.add_argument("--only", default=""); s.add_argument("--name", default="all")
    sub.add_parser("decide")
    a = ap.parse_args()
    {"measure": cmd_measure, "sheets": cmd_sheets, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
