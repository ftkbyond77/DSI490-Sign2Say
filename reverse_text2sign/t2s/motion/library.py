"""The motion library: every selected exemplar retargeted once, offline, into one memory-mappable file.

models/motion/
  q.bin        int16 [N_frames, B, 4]   local bone quaternions · 32767 (rig.BONES order)
  m.bin        uint8 [N_frames, M]      face morphs · 255 (retarget.MORPHS order)
  index.json   {"bones", "morphs", "fps", "clips": {key: {off, n, core:[a,b], ...descriptors, fidelity}}}
A clip covers its signing span plus PAD frames each side (the composer trims / blends them). The API maps the files with
np.memmap, so RAM holds only the signs actually used (OS page cache) and start-up is instant.

Descriptors (for the IR: hand used, location, movement, two-handedness) are measured on the avatar's own forward kinematics, so
they describe what the viewer will see. Fidelity = 2D distance between the avatar's joints and the signer's landmarks in the
same normalised frame (shoulder width units; fingertips in palm lengths) — the retargeting error, per clip.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import paths
from ..data.quality import L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, iso3, load_raw, present
from .clean import clean, mirror
from .retarget import MORPHS, retarget
from .rig import BONES, get_rig

PAD = 4                       # frames kept before / after the signing span (0.16 s) for blending
LIB = paths.MODELS / "motion"
FPS = 25.0


def _fk_2d(q, rig):
    """avatar joints → normalised 2D like the image (x right = model −X, y down = model −Y), mid-shoulder origin, sw units."""
    names = ["leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand", "head"] + \
            [f"{s}{f}Tip" for s in ("left", "right") for f in ("Thumb", "Index", "Middle", "Ring", "Little")] + \
            ["leftMiddleProximal", "rightMiddleProximal"]
    fk = rig.fk(q, names=names)
    P = {k: np.stack([-v[0][:, 0], -v[0][:, 1]], 1) for k, v in fk.items()}
    mid = (P["leftUpperArm"] + P["rightUpperArm"]) / 2
    return {k: (v - mid) / rig.sw for k, v in P.items()}, fk


def fidelity(raw, q, f0, f1, rig):
    """mean 2D joint errors on the span: elbows / wrists (sw), fingertips relative to the wrist (palm lengths)."""
    hw = tuple(int(v) for v in raw["hw"])
    P = iso3(raw["pose"], hw)[:, :, :2]
    mid = (P[:, L_SH] + P[:, R_SH]) / 2
    sw = np.median(np.linalg.norm(P[:, L_SH] - P[:, R_SH], axis=1))
    A, _ = _fk_2d(q, rig)
    s = slice(f0, f1)
    out = {}
    for name, si, ai in (("elbow", (L_EL, R_EL), ("leftLowerArm", "rightLowerArm")), ("wrist", (L_WR, R_WR), ("leftHand", "rightHand"))):
        e = [np.linalg.norm((P[s, j] - mid[s]) / sw - A[a][s], axis=1) for j, a in zip(si, ai)]
        out[name] = float(np.mean(np.concatenate(e)))
    fe = []
    for side, key in (("left", "lh"), ("right", "rh")):
        h = iso3(raw[key], hw)[s, :, :2]
        m = present(raw[key][s])
        if m.sum() < 3:
            continue
        palm = np.median(np.linalg.norm(h[m, 9] - h[m, 0], axis=1))
        apalm = np.median(np.linalg.norm(A[f"{side}MiddleProximal"][s] - A[f"{side}Hand"][s], axis=1))
        for f, lm in zip(("Thumb", "Index", "Middle", "Ring", "Little"), (4, 8, 12, 16, 20)):
            sig = (h[m, lm] - h[m, 0]) / palm
            av = (A[f"{side}{f}Tip"][s][m] - A[f"{side}Hand"][s][m]) / apalm
            fe.append(np.linalg.norm(sig - av, axis=1))
    out["fingertip"] = float(np.mean(np.concatenate(fe))) if fe else None
    return out


def descriptors(q, f0, f1, rig):
    """what the viewer sees: dominant hand, two-handedness, location (head / face / chest / neutral space), movement size."""
    A, fk = _fk_2d(q, rig)
    s = slice(f0, f1)
    rw, lw = A["rightHand"][s], A["leftHand"][s]
    rest_r, rest_l = A["rightHand"][0], A["leftHand"][0]
    mv_r = float(np.linalg.norm(np.diff(rw, axis=0), axis=1).sum()); mv_l = float(np.linalg.norm(np.diff(lw, axis=0), axis=1).sum())
    up_r = float((rw[:, 1] < 1.2).mean()); up_l = float((lw[:, 1] < 1.2).mean())   # y down: < 1.2 sw below shoulders = raised
    dom = "right" if (mv_r + up_r) >= (mv_l + up_l) else "left"
    two = bool(min(up_r, up_l) > 0.5 and min(mv_r, mv_l) > 0.25 * max(mv_r, mv_l, 1e-6))
    w = rw if dom == "right" else lw
    n = len(w)
    mid = w[int(0.2 * n):max(int(0.8 * n), int(0.2 * n) + 1)]       # the sign itself, not the lift from / drop to rest
    y = float(np.median(mid[:, 1]))                                  # sw below the shoulder line (negative = above)
    # avatar: mouth ≈ −0.46 sw, chin ≈ −0.37, top of head ≈ −0.9 (rig.py rest positions)
    loc = "head" if y < -0.55 else "face" if y < -0.12 else "chest" if y < 0.75 else "low"
    path = mv_r if dom == "right" else mv_l
    from scipy.signal import savgol_filter
    ws = savgol_filter(w, min(9, n - (1 - n % 2)), 2, axis=0) if n >= 7 else w
    d = np.diff(ws, axis=0)
    sp = np.linalg.norm(d, axis=1)
    turns = int((((d[1:] * d[:-1]).sum(1) < 0) & (sp[1:] > 0.01)).sum())
    return dict(dominant=dom, two_handed=two, location=loc, movement=round(path, 2),
                movement_kind="static" if path < 0.4 else ("repeated" if turns >= 4 else "path"))


def _build_one(rec):
    rig = get_rig()
    try:
        raw = load_raw(rec["raw_file"])
        ta = rec.get("trim_a")
        if ta is not None and ta == ta:                                 # YouTube lesson: the annotated sign span only (as audited)
            ta, tb = int(ta), int(rec["trim_b"])
            T0 = len(raw["t"])
            raw = {k: (v[ta:tb] if isinstance(v, np.ndarray) and v.ndim and len(v) == T0 else v) for k, v in raw.items()}
        if rec.get("mirrored"):
            raw = mirror(raw)
        cr, rep = clean(raw)
        out = retarget(cr, rig)
        T = len(out["q"])
        a = max(0, int(rec["span_f0"]) - PAD); b = min(T, int(rec["span_f1"]) + PAD)
        fid = fidelity(cr, out["q"], int(rec["span_f0"]), int(rec["span_f1"]), rig)
        desc = descriptors(out["q"], int(rec["span_f0"]), int(rec["span_f1"]), rig)
        return rec["key"], dict(q=out["q"][a:b], m=out["morph"][a:b], core=[int(rec["span_f0"]) - a, int(rec["span_f1"]) - a],
                                fidelity=fid, desc=desc, meta=out["meta"], clean=rep)
    except Exception as e:                                           # noqa: BLE001 — reported, never silently dropped
        return rec["key"], dict(error=f"{type(e).__name__}: {e}")


def build(records, out_dir=LIB, procs=6):
    """records: list of dicts with key, raw_file, mirrored, span_f0, span_f1 (+ any metadata copied into the index).
    Written to <out_dir>.new and swapped in at the end (a memory-mapped q.bin cannot be overwritten on Windows: stop the API first)."""
    final = Path(out_dir)
    out_dir = final.with_name(final.name + ".new"); out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(procs) as pool:
        res = pool.map(_build_one, records, chunksize=16)
    meta = {r["key"]: r for r in records}
    Q, M, index, errors, off = [], [], {}, {}, 0
    for key, r in res:
        if "error" in r:
            errors[key] = r["error"]; continue
        n = len(r["q"])
        Q.append(np.clip(np.round(r["q"] * 32767), -32767, 32767).astype("<i2"))
        M.append(np.clip(np.round(r["m"] * 255), 0, 255).astype(np.uint8))
        extra = {k: v for k, v in meta[key].items() if k not in ("key", "raw_file", "span_f0", "span_f1", "source_file", "trim_a", "trim_b")}
        index[key] = dict(off=off, n=n, core=r["core"], fidelity=r["fidelity"], desc=r["desc"],
                          alpha=round(r["meta"]["alpha"], 3), **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                                 for k, v in extra.items()})
        off += n
    np.concatenate(Q).tofile(out_dir / "q.bin")
    np.concatenate(M).tofile(out_dir / "m.bin")
    (out_dir / "index.json").write_text(json.dumps(dict(bones=BONES, morphs=MORPHS, fps=FPS, frames=off, clips=index),
                                                   ensure_ascii=False, default=_js), encoding="utf-8")
    import shutil
    if final.exists():
        shutil.rmtree(final.with_name(final.name + ".old"), ignore_errors=True)
        final.rename(final.with_name(final.name + ".old"))
    out_dir.rename(final)
    shutil.rmtree(final.with_name(final.name + ".old"), ignore_errors=True)
    print(f"library: {len(index)} clips, {off} frames, {len(errors)} errors")
    if errors:
        (out_dir.parent / "motion_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=1), encoding="utf-8")
    return index, errors


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


class Library:
    """read side (API): memory-mapped clips."""

    def __init__(self, d=LIB):
        d = Path(d)
        self.index = json.loads((d / "index.json").read_text(encoding="utf-8"))
        self.bones, self.morphs = self.index["bones"], self.index["morphs"]
        F, B, Mn = self.index["frames"], len(self.bones), len(self.morphs)
        self.q = np.memmap(d / "q.bin", "<i2", "r", shape=(F, B, 4))
        self.m = np.memmap(d / "m.bin", np.uint8, "r", shape=(F, Mn))
        self.clips = self.index["clips"]

    def __contains__(self, key):
        return key in self.clips

    def get(self, key):
        c = self.clips[key]
        a, n = c["off"], c["n"]
        return (np.asarray(self.q[a:a + n], np.float32) / 32767.0, np.asarray(self.m[a:a + n], np.float32) / 255.0, c)
