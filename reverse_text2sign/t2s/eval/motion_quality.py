"""Motion integrity on whole sentences (v2, result_v2.md §5.6): what the viewer would see go wrong, measured on the avatar.

For every Thai sentence of tests/conversation.tsv (phrasebook off: generated from signs), forward kinematics of the composed
timeline, split into stroke frames (recorded) and transition frames (synthesised):
  torso_pen   a hand point (palm centre, 5 finger tips) inside the torso (behind its front surface, |x| < 12 cm, chest height)
  face_pen    a hand point inside the head (behind the nose plane by > 2 cm, inside the face box)
  hand_pen    the two palm centres closer than 3 cm (hands passing through each other)
  pop         a body / arm / wrist bone rotating > 35° in one frame (25 fps)
  flicker     a finger bone jumping > 25° and back within two frames
  jerk        mean |third derivative| of the wrist path in transitions (m/s³, lower = smoother)
Run on any engine (v1's code in a git worktree vs v2) with the same sentences.
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from .. import paths

TIPS = ["MiddleProximal"] + [f"{f}Tip" for f in ("Thumb", "Index", "Middle", "Ring", "Little")]
NOSE = np.array([0.0, 1.574, -0.084])
CHEST_Z = -0.118


def sentence_metrics(q, segs, rig, fps=25.0):
    from ..motion.codec import decode  # noqa: F401
    T = len(q)
    names = ["head"] + [f"{s}{p}" for s in ("left", "right") for p in TIPS] + ["leftHand", "rightHand"]
    fk = rig.fk(q.astype(np.float64), names)
    hp, hr = fk["head"]
    stroke = np.zeros(T, bool)
    for s in segs:
        if s["kind"] != "transition":
            stroke[s["start"]:s["end"]] = True
    trans = ~stroke
    P = np.stack([fk[f"{s}{p}"][0] for s in ("left", "right") for p in TIPS], 1)      # [T, 12, 3]
    torso = (np.abs(P[..., 0]) < 0.12) & (P[..., 1] > 1.12) & (P[..., 1] < 1.45) & (P[..., 2] > CHEST_Z + 0.01)
    L = np.stack([hr.inv().apply(P[:, k] - hp) for k in range(P.shape[1])], 1)
    zf = (NOSE - rig.rest["head"])[2]
    face = (np.abs(L[..., 0]) < 0.06) & (L[..., 1] > -0.05) & (L[..., 1] < 0.12) & (L[..., 2] > zf + 0.02)
    pc = np.linalg.norm(fk["leftMiddleProximal"][0] - fk["rightMiddleProximal"][0], axis=1)
    hand = pc < 0.03
    ang = np.degrees(2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(-1)), 0, 1)))
    from ..motion.rig import BONES
    body = [i for i, b in enumerate(BONES) if not any(f in b for f in ("Thumb", "Index", "Middle", "Ring", "Little"))]
    fing = [i for i in range(len(BONES)) if i not in body]
    back = np.degrees(2 * np.arccos(np.clip(np.abs((q[2:] * q[:-2]).sum(-1)), 0, 1)))
    flick = ((ang[:-1, fing] > 25) & (ang[1:, fing] > 25) & (back[:, fing] < 10)).sum()
    W = np.stack([fk["leftHand"][0], fk["rightHand"][0]], 1)
    jerk = np.linalg.norm(np.diff(W, 3, axis=0), axis=-1).max(1) * fps ** 3 if T > 4 else np.zeros(1)
    tj = trans[3:]
    return dict(frames=T, torso_pen_stroke=int((torso.any(1) & stroke).sum()), torso_pen_trans=int((torso.any(1) & trans).sum()),
                face_pen_stroke=int((face.any(1) & stroke).sum()), face_pen_trans=int((face.any(1) & trans).sum()),
                hand_pen_stroke=int((hand & stroke).sum()), hand_pen_trans=int((hand & trans).sum()),
                pop=int((ang[:, body] > 35).any(1).sum()), flicker=int(flick),
                jerk_trans=float(np.median(jerk[tj[:len(jerk)]])) if tj[:len(jerk)].any() else 0.0, seconds=T / fps)


def run(engine, rig, lang="th"):
    rows = []
    from ..motion.codec import decode
    for r in csv.DictReader(open(paths.HERE / "tests" / "conversation.tsv", encoding="utf-8"), delimiter="\t"):
        if r["lang"] != lang:
            continue
        ir, mo = engine.run(r["text"], phrasebook=False)
        q, _ = decode(mo)
        rows.append(dict(text=r["text"], **sentence_metrics(q, mo["segments"], rig)))
    d = pd.DataFrame(rows)
    tot = d[[c for c in d.columns if c not in ("text", "jerk_trans", "seconds")]].sum().to_dict()
    tot.update(sentences=len(d), seconds_median=round(float(d.seconds.median()), 2), jerk_trans_median=round(float(d.jerk_trans.median()), 1),
               sentences_with_any_penetration=int(((d.filter(like="_pen_").sum(1)) > 0).sum()))
    return d, tot
