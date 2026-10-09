"""Sentence-level non-manual marking (prompt point 4). Lexical non-manuals (mouthing, the face of a sign) come with each dictionary
clip; only grammatical marks that span a clause are added here, and each one records its evidence in the IR:

  wh_question  brows raised over the clause, strongest at the end — DATA: TSL51 researcher sentences, 15 WH-questions vs 237
               statements, brow-raise blendshapes +0.069 vs +0.014 (first 40 %) and +0.083 vs +0.044 (last 40 %), brow-down lower;
               Mann-Whitney p = 0.001 (one signer — to be confirmed by the Deaf reviewers).
  yn_question  the same brow raise + a small forward head tilt on the final sign — RULE: Thai ไหม/หรือเปล่า has no recorded sign
               and no yes/no sentences exist in the data; marked `source: rule` for expert validation.
Amplitudes are kept within the range the signers' own faces show (no cartoon exaggeration).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .retarget import MORPHS
from .rig import BONE_INDEX

EVIDENCE = {
    "wh_question": dict(brows="raise", source="data", ref="TSL51 15 WH vs 237 statements, brow-raise +0.07 (p=0.001, 1 signer)"),
    "yn_question": dict(brows="raise", head="forward tilt", source="rule", ref="no yes/no question data; typological default"),
}


def _envelope(T, a, b, ramp):
    e = np.zeros(T, np.float32)
    if b <= a:
        return e
    for t in range(max(0, a - ramp), min(T, b + ramp)):
        if t < a:
            e[t] = 0.5 - 0.5 * np.cos(np.pi * (t - (a - ramp)) / ramp)
        elif t >= b:
            e[t] = 0.5 + 0.5 * np.cos(np.pi * (t - b + 1) / ramp)
        else:
            e[t] = 1.0
    return e


def apply_overlays(q, m, overlays, segs, fps=25.0):
    """overlays: [{type, units: [unit indices]}] (from the IR clauses)."""
    q = q.copy(); m = m.copy()
    T = len(q)
    ramp = int(0.2 * fps)
    bi = MORPHS.index("Fcl_BRW_Surprised")
    for ov in overlays:
        typ = ov["type"]
        if typ not in EVIDENCE:
            continue
        us = [s for s in segs if s["unit"] in ov["units"] and s["kind"] != "transition"]
        if not us:
            continue
        a, b = min(s["start"] for s in us), max(s["end"] for s in us)
        env = _envelope(T, a, b, ramp)
        # rising toward the end of the clause (the data shows the raise is larger late)
        rise = np.clip(0.6 + 0.4 * (np.arange(T) - a) / max(b - a, 1), 0.6, 1.0)
        m[:, bi] = np.maximum(m[:, bi], 0.38 * env * rise)
        if typ == "yn_question":
            last = max(us, key=lambda s: s["end"])
            e2 = _envelope(T, last["start"], last["end"], ramp)
            tilt = R.from_rotvec(np.outer(e2, [np.deg2rad(-7), 0, 0]))         # chin slightly down / head forward (+X lifts the chin)
            h = BONE_INDEX["head"]
            q[:, h] = (R.from_quat(q[:, h]) * tilt).as_quat()
    return q.astype(np.float32), m.astype(np.float32)
