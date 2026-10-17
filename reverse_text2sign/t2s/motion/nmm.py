"""Non-manual features (prompt point 4; v2: result_v2.md §5.5) — two levels, kept apart and both recorded in the IR:

  lexical    the face and head of the dictionary signer, carried by each sign's clip (mouth gestures, a sign's own brows, the
             head-shake of a negation sign). Mouth channels stay lexical everywhere.
  sentence   marks that span part of a clause, added on top with explicit SCOPE (which signs) and EVIDENCE (measured on
             recorded sentences by t2s/eval/nmm_data.py; numbers below). Blend rule: brows = max(lexical, sentence); head =
             lexical rotation ∘ sentence rotation; the mouth is never touched. Nothing spans a whole sentence by default.

  wh_question  brow raise from the 2nd sign of the clause to its end (not on the first sign): TSL51 15 WH vs 72 subject-first
               statements — brow bins 0.002 · 0.054 · 0.048 · 0.062 · 0.050 vs 0.015 · 0.015 · 0.016 · 0.024 · 0.049 (one signer)
  yn_question  brow raise + forward head tilt on the final sign: 4 recorded yes/no phrases vs 19 statements (several signers) —
               head pitch at the end p = 0.03, brows p = 0.09 (weak: n = 4; flagged `weak` for the reviewers)
  negation     head-shake on the negation sign itself, ~5° at ~2.5 Hz, TOPPED UP only when the clip's own shake is weaker:
               dictionary negation signs shake twice as much as other signs (109 clips, 12 signers, p = 6e-17); 61 % of them
               shake only weakly, so a sentence would otherwise lose the mark. No spreading over the predicate (no sentence data).
  topic        NOT applied: topic-first TSL51 sentences show no brow difference on the first sign (165 vs 72, p = 0.53).
Amplitudes stay inside the range the signers' own faces show (no cartoon exaggeration).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .retarget import MORPHS
from .rig import BONE_INDEX

EVIDENCE = {
    "wh_question": dict(brows="raise", scope="2nd sign → end of clause", source="data",
                        ref="TSL51 15 WH vs 72 statements: brows up from the 2nd fifth on (one signer)"),
    "yn_question": dict(brows="raise", head="forward tilt", scope="final sign", source="data (weak, n = 4)",
                        ref="4 recorded yes/no phrases vs 19 statements: head pitch p = 0.03, brows p = 0.09"),
    "negation": dict(head="shake", scope="negation sign", source="data",
                     ref="109 dictionary negation signs, 12 signers: head-shake 2× other signs (p = 6e-17); ~5° at 2.5 Hz"),
}
NOT_APPLIED = {"topic": "no evidence: topic-first TSL51 sentences show no brow change on the first sign (p = 0.53)"}
SHAKE_DEG, SHAKE_HZ = 5.0, 2.5


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


def _yaw_osc(q, a, b, fps):
    """RMS (degrees) of the head's own yaw oscillation in [a, b) (0.6 s moving average removed)."""
    hq = R.from_quat(q[a:b, BONE_INDEX["head"]]) * R.from_quat(q[a:b, BONE_INDEX["neck"]])
    yaw = np.degrees(hq.as_euler("YXZ")[:, 0])
    k = max(3, int(0.6 * fps))
    if len(yaw) < k:
        return 0.0
    tr = np.convolve(yaw, np.ones(k) / k, "same")
    return float(np.sqrt(np.mean((yaw - tr)[k // 2: -k // 2 or None] ** 2))) if len(yaw) > k else 0.0


def apply_overlays(q, m, overlays, segs, fps=25.0):
    """overlays: [{type, units: [unit indices], neg_units: [...]}] (IR clauses). Returns q, m and the list of marks applied
    (type, frames, scope, evidence) for the IR."""
    q = q.copy(); m = m.copy()
    T = len(q)
    ramp = int(0.2 * fps)
    bi = MORPHS.index("Fcl_BRW_Surprised")
    h = BONE_INDEX["head"]
    applied = []
    for ov in overlays:
        typ = ov["type"]
        us = sorted((s for s in segs if s["unit"] in ov["units"] and s["kind"] != "transition"), key=lambda s: s["start"])
        if typ == "wh_question" and us:
            scope = us[1:] if len(us) > 1 else us
            a, b = scope[0]["start"], us[-1]["end"]
            env = _envelope(T, a, b, ramp)
            m[:, bi] = np.maximum(m[:, bi], 0.38 * env)
            applied.append(dict(type=typ, frames=[int(a), int(b)], units=[s["unit"] for s in scope], **EVIDENCE[typ]))
        if typ == "yn_question" and us:
            last = us[-1]
            a, b = last["start"], last["end"]
            env = _envelope(T, a, b, ramp)
            m[:, bi] = np.maximum(m[:, bi], 0.3 * env)
            tilt = R.from_rotvec(np.outer(env, [np.deg2rad(-7), 0, 0]))         # chin slightly down / head forward (+X lifts the chin)
            q[:, h] = (R.from_quat(q[:, h]) * tilt).as_quat()
            applied.append(dict(type=typ, frames=[int(a), int(b)], units=[last["unit"]], **EVIDENCE[typ]))
        for s in (x for x in segs if x["unit"] in ov.get("neg_units", ()) and x["kind"] != "transition"):
            a, b = s["start"], s["end"]
            own = _yaw_osc(q, a, b, fps)
            target_rms = SHAKE_DEG / np.sqrt(2)
            if own >= target_rms:
                applied.append(dict(type="negation", frames=[int(a), int(b)], units=[s["unit"]], added=False,
                                    note=f"the clip's own head-shake ({own:.1f}° rms) kept", **EVIDENCE["negation"]))
                continue
            amp = np.sqrt(2) * (target_rms - own)
            env = _envelope(T, a, b, max(2, ramp // 2))
            t = np.arange(T) / fps
            yaw = np.deg2rad(amp) * env * np.sin(2 * np.pi * SHAKE_HZ * (t - a / fps))
            q[:, h] = (R.from_quat(q[:, h]) * R.from_rotvec(np.outer(yaw, [0, 1.0, 0]))).as_quat()
            applied.append(dict(type="negation", frames=[int(a), int(b)], units=[s["unit"]], added=True,
                                note=f"head-shake topped up by {amp:.1f}° (clip's own {own:.1f}° rms)", **EVIDENCE["negation"]))
    return q.astype(np.float32), m.astype(np.float32), applied
