"""Fit a motion made for the library avatar (models/avatar.vrm) onto another VRM body, so every character signs the same sign.

The motion library stores local rotations of normalized humanoid bones. Copying them to a body with other proportions keeps the
*angles* but not the *places*: a hand at the chin of the library avatar ends up at the throat of a character with relatively
shorter arms (the delivered extra models have arm/torso ratios 0.87–0.88 vs 0.95). In sign language the place is part of the
sign, so the arms are re-solved per frame:

  1. forward kinematics of the library avatar → each wrist position, elbow plane and hand orientation;
  2. the wrist is expressed relative to nearby body landmarks (head, neck, chest, shoulders), weighted by closeness, and the
     same offsets — scaled by the torso-length ratio — are re-attached to the new body's landmarks → the wrist target;
  3. two-bone IK (upper arm, forearm) on the new body reaches the target in the library elbow plane; the hand keeps its world
     orientation and the fingers / spine / head keep their rotations.

Spine, neck, head, shoulders and fingers are unchanged, so for the library avatar itself fitting is the identity (skipped).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rig import BONE_INDEX, BONES, Rig, safe_quat

ANCHORS = ["head", "neck", "upperChest", "chest", "leftShoulder", "rightShoulder"]
EPS = 0.06                     # m (library scale): landmark weights 1 / (d² + EPS²) — the nearest landmarks dominate


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _swing(a, b):
    """minimal rotations taking unit vectors a → b (rows)."""
    c = np.cross(a, b)
    s = np.linalg.norm(c, axis=-1)
    d = np.einsum("ij,ij->i", a, b)
    ang = np.arctan2(s, d)
    axis = c / np.maximum(s, 1e-12)[:, None]
    flip = (s < 1e-9) & (d < 0)                                     # antiparallel: any axis ⟂ a
    if flip.any():
        alt = np.cross(a[flip], np.array([0.0, 1.0, 0.0]))
        alt[np.linalg.norm(alt, axis=-1) < 1e-6] = [1.0, 0.0, 0.0]
        axis[flip] = _unit(alt)
    return R.from_rotvec(axis * ang[:, None])


class Fit:
    def __init__(self, body: Rig, ref: Rig):
        self.body, self.ref = body, ref
        need = set(ANCHORS) | {f"{s}{b}" for s in ("left", "right") for b in ("Shoulder", "UpperArm", "LowerArm", "Hand")} | {"spine"}
        missing = need - set(body.rest)
        if missing:
            raise ValueError(f"humanoid bones missing: {sorted(missing)}")
        self.scale = float(np.linalg.norm(body.rest["head"] - body.rest["spine"]) / np.linalg.norm(ref.rest["head"] - ref.rest["spine"]))
        self.identity = all(np.allclose(body.offset(b), ref.offset(b), atol=1e-4) for b in BONES)
        self.len = {s: (float(np.linalg.norm(body.offset(f"{s}LowerArm"))), float(np.linalg.norm(body.offset(f"{s}Hand"))))
                    for s in ("left", "right")}

    def __call__(self, q):
        """q [T, len(BONES), 4] library rotations → the same signing on this body (float32)."""
        if self.identity:
            return q
        q = np.array(q, np.float64)
        T = q.shape[0]
        names = ANCHORS + [f"{s}{b}" for s in ("left", "right") for b in ("Shoulder", "UpperArm", "LowerArm", "Hand")]
        fr = self.ref.fk(q, names)
        fb = self.body.fk(q, ANCHORS + ["leftUpperArm", "rightUpperArm"])
        for s in ("left", "right"):
            wrist = fr[f"{s}Hand"][0]
            num, den = np.zeros((T, 3)), np.zeros((T, 1))
            for a in ANCHORS:
                off = wrist - fr[a][0]
                w = 1.0 / (np.einsum("ij,ij->i", off, off) + EPS ** 2)[:, None]
                num += w * (fb[a][0] + self.scale * off)
                den += w
            target = num / den
            # two-bone IK in the library elbow plane
            S = fb[f"{s}UpperArm"][0]
            la, lb = self.len[s]
            v = target - S
            dist = np.clip(np.linalg.norm(v, axis=-1), abs(la - lb) + 1e-4, (la + lb) * (1 - 1e-4))
            u = _unit(v)
            Sr, Er = fr[f"{s}UpperArm"][0], fr[f"{s}LowerArm"][0]
            ur = _unit(fr[f"{s}Hand"][0] - Sr)
            pole = (Er - Sr) - np.einsum("ij,ij->i", Er - Sr, ur)[:, None] * ur
            pole = pole - np.einsum("ij,ij->i", pole, u)[:, None] * u
            weak = np.linalg.norm(pole, axis=-1) < 1e-6                 # straight library arm: bend downward / backward
            if weak.any():
                alt = np.array([0.0, -1.0, 0.3])
                pole[weak] = alt - (u[weak] @ alt)[:, None] * u[weak]
            n = _unit(pole)
            x = (la ** 2 - lb ** 2 + dist ** 2) / (2 * dist)
            h = np.sqrt(np.maximum(la ** 2 - x ** 2, 0.0))
            E = S + u * x[:, None] + n * h[:, None]
            W = S + u * dist[:, None]
            Wsh, Wu, Wl, Wh = (fr[f"{s}{b}"][1] for b in ("Shoulder", "UpperArm", "LowerArm", "Hand"))
            Wu_new = _swing(_unit(Wu.apply(np.repeat(_unit(self.body.offset(f"{s}LowerArm"))[None], T, 0))), _unit(E - S)) * Wu
            Wl0 = Wu_new * R.from_quat(safe_quat(q[:, BONE_INDEX[f"{s}LowerArm"]]))
            Wl_new = _swing(_unit(Wl0.apply(np.repeat(_unit(self.body.offset(f"{s}Hand"))[None], T, 0))), _unit(W - E)) * Wl0
            q[:, BONE_INDEX[f"{s}UpperArm"]] = (Wsh.inv() * Wu_new).as_quat()
            q[:, BONE_INDEX[f"{s}LowerArm"]] = (Wu_new.inv() * Wl_new).as_quat()
            q[:, BONE_INDEX[f"{s}Hand"]] = (Wl_new.inv() * Wh).as_quat()
        # keep quaternion signs continuous for the browser's per-frame slerp
        for b in range(q.shape[1]):
            d = np.einsum("ij,ij->i", q[1:, b], q[:-1, b]) < 0
            if d.any():
                sign = np.cumprod(np.where(np.concatenate([[False], d]), -1.0, 1.0))
                q[:, b] *= sign[:, None]
        return q.astype(np.float32)
