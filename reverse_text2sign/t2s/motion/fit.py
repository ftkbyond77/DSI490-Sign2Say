"""Fit a motion made for the library avatar (models/avatar.vrm) onto another VRM body, so every character signs the same sign.

The motion library stores local rotations of normalized humanoid bones. Copying them to a body with other proportions keeps the
*angles* but not the *places*: a hand at the chin of the library avatar ends up at the throat of a character with relatively
shorter arms, and next to the chest of a character with wider shoulders. In sign language the place is part of the sign, so the
arms are re-solved per frame:

  1. forward kinematics of the library avatar → each wrist position, elbow plane and hand orientation;
  2. the wrist is expressed relative to nearby body landmarks (eyes, head, neck, chest, shoulders), weighted by closeness; each
     offset is scaled by the proportion that governs its region — head size around the face (anime heads differ most),
     shoulder width sideways and torso length up / forward around the torso — and re-attached to the new body's landmarks
     → the wrist target;
  3. two-bone IK (upper arm, forearm) on the new body reaches the target in the library elbow plane; the hand keeps its world
     orientation and the fingers / spine / head keep their rotations.

Bodies may lack optional in-between bones (VRM 1.0 exports without upperChest, …): their rotation is folded into the children so
the bones that exist keep the library's world rotations (the browser skips absent bones). Fitting the library avatar itself is
the identity (skipped). Accuracy per character: tests/test_motion.py::test_fit_to_character (result_v2.md §6).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rig import BONE_INDEX, BONES, Rig, safe_quat

ANCHORS = ["eyes", "head", "neck", "upperChest", "chest", "leftShoulder", "rightShoulder"]
FACE = {"eyes", "head"}
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


def anchor_positions(rig: Rig, q, anchors):
    """world positions [T,3] of the anchors (`eyes` = midpoint of the eye bones, carried by the head)."""
    names = [a for a in anchors if a != "eyes"] + (["leftEye", "rightEye"] if "eyes" in anchors else [])
    fk = rig.fk(q, names)
    out = {a: fk[a][0] for a in anchors if a != "eyes"}
    if "eyes" in anchors:
        out["eyes"] = 0.5 * (fk["leftEye"][0] + fk["rightEye"][0])
    return out


def _eyes(rig):
    return 0.5 * (rig.rest["leftEye"] + rig.rest["rightEye"]) if {"leftEye", "rightEye"} <= set(rig.rest) else None


class Fit:
    def __init__(self, body: Rig, ref: Rig):
        self.body, self.ref = body, ref
        need = {f"{s}{b}" for s in ("left", "right") for b in ("UpperArm", "LowerArm", "Hand")} | {"spine", "head"}
        missing = need - set(body.rest)
        if missing:
            raise ValueError(f"humanoid bones missing: {sorted(missing)}")
        # optional in-between bones the body lacks (upperChest, chest, neck, shoulders): their rotation is folded into the
        # children, top-down, so the bones that exist end up with the same world rotation (the browser skips absent bones)
        self.folded = [b for b in ref.order if b in BONE_INDEX and b not in body.rest]
        has = lambda r, a: (_eyes(r) is not None) if a == "eyes" else a in r.rest            # noqa: E731
        self.anchors = [a for a in ANCHORS if has(body, a) and has(ref, a)]
        torso = float(np.linalg.norm(body.rest["head"] - body.rest["spine"]) / np.linalg.norm(ref.rest["head"] - ref.rest["spine"]))
        self.scale = torso
        if "eyes" in self.anchors and "neck" in body.rest and "neck" in ref.rest:
            head = float(np.linalg.norm(_eyes(body) - body.rest["neck"]) / np.linalg.norm(_eyes(ref) - ref.rest["neck"]))
        else:
            head = torso
        sw = body.sw / ref.sw
        self.head_scale, self.sw_scale = head, sw
        self.anchor_scale = {a: (np.full(3, head) if a in FACE else
                                 np.full(3, 0.5 * (head + torso)) if a == "neck" else np.array([sw, torso, torso]))
                             for a in self.anchors}
        self.identity = not self.folded and all(np.allclose(body.offset(b), ref.offset(b), atol=1e-4) for b in BONES)
        self.len = {s: (float(np.linalg.norm(body.offset(f"{s}LowerArm"))), float(np.linalg.norm(body.offset(f"{s}Hand"))))
                    for s in ("left", "right")}

    def fold(self, q):
        """move the rotation of every absent bone onto its children (in place on a float64 copy)."""
        for m in self.folded:
            rm = R.from_quat(safe_quat(q[:, BONE_INDEX[m]]))
            for c in BONES:
                if self.ref.parent.get(c) == m:
                    q[:, BONE_INDEX[c]] = (rm * R.from_quat(safe_quat(q[:, BONE_INDEX[c]]))).as_quat()
            q[:, BONE_INDEX[m]] = (0.0, 0.0, 0.0, 1.0)
        return q

    def targets(self, q0, q):
        """wrist targets on the body {side: [T,3]} from the library pose q0 (body pose q = folded q0)."""
        T = q0.shape[0]
        ar = anchor_positions(self.ref, q0, self.anchors)
        ab = anchor_positions(self.body, q, self.anchors)
        fr = self.ref.fk(q0, ["leftHand", "rightHand"])
        out = {}
        for s in ("left", "right"):
            wrist = fr[f"{s}Hand"][0]
            num, den = np.zeros((T, 3)), np.zeros((T, 1))
            for a in self.anchors:
                off = wrist - ar[a]
                w = 1.0 / (np.einsum("ij,ij->i", off, off) + EPS ** 2)[:, None]
                num += w * (ab[a] + self.anchor_scale[a] * off)
                den += w
            out[s] = num / den
        return out

    def __call__(self, q):
        """q [T, len(BONES), 4] library rotations → the same signing on this body (float32)."""
        if self.identity:
            return q
        q0 = np.array(q, np.float64)
        q = self.fold(q0.copy())
        T = q.shape[0]
        fr = self.ref.fk(q0, [f"{s}{b}" for s in ("left", "right") for b in ("UpperArm", "LowerArm", "Hand")])
        fb = self.body.fk(q, list(dict.fromkeys(["leftUpperArm", "rightUpperArm"] +
                                                [self.body.parent[f"{s}UpperArm"] for s in ("left", "right")])))
        tg = self.targets(q0, q)
        for s in ("left", "right"):
            target = tg[s]
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
            Wu, Wl, Wh = (fr[f"{s}{b}"][1] for b in ("UpperArm", "LowerArm", "Hand"))
            Wpar = fb[self.body.parent[f"{s}UpperArm"]][1]          # the body's own parent (shoulder, or the chest without one)
            Wu_new = _swing(_unit(Wu.apply(np.repeat(_unit(self.body.offset(f"{s}LowerArm"))[None], T, 0))), _unit(E - S)) * Wu
            Wl0 = Wu_new * R.from_quat(safe_quat(q0[:, BONE_INDEX[f"{s}LowerArm"]]))
            Wl_new = _swing(_unit(Wl0.apply(np.repeat(_unit(self.body.offset(f"{s}Hand"))[None], T, 0))), _unit(W - E)) * Wl0
            q[:, BONE_INDEX[f"{s}UpperArm"]] = (Wpar.inv() * Wu_new).as_quat()
            q[:, BONE_INDEX[f"{s}LowerArm"]] = (Wu_new.inv() * Wl_new).as_quat()
            q[:, BONE_INDEX[f"{s}Hand"]] = (Wl_new.inv() * Wh).as_quat()
        # keep quaternion signs continuous for the browser's per-frame slerp
        for b in range(q.shape[1]):
            d = np.einsum("ij,ij->i", q[1:, b], q[:-1, b]) < 0
            if d.any():
                sign = np.cumprod(np.where(np.concatenate([[False], d]), -1.0, 1.0))
                q[:, b] *= sign[:, None]
        return q.astype(np.float32)

    def placement_error(self, q0, q_fit):
        """per frame and side: distance (m, library scale) between the fitted wrist and its target, and the target expressed at
        the NEAREST anchor (how far the hand lands from where the library avatar puts it relative to that body part)."""
        q0 = np.asarray(q0, np.float64)
        ab = anchor_positions(self.body, np.asarray(q_fit, np.float64), self.anchors)
        ar = anchor_positions(self.ref, q0, self.anchors)
        fr = self.ref.fk(q0, ["leftHand", "rightHand"])
        fb = self.body.fk(np.asarray(q_fit, np.float64), ["leftHand", "rightHand"])
        out = {}
        for s in ("left", "right"):
            wr, wb = fr[f"{s}Hand"][0], fb[f"{s}Hand"][0]
            D = np.stack([np.linalg.norm(wr - ar[a], axis=1) for a in self.anchors], 1)
            near = D.argmin(1)
            err = np.empty(len(wr))
            for i, a in enumerate(self.anchors):
                m = near == i
                if m.any():
                    err[m] = np.linalg.norm((wb[m] - ab[a][m]) / self.anchor_scale[a] - (wr[m] - ar[a][m]), axis=1)
            out[s] = (err, np.array(self.anchors)[near])
        return out
