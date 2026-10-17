"""The avatar skeleton (VRM humanoid) in the model's own frame, and forward kinematics.

Convention (= three-vrm normalized humanoid bones of a VRM 0.x model, which are children of the 180°-rotated scene):
  model frame: +Y up, the avatar faces −Z, its LEFT side is −X  (raw VRM 0.x glTF coordinates)
  every normalized bone has identity rest rotation and world-aligned axes, so a bone's LOCAL quaternion q_b gives
  world_rot[b] = world_rot[parent] ⊗ q_b and world_pos[b] = world_pos[parent] + world_rot[parent] · (rest[b] − rest[parent]).
The browser sets exactly these local quaternions on `vrm.humanoid.getNormalizedBoneNode(name)` — no axis flips anywhere.
"""
from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .. import paths

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
SEGS = ["Proximal", "Intermediate", "Distal"]
SIDES = ["left", "right"]

# the animated bones, in the fixed order of the motion arrays (hips / legs are never moved: half-body signing avatar)
BONES = (["spine", "chest", "upperChest", "neck", "head"] +
         [f"{s}{b}" for s in SIDES for b in ["Shoulder", "UpperArm", "LowerArm", "Hand"]] +
         [f"{s}{f}{g}" for s in SIDES for f in FINGERS for g in SEGS])
BONE_INDEX = {b: i for i, b in enumerate(BONES)}
# the motion uses VRM 0.x thumb names (Proximal · Intermediate · Distal); VRM 1.0 — and three-vrm for EVERY model, it renames
# VRM 0.x thumbs on load — calls the same three joints Metacarpal · Proximal · Distal. Used by the rig loader (VRM 1.0 files),
# the browser binding (service/web/src/avatar.ts mirrors this table) and the .vrma export.
TO_VRM1 = {f"{s}Thumb{a}": f"{s}Thumb{b}" for s in SIDES for a, b in (("Proximal", "Metacarpal"), ("Intermediate", "Proximal"))}
FROM_VRM1 = {v: k for k, v in TO_VRM1.items()}
# what a body needs to sign (VRM requires the limbs; fingers are optional in the spec but signing is impossible without them).
# Optional in-between bones (chest, upperChest, neck, shoulders) may be absent: Fit folds their rotation into the nearest
# present ancestor / child so the visible pose is the same.
REQUIRED = ({"hips", "spine", "head"} | {f"{s}{b}" for s in SIDES for b in ("UpperArm", "LowerArm", "Hand")} |
            {f"{s}{f}{g}" for s in SIDES for f in FINGERS for g in SEGS if not (f == "Thumb" and g == "Proximal")})


class Rig:
    def __init__(self, vrm_path=None):
        vrm_path = Path(vrm_path or (paths.AVATAR if paths.AVATAR.exists() else paths.AVATAR_SRC))
        b = vrm_path.read_bytes()
        cl = struct.unpack("<I", b[12:16])[0]
        j = json.loads(b[20:20 + cl])
        nodes = j["nodes"]
        par = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
        ext = j["extensions"]
        hb = ({x["bone"]: x["node"] for x in ext["VRM"]["humanoid"]["humanBones"]} if "VRM" in ext
              else {FROM_VRM1.get(k, k): v["node"] for k, v in ext["VRMC_vrm"]["humanoid"]["humanBones"].items()})
        self.version = "0" if "VRM" in ext else "1"
        missing = REQUIRED - set(hb)
        if missing:
            raise ValueError(f"humanoid bones missing: {sorted(missing)}")
        # normalized humanoid bones (three-vrm / UniVRM) sit at the raw bones' REST WORLD positions with identity rotation, so
        # only the world positions matter: full TRS chain (Blender / other exporters store rest rotations and scales, VRoid does not)
        memo = {}

        def world(i):
            if i in memo:
                return memo[i]
            n = nodes[i]
            if "matrix" in n:
                M = np.array(n["matrix"], float).reshape(4, 4).T
            else:
                M = np.eye(4)
                M[:3, :3] = R.from_quat(n.get("rotation", [0, 0, 0, 1])).as_matrix() * np.array(n.get("scale", [1, 1, 1]), float)
                M[:3, 3] = n.get("translation", [0, 0, 0])
            p = par.get(i)
            memo[i] = M if p is None else world(p) @ M
            return memo[i]
        node2bone = {v: k for k, v in hb.items()}
        self.rest = {k: world(v)[:3, 3].copy() for k, v in hb.items()}
        self.present = set(hb)
        self.parent = {}
        for k, v in hb.items():                       # nearest humanoid ancestor
            p = par.get(v)
            while p is not None and p not in node2bone:
                p = par.get(p)
            self.parent[k] = node2bone.get(p)
        if self.version == "1":                       # VRM 1.0 faces +Z: express everything in the VRM 0.x frame used here
            for k in self.rest:
                self.rest[k] = self.rest[k] * np.array([-1, 1, -1])
        # finger tips: the distal bone has no child in VRoid → extend it by its own intermediate→distal length
        for s in SIDES:
            for f in FINGERS:
                a, d = self.rest[f"{s}{f}Intermediate"], self.rest[f"{s}{f}Distal"]
                self.rest[f"{s}{f}Tip"] = d + (d - a) * 0.9
                self.parent[f"{s}{f}Tip"] = f"{s}{f}Distal"
        self.sw = float(np.linalg.norm(self.rest["leftUpperArm"] - self.rest["rightUpperArm"]))
        self.order = self._topo()

    def _topo(self):
        out, seen = [], set()

        def visit(b):
            if b in seen or b is None:
                return
            visit(self.parent.get(b))
            seen.add(b); out.append(b)
        for b in self.rest:
            visit(b)
        return out

    def offset(self, b):
        p = self.parent.get(b)
        return self.rest[b] - (self.rest[p] if p else 0.0)

    def fk(self, local_q, names=None):
        """local_q: [T, len(BONES), 4] (x, y, z, w) → dict bone → (world_pos [T,3], world_rot Rotation[T]).
        Bones not in BONES keep identity. names: the bones to return (default all incl. finger tips)."""
        T = local_q.shape[0]
        wr, wp = {}, {}
        ident = R.identity(T)
        for b in self.order:
            p = self.parent.get(b)
            lq = R.from_quat(safe_quat(local_q[:, BONE_INDEX[b]])) if b in BONE_INDEX else ident
            if p is None:
                wr[b] = lq; wp[b] = np.repeat(self.rest[b][None], T, 0)
            else:
                wr[b] = wr[p] * lq
                wp[b] = wp[p] + wr[p].apply(np.repeat(self.offset(b)[None], T, 0))
        names = names or list(self.rest)
        return {n: (wp[n], wr[n]) for n in names}


def safe_quat(q):
    """NaN / zero rows → identity (missing data stays at the rest pose in FK)."""
    q = np.asarray(q, np.float64).copy()
    bad = ~np.isfinite(q).all(-1) | (np.linalg.norm(np.nan_to_num(q), axis=-1) < 1e-9)
    q[bad] = (0, 0, 0, 1)
    return q


@lru_cache(maxsize=2)
def get_rig(path=None):
    return Rig(path)


def identity_pose(T):
    q = np.zeros((T, len(BONES), 4), np.float32)
    q[..., 3] = 1.0
    return q
