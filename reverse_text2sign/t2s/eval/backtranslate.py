"""Motion level, automatic proxy for "is the avatar's sign recognisable?": the forward ThaiSLM v8 recogniser (Thai-only landmark
encoder, trained on real signers) reads the AVATAR's motion and names the sign.

  avatar FK → MediaPipe-style landmarks (pose 33 incl. face points carried by the head bone, hands 21 from the finger bones;
  orthographic camera in front of the avatar) → forward Standard Schema → canonical hands → v8 encoder embedding
  → rank of the true concept among the forward bank's concepts, with the SOURCE clip removed from the bank (leave-source-out:
  the avatar must look like the sign as other people sign it, not like a copy of its own clip).
Upper bound: the same score for the real signer's landmarks of the same clip (same leave-source-out bank).
Ratio avatar / human = how much of the recognisable signal survives retargeting. It is a proxy — the Deaf reviewers' judgement
(comprehension level) is the real test.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from .. import paths
from ..motion.library import Library
from ..motion.rig import FINGERS, SEGS, get_rig

# face points relative to the head bone (rig frame, metres): nose tip / eyes / ears / mouth corners measured on the VRoid mesh
FACE = {0: (0.0, 0.013, -0.111), 2: (-0.033, 0.06, -0.075), 5: (0.033, 0.06, -0.075), 1: (-0.022, 0.06, -0.078),
        3: (-0.045, 0.06, -0.07), 4: (0.022, 0.06, -0.078), 6: (0.045, 0.06, -0.07), 7: (-0.072, 0.04, -0.005),
        8: (0.072, 0.04, -0.005), 9: (-0.018, -0.02, -0.092), 10: (0.018, -0.02, -0.092)}
POSE_BONES = {11: "leftUpperArm", 12: "rightUpperArm", 13: "leftLowerArm", 14: "rightLowerArm", 15: "leftHand", 16: "rightHand"}
CROP_C, CROP_H = np.array([0.0, 1.30]), 1.30         # square virtual camera: centre (x, y), height in metres


def avatar_raw(q, rig=None):
    """local quaternions [T,B,4] → MediaPipe raw dict (normalised image coords of a 720×720 orthographic front camera)."""
    rig = rig or get_rig()
    names = list(POSE_BONES.values()) + ["head", "hips"] + [f"{s}{f}{g}" for s in ("left", "right") for f in FINGERS for g in SEGS] + \
        [f"{s}{f}Tip" for s in ("left", "right") for f in FINGERS]
    fk = rig.fk(q, names=names)
    T = len(q)

    def img(P):
        x = (-P[..., 0] - (CROP_C[0] - CROP_H / 2)) / CROP_H
        y = ((CROP_C[1] + CROP_H / 2) - P[..., 1]) / CROP_H
        z = P[..., 2] / CROP_H
        return np.stack([x, y, z], -1)
    pose = np.zeros((T, 33, 4), np.float32)
    hp, hr = fk["head"]
    hp_rest = rig.rest["head"]
    for i, off in FACE.items():
        # FACE offsets are written with the avatar's left at −X (rig frame): left eye (2) has x < 0
        o = np.array(off) + 0 * hp_rest
        pose[:, i, :3] = img(hp + hr.apply(np.repeat(o[None], T, 0)))
    for i, b in POSE_BONES.items():
        pose[:, i, :3] = img(fk[b][0])
    hips = fk["hips"][0]
    pose[:, 23, :3] = img(hips + np.array([-0.09, 0, 0])); pose[:, 24, :3] = img(hips + np.array([0.09, 0, 0]))
    pose[:, :, 3] = 1.0
    pose[:, 17:23, 3] = 0.0
    hands = {}
    for side in ("left", "right"):
        h = np.zeros((T, 21, 3), np.float32)
        h[:, 0] = img(fk[f"{side}Hand"][0])
        for f, base in zip(FINGERS, (1, 5, 9, 13, 17)):
            seq = [f"{side}{f}{g}" for g in SEGS] + [f"{side}{f}Tip"]
            for k, b in enumerate(seq):
                h[:, base + k] = img(fk[b][0])
        h[..., 2] -= h[:, :1, 2]
        hands[side] = h
    return dict(t=np.arange(T) / 25.0, hw=np.array([720, 720]), pose=pose, lh=hands["left"], rh=hands["right"],
                face=np.zeros((T, 68, 2), np.float32), bs=np.zeros((T, 52), np.float32), src_fps=np.float32(25))


class Forward:
    """the forward project's deployed encoder + bank (read-only)."""

    def __init__(self):
        sys.path.insert(0, str(paths.REPO))
        import json
        from modules.runtime import make_backend
        self.be = make_backend(paths.REPO / "models", "onnx", "cpu")
        b = np.load(paths.REPO / "models" / "bank.npz", allow_pickle=True)
        self.Z = b["Z"].astype(np.float32)
        self.c = b["c"].astype(np.int64)
        self.clip = b["clip"].astype(str) if "clip" in b.files else np.array([""] * len(self.c))
        self.concepts = json.loads((paths.REPO / "models" / "concepts.json").read_text(encoding="utf-8"))
        self.cidx = {c: i for i, c in enumerate(self.concepts)}

    def embed_raw(self, raw, a=None, b=None):
        from modules.mediapipe_pose import to_standard
        from modules.parts import to_iso
        from modules.schema import canonical_hands
        d = to_standard(raw)
        hw = tuple(int(x) for x in d["hw"])
        kp, sc, _ = canonical_hands(d["kp"], d["sc"], hw)
        kpi = to_iso(kp, hw)
        a = a or 0; b = b or len(kpi)
        z = self.be.embed([self.be.prep(kpi[a:b], sc[a:b])])[0]
        return z / (np.linalg.norm(z) + 1e-8)

    def rank(self, z, concept, exclude_clips):
        """rank (1 = best) of `concept` by per-concept max similarity, excluding the given bank clips; None if not rankable."""
        if concept not in self.cidx:
            return None
        keep = ~np.isin(self.clip, list(exclude_clips))
        S = self.Z[keep] @ z
        c = self.c[keep]
        if not (c == self.cidx[concept]).any():
            return None                                    # no other recording of this concept left in the bank
        best = np.full(len(self.concepts), -2.0, np.float32)
        np.maximum.at(best, c, S)
        return int((best > best[self.cidx[concept]]).sum()) + 1


def run(n=600, seed=0):
    from ..data.quality import load_raw
    from ..motion.clean import clean, mirror
    lib = Library()
    fw = Forward()
    rig = get_rig()
    sel = pd.read_parquet(paths.WORK / "select" / "exemplars.parquet")
    sel = sel[sel.dataset.isin(["ttrs", "th_sl"]) & sel.concept.isin(fw.cidx) & sel.concept.isin(lib.clips)]
    # only concepts that keep ≥ 1 OTHER recording in the forward bank after removing the source clip
    counts = pd.Series(fw.c).value_counts()
    sel = sel[[counts.get(fw.cidx[c], 0) >= 2 for c in sel.concept]]
    sel = sel.sample(min(n, len(sel)), random_state=seed)
    rows = []
    for r in sel.itertuples():
        q, m, info = lib.get(r.concept)
        a, b = info["stroke"]
        za = fw.embed_raw(avatar_raw(q, rig), a, b)
        src = r.clip_id
        ra = fw.rank(za, r.concept, {src})
        raw = load_raw(r.raw_file)
        if r.mirrored:
            raw = mirror(raw)
        raw = clean(raw)[0]
        zh = fw.embed_raw(raw, int(r.span_f0), int(r.span_f1))
        rh = fw.rank(zh, r.concept, {src})
        rows.append(dict(concept=r.concept, clip=src, dataset=r.dataset, tier=r.tier, rank_avatar=ra, rank_human=rh,
                         cos_avatar_human=float(za @ zh), wrist_err=info["fidelity"]["wrist"], finger_err=info["fidelity"]["fingertip"]))
    d = pd.DataFrame(rows).dropna(subset=["rank_avatar", "rank_human"])
    (paths.WORK / "eval").mkdir(parents=True, exist_ok=True)
    d.to_csv(paths.WORK / "eval" / "backtranslate.csv", index=False, encoding="utf-8-sig")
    out = {}
    for who in ("avatar", "human"):
        rk = d[f"rank_{who}"]
        out[who] = dict(top1=float((rk <= 1).mean()), top5=float((rk <= 5).mean()), top10=float((rk <= 10).mean()),
                        median_rank=float(rk.median()))
    out["n"] = len(d)
    out["cos_avatar_human_median"] = float(d.cos_avatar_human.median())
    return out, d
