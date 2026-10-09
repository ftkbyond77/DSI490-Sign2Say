"""Motion timeline → .vrma (VRM Animation 1.0, glTF binary): open it in Unity with UniVRM (VRM-1.0 package → VRMA import) or in
any three-vrm app, to inspect the signing frame by frame on any VRM avatar.

The file holds a humanoid skeleton with identity rest rotations (T-pose, positions of our avatar) in the VRM 1.0 / glTF frame
(facing +Z), one rotation track per animated bone, and expression tracks (VRMC_vrm_animation convention: an expression's weight
is the x translation of its node). Our rotations live in the VRM 0.x model frame (facing −Z): a 180° turn about Y maps them
(x, y, z, w) → (−x, y, −z, w); rest positions (x, y, z) → (−x, y, −z).
"""
from __future__ import annotations

import json
import struct

import numpy as np

from .codec import decode
from .rig import get_rig

EXPR = {"Fcl_MTH_A": "aa", "Fcl_MTH_I": "ih", "Fcl_MTH_U": "ou", "Fcl_MTH_O": "oh", "Fcl_EYE_Close_L": "blinkLeft",
        "Fcl_EYE_Close_R": "blinkRight", "Fcl_BRW_Surprised": "surprised", "Fcl_MTH_Fun": "happy"}
EXPR_GAIN = {"surprised": 0.6, "happy": 0.6}


def to_vrma(payload) -> bytes:
    q, m = decode(payload)
    fps, bones, morphs = payload["fps"], payload["bones"], payload["morphs"]
    T = q.shape[0]
    rig = get_rig()
    skel = [b for b in rig.order if not b.endswith("Tip")]
    idx = {b: i for i, b in enumerate(skel)}
    flip = np.array([-1, 1, -1], np.float32)
    nodes = []
    for b in skel:
        p = rig.parent.get(b)
        off = (rig.rest[b] - (rig.rest[p] if p else 0.0)) * flip
        nodes.append(dict(name=b, translation=[float(x) for x in off], children=[]))
    roots = []
    for b in skel:
        p = rig.parent.get(b)
        (nodes[idx[p]]["children"] if p else roots).append(idx[b])
    expr_nodes = {}
    for k, name in EXPR.items():
        if k in morphs:
            expr_nodes[name] = len(nodes)
            nodes.append(dict(name=f"expression_{name}"))
            roots.append(expr_nodes[name])
    for n in nodes:
        if "children" in n and not n["children"]:
            del n["children"]

    blob = bytearray()
    views, accs = [], []

    def add(arr, typ, minmax=False):
        a = np.ascontiguousarray(arr, np.float32)
        while len(blob) % 4:
            blob.append(0)
        views.append(dict(buffer=0, byteOffset=len(blob), byteLength=a.nbytes))
        blob.extend(a.tobytes())
        acc = dict(bufferView=len(views) - 1, componentType=5126, count=int(a.shape[0]), type=typ)
        if minmax:
            acc["min"] = [float(a.min())]; acc["max"] = [float(a.max())]
        accs.append(acc)
        return len(accs) - 1

    t_acc = add(np.arange(T, dtype=np.float32) / fps, "SCALAR", minmax=True)
    samplers, channels = [], []
    for bi, b in enumerate(bones):
        if b not in idx:
            continue
        qq = q[:, bi] * np.array([-1, 1, -1, 1], np.float32)
        samplers.append(dict(input=t_acc, output=add(qq, "VEC4"), interpolation="LINEAR"))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=idx[b], path="rotation")))
    for k, name in EXPR.items():
        if k not in morphs:
            continue
        w = np.clip(m[:, morphs.index(k)] * EXPR_GAIN.get(name, 1.0), 0, 1)
        tr = np.stack([w, np.zeros(T), np.zeros(T)], 1)
        samplers.append(dict(input=t_acc, output=add(tr, "VEC3"), interpolation="LINEAR"))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=expr_nodes[name], path="translation")))
    while len(blob) % 4:
        blob.append(0)
    gltf = dict(
        asset=dict(version="2.0", generator="ThaiSLM reverse t2s"),
        extensionsUsed=["VRMC_vrm_animation"],
        extensions=dict(VRMC_vrm_animation=dict(
            specVersion="1.0",
            humanoid=dict(humanBones={b: dict(node=idx[b]) for b in skel}),
            expressions=dict(preset={n: dict(node=i) for n, i in expr_nodes.items()}))),
        scene=0, scenes=[dict(nodes=roots)], nodes=nodes,
        animations=[dict(name="signing", samplers=samplers, channels=channels)],
        buffers=[dict(byteLength=len(blob))], bufferViews=views, accessors=accs)
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js +
            struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))
