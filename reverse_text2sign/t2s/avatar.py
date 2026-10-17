"""Avatars: the web-optimised library avatar (models/avatar.vrm) and the character catalogue of the web picker.

optimise()   same skeleton, morphs and materials, smaller textures.
catalogue()  every *.vrm in vrm/ (any file name) becomes a selectable character: `default.vrm` is "Default", the others
             "Model 1", "Model 2", ... in file-name order. Each gets an optimised copy and a thumbnail (from the VRM meta) in a
             content-addressed cache, plus a body Fit so it signs like the library avatar (t2s/motion/fit.py). Rescanned on every
             call (one stat per file): dropping a file into vrm/ adds a character without a restart.

The delivered VRoid file is 20 MB, 12.6 MB of it PNG textures up to 4096². The avatar is shown half-body at ≤ ~1000 px, so
textures are capped at 2048 px; opaque ones become JPEG (q 90), textures with real transparency stay PNG. The glTF JSON is kept
as is except image mime types / buffer views, so three-vrm / UniVRM read it exactly like the original.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import struct
import threading
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from . import paths


def optimise(src=None, dst=None, max_side=2048, jpeg_q=90):
    src = src or paths.AVATAR_SRC
    dst = dst or paths.AVATAR
    b = open(src, "rb").read()
    cl = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + cl])
    off = 20 + cl
    bl = struct.unpack("<I", b[off:off + 4])[0]
    binc = b[off + 8:off + 8 + bl]
    views = j["bufferViews"]
    image_views = {im["bufferView"]: k for k, im in enumerate(j.get("images", []))}
    new_bin = bytearray()
    new_views = []
    for vi, v in enumerate(views):
        data = binc[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
        if vi in image_views:
            im = j["images"][image_views[vi]]
            img = Image.open(io.BytesIO(data))
            img.load()
            if max(img.size) > max_side:
                s = max_side / max(img.size)
                img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))), Image.LANCZOS)
            has_alpha = img.mode in ("RGBA", "LA") and np.asarray(img.getchannel("A")).min() < 255
            out = io.BytesIO()
            if has_alpha:
                img.save(out, "PNG", optimize=True)
                im["mimeType"] = "image/png"
            else:
                img.convert("RGB").save(out, "JPEG", quality=jpeg_q, optimize=True)
                im["mimeType"] = "image/jpeg"
            if len(out.getvalue()) < len(data):
                data = out.getvalue()
            else:
                im["mimeType"] = "image/png" if data[:4] == b"\x89PNG" else im["mimeType"]
        while len(new_bin) % 4:
            new_bin.append(0)
        nv = dict(v)
        nv["byteOffset"] = len(new_bin)
        nv["byteLength"] = len(data)
        new_bin.extend(data)
        new_views.append(nv)
    while len(new_bin) % 4:
        new_bin.append(0)
    j["bufferViews"] = new_views
    j["buffers"][0]["byteLength"] = len(new_bin)
    js = json.dumps(j, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(new_bin)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js +
                struct.pack("<I4s", len(new_bin), b"BIN\0") + bytes(new_bin))
    return len(b), total


# ------------------------------------------------------------------------------------------------------------- catalogue
SRC_DIR = Path(os.environ.get("T2S_VRM_DIR", paths.HERE / "vrm"))
CACHE_DIR = Path(os.environ.get("T2S_AVATAR_CACHE", paths.WORK / "avatars"))
_MEMO: dict = {}                    # file name -> (size, mtime_ns, entry, fit, content hash)
_LOCK = threading.Lock()
_QUEUE: "queue.Queue" = queue.Queue()
_WORKER = None
_REF = None


def _glb(b):
    cl = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + cl])
    off = 20 + cl
    bl = struct.unpack("<I", b[off:off + 4])[0]
    return j, b[off + 8:off + 8 + bl]


def _thumbnail(j, binc, stem, side=160):
    """the VRM's own thumbnail (VRM 0.x meta.texture / VRM 1.0 meta.thumbnailImage) -> small JPEG / PNG; None if it has none."""
    ext = j.get("extensions", {})
    try:
        if "VRM" in ext:
            img_i = j["textures"][ext["VRM"]["meta"]["texture"]]["source"]
        else:
            img_i = ext["VRMC_vrm"]["meta"]["thumbnailImage"]
        v = j["bufferViews"][j["images"][img_i]["bufferView"]]
        img = Image.open(io.BytesIO(binc[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]))
        img.load()
    except (KeyError, IndexError, TypeError, OSError):
        return None
    img.thumbnail((side, side * 2), Image.LANCZOS)
    alpha = img.mode in ("RGBA", "LA") and np.asarray(img.getchannel("A")).min() < 255
    out = stem.with_name(stem.name + (".png" if alpha else ".jpg"))
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    if alpha:
        img.save(tmp, "PNG", optimize=True)
    else:
        img.convert("RGB").save(tmp, "JPEG", quality=85, optimize=True)
    os.replace(tmp, out)
    return out.name


def _optimise_worker():
    """background: optimised copies, one at a time. A lock file keeps several API workers from compressing the same model."""
    while True:
        f, h = _QUEUE.get()
        dst, lock = CACHE_DIR / f"{h}.vrm", CACHE_DIR / f"{h}.lock"
        try:
            if dst.exists():
                continue
            if lock.exists() and time.time() - lock.stat().st_mtime > 600:
                lock.unlink(missing_ok=True)                  # left behind by a killed worker
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue                                      # another worker is on it; the raw file is served meanwhile
            tmp = CACHE_DIR / f"{h}.vrm.{os.getpid()}.tmp"
            optimise(f, tmp)
            os.replace(tmp, dst)
            lock.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"[avatars] optimise {f.name}: {type(e).__name__}: {e}", flush=True)
            lock.unlink(missing_ok=True)


def _build(f):
    from .motion.fit import Fit
    from .motion.rig import Rig
    global _REF, _WORKER
    b = f.read_bytes()
    h = hashlib.sha1(b).hexdigest()[:16]
    j, binc = _glb(b)
    ext = j.get("extensions", {})
    if "VRM" in ext:
        m = ext["VRM"].get("meta", {})
        title, author, version = m.get("title") or "", m.get("author") or "", "0"
    elif "VRMC_vrm" in ext:
        m = ext["VRMC_vrm"].get("meta", {})
        title, author, version = m.get("name") or "", ", ".join(m.get("authors") or []), "1"
    else:
        raise ValueError("not a VRM (no VRM / VRMC_vrm extension)")
    if _REF is None:
        _REF = Rig()
    fit = Fit(Rig(f), _REF)                          # raises on bodies the signing rig cannot drive
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    thumbs = [p.name for p in (CACHE_DIR / f"{h}.jpg", CACHE_DIR / f"{h}.png") if p.exists()]
    thumb = thumbs[0] if thumbs else _thumbnail(j, binc, CACHE_DIR / h)
    if not (CACHE_DIR / f"{h}.vrm").exists():
        if _WORKER is None:
            _WORKER = threading.Thread(target=_optimise_worker, name="avatar-optimise", daemon=True)
            _WORKER.start()
        _QUEUE.put((f, h))
    entry = dict(id=f.stem, file=f.name, title=title, author=author, vrm_version=version,
                 thumb=f"/avatars/{thumb}" if thumb else None, scale=round(fit.scale, 4), check=check(j, fit, m, version))
    return entry, fit, h


# the motion's face channels → the morphs a VRoid model carries, or the standard VRM expression the browser uses instead
FACE_CHANNELS = {"Fcl_BRW_Surprised": "surprised", "Fcl_BRW_Angry": "angry", "Fcl_BRW_Sorrow": "sad", "Fcl_EYE_Close_L": "blinkLeft",
                 "Fcl_EYE_Close_R": "blinkRight", "Fcl_EYE_Spread": None, "Fcl_MTH_A": "aa", "Fcl_MTH_I": "ih", "Fcl_MTH_U": "ou",
                 "Fcl_MTH_O": "oh", "Fcl_MTH_Fun": "happy", "Fcl_MTH_Angry": None}
V0_PRESET = {"a": "aa", "i": "ih", "u": "ou", "o": "oh", "e": "ee", "blink_l": "blinkLeft", "blink_r": "blinkRight", "joy": "happy",
             "sorrow": "sad", "fun": "relaxed"}


def check(j, fit, meta, version):
    """what this character can show of the motion — computed when the file appears in vrm/, so a new model is validated on
    arrival (result_v2.md §6): the rig drives the signing bones (Fit raised otherwise), which optional bones were folded, how
    many face channels it can show (VRoid morphs directly, else the nearest standard expression), and its licence terms."""
    morphs = {n for mesh in j.get("meshes", []) for n in (mesh.get("extras", {}).get("targetNames") or [])}
    ext = j.get("extensions", {})
    if "VRM" in ext:
        presets = {V0_PRESET.get((g.get("presetName") or "").lower(), (g.get("presetName") or "").lower())
                   for g in ext["VRM"].get("blendShapeMaster", {}).get("blendShapeGroups", [])}
        lic = dict(commercial=meta.get("commercialUssageName"), licence=meta.get("licenseName"))
    else:
        presets = set(ext.get("VRMC_vrm", {}).get("expressions", {}).get("preset", {}))
        lic = dict(commercial=meta.get("commercialUsage"), licence=meta.get("licenseUrl"))
    direct = [c for c in FACE_CHANNELS if c in morphs]
    via = [c for c, e in FACE_CHANNELS.items() if c not in morphs and e and e in presets]
    missing = [c for c in FACE_CHANNELS if c not in direct and c not in via]
    warn = []
    if fit.folded:
        warn.append(f"no {', '.join(fit.folded)} bone — its rotation is carried by the neighbouring bones")
    if len(direct) < len(FACE_CHANNELS):
        warn.append(f"face: {len(direct)} channels as morphs, {len(via)} through whole-face expressions, {len(missing)} not shown")
    return dict(ok=True, vrm=version, folded=fit.folded, face_direct=len(direct), face_expression=len(via),
                face_missing=missing, licence=lic, warnings=warn,
                proportions=dict(torso=round(fit.scale, 3), head=round(fit.head_scale, 3), shoulders=round(fit.sw_scale, 3)))


def catalogue():
    """[(entry, fit)] in picker order: default.vrm first, then the other files by name. Unreadable files are skipped (logged).
    entry.url is the optimised copy once it exists, the original file (served from /avatars/src/) until then."""
    files = sorted((p for p in SRC_DIR.glob("*.vrm") if p.is_file()),
                   key=lambda p: (p.name.lower() != "default.vrm", p.name.lower()))
    out = []
    with _LOCK:
        for f in files:
            st = f.stat()
            got = _MEMO.get(f.name)
            if not got or got[:2] != (st.st_size, st.st_mtime_ns):
                try:
                    _MEMO[f.name] = (st.st_size, st.st_mtime_ns, *_build(f))
                except Exception as e:  # noqa: BLE001 - one broken file must not take the picker down
                    print(f"[avatars] skip {f.name}: {type(e).__name__}: {e}", flush=True)
                    _MEMO[f.name] = (st.st_size, st.st_mtime_ns, None, None, None)
            _, _, e, fit, h = _MEMO[f.name]
            if e is None:
                continue
            if not e.get("thumb"):                    # a headshot stored since (by any API worker: the cache is shared)
                made = CACHE_DIR / f"{h}.png"
                if made.exists():
                    e["thumb"] = f"/avatars/{made.name}"
            opt = CACHE_DIR / f"{h}.vrm"
            if opt.exists():
                e.update(url=f"/avatars/{opt.name}", bytes=opt.stat().st_size, optimised=True)
            else:
                e.update(url=f"/avatars/src/{quote(f.name)}?v={h}", bytes=st.st_size, optimised=False)
            out.append((e, fit))
        for gone in set(_MEMO) - {f.name for f in files}:
            del _MEMO[gone]
    for k, (e, _) in enumerate(out):
        e["label"] = "Default" if k == 0 else f"Model {k}"
    return out


def set_thumbnail(avatar_id, data, side=160):
    """store a picker thumbnail for a character whose file has none (VRM 1.0 exports often omit meta.thumbnailImage): the
    browser renders a headshot once and posts it here; kept in the content-addressed cache, so it is made only once. A
    character that already has a thumbnail (embedded or made before) is never overwritten. → the thumbnail URL or None."""
    catalogue()
    with _LOCK:
        for name, (size, mtime, e, fit, h) in _MEMO.items():
            if e is None or e["id"] != avatar_id:
                continue
            if e.get("thumb"):
                return e["thumb"]
            img = Image.open(io.BytesIO(data))
            img.load()
            if img.width > 1024 or img.height > 1024:
                raise ValueError("thumbnail too large")
            img.thumbnail((side, side * 2), Image.LANCZOS)
            out = CACHE_DIR / f"{h}.png"
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
            img.convert("RGBA").save(tmp, "PNG", optimize=True)
            os.replace(tmp, out)
            e["thumb"] = f"/avatars/{out.name}"
            return e["thumb"]
    return None


def fit_for(avatar_id):
    """the body Fit of a catalogue character (None = the library avatar or an unknown id -> motion as is)."""
    if not avatar_id:
        return None
    for e, fit in catalogue():
        if e["id"] == avatar_id:
            return None if fit.identity else fit
    return None
