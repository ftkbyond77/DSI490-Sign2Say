"""L2 Frozen frame encoder (DINOv2-S fallback of DINOv3) + embedding cache.

Cache layout (per clip, resumable):
  artifacts/tracks/<clip>.npz                     kpts [T,17,3] · geom [T,3,3] · valid [T,3] · hw · start_s
  artifacts/cache/<profile>/<version>/<clip>.npy  float16 [T,3,768]  ([CLS ‖ patch-mean] for hand_l, hand_r, face)

The cache job uses N CPU worker processes (decode → pose → crops) feeding one GPU process
(augment → encode), so the GPU never waits on pose and pose never waits on the GPU."""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import ART, get_logger, load_cfg, safe_id

log = get_logger("encoder")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


class FrozenEncoder:
    def __init__(self, name: str = "facebook/dinov2-small", device: str = "cuda", res: int = 112, dtype=torch.float16):
        from transformers import Dinov2Model
        self.device, self.res, self.dtype = device, res, dtype if device == "cuda" else torch.float32
        self.m = Dinov2Model.from_pretrained(name, local_files_only=True, attn_implementation="sdpa",
                                             torch_dtype=self.dtype).to(device).eval()
        self.mean, self.std = _MEAN.to(device, self.dtype), _STD.to(device, self.dtype)
        self.dim = 2 * self.m.config.hidden_size

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, batch: int = 512) -> torch.Tensor:
        """x [..., S, S, 3] BGR uint8/float (0..255) → [..., 768] (fp16 on GPU)."""
        lead = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        outs = []
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device, non_blocking=True)
            b = b.flip(-1).permute(0, 3, 1, 2).to(self.dtype)
            if b.shape[-1] != self.res:
                b = F.interpolate(b, (self.res, self.res), mode="bilinear", align_corners=False)
            b = (b - self.mean) / self.std
            h = self.m(pixel_values=b).last_hidden_state
            outs.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))
        return torch.cat(outs).reshape(*lead, self.dim)


def profile_name(cfg=None) -> str:
    cfg = cfg or load_cfg("data")
    return f"dinov2s_{cfg['crop_size']}"


def cache_path(clip_id: str, version: str = "base", profile: str | None = None) -> Path:
    return ART / "cache" / (profile or profile_name()) / version / f"{safe_id(clip_id)}.npy"


def track_path(clip_id: str) -> Path:
    return ART / "tracks" / f"{safe_id(clip_id)}.npz"


def load_clip(clip_id: str, version: str = "base", profile: str | None = None):
    emb = np.load(cache_path(clip_id, version, profile))
    tr = np.load(track_path(clip_id))
    return emb, dict(tr)


# ----------------------------------------------------------------------------- packed store
def pack(set_name: str, clip_ids: list[str], versions: list[str], mode_for=lambda cid: "clip"):
    """Concatenate per-clip caches into memory-mappable arrays for training.
    Writes <profile>/packed/<set>/{<version>.npy, pose_base.npy, pose_flip.npy, valid.npy, active.npy, index.parquet}."""
    import pandas as pd
    from .articulators import active_mask, mirror_kpts, pose_features
    prof = profile_name()
    out = ART / "cache" / prof / "packed" / set_name
    out.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for cid in clip_ids:
        if not (track_path(cid).exists() and all(cache_path(cid, v, prof).exists() for v in versions)):
            continue
        T = len(np.load(track_path(cid))["kpts"])
        if T < 4:
            continue
        rows.append((cid, n, T)); n += T
    idx = pd.DataFrame(rows, columns=["clip_id", "start", "n"])
    arrs = {v: np.lib.format.open_memmap(out / f"{v}.npy", "w+", np.float16, (n, 3, 768)) for v in versions}
    pose_b = np.zeros((n, 44), np.float32); pose_f = np.zeros((n, 44), np.float32)
    valid = np.zeros((n, 3), np.uint8); active = np.zeros(n, bool)
    for cid, s, T in rows:
        tr = np.load(track_path(cid))
        k = tr["kpts"]
        for v in versions:
            e = np.load(cache_path(cid, v, prof))
            arrs[v][s:s + T] = e[:T]
        pose_b[s:s + T] = pose_features(k)
        pose_f[s:s + T] = pose_features(mirror_kpts(k, int(tr["hw"][1])))
        valid[s:s + T] = tr["valid"]
        active[s:s + T] = active_mask(k, mode=mode_for(cid))
    for a in arrs.values():
        a.flush()
    np.save(out / "pose_base.npy", pose_b); np.save(out / "pose_flip.npy", pose_f)
    np.save(out / "valid.npy", valid); np.save(out / "active.npy", active)
    idx.to_parquet(out / "index.parquet", index=False)
    log.info("packed %s: %d clips, %d frames, versions=%s, active=%.2f", set_name, len(idx), n, versions, active.mean())


class FeatureStore:
    """Read-only view over a packed set (memory-mapped)."""

    def __init__(self, set_name: str, versions: list[str] | None = None):
        import pandas as pd
        self.dir = ART / "cache" / profile_name() / "packed" / set_name
        self.index = pd.read_parquet(self.dir / "index.parquet")
        self.versions = versions or [p.stem for p in self.dir.glob("*.npy")
                                     if p.stem not in ("pose_base", "pose_flip", "valid", "active")]
        self.feat = {v: np.load(self.dir / f"{v}.npy", mmap_mode="r") for v in self.versions}
        self.pose = {"base": np.load(self.dir / "pose_base.npy"), "flip": np.load(self.dir / "pose_flip.npy")}
        self.valid = np.load(self.dir / "valid.npy")
        self.active = np.load(self.dir / "active.npy")
        self.row = {c: (s, n) for c, s, n in self.index.itertuples(index=False)}

    def clip(self, cid: str, version: str = "base", active_only: bool = False):
        s, n = self.row[cid]
        sl = slice(s, s + n)
        f = np.asarray(self.feat[version][sl], dtype=np.float32)
        p = self.pose["flip" if version == "flip" else "base"][sl]
        v = self.valid[sl][:, [1, 0, 2]] if version == "flip" else self.valid[sl]
        a = self.active[sl]
        if active_only and a.sum() >= 4:
            i0, i1 = np.flatnonzero(a)[[0, -1]]
            f, p, v, a = f[i0:i1 + 1], p[i0:i1 + 1], v[i0:i1 + 1], a[i0:i1 + 1]
        return f, p, v, a


# ----------------------------------------------------------------------------- cache job
def _worker(task_q, out_q, crop_size, threads, wid):
    import os
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from .articulators import Articulators, chroma_masks, mask_regions
    from .data import decode_frames
    cfg = load_cfg("data")
    art = Articulators(device="cpu", threads=threads, crop_size=crop_size, cfg=cfg)
    lo, hi, fill = cfg["ttrs"]["chroma_hsv_lo"], cfg["ttrs"]["chroma_hsv_hi"], tuple(cfg["ttrs"]["fill_bgr"])
    while True:
        item = task_q.get()
        if item is None:
            break
        cid, path, regions, zone, chroma, start_s, max_s = item
        try:
            art.reset(signer_zone=zone)
            K, G, V, C, Gc = [], [], [], [], []
            ci, hw = 0, None
            for ch in decode_frames(path, start_s=start_s, max_s=max_s, chunk=64):
                hw = ch.shape[1:3]
                ch = mask_regions(ch, regions, fill)
                for f in ch:
                    kp, g, v = art.step(f)
                    K.append(kp); G.append(g); V.append(v); C.append(art.crops(f, g)); Gc.append(g)
                if len(C) >= 192:
                    c = np.stack(C)
                    out_q.put(("chunk", cid, ci, c, chroma_masks(c, lo, hi) if chroma else None, np.stack(Gc), hw))
                    ci += 1; C, Gc = [], []
            if C:
                c = np.stack(C)
                out_q.put(("chunk", cid, ci, c, chroma_masks(c, lo, hi) if chroma else None, np.stack(Gc), hw))
            if not K:
                out_q.put(("error", cid, "no frames decoded"))
                continue
            out_q.put(("done", cid, np.stack(K), np.stack(G), np.stack(V), hw, start_s))
        except Exception:
            out_q.put(("error", cid, traceback.format_exc()[-800:]))


def run_cache(items: list[tuple], versions_for, workers: int = 8, threads: int = 1, batch: int = 768):
    """items: (clip_id, media_path, mask_regions, signer_zone, chroma, start_s, max_s)
    versions_for(clip_id) → list of versions among base/flip/aug1/aug2."""
    from .augment import BackgroundPool, chroma_replace, clip_rng, flip_swap, photometric
    cfg = load_cfg("data")
    S = cfg["crop_size"]
    prof = profile_name(cfg)
    todo = [it for it in items if not (track_path(it[0]).exists() and
                                       all(cache_path(it[0], v, prof).exists() for v in versions_for(it[0])))]
    log.info("cache job: %d/%d clips to process (profile %s, %d workers)", len(todo), len(items), prof, workers)
    if not todo:
        return
    enc = FrozenEncoder(cfg["cache"]["backbone"], "cuda", S)
    bgpool = BackgroundPool(device="cuda")
    ctx = mp.get_context("spawn")
    task_q, out_q = ctx.Queue(), ctx.Queue(maxsize=workers * 3)
    procs = [ctx.Process(target=_worker, args=(task_q, out_q, S, threads, i), daemon=True) for i in range(workers)]
    for p in procs:
        p.start()
    for it in todo:
        task_q.put(it)
    for _ in procs:
        task_q.put(None)
    acc: dict[str, dict] = {}
    done = errors = frames = 0
    t0 = time.time()
    while done + errors < len(todo):
        try:
            msg = out_q.get(timeout=600)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                log.error("all workers died"); break
            continue
        kind, cid = msg[0], msg[1]
        if kind == "chunk":
            _, _, ci, crops, masks, geom, hw = msg
            vers = versions_for(cid)
            st = acc.setdefault(cid, {v: [] for v in vers})
            x = torch.from_numpy(crops).cuda()
            for v in vers:
                if v == "base":
                    e = enc(x, batch)
                elif v == "flip":
                    e = enc(flip_swap(x), batch)
                else:
                    bgs = st.setdefault("_bg", {})
                    if v not in bgs:
                        rng = clip_rng(cid, v)
                        bgs[v] = (bgpool.sample(rng), rng, rng.get_state())
                    bg, rng, state = bgs[v]
                    rng.set_state(state)  # same photometric params for every chunk of the clip
                    y = chroma_replace(x, torch.from_numpy(masks).cuda(), geom, hw, bg) if masks is not None else x.float()
                    e = enc(photometric(y, rng, S), batch)
                st[v].append(e.to(torch.float16).cpu().numpy())
        elif kind == "done":
            _, _, K, G, V, hw, start_s = msg
            st = acc.pop(cid, {})
            tp = track_path(cid); tp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tp, kpts=K, geom=G, valid=V, hw=np.array(hw), start_s=np.array(start_s))
            for v in versions_for(cid):
                cp = cache_path(cid, v, prof); cp.parent.mkdir(parents=True, exist_ok=True)
                arr = np.concatenate(st[v]) if st.get(v) else np.zeros((0, 3, enc.dim), np.float16)
                if len(arr) != len(K):
                    log.warning("%s %s: %d emb vs %d kpts", cid, v, len(arr), len(K))
                np.save(cp, arr)
            done += 1; frames += len(K)
            if done % 100 == 0 or done == len(todo):
                el = time.time() - t0
                log.info("%d/%d clips · %d frames · %.0f fr/s · eta %.1f min", done, len(todo), frames, frames / el,
                         el / done * (len(todo) - done) / 60)
        else:
            errors += 1
            acc.pop(cid, None)
            log.error("clip %s failed: %s", cid, msg[2])
    for p in procs:
        p.join(timeout=5)
    log.info("cache job finished: %d ok, %d errors, %.1f min", done, errors, (time.time() - t0) / 60)
