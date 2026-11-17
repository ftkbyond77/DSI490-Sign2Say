"""v11 training data (scripts/train_v11.py): augmentation pack R3, the cross-signer sampler R2 and the batch iterator — an importable
module so spawn-based loader workers (Windows) can unpickle them."""
from __future__ import annotations

import numpy as np
import torch

from . import encoder as E


# ----------------------------------------------------------------------------- augmentation pack (R3)
def augment_v11(kp, sc, hw, rng, fps_lo=10, fps_hi=25):
    """Coherent virtual signer per view: bone lengths drawn jointly (one 'body build' factor + small per-bone noise), signing-space
    scale 0.8–1.25 and offset ±0.15 shoulder widths (arms + hands around the shoulder centre), tempo 0.7–1.8×, left-handed 12 %;
    structured hand drop-out (a whole hand for 0.2–1.5 s · the rear hand where the wrists are < 0.45 shoulder widths apart ·
    a bottom-of-frame cut); 10–25 fps capture; small affine + jitter (v10)."""
    from .encoder import _MIRROR, frame_crop
    from .parts import resample_len, to_iso
    kp = to_iso(kp, hw).copy(); sc = sc.copy()
    ok = (sc[:, 5] > 0) & (sc[:, 6] > 0)
    if ok.any():
        mid = ((kp[ok, 5] + kp[ok, 6]) / 2).mean(0)
        sw = float(np.linalg.norm(kp[ok, 5] - kp[ok, 6], axis=1).mean()) or 0.2
        build = rng.normal(0, 0.06)
        ws = 1 + build + rng.normal(0, 0.02)
        space = rng.uniform(0.8, 1.25); off = rng.uniform(-0.15, 0.15, 2) * sw
        for sh, el, wr, hand in ((5, 7, 9, slice(91, 112)), (6, 8, 10, slice(112, 133))):
            sh0, el0, wr0 = kp[:, sh].copy(), kp[:, el].copy(), kp[:, wr].copy()
            sh1 = mid + (sh0 - mid) * ws
            el1 = sh1 + (el0 - sh0) * (1 + build + rng.normal(0, 0.03))
            wr1 = el1 + (wr0 - el0) * (1 + build + rng.normal(0, 0.03))
            wr1 = mid + (wr1 - mid) * space + off; el1 = mid + (el1 - mid) * (0.5 + 0.5 * space) + off * 0.5
            root = kp[:, hand][:, :1]
            hs = 1 + build + rng.normal(0, 0.04)
            kp[:, hand] = wr1[:, None] + (kp[:, hand] - root) * hs + (root - wr0[:, None])
            kp[:, sh], kp[:, el], kp[:, wr] = sh1, el1, wr1
    if rng.rand() < 0.12:                                                    # left-handed signer
        cx = float(((kp[ok, 5, 0] + kp[ok, 6, 0]) / 2).mean()) if ok.any() else 0.5
        kp[..., 0] = np.where(sc > 0, 2 * cx - kp[..., 0], kp[..., 0])
        kp, sc = kp[:, _MIRROR], sc[:, _MIRROR]
    n = len(kp)
    f = rng.uniform(0.7, 1.8) if rng.rand() < 0.85 else 1.0                   # tempo
    kp, sc = resample_len(kp, sc, max(4, int(round(n / f))))
    n = len(kp)
    if n > 8 and rng.rand() < 0.5:                                            # 10–25 fps capture → 25 fps (nearest-frame duplicates)
        fr = rng.uniform(fps_lo, fps_hi) / 25.0
        m = max(4, int(round(n * fr)))
        src = np.round(np.linspace(0, n - 1, m)).astype(int)
        back = src[np.clip(np.round(np.linspace(0, m - 1, n)).astype(int), 0, m - 1)]
        kp, sc = kp[back], sc[back]
    ang = np.deg2rad(rng.uniform(-10, 10)); s = rng.uniform(0.9, 1.1)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    R = R @ np.array([[1, rng.uniform(-0.08, 0.08)], [0, 1]])
    live = sc > 0
    c = kp[live].mean(0) if live.any() else np.zeros(2)
    kp = ((kp - c) @ R.T + c + rng.uniform(-0.05, 0.05, 2)).astype(np.float32)
    if rng.rand() < 0.5:
        kp = kp + rng.normal(0, 0.004, kp.shape).astype(np.float32)
    if rng.rand() < 0.4:
        for hand in (slice(91, 112), slice(112, 133)):
            kp[:, hand] += rng.normal(0, rng.uniform(0.002, 0.006), kp[:, hand].shape).astype(np.float32)
    if rng.rand() < 0.5:
        sc = np.where(sc > 0.3, 1.0, 0.0).astype(np.float32)
    if rng.rand() < 0.3 and n > 6:                                            # whole hand lost for 0.2–1.5 s
        hand = slice(91, 112) if rng.rand() < 0.5 else slice(112, 133)
        L = min(n, rng.randint(5, 38)); a = rng.randint(0, max(1, n - L + 1)); sc[a:a + L, hand] = 0.0
    if rng.rand() < 0.3:                                                      # rear hand hidden where the hands are close / crossing
        okb = (sc[:, 9] > 0) & (sc[:, 10] > 0)
        swn = float(np.median(np.linalg.norm(kp[:, 5] - kp[:, 6], axis=1))) or 0.2
        close = okb & (np.linalg.norm(kp[:, 9] - kp[:, 10], axis=1) < 0.45 * swn)
        if close.any():
            rear = slice(91, 112) if rng.rand() < 0.5 else slice(112, 133)
            sc[close, rear] = 0.0
    if rng.rand() < 0.15:                                                     # bottom-of-frame cut
        kp, sc = frame_crop(kp, sc, rng, lo=0.4, hi=1.2)
    if rng.rand() < 0.3 and n > 6:                                            # short flicker (≤ 4 frames)
        hand = slice(91, 112) if rng.rand() < 0.5 else slice(112, 133)
        for _ in range(rng.randint(1, 4)):
            a = rng.randint(0, n - 2); sc[a:a + rng.randint(1, 5), hand] = 0.0
    return kp.astype(np.float32), sc.astype(np.float32)



class V11Data(E.ClipDataset):
    def view(self, i, rng, aug=True, use_aux=None):
        if not (aug and self.aug_v11):
            return super().view(i, rng, aug, use_aux)
        r = self.t[i]
        row = r["row"]
        if r.get("aux", -1) >= 0 and (use_aux if use_aux is not None else rng.rand() < self.aux_p):
            row = r["aux"]
        n = self.store.length(row)
        f0, f1 = min(r["f0"], max(0, n - 4)), min(r["f1"], n)
        L = f1 - f0; j = self.jitter
        f0 = int(np.clip(f0 + rng.randint(-int(j * L) - 1, int(j * L) + 2), 0, max(0, f1 - 4)))
        f1 = int(np.clip(f1 + rng.randint(-int(j * L) - 1, int(j * L) + 2), f0 + 4, n))
        if rng.rand() < 0.25 and f1 - f0 > 10:                       # on-set / off-set truncation 10–35 % (co-articulated sign)
            cut = int(round((f1 - f0) * rng.uniform(0.1, 0.35)))
            if rng.rand() < 0.5:
                f0 += cut
            else:
                f1 -= cut
        kp, sc, hw = self.store.get(row, f0, f1)
        kp, sc = augment_v11(kp, sc, hw, rng)
        if self.context_p and rng.rand() < self.context_p:
            kp, sc = self._context(kp, sc, rng)
        return self.prep(kp, sc)

    # ---- R2 cross-signer sampler
    def setup_cross(self, groups, words_multi, daily_words, tail_rows):
        self.cells = {}                                        # word → form → signer → [row idx]
        for i, r in enumerate(self.t):
            if r["y"] < 0 or r["c"] == self.null_c:
                continue
            self.cells.setdefault(r["y"], {}).setdefault(r["form"], {}).setdefault(r["signer"], []).append(i)
        self.by_signer = {}
        for i, r in enumerate(self.t):
            if r["y"] >= 0 and r["c"] != self.null_c:
                self.by_signer.setdefault(r["signer"], []).append(i)
        self.groups = [[w for w in g if w in self.cells] for g in groups]
        self.groups = [g for g in self.groups if len(g) >= 2]
        self.words_multi = [w for w in words_multi if w in self.cells]
        wm = np.array([3.0 if w in daily_words else 1.0 for w in self.words_multi]); self.wm = wm / wm.sum()
        self.tail_rows = tail_rows

    def _pick_cell(self, w, rng, signer=None):
        forms = self.cells[w]
        fk = list(forms)
        p = np.sqrt([sum(len(v) for v in forms[f].values()) for f in fk]); p /= p.sum()
        f = fk[rng.choice(len(fk), p=p)] if signer is None else next((f for f in fk if signer in forms[f]), fk[0])
        sg = list(forms[f])
        s1 = signer if signer in forms[f] else sg[rng.randint(len(sg))]
        others = [s for s in sg if s != s1]
        if others:                                             # different source first, signer drawn uniformly (≤ ⅓ per person)
            src1 = self.t[forms[f][s1][0]]["dataset"]
            diff = [s for s in others if self.t[forms[f][s][0]]["dataset"] != src1]
            pool = diff if (diff and rng.rand() < 0.7) else others
            s2 = pool[rng.randint(len(pool))]
        else:
            s2 = s1
        i1 = forms[f][s1][rng.randint(len(forms[f][s1]))]
        i2 = forms[f][s2][rng.randint(len(forms[f][s2]))]
        return [i1, i1, i2, i2]                                 # 2 signers × 2 views

    def batch_cross(self, P, K, null_frac, tail_frac=0.06):
        if self._n_batches % 20 == 0:
            self._refresh_neighbours()
        self._n_batches += 1
        rng = self.rng
        out, seen = [], set()
        n_grp = int(round(P * rng.uniform(0.25, 0.5)))
        tries = 0
        while len(out) < n_grp and self.groups and tries < 50:
            tries += 1
            g = self.groups[rng.randint(len(self.groups))]
            for w in rng.permutation(g)[:rng.randint(2, len(g) + 1)]:
                if int(w) not in seen and len(out) < n_grp:
                    out.append((int(w), None)); seen.add(int(w))
        while len(out) < P:
            w = int(self.words_multi[rng.choice(len(self.words_multi), p=self.wm)])
            if w in seen:
                continue
            out.append((w, None)); seen.add(w)
            if rng.rand() < self.hard_frac * 0.5 and len(out) < P:    # another word of the same signer (hard negative)
                i1 = self._pick_cell(w, rng)[0]; s = self.t[i1]["signer"]
                cand = self.by_signer.get(s, [])
                if cand:
                    j = cand[rng.randint(len(cand))]; w2 = self.t[j]["y"]
                    if w2 not in seen and w2 in self.cells:
                        out.append((w2, s)); seen.add(w2)
            if rng.rand() < self.hard_frac * 0.5 and len(out) < P:    # nearest word by the current proxies
                nb = [x for x in self.neighbours.get(w, []) if x in self.cells and x not in seen]
                if nb:
                    x = nb[rng.randint(min(len(nb), 5))]; out.append((x, None)); seen.add(x)
        parts, ys, fs, sg = [], [], [], []
        for w, s in out[:P]:
            for k, i in enumerate(self._pick_cell(w, rng, s)[:K]):
                parts.append(self.view(i, rng)); ys.append(w); fs.append(self.t[i]["form"])
                sg.append(i if getattr(self, "phon_rows", False) else self.t[i]["sid"])
        for _ in range(int(round(P * K * tail_frac))):              # Tail / single-signer concepts: negatives only (unique labels)
            i = self.tail_rows[rng.randint(len(self.tail_rows))]
            parts.append(self.view(i, rng)); ys.append(-1); fs.append(0)
            sg.append(i if getattr(self, "phon_rows", False) else self.t[i]["sid"])
        for _ in range(int(round(P * K * null_frac))):
            parts.append(self.null_view(rng)); ys.append(self.null_y); fs.append(0); sg.append(-1)
        return parts, np.array(ys), np.array(fs), np.array(sg)

    def batch_old(self, P, K, multi, allc, null_frac):
        parts, labels = self.batch(P, K, multi, allc, null_frac)
        idx = getattr(self, "_last_rows", None)
        ys = np.array([self.lut[c] for c in labels])
        return parts, ys, np.zeros(len(ys), int), np.full(len(ys), -1)


class Iter(torch.utils.data.IterableDataset):
    def __init__(self, ds, P, K, multi, allc, null_frac, seed, mode):
        super().__init__()
        self.ds, self.P, self.K, self.multi, self.allc, self.null_frac, self.seed, self.mode = ds, P, K, multi, allc, null_frac, seed, mode

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        self.ds.rng = np.random.RandomState(self.seed + (wi.id if wi else 0))
        while True:
            if self.mode == "cross":
                yield self.ds.batch_cross(self.P, self.K, self.null_frac)
            else:
                yield self.ds.batch_old(self.P, self.K, self.multi, self.allc, self.null_frac)


def collate(b):
    from .lmenc import collate_np
    parts, ys, fs, sg = b
    x, m = collate_np(parts)
    return (torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(ys.astype(np.int64)), torch.from_numpy(fs.astype(np.int64)),
            torch.from_numpy(sg.astype(np.int64)))


