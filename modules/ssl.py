"""L3 Temporal SSL (SignDINO-style): per-stream temporal Transformers over frozen frame features,
trained with temporal DINO (tCLS) + iBOT (masked frames) + KoLeo, then Gram anchoring (stage 2).

Streams: hand_l, hand_r, face (768-d [CLS‖patch-mean]) + pose (44-d). NAP is applied to RGB inputs."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ART, get_logger

log = get_logger("ssl")
RGB_STREAMS = ("hand_l", "hand_r", "face")
STREAMS = RGB_STREAMS + ("pose",)


# ----------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, i, o, bias=True):
        super().__init__()
        self.base = nn.Linear(i, o, bias=bias)
        self.r = 0

    def enable_lora(self, r: int, alpha: float):
        dev = self.base.weight.device
        self.A = nn.Parameter(torch.randn(r, self.base.in_features, device=dev) * (1 / r))
        self.B = nn.Parameter(torch.zeros(self.base.out_features, r, device=dev))
        self.r, self.scale = r, alpha / r

    def forward(self, x):
        y = self.base(x)
        if self.r:
            y = y + (x @ self.A.t() @ self.B.t()) * self.scale
        return y


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4, drop_path=0.0):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = LoRALinear(d, d), nn.Linear(d, d), LoRALinear(d, d)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.dp = drop_path

    def _dp(self, x):
        if not self.training or self.dp == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) > self.dp).to(x.dtype)
        return x * keep / (1 - self.dp)

    def forward(self, x, attn_mask):
        B, T, D = x.shape
        y = self.n1(x)
        q = self.q(y).view(B, T, self.h, -1).transpose(1, 2)
        k = self.k(y).view(B, T, self.h, -1).transpose(1, 2)
        v = self.v(y).view(B, T, self.h, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self._dp(self.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + self._dp(self.mlp(self.n2(x)))


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, d, layers, heads, drop_path=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.missing = nn.Parameter(torch.zeros(d))
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, drop_path=drop_path * i / max(layers - 1, 1)) for i in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.d = d
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, valid=None, pad=None, fmask=None):
        """x [B,T,in] · valid [B,T] bool · pad [B,T] bool (True=real frame) · fmask [B,T] bool (iBOT)."""
        B, T, _ = x.shape
        h = self.inp(x)
        if valid is not None:
            h = torch.where(valid[..., None], h, self.missing.to(h.dtype))
        if fmask is not None:
            h = torch.where(fmask[..., None], self.mask_tok.to(h.dtype), h)
        h = h + sinusoid(T, self.d, x.device).to(h.dtype)
        h = torch.cat([self.cls.expand(B, 1, -1).to(h.dtype), h], 1)
        am = None
        if pad is not None:
            am = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), pad], 1)[:, None, None, :]
        for b in self.blocks:
            h = b(h, am)
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class SignEncoder(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, d_pose=128, layers_pose=2, heads_pose=4, nap_N=None, drop_path=0.1):
        super().__init__()
        self.enc = nn.ModuleDict({s: TemporalEncoder(768, d, layers, heads, drop_path) for s in RGB_STREAMS})
        self.enc["pose"] = TemporalEncoder(44, d_pose, layers_pose, heads_pose, drop_path)
        self.register_buffer("nap", torch.as_tensor(nap_N) if nap_N is not None else torch.zeros(3, 0, 768))
        self.cfg = dict(d=d, layers=layers, heads=heads, d_pose=d_pose, layers_pose=layers_pose, heads_pose=heads_pose)

    def out_dims(self):
        return {s: self.enc[s].d for s in STREAMS}

    def apply_nap(self, feats):
        if self.nap.shape[1] == 0:
            return feats
        N = self.nap.to(feats.dtype)
        proj = torch.einsum("btkd,kjd->btkj", feats, N)
        return feats - torch.einsum("btkj,kjd->btkd", proj, N)

    def forward(self, feats, pose, valid, pad=None, fmask=None, streams=STREAMS):
        """feats [B,T,3,768] · pose [B,T,44] · valid [B,T,3] bool → {stream: (cls [B,d], tokens [B,T,d])}"""
        feats = self.apply_nap(feats)
        out = {}
        for k, s in enumerate(RGB_STREAMS):
            if s in streams:
                out[s] = self.enc[s](feats[:, :, k], valid[:, :, k], pad, fmask)
        if "pose" in streams:
            out["pose"] = self.enc["pose"](pose, None, pad, fmask)
        return out

    def enable_lora(self, r=4, alpha=8):
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.enable_lora(r, alpha)


class DINOHead(nn.Module):
    def __init__(self, d, hidden=1024, bottleneck=256, K=4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bottleneck))
        self.proto = nn.Parameter(torch.randn(K, bottleneck) * 0.02)

    def forward(self, x):
        z = F.normalize(self.mlp(x).float(), dim=-1)
        return z @ F.normalize(self.proto.float(), dim=-1).t()


# ----------------------------------------------------------------------------- losses
@torch.no_grad()
def sinkhorn(logits, temp, iters=3):
    Q = torch.exp(logits.float() / temp).t()  # K × B
    Q /= Q.sum()
    K, B = Q.shape
    for _ in range(iters):
        Q /= Q.sum(1, keepdim=True); Q /= K
        Q /= Q.sum(0, keepdim=True); Q /= B
    return (Q * B).t()


def koleo(x, eps=1e-8):
    x = F.normalize(x.float(), dim=-1)
    with torch.no_grad():
        d = x @ x.t()
        d.fill_diagonal_(-2)
        nn_idx = d.argmax(1)
    dist = (x - x[nn_idx]).norm(dim=-1)
    return -torch.log(dist + eps).mean()


# ----------------------------------------------------------------------------- sampler
class SSLSampler:
    """Multi-temporal-crop sampler over packed stores. Views of one sample come from different
    cache versions (base / aug*), or all from `flip` (mirror) — never mixed with flip."""

    def __init__(self, stores: dict, weights: dict, clip_lists: dict, chunk=96, n_global=2, n_local=6,
                 global_len=(24, 48), local_len=(6, 16), seed=0):
        self.stores, self.clips = stores, clip_lists
        names = [n for n in stores if len(clip_lists[n])]
        w = np.array([weights[n] for n in names], dtype=np.float64)
        self.names, self.w = names, w / w.sum()
        self.chunk, self.ng, self.nl, self.gl, self.ll = chunk, n_global, n_local, global_len, local_len
        self.rng = np.random.RandomState(seed)

    def _window(self, store, cid, active_only):
        s, n = store.row[cid]
        a = store.active[s:s + n]
        if active_only and a.sum() >= 8:
            i0, i1 = np.flatnonzero(a)[[0, -1]]
            s, n = s + i0, i1 - i0 + 1
        if n > self.chunk:
            o = self.rng.randint(0, n - self.chunk + 1)
            s, n = s + o, self.chunk
        return s, n

    def batch(self, B):
        Lg = self.rng.randint(self.gl[0], self.gl[1] + 1)
        Ll = self.rng.randint(self.ll[0], self.ll[1] + 1)
        V = self.ng + self.nl
        lens = [Lg] * self.ng + [Ll] * self.nl
        feats = [np.zeros((B, L, 3, 768), np.float16) for L in lens]
        pose = [np.zeros((B, L, 44), np.float32) for L in lens]
        valid = [np.zeros((B, L, 3), bool) for L in lens]
        pad = [np.zeros((B, L), bool) for L in lens]
        for b in range(B):
            name = self.names[self.rng.choice(len(self.names), p=self.w)]
            st = self.stores[name]
            cid = self.clips[name][self.rng.randint(len(self.clips[name]))]
            s, n = self._window(st, cid, active_only=(name == "ttrs"))
            mirror = "flip" in st.feat and self.rng.rand() < 0.25
            vers = ["flip"] if mirror else [v for v in st.versions if v != "flip"]
            for vi, L in enumerate(lens):
                speed = self.rng.uniform(0.8, 1.25)
                span = min(n, max(2, int(round(L * speed))))
                o = self.rng.randint(0, n - span + 1)
                idx = s + o + np.round(np.linspace(0, span - 1, min(L, span))).astype(int)
                m = len(idx)
                ver = vers[self.rng.randint(len(vers))]
                feats[vi][b, :m] = st.feat[ver][idx]
                pose[vi][b, :m] = st.pose["flip" if ver == "flip" else "base"][idx]
                vv = st.valid[idx].astype(bool)
                valid[vi][b, :m] = vv[:, [1, 0, 2]] if ver == "flip" else vv
                pad[vi][b, :m] = True
                if self.rng.rand() < 0.1:  # stream dropout
                    valid[vi][b, :m, self.rng.randint(3)] = False
        return [(torch.from_numpy(f), torch.from_numpy(p), torch.from_numpy(v), torch.from_numpy(pd))
                for f, p, v, pd in zip(feats, pose, valid, pad)]


# ----------------------------------------------------------------------------- training
def train_ssl(sampler: SSLSampler, nap_N, steps=6000, batch=48, lr=5e-4, wd=0.04, K=4096, model_kw=None,
              gram_from=0.75, gram_w=2.0, koleo_w=0.1, ibot_ratio=(0.3, 0.5), ckpt_path=None, eval_fn=None,
              eval_every=1000, device="cuda"):
    model_kw = model_kw or {}
    student = SignEncoder(nap_N=nap_N, **model_kw).to(device)
    heads = nn.ModuleDict({s: DINOHead(d, K=K) for s, d in student.out_dims().items()}).to(device)
    teacher = copy.deepcopy(student).eval()
    t_heads = copy.deepcopy(heads).eval()
    for p in list(teacher.parameters()) + list(t_heads.parameters()):
        p.requires_grad_(False)
    params = list(student.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    gram_teacher = None
    hist = []
    t0 = time.time()
    rng = np.random.RandomState(1)
    for it in range(steps):
        views = [[t.to(device, non_blocking=True) for t in v] for v in sampler.batch(batch)]
        views = [(f.float(), p, v, pd) for f, p, v, pd in views]
        frac = it / steps
        t_temp = 0.04 + 0.03 * min(1.0, frac / 0.3)
        mom = 1 - (1 - 0.994) * (math.cos(math.pi * frac) + 1) / 2
        if gram_teacher is None and frac >= gram_from:
            gram_teacher = copy.deepcopy(teacher).eval()
            log.info("stage 2: Gram anchoring enabled at step %d", it)
        ng = sampler.ng
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                t_out = [teacher(*views[g]) for g in range(ng)]
                g_out = [gram_teacher(*views[g]) for g in range(ng)] if gram_teacher is not None else None
            fmasks = []
            for g in range(ng):
                pad = views[g][3]
                r = rng.uniform(*ibot_ratio)
                fmasks.append((torch.rand(pad.shape, device=device) < r) & pad)
            s_out = [student(*views[g], fmask=fmasks[g]) for g in range(ng)] + \
                    [student(*views[v]) for v in range(ng, len(views))]
        loss_d = loss_i = loss_k = loss_g = 0.0
        for s in STREAMS:
            tp = [sinkhorn(t_heads[s](t_out[g][s][0]), t_temp) for g in range(ng)]
            for vi, so in enumerate(s_out):
                ls = F.log_softmax(heads[s](so[s][0]) / 0.1, -1)
                for g in range(ng):
                    if g != vi:
                        loss_d = loss_d + (-(tp[g] * ls).sum(-1).mean()) / (ng * (len(s_out) - 1))
            for g in range(ng):
                m = fmasks[g]
                if m.sum() > 0:
                    tt = sinkhorn(t_heads[s](t_out[g][s][1][m]), t_temp)
                    st_ = F.log_softmax(heads[s](s_out[g][s][1][m]) / 0.1, -1)
                    loss_i = loss_i + (-(tt * st_).sum(-1).mean()) / ng
                if gram_teacher is not None:
                    pad = views[g][3]
                    zs = F.normalize(s_out[g][s][1].float(), dim=-1)
                    zt = F.normalize(g_out[g][s][1].float(), dim=-1)
                    pm = (pad[:, :, None] & pad[:, None, :]).float()
                    loss_g = loss_g + (((zs @ zs.transpose(1, 2) - zt @ zt.transpose(1, 2)) ** 2) * pm).sum() / pm.sum() / ng
            loss_k = loss_k + koleo(s_out[0][s][0])
        loss = loss_d + loss_i + koleo_w * loss_k + (gram_w * loss_g if gram_teacher is not None else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 3.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for ps, pt in zip(list(student.parameters()) + list(heads.parameters()),
                              list(teacher.parameters()) + list(t_heads.parameters())):
                pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
        if it % 100 == 0 or it == steps - 1:
            with torch.no_grad():
                ent = 0.0
                for s in STREAMS:
                    pr = sinkhorn(t_heads[s](t_out[0][s][0]), t_temp).mean(0)
                    ent += float(-(pr * torch.log(pr + 1e-9)).sum()) / len(STREAMS)
            rec = dict(step=it, loss=float(loss), dino=float(loss_d), ibot=float(loss_i), koleo=float(loss_k),
                       gram=float(loss_g) if gram_teacher is not None else 0.0, proto_entropy=ent,
                       lr=sched.get_last_lr()[0], sec=time.time() - t0)
            hist.append(rec)
            log.info("step %d loss %.3f dino %.3f ibot %.3f koleo %.3f gram %.4f H %.2f (%.0fs)", it, rec["loss"],
                     rec["dino"], rec["ibot"], rec["koleo"], rec["gram"], ent, rec["sec"])
        if eval_fn is not None and (it + 1) % eval_every == 0:
            r = eval_fn(teacher)
            hist.append(dict(step=it, eval=r))
            log.info("step %d eval %s", it, r)
            teacher.eval()
    if ckpt_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=teacher.state_dict(), cfg=teacher.cfg, hist=hist), ckpt_path)
    return teacher, hist


def load_encoder(path, device="cuda") -> SignEncoder:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SignEncoder(nap_N=ck["model"]["nap"].cpu().numpy(), **ck["cfg"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()
