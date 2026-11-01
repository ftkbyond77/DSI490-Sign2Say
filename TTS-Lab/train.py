"""ThonburianTTS (F5-TTS, 337M) fine-tune, v4:  multi-grade data, male-balanced + youth-weighted sampling, prompt-format samples, onset-weighted loss,
emotion tags, gap samples.  One planned run, time-budgeted, validated in-run.

  Vertex A100 :  python train.py --data /tmp/data --ckpt v3.safetensors --vocab vocab.txt --out /gcs/.../out --minutes 55
  local check :  python train.py --preflight ...

Sample kinds (all mel-domain, computed in the DataLoader workers):
  real   one clip, F5's random-span infill (loss on the masked span)
  pair   [clip A of speaker S][clip B of speaker S] with A as the prompt and loss on B only -- exactly the inference layout (reference + new text)
  gap    "A [murmur/pause] B" synthetic samples that teach the "…" marker
Emotion: the text of a tagged sample starts with one tag symbol (config 'emotion_tags': unused vocab symbols); tag dropout keeps the untagged path.
Loss: flow-matching MSE on the generated span, x ONSET_W on the first ONSET_N frames after the prompt boundary (the male "lost first word" fix).
"""
import argparse, copy, json, math, random, time
from pathlib import Path

import numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import butter, sosfilt
from torch.utils.data import DataLoader, Dataset, SequentialSampler

SR, HOP = 24000, 256
ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)   # identical to the baseline loader
GAP = "…"
ONSET_N, ONSET_W = 12, 2.0                                                          # first 12 frames (~128 ms) of the generated span count double


# ----------------------------------------------------------------------------- gap synthesis (unchanged from v1)
def _fade(x, n=int(0.02 * SR)):
    x = x.copy(); n = min(n, len(x) // 2); w = np.linspace(0, 1, n, dtype=np.float32)
    x[:n] *= w; x[-n:] *= w[::-1]
    return x


def room_tone(y, n, rng):
    """n samples of the clip's own quietest background (tiled, cross-faded)."""
    w = int(0.08 * SR)
    fr = [y[i:i + w] for i in range(0, max(len(y) - w, 1), w // 2)]
    quiet = sorted(fr, key=lambda f: float((f ** 2).mean()))[:3]
    seg = np.concatenate(quiet); seg = seg - seg.mean()
    out = np.zeros(n, np.float32); i = 0; cf = int(0.01 * SR)
    while i < n:
        s = np.roll(seg, int(rng.integers(len(seg)))); k = min(len(s), n - i)
        s = s[:k].copy(); m = min(cf, k); s[:m] *= np.linspace(0, 1, m, dtype=np.float32)
        out[i:i + k] += s; i += max(k - cf, 1)
    return out


def murmur(y, dur, rng):
    """Soft 'mm' hum: the clip's loudest voiced stretch, low-passed, quiet, with a smooth envelope."""
    w = int(dur * SR)
    if len(y) <= w:
        return np.zeros(w, np.float32)
    e = np.convolve(y ** 2, np.ones(w, np.float32) / w, "valid")
    i = int(np.argmax(e[: len(e)])); seg = y[i:i + w]
    seg = sosfilt(butter(4, 500, "low", fs=SR, output="sos"), seg).astype(np.float32)
    rms_t = 0.30 * float(np.sqrt((y ** 2).mean())); seg *= rms_t / (float(np.sqrt((seg ** 2).mean())) + 1e-6)
    return seg * np.hanning(w).astype(np.float32)


def make_gap(ref, length, rng):
    n = int(length * SR); g = room_tone(ref, n, rng)
    if rng.random() < 0.55:                                            # quiet hum inside the pause
        md = float(rng.uniform(0.25, max(0.3, min(0.9, length - 0.1)))); m = murmur(ref, md, rng)
        s = int(rng.uniform(0.04, max(length - md - 0.04, 0.05)) * SR); g[s:s + len(m)] += m[: max(n - s, 0)]
    return _fade(g, int(0.03 * SR))


def join(parts):
    out = parts[0]
    for p in parts[1:]:
        k = int(0.012 * SR); a = out.copy(); b = p.copy()
        a[-k:] *= np.linspace(1, 0, k, dtype=np.float32); b[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        out = np.concatenate([a, b])
    return out


# ----------------------------------------------------------------------------- emotion prosody transform (labels by construction)
# emotion -> (shift of mean F0 in semitones, F0-range scale, speaking-rate factor, gain dB) at strength 1.0 (descriptive templates after
# Banse & Scherer 1996); the strength s in [0.35, 1] scales every deviation linearly, so a tag at alpha=s means "s times this shift"
TEMPLATE = {"happy": (1.5, 1.30, 1.06, 2.0), "sad": (-1.5, 0.60, 0.85, -3.0), "angry": (1.0, 1.35, 1.08, 4.0),
            "fear": (3.0, 1.10, 1.12, -1.0), "disgust": (-1.0, 0.90, 0.90, -1.0), "surprise": (3.5, 1.50, 1.00, 2.0)}


def emotionize(y, emo, s):
    """Praat PSOLA: move the clip's own F0 mean/range, rate and level toward `emo` by strength s. None when the clip has too few pitch points."""
    import parselmouth
    from parselmouth.praat import call
    st, rg, rate, db = TEMPLATE[emo]
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    man = call(snd, "To Manipulation", 0.01, 70, 450)
    pt = call(man, "Extract pitch tier"); n = call(pt, "Get number of points")
    if n < 3:
        return None
    t = np.array([call(pt, "Get time from index", i) for i in range(1, n + 1)]); f = np.array([call(pt, "Get value at index", i) for i in range(1, n + 1)])
    med = np.median(f); f2 = med * 2 ** (s * st / 12) * (f / med) ** (1 + s * (rg - 1))
    new = call("Create PitchTier", "p", 0, snd.xmax)
    for a_, b_ in zip(t, f2):
        call(new, "Add point", float(a_), float(b_))
    call([new, man], "Replace pitch tier")
    dur = call("Create DurationTier", "d", 0, snd.xmax); r = 1 / (1 + s * (rate - 1))
    call(dur, "Add point", 0.0, r); call(dur, "Add point", snd.xmax, r)
    call([dur, man], "Replace duration tier")
    out = call(man, "Get resynthesis (overlap-add)").values[0].astype(np.float32) * 10 ** (s * db / 20)
    return np.clip(out, -0.99, 0.99)


# ----------------------------------------------------------------------------- data
class Clips(Dataset):
    """Items are built once per epoch by `plan()`; mel is computed in the worker. Item = dict(kind, i, j, L, seed, text, dur, tag)."""
    def __init__(self, m, root, mel, tags, a, seed=0):
        self.m, self.root, self.mel, self.tags, self.a = m.reset_index(drop=True), Path(root), mel, tags, a
        self.by = self.m.groupby("speaker_id").indices
        self.items = []
        self.seed = seed
        self.w = self.m.weight.values if "weight" in self.m else np.ones(len(self.m))
        self.w = self.w / self.w.sum()

    def plan(self, epoch):
        a, m = self.a, self.m
        rng = random.Random(self.seed * 1000 + epoch); nrng = np.random.default_rng(self.seed * 1000 + epoch)
        n = len(m)
        pick = nrng.choice(n, size=n, replace=True, p=self.w)                          # weighted epoch: male-hour balance, youth, emotion classes
        self.items = []
        for i in pick:
            r = m.iloc[i]
            emo, alpha, dsp = self.emotion_of(r, rng)
            tag = self.tags[emo] if emo else ""
            kind = "pair" if rng.random() < (0.85 if emo else a.pair_frac) else "real"
            if kind == "pair":
                pool = [j for j in self.by[r.speaker_id] if j != i and m.duration.iloc[j] <= 9 and m.duration.iloc[j] + r.duration <= 17]
                if not pool:
                    kind = "real"
                else:
                    j = rng.choice(pool); b = m.iloc[j]
                    self.items.append(dict(kind="pair", i=int(j), j=int(i), seed=rng.randrange(1 << 30), dur=b.duration + r.duration + 0.05, emo=emo if dsp else None, alpha=alpha,
                                           text=b.text.strip() + ". " + tag + r.text.strip()))   # prompt = clip j (clean), target = clip i (tagged)
                    continue
            self.items.append(dict(kind="real", i=int(i), seed=rng.randrange(1 << 30), dur=r.duration, emo=emo if dsp else None, alpha=alpha, text=tag + r.text.strip()))
        for _ in range(int(a.gap_frac * n)):
            i = int(nrng.choice(n, p=self.w)); x = m.iloc[i]; kind = rng.choices(["mid", "end", "start"], [0.6, 0.25, 0.15])[0]
            pool = [j for j in self.by[x.speaker_id] if j != i]
            if kind == "mid" and not pool:
                kind = "end"
            L = rng.uniform(0.45, 1.4) if kind != "start" else rng.uniform(0.35, 0.9)
            j = rng.choice(pool) if kind == "mid" else i
            if kind == "mid" and x.duration + m.iloc[j].duration > 15:
                continue
            b = m.iloc[j]
            if kind == "mid":
                t, d = f"{x.text.strip()} {GAP} {b.text.strip()}", x.duration + L + b.duration
            elif kind == "end":
                t, d = f"{x.text.strip()} {GAP}", x.duration + L
            else:
                t, d = f"{GAP} {x.text.strip()}", x.duration + L
            self.items.append(dict(kind="gap", gk=kind, i=i, j=int(j), L=L, seed=rng.randrange(1 << 30), dur=d, text=t, tlen=0))

    def emotion_of(self, r, rng):
        """-> (emotion or None, alpha, dsp): a prosody-transformed clip (label by construction, random strength), a natural pseudo-label (SER + prosody agree),
        a plain neutral tag (10 %), or nothing (the untagged path = v3 behaviour)."""
        if not self.tags:
            return None, 1.0, False
        x = rng.random(); a = self.a
        if x < a.dsp_frac:
            return rng.choice(list(TEMPLATE)), rng.uniform(0.35, 1.0), True
        emo = r.get("emo") if hasattr(r, "get") else None
        if isinstance(emo, str) and emo in self.tags and emo != "neutral" and x < a.dsp_frac + a.nat_frac:
            return emo, 1.0, False
        if x > 0.9:
            return "neutral", 1.0, False
        return None, 1.0, False

    def get_frame_len(self, k):
        return self.items[k]["dur"] * SR / HOP

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        y, _ = sf.read(self.root / self.m.iloc[i].path, dtype="float32")
        return y

    def _onset(self, y, lead_ms, rng):
        """normalise the leading silence to a fixed 60-140 ms (trim or pad with the clip's own room tone) -- same onset for every clip."""
        want = int(rng.uniform(0.06, 0.14) * SR); have = int(lead_ms / 1000 * SR)
        if have > want:
            return y[have - want:]
        return np.concatenate([room_tone(y, want - have, rng) if want - have > 0 else y[:0], y])

    def _emo(self, y, it, rng):
        if it.get("emo"):
            z = emotionize(y, it["emo"], it["alpha"])
            if z is not None:
                return z
            it["text"] = it["text"].replace(self.tags[it["emo"]], "", 1); it["alpha"] = 1.0                  # too few pitch points: untagged, original audio
        return y

    def __getitem__(self, k):
        it = self.items[k]; rng = np.random.default_rng(it["seed"])
        if it["kind"] == "real":
            r = self.m.iloc[it["i"]]; y = self._load(it["i"])
            y = self._emo(y, it, rng)
            if self.a.onset_norm:
                y = self._onset(y, r.lead_ms, rng)
            span = None
        elif it["kind"] == "pair":
            ra, rb = self.m.iloc[it["i"]], self.m.iloc[it["j"]]
            ya, yb = self._load(it["i"]), self._load(it["j"])
            yb = self._emo(yb, it, rng)
            if self.a.onset_norm:
                yb = self._onset(yb, rb.lead_ms, rng)
            tail = int(rng.uniform(0.0, 0.12) * SR)                                 # prompt tail: as in preprocess_ref_audio_text (<= ~50 ms of silence)
            ya = ya[: max(len(ya) - max(int(ra.trail_ms / 1000 * SR) - tail, 0), SR // 2)]
            y = np.concatenate([ya, yb]); span = len(ya)
        else:
            a = self._load(it["i"]); g = make_gap(a, it["L"], rng)
            y = join([a, g, self._load(it["j"])]) if it["gk"] == "mid" else (join([a, g]) if it["gk"] == "end" else join([g, a]))
            span = None
        mel = self.mel(torch.from_numpy(y)[None]).squeeze(0)
        return dict(mel_spec=mel, text=it["text"], span_start=-1 if span is None else span // HOP, alpha=float(it.get("alpha", 1.0)))


def collate(b):
    L = torch.LongTensor([x["mel_spec"].shape[-1] for x in b])
    mel = torch.stack([torch.nn.functional.pad(x["mel_spec"], (0, int(L.max()) - x["mel_spec"].shape[-1])) for x in b])
    t = [x["text"] for x in b]
    return dict(mel=mel, mel_lengths=L, text=t, text_lengths=torch.LongTensor([len(x) for x in t]), span_start=torch.LongTensor([x["span_start"] for x in b]), alpha=torch.tensor([x["alpha"] for x in b]))


# ----------------------------------------------------------------------------- model
def build(vocab, ckpt, dev):
    from safetensors.torch import load_file
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer
    cmap, n = get_tokenizer(vocab, "custom")
    model = CFM(transformer=DiT(**ARCH, text_num_embeds=n, mel_dim=100), mel_spec_kwargs=dict(
        n_fft=1024, hop_length=HOP, win_length=1024, n_mel_channels=100, target_sample_rate=SR, mel_spec_type="vocos"), vocab_char_map=cmap)
    sd = load_file(ckpt) if ckpt.endswith("safetensors") else torch.load(ckpt, map_location="cpu")["ema_model_state_dict"]
    sd = {k.replace("ema_model.", ""): v for k, v in sd.items() if k not in ("initted", "step")}
    for k in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        sd.pop(k, None)
    print("load:", model.load_state_dict(sd, strict=False))
    return model.to(dev), cmap


def init_tags(model, cmap, tags):
    """emotion tag rows start at the mean embedding of the Thai characters + small noise, so the tag is a distinct but 'in-distribution' vector."""
    e = model.transformer.text_embed.text_embed.weight
    th = [i + 1 for c, i in cmap.items() if "฀" <= c <= "๿"]
    mean = e.data[th].mean(0)
    g = torch.Generator().manual_seed(0)
    for t in tags.values():
        e.data[cmap[t] + 1] = mean + 0.5 * e.data[th].std(0) * torch.randn(mean.shape, generator=g).to(e.device)


ALPHA = {}


def install_alpha_hook(model, cmap, tags):
    """Tag embeddings live in their own parameter `tag_emb` (own, higher learning rate) and replace the embedding of the tag tokens in a forward hook:
    e = e_neutral + alpha * (e_tag - e_neutral), alpha per sample (the training twin of Speech._set_intensity). save() writes them back into the table."""
    emb = model.transformer.text_embed.text_embed
    names = list(tags)
    ids = [cmap[tags[e]] + 1 for e in names]
    model.tag_emb = torch.nn.Parameter(torch.stack([emb.weight.data[i].clone() for i in ids]))
    lookup = torch.full((emb.weight.shape[0],), -1, dtype=torch.long)
    for k, i in enumerate(ids):
        lookup[i] = k
    model.tag_ids = ids
    neu = names.index("neutral")

    def hook(mod, inp, out):
        al = ALPHA.get("a")
        if al is None:
            return out
        lk = lookup.to(out.device)[inp[0]]
        m = (lk >= 0)[..., None]
        tag = model.tag_emb[lk.clamp(min=0)]
        e0 = model.tag_emb[neu]
        return torch.where(m, e0 + al.to(out.dtype)[:, None, None] * (tag - e0), out)
    emb.register_forward_hook(hook)


def flow_loss(model, mel, text, lens, span_start, dev, alpha=None):
    """F5's CFM.forward with an explicit prompt boundary for 'pair' samples and extra weight on the first frames of the generated span."""
    from f5_tts.model.modules import MelSpec  # noqa: F401
    from f5_tts.model.utils import lens_to_mask, list_str_to_idx, mask_from_frac_lengths
    inp = mel.permute(0, 2, 1).to(dev); B, T, _ = inp.shape
    text = list_str_to_idx(text, model.vocab_char_map).to(dev) if isinstance(text, list) else text
    lens = lens.to(dev); mask = lens_to_mask(lens, length=T)
    frac = torch.zeros((B,), device=dev).float().uniform_(*model.frac_lengths_mask)
    span = mask_from_frac_lengths(lens, frac) & mask
    ss = span_start.to(dev)
    pair = ss >= 0
    idx = torch.arange(T, device=dev)[None]
    span = torch.where(pair[:, None], (idx >= ss[:, None]) & mask, span)
    x1 = inp; x0 = torch.randn_like(x1); t = torch.rand((B,), dtype=inp.dtype, device=dev)
    tt = t[:, None, None]; phi = (1 - tt) * x0 + tt * x1; flow = x1 - x0
    cond = torch.where(span[..., None], torch.zeros_like(x1), x1)
    drop_audio = random.random() < model.audio_drop_prob
    drop_text = False
    if random.random() < model.cond_drop_prob:
        drop_audio, drop_text = True, True
    ALPHA["a"] = alpha.to(dev) if alpha is not None else None
    pred = model.transformer(x=phi, cond=cond, text=text, time=t, drop_audio_cond=drop_audio, drop_text=drop_text, mask=mask)
    l = torch.nn.functional.mse_loss(pred, flow, reduction="none").mean(-1)           # (B, T)
    w = span.float()
    if ONSET_W != 1.0:
        start = torch.where(pair, ss, (span.float().cumsum(1) == 1).float().argmax(1))  # first masked frame (pair: the boundary)
        w = w * (1 + (ONSET_W - 1) * ((idx >= start[:, None]) & (idx < start[:, None] + ONSET_N)).float())
    return (l * w).sum() / w.sum()


def save(model, path):
    from safetensors.torch import save_file
    sd = {"ema_model." + k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items() if k not in ("tag_emb",)}
    if hasattr(model, "tag_emb"):
        w = sd["ema_model.transformer.text_embed.text_embed.weight"]
        for k, i in enumerate(model.tag_ids):
            w[i] = model.tag_emb[k].detach().half().cpu()
    save_file(sd, str(path))


@torch.no_grad()
def val_loss(model, dl, dev):
    model.eval(); torch.manual_seed(1234); random.seed(1234); tot = n = 0
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            tot += flow_loss(model, b["mel"], b["text"], b["mel_lengths"], b["span_start"], dev, b["alpha"]).item(); n += 1
    model.train(); return tot / max(n, 1)


def weights(m):
    """sampling weight per clip: hours-balanced masculine/feminine, youth (estimated age) up-weight, rare-emotion up-weight, speaker-capped."""
    w = np.ones(len(m))
    h = m.groupby("voice").duration.sum()
    for v in ("masculine", "feminine"):
        if v in h:
            w[(m.voice == v).values] *= (h.max() / h[v]) ** 1.0                          # equalise hours
    if "spk_age" in m:
        y = m.spk_age.fillna(m.spk_age.median()).values
        w *= 1 + 1.0 * np.exp(-0.5 * ((y - 21) / 5) ** 2)                               # up to 2x for 16-26 y
    if "emo" in m:
        c = m.emo.value_counts()
        rare = m.emo.map(lambda e: min(8.0, (c.get("neutral", 1) / max(c.get(e, 1), 1)) ** 0.5) if isinstance(e, str) and e != "neutral" else 1.0)
        w *= rare.values
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", default="manifest.csv"); ap.add_argument("--config", default=None, help="model config.json (emotion_tags)")
    ap.add_argument("--minutes", type=float, default=55); ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--frames", type=int, default=19200); ap.add_argument("--accum", type=int, default=2); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gap_frac", type=float, default=0.12); ap.add_argument("--pair_frac", type=float, default=0.5); ap.add_argument("--dsp_frac", type=float, default=0.30); ap.add_argument("--nat_frac", type=float, default=0.08)
    ap.add_argument("--onset_norm", type=int, default=1); ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--preflight", action="store_true"); ap.add_argument("--gate", action="store_true", help="2 optimizer steps on the GPU, then exit (fail-fast check)")
    a = ap.parse_args()
    dev = "cpu" if a.preflight else "cuda"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    tags = json.loads(Path(a.config).read_text(encoding="utf-8")).get("emotion_tags", {}) if a.config else {}
    m = pd.read_csv(Path(a.data) / a.manifest); m = m[m.reason.isna() | (m.reason == "")]
    m["weight"] = 1.0
    mtr, mva = m[m.split == "train"].copy(), m[m.split == "val"].head(300)
    mtr["weight"] = weights(mtr)
    if a.preflight:
        mtr, mva = mtr.head(60), mva.head(8); a.frames, a.accum, a.workers, a.eval_every = 1500, 1, 0, 2
    if a.gate:
        mtr, mva = mtr.head(400), mva.head(8); a.eval_every = 10**9
    model, cmap = build(a.vocab, a.ckpt, dev); model.train()
    if tags:
        miss = [t for t in tags.values() if t not in cmap]
        assert not miss, f"tag symbols missing from the vocab: {miss}"
        if not any(t in open(a.vocab, encoding="utf-8").read() for t in tags.values()):
            raise SystemExit("tag symbols not in vocab")
        init_tags(model, cmap, tags)
        install_alpha_hook(model, cmap, tags)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    from f5_tts.model.dataset import DynamicBatchSampler
    mel = model.mel_spec.cpu()
    tr = Clips(mtr, a.data, mel, tags, a, 0); va = Clips(mva, a.data, mel, tags, a, 1)
    va.a = copy.copy(a); va.a.pair_frac, va.a.gap_frac, va.a.dsp_frac, va.a.nat_frac = 0.5, 0.1, 0.3, 0.0; va.plan(0)           # fixed val set: half pairs, 10 % gaps
    print(f"train clips {len(tr.m)} ({tr.m.duration.sum() / 3600:.1f} h), val items {len(va)}", flush=True)
    vdl = DataLoader(va, collate_fn=collate, batch_sampler=DynamicBatchSampler(SequentialSampler(va), a.frames, 0, drop_residual=False), num_workers=a.workers)
    rest = [p_ for n_, p_ in model.named_parameters() if n_ != "tag_emb"]
    groups = [dict(params=rest, mult=1.0)] + ([dict(params=[model.tag_emb], mult=20.0, weight_decay=0.0)] if hasattr(model, "tag_emb") else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.999), fused=dev == "cuda")
    t0, step, frames, best, bad, log, done = time.time(), 0, a.frames, 1e9, 0, [], False
    budget, warm = a.minutes * 60, 100
    for ep in range(a.epochs):
        tr.plan(ep)
        sampler = DynamicBatchSampler(SequentialSampler(tr), frames, 0, random_seed=ep, drop_residual=False)
        dl = DataLoader(tr, collate_fn=collate, batch_sampler=sampler, num_workers=a.workers)
        micro = 0
        try:
            for b in dl:
                prog = max((time.time() - t0) / budget, ep / a.epochs)
                lr = a.lr * min(1, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1))))
                for g in opt.param_groups:
                    g["lr"] = lr * g["mult"]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    loss = flow_loss(model, b["mel"], b["text"], b["mel_lengths"], b["span_start"], dev, b["alpha"])
                (loss / a.accum).backward(); micro += 1
                if micro % a.accum:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                with torch.no_grad():
                    d = min(0.995, (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - d)
                if step % 10 == 0:
                    print(f"ep {ep} step {step} loss {loss.item():.4f} lr {lr:.2e} {(time.time() - t0) / 60:.1f} min "
                          f"vram {torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0:.1f}G", flush=True)
                if step % a.eval_every == 0 or prog >= 1:
                    vl = val_loss(ema, vdl, dev); log.append(dict(step=step, minutes=(time.time() - t0) / 60, val=vl, train=loss.item()))
                    print(f"== eval step {step} val {vl:.4f} (best {best:.4f})", flush=True)
                    if vl < best:
                        best, bad = vl, 0; save(ema, out / "model_best.safetensors")
                    else:
                        bad += 1
                    (out / "log.json").write_text(json.dumps(log, indent=1))
                    if bad >= a.patience or prog >= 1:
                        done = True; break
                if (a.preflight and step >= 3) or (a.gate and step >= 2):
                    done = True; break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); frames = int(frames * 0.75); print("OOM -> frames", frames, flush=True)
            continue
        if done:
            break
    if a.gate:
        print("gate ok", step, "steps", "vram", torch.cuda.max_memory_allocated() / 1e9); return
    save(ema, out / "model_last.safetensors")
    (out / "log.json").write_text(json.dumps(dict(steps=step, minutes=(time.time() - t0) / 60, best_val=best, frames=frames, accum=a.accum, log=log), indent=1))
    print("done", step, "steps", (time.time() - t0) / 60, "min best val", best)


if __name__ == "__main__":
    main()
