"""EmoSteer-style steering on v4 (no training). Lab only; models/ untouched.

 1) build:  for happy/sad/angry take THAI-SER actors (train split), run the DiT once per clip at fixed t on the clip's own noised mel, record each chosen
    block's mean output over frames. Vector[e][layer] = mean over actors of (mean emotional clips - mean neutral clips of the SAME actor), so speaker cancels.
 2) gen:    neutral reference + fixed tts (no clipping) + vector added to the same blocks' outputs (strength alpha). No tag, no PSOLA.

  HF_HUB_OFFLINE=1 caffeinate -i python -u emo_steer.py --preflight     # 1 clip, prints structure and timing
  HF_HUB_OFFLINE=1 caffeinate -i python -u emo_steer.py                  # build (cached vectors.pt) + generate -> out/index.html
"""
import argparse, csv, html, json, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="../../models/Speech-model-v4-TTS"); ap.add_argument("--data", default="../emo_ref/thai-ser")
ap.add_argument("--out", default="out"); ap.add_argument("--layers", default="1,6,11,16,21"); ap.add_argument("--alphas", default="1,2")
ap.add_argument("--actors", type=int, default=12); ap.add_argument("--per", type=int, default=6); ap.add_argument("--t", type=float, default=0.5)
ap.add_argument("--sents", default="0,4"); ap.add_argument("--preflight", action="store_true"); ap.add_argument("--rebuild", action="store_true")
a = ap.parse_args()
HERE = Path(__file__).resolve().parent; OUT = HERE / a.out; OUT.mkdir(exist_ok=True)
MODEL, DATA = (HERE / a.model).resolve(), (HERE / a.data).resolve(); REF = HERE.parent / "emo_ref"
LAYERS = [int(x) for x in a.layers.split(",")]; ALPHAS = [float(x) for x in a.alphas.split(",")]
EMOS = ["happy", "sad", "angry"]; EMO = {"Neutral": "neutral", "Happy": "happy", "Sad": "sad", "Angry": "angry"}
TEST, VAL = {"003", "004", "005", "009"}, {"018", "019"}      # same held-out actors as training; refs for listening come from TEST actors

sys.path.insert(0, str(HERE))
from tts_fixed import Speech                                   # lab copy of v4 tts.py, peak-scale instead of hard clip
s = Speech(str(MODEL), device="cpu", warm=False)
model = s.model.float().eval()
blocks = model.transformer.transformer_blocks
print(f"DiT blocks: {len(blocks)}; steering layers {LAYERS}", flush=True)

# ---------------------------------------------------------------- 1) vectors
REC = {"on": False, "acc": {}}
def rec_hook(i):
    def f(mod, inp, out):
        if REC["on"]: REC["acc"][i] = out.detach().float().mean(dim=(0, 1))
    return f
for i in LAYERS: blocks[i].register_forward_hook(rec_hook(i))

def wav24(path):
    y, sr = sf.read(str(path), dtype="float32"); y = y.mean(1) if y.ndim > 1 else y; y = torch.from_numpy(y)
    if sr != 24000: y = torchaudio.functional.resample(y, sr, 24000)
    return y * (0.1 / y.pow(2).mean().sqrt().clamp_min(1e-4))

def clip_feat(r):
    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx
    mel = model.mel_spec(wav24(DATA / r["file"])[None]).permute(0, 2, 1)
    g = torch.Generator().manual_seed(0); x0 = torch.randn(mel.shape, generator=g)
    tt = torch.tensor([a.t]); phi = (1 - a.t) * x0 + a.t * mel
    ids = list_str_to_idx(convert_char_to_pinyin([r["script_sent"].strip()]), model.vocab_char_map)
    REC["on"], REC["acc"] = True, {}
    with torch.inference_mode():
        model.transformer(x=phi, cond=torch.zeros_like(mel), text=ids, time=tt, drop_audio_cond=False, drop_text=False, mask=None)
    REC["on"] = False
    return torch.stack([REC["acc"][i] for i in LAYERS])         # [L, 1024]

def build():
    rows = [r for r in csv.DictReader(open(DATA / "meta.csv", encoding="utf-8"))
            if r["turn_type"] == "script" and r["assigned_emo"] in EMO and r["majority_emo"] == r["assigned_emo"] and float(r["agreement"] or 0) >= 0.7]
    by = {}
    for r in rows: by.setdefault(r["actor_id"], {}).setdefault(EMO[r["assigned_emo"]], []).append(r)
    ids = sorted(k for k, v in by.items() if k not in TEST | VAL and "neutral" in v and all(e in v for e in EMOS))
    random.Random(0).shuffle(ids); ids = ids[:a.actors]
    print(f"actors for vectors: {len(ids)}", flush=True)
    diffs = {e: [] for e in EMOS}; t0 = time.time()
    for n, k in enumerate(ids):
        rng = random.Random(k); feat = {}
        for e in ["neutral"] + EMOS:
            feat[e] = torch.stack([clip_feat(r) for r in rng.sample(by[k][e], min(a.per, len(by[k][e])))]).mean(0)
            if a.preflight: print("preflight ok", feat[e].shape, f"{time.time()-t0:.1f}s", flush=True); return None
        for e in EMOS: diffs[e].append(feat[e] - feat["neutral"])
        print(f"  actor {k} ({n+1}/{len(ids)}) {time.time()-t0:.0f}s", flush=True)
    V = {e: torch.stack(diffs[e]).mean(0) for e in EMOS}         # [L, 1024]
    # consistency: mean cosine between per-actor differences (high = a shared direction, low = noise)
    for e in EMOS:
        d = torch.nn.functional.normalize(torch.stack(diffs[e]), dim=-1)           # [A, L, D]
        m = d.mean(0); print(f"  {e}: |v| per layer {[round(float(V[e][i].norm()), 2) for i in range(len(LAYERS))]}, direction agreement {[round(float(m[i].norm()), 2) for i in range(len(LAYERS))]}", flush=True)
    torch.save({"V": V, "layers": LAYERS, "t": a.t}, OUT / "vectors.pt"); return V

vp = OUT / "vectors.pt"
if vp.exists() and not a.rebuild and not a.preflight:
    ck = torch.load(vp); assert ck["layers"] == LAYERS, "layers changed: use --rebuild"; V = ck["V"]; print("vectors loaded", flush=True)
else:
    V = build()
if a.preflight or V is None: sys.exit()

# ---------------------------------------------------------------- 2) generate
CUR = {"e": None, "k": 0.0}
def steer_hook(j):
    def f(mod, inp, out):
        return out if CUR["e"] is None else out + CUR["k"] * V[CUR["e"]][j].to(out.dtype)
    return f
for j, i in enumerate(LAYERS):
    blocks[i].register_forward_hook(steer_hook(j))               # added after the recorder; recorder is off (REC["on"] False)

sents = [x.strip() for x in open(REF / "sentences.txt", encoding="utf-8") if x.strip()]; SI = [int(x) for x in a.sents.split(",")]
refs = [r for r in csv.DictReader(open(REF / "refs.csv", encoding="utf-8")) if r["emotion"] == "neutral"]
for r in refs: s.voices[r["speaker"]] = {"ref": str((REF / r["path"]).resolve()), "ref_text": r["ref_text"]}
VAR = [("neutral", None, 0.0)] + [(f"{e}_x{k:g}", e, k) for e in EMOS for k in ALPHAS]
for r in refs:
    for i in SI:
        for name, e, k in VAR:
            f = OUT / f"{r['speaker']}_s{i}_{name}.wav"
            if f.exists(): continue
            CUR.update(e=e, k=k)
            wav, sr = s.say(sents[i], voice=r["speaker"], seed=0)
            CUR["e"] = None; s.save(f, wav, sr); print(f.name, flush=True)
rows = ["<meta charset=utf-8><style>body{font-family:sans-serif;margin:1em}td{padding:.3em;border-bottom:1px solid #ddd}</style>"
        "<h2>EmoSteer แบบย่อบน v4 (ไม่เทรน, ไม่มี PSOLA, ref neutral) — เทียบ neutral กับ x1 / x2 ของแต่ละอารมณ์</h2><table><tr><th>คน / ประโยค</th>" + "".join(f"<th>{n}</th>" for n, _, _ in VAR) + "</tr>"]
for r in refs:
    for i in SI:
        rows.append(f"<tr><td>{r['speaker']}<br>{html.escape(sents[i])}</td>" + "".join(f"<td><audio controls preload=none src='{r['speaker']}_s{i}_{n}.wav' style='width:150px'></audio></td>" for n, _, _ in VAR) + "</tr>")
(OUT / "index.html").write_text("\n".join(rows) + "</table>", encoding="utf-8"); print("->", OUT / "index.html")
