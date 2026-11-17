"""Emotion by reference (B) vs v4 tag + PSOLA (A), same speaker for both.

refs.csv columns: speaker,gender,emotion,path,ref_text   (emotion in neutral|happy|sad|angry; one row per speaker x emotion)
  A = neutral reference of the speaker + emotion tag + PSOLA (what the web app does now)
  B = that speaker's emotional reference, no tag, no PSOLA
Sentences are emotion-neutral on purpose: listeners must judge the voice, not the words.
Output: out/<blind_id>.wav, out/key.csv (answers; do not show listeners), out/clips.csv (blind ids only).

  python gen.py --model ../../models/Speech-model-v4-TTS --refs refs.csv --intensity 0.75
"""
import argparse, csv, random, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="../../models/Speech-model-v4-TTS")
ap.add_argument("--refs", default="refs.csv")
ap.add_argument("--sentences", default="sentences.txt")
ap.add_argument("--out", default="out")
ap.add_argument("--intensity", type=float, default=0.75)
ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: cuda if available else cpu)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="only the first N sentences (smoke test)")
a = ap.parse_args()

sys.path.insert(0, str(Path(a.model).resolve()))
import numpy as np  # noqa: E402
import tts as _tts  # noqa: E402
from tts import Speech  # noqa: E402

_emotionize = _tts.emotionize
def _emotionize_headroom(y, emo, s, sr=24000):
    """v4's PSOLA adds up to +4 dB and then hard-clips at 0.99 (audible crackle on loud voices). Same transform, 12 dB headroom, no clip."""
    return _emotionize(np.asarray(y) * 0.25, emo, s, sr) * 4.0
_tts.emotionize = _emotionize_headroom
PEAK = 10 ** (-1 / 20)                                                    # -1 dBFS

def cap(w):
    m = float(np.max(np.abs(w))) if len(w) else 0.0
    return w * (PEAK / m) if m > PEAK else w

rows = list(csv.DictReader(open(a.refs, encoding="utf-8")))
refs = {(r["speaker"], r["emotion"]): r for r in rows}
speakers = sorted({r["speaker"] for r in rows})
sents = [s.strip() for s in open(a.sentences, encoding="utf-8") if s.strip()]
if a.limit:
    sents = sents[:a.limit]
out = Path(a.out); out.mkdir(exist_ok=True)

s = Speech(a.model, device=a.device, warm=False)        # v4 refs/ are not needed: every voice here is a THAI-SER reference
for (spk, emo), r in refs.items():                       # register every reference as a voice profile (absolute path)
    s.voices[f"{spk}__{emo}"] = {"ref": str(Path(r["path"]).resolve()), "ref_text": r["ref_text"]}
s.say("ทดสอบ", voice=f"{speakers[0]}__neutral", nfe=4)                   # warm-up (kernels, caches)

jobs = []
for spk in speakers:
    g = next(r["gender"] for r in rows if r["speaker"] == spk)
    if (spk, "neutral") not in refs:
        print(f"skip {spk}: no neutral reference"); continue
    for i, t in enumerate(sents):
        jobs.append(dict(speaker=spk, gender=g, sent=i, text=t, emotion="neutral", method="neutral", voice=f"{spk}__neutral", emo_arg=None))
        for emo in ("happy", "sad", "angry"):
            jobs.append(dict(speaker=spk, gender=g, sent=i, text=t, emotion=emo, method="A_tag_psola", voice=f"{spk}__neutral", emo_arg=emo))
            if (spk, emo) in refs:
                jobs.append(dict(speaker=spk, gender=g, sent=i, text=t, emotion=emo, method="B_emo_ref", voice=f"{spk}__{emo}", emo_arg=None))

rng = random.Random(a.seed)
ids = rng.sample(range(1000, 9999), len(jobs))
cols = ["id", "speaker", "gender", "sent", "text", "emotion", "method", "voice", "sec"]
for j, bid in zip(jobs, ids):
    j["id"] = f"c{bid}"
with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:        # written up front: same command = same ids, so a stopped run resumes
    w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(jobs)
t0 = time.time()
for k, j in enumerate(jobs):
    if (out / f"{j['id']}.wav").exists():
        continue                                                     # resume: already generated
    t1 = time.time()
    wav, sr = s.say(j["text"], voice=j["voice"], emotion=j["emo_arg"], intensity=a.intensity, seed=a.seed)
    s.save(out / f"{j['id']}.wav", cap(wav), sr)
    print(f"[{k+1}/{len(jobs)}] {j['id']} {j['speaker']} {j['method']} {j['emotion']}  {time.time()-t1:.1f}s/clip  total {time.time()-t0:.0f}s", flush=True)
order = jobs[:]; rng.shuffle(order)
with open(out / "clips.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "text"]); [w.writerow([j["id"], j["text"]]) for j in order]
print(f"done: {len(jobs)} clips in {time.time()-t0:.0f}s -> {out}/")
