"""Round 3: the parts not yet working or never tested, on sentences 0 and 4.
  happy_C  = emotional reference (B) + a mild PSOLA happy layer on top (strength --psola)
  fear / disgust / surprise = v4 tag + PSOLA (A), never listened to before
Adds clips to out/, writes out/key3.csv (answers), out/session3_order.csv, out/play3.html (64 clips: these 32 + 32 existing).
  HF_HUB_OFFLINE=1 caffeinate -i python -u gen3.py
"""
import argparse, csv, json, random, re, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="../../models/Speech-model-v4-TTS"); ap.add_argument("--out", default="out")
ap.add_argument("--sents", default="0,4"); ap.add_argument("--psola", type=float, default=0.6); ap.add_argument("--intensity", type=float, default=0.75)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.model).resolve()))
import tts as _tts
from tts import Speech
_emo = _tts.emotionize
def emo_headroom(y, emo, s, sr=24000):                       # same PSOLA as v4, 12 dB headroom, no hard clip
    return _emo(np.asarray(y) * 0.25, emo, s, sr) * 4.0
_tts.emotionize = emo_headroom
PEAK = 10 ** (-1 / 20)
def cap(w):
    m = float(np.max(np.abs(w))) if len(w) else 0.0
    return w * (PEAK / m) if m > PEAK else w

out = Path(a.out); keep = [int(x) for x in a.sents.split(",")]
refs = list(csv.DictReader(open("refs.csv", encoding="utf-8")))
sents = [s.strip() for s in open("sentences.txt", encoding="utf-8") if s.strip()]
old = list(csv.DictReader(open(out / "key.csv", encoding="utf-8")))
speakers = sorted({r["speaker"] for r in refs}); gender = {r["speaker"]: r["gender"] for r in refs}

jobs = []
for spk in speakers:
    for i in keep:
        jobs.append(dict(speaker=spk, gender=gender[spk], sent=i, text=sents[i], emotion="happy", method="C_ref_psola"))
        for e in ("fear", "disgust", "surprise"):
            jobs.append(dict(speaker=spk, gender=gender[spk], sent=i, text=sents[i], emotion=e, method="A_tag_psola"))
rng = random.Random(3)
used = {k["id"] for k in old}
for j in jobs:
    while True:
        cid = f"d{rng.randrange(1000, 9999)}"
        if cid not in used: used.add(cid); j["id"] = cid; break
cols = ["id", "speaker", "gender", "sent", "text", "emotion", "method"]
with open(out / "key3.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(jobs)

POL = {("neutral", "neutral"), ("B_emo_ref", "happy"), ("A_tag_psola", "sad"), ("B_emo_ref", "angry")}
ids = [k["id"] for k in old if int(k["sent"]) in keep and (k["method"], k["emotion"]) in POL] + [j["id"] for j in jobs]
random.Random(33).shuffle(ids)
with open(out / "session3_order.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["no", "id"]); [w.writerow([i + 1, c]) for i, c in enumerate(ids)]
h = re.sub(r"const C=\[[^\]]*\]", "const C=" + json.dumps(ids), open(out / "play.html", encoding="utf-8").read())
(out / "play3.html").write_text(h, encoding="utf-8")

todo = [j for j in jobs if not (out / f"{j['id']}.wav").exists()]
print(f"{len(jobs)} new clips ({len(todo)} to generate), session = {len(ids)} clips", flush=True)
if todo:
    s = Speech(a.model, warm=False)
    for r in refs:
        s.voices[f"{r['speaker']}__{r['emotion']}"] = {"ref": str(Path(r["path"]).resolve()), "ref_text": r["ref_text"]}
    s.say("ทดสอบ", voice=f"{speakers[0]}__neutral", nfe=4)
    t0 = time.time()
    for k, j in enumerate(todo):
        if j["method"] == "C_ref_psola":
            wav, sr = s.say(j["text"], voice=f"{j['speaker']}__happy", seed=0)
            wav = emo_headroom(wav, "happy", a.psola, sr)
        else:
            wav, sr = s.say(j["text"], voice=f"{j['speaker']}__neutral", emotion=j["emotion"], intensity=a.intensity, seed=0)
        s.save(out / f"{j['id']}.wav", cap(wav), sr)
        print(f"[{k+1}/{len(todo)}] {j['id']} {j['speaker']} {j['method']} {j['emotion']}  total {time.time()-t0:.0f}s", flush=True)
print(f"done -> open {out}/play3.html")
