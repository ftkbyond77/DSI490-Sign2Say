"""Measure how much each emotion differs from neutral in THAI-SER (same actor, same scripted sentence) -> templates_measured.json
Same 4 numbers as TEMPLATE in tts.py: (F0 median shift in semitones, F0 range scale, speaking-rate factor, level dB).
  python measure_templates.py          (needs praat-parselmouth, already used by tts.py; ~2-4 min)
"""
import csv, json, statistics as st, sys
from collections import defaultdict
import numpy as np, soundfile as sf, parselmouth
from parselmouth.praat import call

EMO = {"Neutral": "neutral", "Happy": "happy", "Sad": "sad", "Angry": "angry"}
rows = [r for r in csv.DictReader(open("thai-ser/meta.csv", encoding="utf-8"))
        if r["turn_type"] == "script" and r["assigned_emo"] in EMO and r["majority_emo"] == r["assigned_emo"] and float(r["agreement"] or 0) >= 0.7]
def feat(path):
    y, sr = sf.read("thai-ser/" + path, dtype="float64"); y = y.mean(1) if y.ndim > 1 else y
    snd = parselmouth.Sound(y, sr)
    f = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=450).selected_array["frequency"]; f = f[f > 0]
    if len(f) < 15: return None
    s = 12 * np.log2(f)
    db = call(snd.to_intensity(), "Get mean", 0, 0, "energy")
    return dict(med=float(np.median(s)), sd=float(np.std(s)), dur=len(y) / sr, db=float(db))
by = defaultdict(lambda: defaultdict(list))                    # (actor, gender, sentence) -> emotion -> [feat]
for i, r in enumerate(rows):
    ft = feat(r["file"])
    if ft: by[(r["actor_id"], r["actor_gender"], r["script_sent"])][EMO[r["assigned_emo"]]].append(ft)
    if i % 200 == 0: print(f"{i}/{len(rows)}", flush=True)
per = defaultdict(lambda: defaultdict(list))                    # emotion -> gender -> [per-(actor,sentence) deltas]
for (act, g, sent), em in by.items():
    if "neutral" not in em: continue
    n = {k: float(np.mean([x[k] for x in em["neutral"]])) for k in ("med", "sd", "dur", "db")}
    for e in ("happy", "sad", "angry"):
        if e not in em: continue
        m = {k: float(np.mean([x[k] for x in em[e]])) for k in ("med", "sd", "dur", "db")}
        per[e][g].append((m["med"] - n["med"], m["sd"] / max(n["sd"], 1e-3), n["dur"] / max(m["dur"], 1e-3), m["db"] - n["db"]))
out, rep = {}, []
for e in ("happy", "sad", "angry"):
    allv = [v for g in per[e] for v in per[e][g]]
    med = [round(float(np.median([v[k] for v in allv])), 2) for k in range(4)]
    out[e] = med; rep.append(f"{e:6s} n={len(allv):3d}  all {med}  " + "  ".join(f"{g}: {[round(float(np.median([v[k] for v in per[e][g]])), 2) for k in range(4)]}" for g in per[e]))
json.dump({"doc": "median over (actor, sentence) of emotion minus neutral: [semitones, F0-range ratio, rate factor (neutral dur / emotion dur), dB]; THAI-SER scripted turns, agreement>=0.7", "templates": out},
          open("templates_measured.json", "w", encoding="utf-8"), indent=1)
print("\n".join(rep)); print("old TEMPLATE: happy (1.5,1.3,1.06,2.0) sad (-1.5,0.6,0.85,-3.0) angry (1.0,1.35,1.08,4.0)"); print("-> templates_measured.json")
