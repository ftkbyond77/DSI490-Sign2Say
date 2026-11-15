"""Pick emotion references from the THAI-SER subset -> refs.csv (speaker,gender,emotion,path,ref_text,agreement,dur).

Rules: scripted turns only (known transcript, same sentence in every emotion); listeners' majority label must equal the acted
emotion; emotional clips prefer the 'High' intensity take; 2.5-8 s; per actor choose the script sentence whose 4 clips
(neutral/happy/sad/angry) have the highest mean agreement; keep the --top best actors per gender.
  python select_refs.py --top 2
"""
import argparse, csv, json, subprocess
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=2); ap.add_argument("--meta", default="thai-ser/meta.csv")
a = ap.parse_args()
EMO = {"Neutral": "neutral", "Happy": "happy", "Sad": "sad", "Angry": "angry"}

def dur(p):
    try:
        return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
    except Exception:
        import soundfile as sf; return sf.info(p).duration

def clipped(p):
    import re
    o = subprocess.run(["ffmpeg", "-v", "info", "-i", p, "-af", "astats=metadata=0:reset=0", "-f", "null", "-"], capture_output=True, text=True).stderr
    f = re.findall(r"Flat factor: *(\S+)", o)
    return bool(f) and float(f[-1]) > 0                          # flat tops = the take was clipped before the dataset normalised it

rows = [r for r in csv.DictReader(open(a.meta, encoding="utf-8"))
        if r["turn_type"] == "script" and r["assigned_emo"] in EMO and r["majority_emo"] == r["assigned_emo"]]
best = defaultdict(dict)                      # (actor, gender, sentence) -> emotion -> row
for r in rows:
    d = dur("thai-ser/" + r["file"])
    if not 2.5 <= d <= 8.0 or clipped("thai-ser/" + r["file"]):
        continue
    r["dur"] = round(d, 2); r["agreement"] = float(r["agreement"] or 0)
    k = (r["actor_id"], r["actor_gender"], r["script_sent"]); e = EMO[r["assigned_emo"]]
    score = r["agreement"] + (0.05 if (e == "neutral" or r["script_intensity"] == "High") else 0)
    if e not in best[k] or score > best[k][e]["_s"]:
        r["_s"] = score; best[k][e] = r
cands = defaultdict(list)
for (act, g, sent), em in best.items():
    if len(em) == 4:
        cands[g].append((sum(x["agreement"] for x in em.values()) / 4, act, sent, em))
out, report = [], []
for g in ("Female", "Male"):
    seen = set()
    for sc, act, sent, em in sorted(cands[g], key=lambda x: -x[0]):
        if act in seen: continue
        seen.add(act)
        if len(seen) > a.top: break
        report.append(f"{g:6s} actor{act} mean_agreement={sc:.3f}  '{sent}'")
        for e in ("neutral", "happy", "sad", "angry"):
            r = em[e]
            out.append(dict(speaker=f"{g[0].lower()}{act}", gender=g.lower(), emotion=e, path="thai-ser/" + r["file"], ref_text=sent,
                            agreement=r["agreement"], dur=r["dur"], intensity=r["script_intensity"]))
with open("refs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
print("\n".join(report)); print(f"-> refs.csv ({len(out)} rows)")
