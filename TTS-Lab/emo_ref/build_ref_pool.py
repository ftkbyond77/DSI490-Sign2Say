"""Build a POOL of emotional references per (actor, emotion) from THAI-SER -> refs_pool.csv + refs_pool/*.flac + refs_pool/index.html.

rank 0 = the reference already in refs_final.csv (unchanged behaviour); rank 1..K = extra clean candidates (scripted, listeners' majority
= acted emotion, agreement >= 0.7, 2.5-8 s, not clipped), preferring other sentences. MixSpeech uses seed % n_variants, so a guard retry
(or a different seed) switches to another reference; later a classifier can pick the best of N.
  python build_ref_pool.py --k 4
"""
import argparse, csv, re, shutil, subprocess
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=4); ap.add_argument("--min_agree", type=float, default=0.7)
a = ap.parse_args()
HERE = Path(__file__).resolve().parent
EMO = {"Neutral": "neutral", "Happy": "happy", "Sad": "sad", "Angry": "angry"}
base = list(csv.DictReader(open(HERE / "refs_final.csv", encoding="utf-8")))

def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)

def clipped(p):
    o = subprocess.run(["ffmpeg", "-v", "info", "-i", p, "-af", "astats=metadata=0:reset=0", "-f", "null", "-"], capture_output=True, text=True).stderr
    f = re.findall(r"Flat factor: *(\S+)", o)
    return bool(f) and float(f[-1]) > 0

meta = [r for r in csv.DictReader(open(HERE / "thai-ser/meta.csv", encoding="utf-8")) if r["turn_type"] == "script" and r["assigned_emo"] in EMO and r["majority_emo"] == r["assigned_emo"]]
pool = HERE / "refs_pool"; pool.mkdir(exist_ok=True)
out = []
for b in base:                                                     # rank 0 = what is in use now
    src = HERE / b["path"]; dst = pool / src.name; shutil.copy(src, dst)
    out.append(dict(b, path=f"refs_pool/{src.name}", rank=0))
for spk in sorted({b["speaker"] for b in base}):
    g, act = spk[0], spk[1:]
    for emo in ("neutral", "happy", "sad", "angry"):
        used = {Path(r["path"]).name for r in out if r["speaker"] == spk and r["emotion"] == emo}
        used_sent = {r["ref_text"] for r in out if r["speaker"] == spk and r["emotion"] == emo}
        c = [r for r in meta if r["actor_id"] == act and r["actor_gender"][0].lower() == g and EMO[r["assigned_emo"]] == emo
             and float(r["agreement"] or 0) >= a.min_agree and Path(r["file"]).name not in used]
        c.sort(key=lambda r: -(float(r["agreement"]) + (0.05 if r["script_intensity"] == "High" else 0) + (0.2 if r["script_sent"] not in used_sent else 0)))
        n = 0
        for r in c:
            p = HERE / "thai-ser" / r["file"]; d = dur(str(p))
            if not 2.5 <= d <= 8.0 or clipped(str(p)): continue
            n += 1; shutil.copy(p, pool / p.name); used_sent.add(r["script_sent"])
            out.append(dict(speaker=spk, gender=b["gender"] if False else ("female" if g == "f" else "male"), emotion=emo, path=f"refs_pool/{p.name}",
                            ref_text=r["script_sent"], agreement=r["agreement"], dur=round(d, 2), intensity=r["script_intensity"], rank=n))
            if n >= a.k: break
with open(HERE / "refs_pool.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["speaker", "gender", "emotion", "path", "ref_text", "agreement", "dur", "intensity", "rank"]); w.writeheader(); w.writerows(out)
rows = "".join(f"<tr><td>{r['speaker']}</td><td>{r['emotion']}</td><td>{r['rank']}</td><td>{r['ref_text']}</td><td>{float(r['agreement']):.2f}</td><td><audio controls preload=none src='{Path(r['path']).name}'></audio></td></tr>" for r in out)
(pool / "index.html").write_text(f"<meta charset=utf-8><style>td,th{{padding:.3em;border-bottom:1px solid #ddd;text-align:left}}</style><h3>Reference pool (rank 0 = in use now)</h3><table><tr><th>speaker</th><th>emotion</th><th>rank</th><th>text</th><th>agree</th><th></th></tr>{rows}</table>", encoding="utf-8")
print(len(out), "rows ->", HERE / "refs_pool.csv")
