"""Listening page for REFERENCE candidates (original THAI-SER clips, no TTS): angry + happy for f004/f005/m003/m009, same actor.
Listen, note which are clean (no crackle), then tell Claude the ids. Output: ref_cands/index.html (+ copies of the clips).
  python ref_candidates.py
"""
import csv, html, shutil, subprocess
from pathlib import Path
ACT = {"f004": "004", "f005": "005", "m003": "003", "m009": "009"}; EMO = {"Happy": "happy", "Angry": "angry"}
cur = {(r["speaker"], r["emotion"]): r["path"] for r in csv.DictReader(open("refs.csv", encoding="utf-8"))}
def dur(p):
    try: return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
    except Exception: return 0.0
rows = [r for r in csv.DictReader(open("thai-ser/meta.csv", encoding="utf-8")) if r["turn_type"] == "script" and r["assigned_emo"] in EMO and r["majority_emo"] == r["assigned_emo"]]
D = Path("ref_cands"); D.mkdir(exist_ok=True); out = ["<meta charset=utf-8><style>body{font-family:sans-serif;margin:1em}td{padding:.3em;border-bottom:1px solid #ddd}</style><h2>เลือก ref ที่ไม่แตก (เสียงต้นฉบับ THAI-SER)</h2><table><tr><th>คน/อารมณ์</th><th>id</th><th>ประโยค</th><th>ความแรง</th><th>ฟัง</th></tr>"]
for sp, act in ACT.items():
    for emo in ("angry", "happy"):
        c = [r for r in rows if r["actor_id"] == act and EMO[r["assigned_emo"]] == emo and 2.5 <= dur("thai-ser/" + r["file"]) <= 8.0]
        c.sort(key=lambda r: (-float(r["agreement"] or 0), r["script_intensity"] != "Low"))
        for r in c[:6]:
            cid = Path(r["file"]).stem.split("_")[0]; dst = D / f"{sp}_{emo}_{cid}.flac"; shutil.copy("thai-ser/" + r["file"], dst)
            mark = " <b>(ref ปัจจุบัน)</b>" if cur.get((sp, emo)) == "thai-ser/" + r["file"] else ""
            out.append(f"<tr><td>{sp} {emo}</td><td>{cid}{mark}</td><td>{html.escape(r['script_sent'])}</td><td>{r['script_intensity']}</td><td><audio controls preload=none src='{dst.name}'></audio></td></tr>")
(D / "index.html").write_text("\n".join(out) + "</table>", encoding="utf-8"); print("->", D / "index.html", len(out) - 1, "clips")
