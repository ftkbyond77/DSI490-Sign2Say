"""Side-by-side check page (NOT blind; for us only): per speaker x sentence -> reference clips, neutral, then A vs B per emotion.
  python compare.py --out out_mps   ->  out_mps/compare.html
"""
import argparse, csv, html, os
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out_mps"); ap.add_argument("--refs", default="refs.csv"); a = ap.parse_args()
out = Path(a.out); key = list(csv.DictReader(open(out / "key.csv", encoding="utf-8")))
refs = {(r["speaker"], r["emotion"]): r["path"] for r in csv.DictReader(open(a.refs, encoding="utf-8"))}
by = {(k["speaker"], k["sent"], k["emotion"], k["method"]): k for k in key}
def au(p, label):
    ok = (out / p).exists() if not p.startswith("..") else os.path.exists(out / p)
    return f"<div><small>{label}</small><br>" + (f"<audio controls preload=none src='{p}'></audio>" if ok else "<i>ยังไม่เสร็จ</i>") + "</div>"
rows = ["<meta charset=utf-8><title>compare</title><style>body{font-family:sans-serif;margin:1em}td{padding:.4em;vertical-align:top;border-bottom:1px solid #ddd}"
        "h3{margin-top:2em}</style><h2>A (แท็ก+PSOLA) vs B (เสียงอ้างอิงมีอารมณ์) — ไม่ blind ใช้เช็กเอง</h2>"]
for spk, sent in sorted({(k["speaker"], k["sent"]) for k in key}):
    text = next(k["text"] for k in key if k["speaker"] == spk and k["sent"] == sent)
    n = by.get((spk, sent, "neutral", "neutral"))
    rows.append(f"<h3>{spk} — “{html.escape(text)}”</h3>" + (au(n["id"] + ".wav", "neutral (ใช้ร่วม A/B)") if n else ""))
    rows.append("<table><tr><th>อารมณ์</th><th>เสียงอ้างอิงจริง (THAI-SER)</th><th>A แท็ก+PSOLA</th><th>B เสียงอ้างอิงมีอารมณ์</th></tr>")
    for e in ("happy", "sad", "angry"):
        A, B = by.get((spk, sent, e, "A_tag_psola")), by.get((spk, sent, e, "B_emo_ref"))
        ref = refs.get((spk, e))
        rows.append(f"<tr><td><b>{e}</b></td><td>{au(os.path.relpath(ref, out), 'ref') if ref else ''}</td>"
                    f"<td>{au(A['id'] + '.wav', 'A') if A else ''}</td><td>{au(B['id'] + '.wav', 'B') if B else ''}</td></tr>")
    rows.append("</table>")
(out / "compare.html").write_text("\n".join(rows), encoding="utf-8"); print(f"-> {out}/compare.html")
