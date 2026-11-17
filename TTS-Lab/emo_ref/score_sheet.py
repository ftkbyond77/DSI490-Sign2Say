"""Score the shared-session sheet (downloaded as CSV: ข้อ, คน 1 อารมณ์, คน 1 คะแนน, ... คน 4 ...).
  python score_sheet.py --out out sheet.csv
"""
import argparse, csv
from collections import defaultdict
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("sheet"); a = ap.parse_args()
TH = {"ปกติ": "neutral", "ดีใจ": "happy", "เศร้า": "sad", "โกรธ": "angry"}
order = {int(r["no"]): r["id"] for r in csv.DictReader(open(f"{a.out}/session_order.csv", encoding="utf-8"))}
key = {r["id"]: r for r in csv.DictReader(open(f"{a.out}/key.csv", encoding="utf-8"))}
rows = list(csv.reader(open(a.sheet, encoding="utf-8-sig")))
head, body = rows[0], rows[1:]
agg = defaultdict(lambda: [0, 0, 0.0, 0]); n_ans = 0; conf = defaultdict(int)
for r in body:
    if not r or not r[0].strip().isdigit(): continue
    k = key[order[int(r[0])]]
    for p in range(4):
        emo, mos = (r[1 + 2 * p] if len(r) > 1 + 2 * p else "").strip(), (r[2 + 2 * p] if len(r) > 2 + 2 * p else "").strip()
        if emo not in TH: continue
        n_ans += 1; heard = TH[emo]; conf[(k["method"], k["emotion"], heard)] += 1
        for g in [(k["method"], "all", "all"), (k["method"], k["gender"], "all"), (k["method"], "all", k["emotion"]), (k["method"], k["gender"], k["emotion"])]:
            x = agg[g]; x[0] += 1; x[1] += heard == k["emotion"]
            if mos.replace(".", "").isdigit(): x[2] += float(mos); x[3] += 1
print(f"{n_ans} answers")
print(f"{'method':12s} {'gender':7s} {'emotion':8s} {'n':>4s} {'emo_acc':>8s} {'MOS':>5s}")
for (m, g, e), (n, c, s, nm) in sorted(agg.items()):
    print(f"{m:12s} {g:7s} {e:8s} {n:4d} {c/n:8.3f} {(s/nm if nm else float('nan')):5.2f}")
print("\nconfusion (method, played -> heard counts):")
for m in ("A_tag_psola", "B_emo_ref", "neutral"):
    for e in ("neutral", "happy", "sad", "angry"):
        c = {h: conf[(m, e, h)] for h in ("neutral", "happy", "sad", "angry")}
        if sum(c.values()): print(f"  {m:12s} {e:8s} -> {c}")
print("chance = 0.25")
