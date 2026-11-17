"""Score round 3 (7 choices). python score3.py session3.csv   (columns: no,p1[,p2,p3],note)"""
import csv, sys
from collections import defaultdict
TH = {"ปกติ": "neutral", "ดีใจ": "happy", "เศร้า": "sad", "โกรธ": "angry", "กลัว": "fear", "รังเกียจ": "disgust", "ประหลาดใจ": "surprise"}
order = {int(r["no"]): r["id"] for r in csv.DictReader(open("out/session3_order.csv", encoding="utf-8"))}
key = {r["id"]: r for f in ("out/key.csv", "out/key3.csv") for r in csv.DictReader(open(f, encoding="utf-8"))}
agg = defaultdict(lambda: [0, 0]); conf = defaultdict(lambda: defaultdict(int))
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    k = key[order[int(r["no"])]]; cell = (k["emotion"], k["method"])
    for p in ("p1", "p2", "p3"):
        if TH.get((r.get(p) or "").strip()) is None: continue
        h = TH[r[p].strip()]; conf[cell][h] += 1
        for g in [cell + ("all",), cell + (k["gender"],)]:
            agg[g][0] += 1; agg[g][1] += h == k["emotion"]
        if r.get("note"): print("note", r["no"], cell, k["speaker"], "heard", h, "|", r["note"])
for g, (n, c) in sorted(agg.items()): print(f"{g[0]:9s} {g[1]:12s} {g[2]:7s} {c}/{n} = {c/n:.2f}")
for cell, d in sorted(conf.items()): print(" ", cell, dict(d))
print("chance = 1/7 = 0.14")
