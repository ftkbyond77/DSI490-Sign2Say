"""Join listeners' answers with the key: emotion accuracy and MOS per method x gender (x emotion).
  python score.py --out out answers_*.csv
"""
import argparse, csv, glob
from collections import defaultdict
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("answers", nargs="+"); a = ap.parse_args()
key = {r["id"]: r for r in csv.DictReader(open(f"{a.out}/key.csv", encoding="utf-8"))}
ans = [r for f in a.answers for g in glob.glob(f) for r in csv.DictReader(open(g, encoding="utf-8"))]
agg = defaultdict(lambda: [0, 0, 0.0])
for r in ans:
    k = key[r["id"]]
    for grp in [(k["method"], "all", "all"), (k["method"], k["gender"], "all"), (k["method"], k["gender"], k["emotion"])]:
        x = agg[grp]; x[0] += 1; x[1] += r["heard"] == k["emotion"]; x[2] += float(r["mos"])
print(f"{len(ans)} answers from {len({r['listener'] for r in ans})} listeners")
print(f"{'method':12s} {'gender':7s} {'emotion':8s} {'n':>4s} {'emo_acc':>8s} {'MOS':>5s}")
for (m, g, e), (n, c, s) in sorted(agg.items()):
    print(f"{m:12s} {g:7s} {e:8s} {n:4d} {c/n:8.3f} {s/n:5.2f}")
print("chance for emotion = 0.25 (4 choices). neutral clips are shared by A and B.")
