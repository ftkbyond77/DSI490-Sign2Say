"""pitty: the team metric table (WER · Segment F1 · Recall@1/@5 · accepted correct · count error · latency), baseline vs tuned.

baseline = v8 before tuning (accept_min 0.8, post.p_sign_min 0.65, no min_hand_frac) · tuned = models/spotting.json now.
Reads results already on disk (no model run):
  TSL51 test   cache/pitty_tune/pairs_test_x{1,2}.0.json (baseline) · cache/pitty_tune/final_test_x{1,2}.json (tuned)
  real videos  cache/pitty_eval/<set>_base/metrics.json (baseline) · cache/pitty_eval/<set>/metrics.json (tuned)
Recall@1 / @5 are ordered (the WER alignment: the right word at that position / in that position's top 5), not the old
order-free count, which credited a word found anywhere in the output. Segment F1: a detected sign matches a hand-marked one
when their overlap (IoU) ≥ 0.5, one-to-one — TSL51 only (real videos have no marked boundaries)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from wer_ops import align

ROOT = Path(__file__).resolve().parents[2]
T, E = ROOT / "cache/pitty_tune", ROOT / "cache/pitty_eval"
A = json.loads((ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json").read_text(encoding="utf-8"))


def _iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def tsl51(f, speed, thr):
    R = json.loads((T / f).read_text(encoding="utf-8"))["results"]
    n = sum(len(r["ref"]) for r in R); C = H = tp = npred = ngt = 0; acc = []
    for r in R:
        for o in align(r["ref"], r["top1"]):
            C += o[0] == "C"; H += o[0] in "CS" and o[1] in r["top5"][o[3]]
        gt = [(g["f0"] / speed, g["f1"] / speed) for g in A[r["clip"]]["segments"]]; used = set()
        for s in r["seg_f"]:
            j = max(range(len(gt)), key=lambda k: _iou(s, gt[k]) if k not in used else -1)
            if j not in used and _iou(s, gt[j]) >= 0.5:
                used.add(j); tp += 1
        npred += len(r["seg_f"]); ngt += len(gt)
        acc += [w in r["ref"] for w, p in zip(r["top1"], r["p"]) if p >= thr]
    P, Rc = tp / max(npred, 1), tp / max(ngt, 1)
    return dict(WER=np.mean([r["WER"] for r in R]), F1=2 * P * Rc / max(P + Rc, 1e-9), R1=C / n, R5=H / n,
                acc=f"{sum(acc)}/{len(acc)}", count=np.mean([abs(r["n_pred"] - r["n_ref"]) for r in R]))


def real(s):
    m = json.loads((E / s / "metrics.json").read_text(encoding="utf-8"))["summary"]; n = m["ref_words"]
    return dict(WER=m["WER"], R1=int(m["right"].split("/")[0]) / n, R5=int(m["top5_hit"].split("/")[0]) / n, acc=m["accepted"],
                count=m["count_err"], lat=m["latency_s"])


def table():
    """→ one row per (metric, test set): before / after tuning and the change."""
    t = {k: (tsl51(f"pairs_test_x{k}.0.json", float(k), 0.8), tsl51(f"final_test_x{k}.json", float(k), 0.7)) for k in ("1", "2")}
    v = {s: (real(f"{s}_base"), real(s)) for s in ("data_test", "mine", "mine2")}
    lab = {"data_test": "data_test (5 คลิป)", "mine": "คลิปที่อัดเอง (5 คลิป)", "mine2": "คลิปที่อัดเอง ตัดหัวท้าย (7 คลิป)"}
    rows = []
    def add(metric, name, b, a, fmt, better):
        if isinstance(b, str):
            d = "เท่าเดิม" if b == a else "เพิ่มขึ้น"
        else:
            d = "เท่าเดิม" if abs(a - b) < 0.01 or fmt(a) == fmt(b) else ("ดีขึ้น" if (a < b) == (better == "low") else "แย่ลง")
        rows.append({"ตัววัด": metric, "ชุดทดสอบ": name, "ก่อนจูน": fmt(b), "หลังจูน": fmt(a), "ผล": d})
    f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"; same = lambda x: x
    for k, nm in (("1", "TSL51 ปกติ (1×)"), ("2", "TSL51 เร็ว (2×)")):
        add("WER (ต่ำ = ดี)", nm, t[k][0]["WER"], t[k][1]["WER"], f3, "low")
    for s in v:
        add("WER (ต่ำ = ดี)", lab[s], v[s][0]["WER"], v[s][1]["WER"], f3, "low")
    for k, nm in (("1", "TSL51 ปกติ (1×)"), ("2", "TSL51 เร็ว (2×)")):
        add("Segment F1 (สูง = ดี)", nm, t[k][0]["F1"], t[k][1]["F1"], f3, "high")
    for k, nm in (("1", "TSL51 ปกติ (1×)"), ("2", "TSL51 เร็ว (2×)")):
        add("Recall@1 (สูง = ดี)", nm, t[k][0]["R1"], t[k][1]["R1"], f2, "high")
    for s in v:
        add("Recall@1 (สูง = ดี)", lab[s], v[s][0]["R1"], v[s][1]["R1"], f2, "high")
    for k, nm in (("1", "TSL51 ปกติ (1×)"), ("2", "TSL51 เร็ว (2×)")):
        add("คำที่ระบบยืนยัน (ถูก/ทั้งหมด)", nm, t[k][0]["acc"], t[k][1]["acc"], same, "high")
    for s in v:
        add("คำที่ระบบยืนยัน (ถูก/ทั้งหมด)", lab[s], v[s][0]["acc"], v[s][1]["acc"], same, "high")
    for k, nm in (("1", "TSL51 ปกติ (1×)"), ("2", "TSL51 เร็ว (2×)")):
        add("Count error (ต่ำ = ดี)", nm, t[k][0]["count"], t[k][1]["count"], f2, "low")
    for s in v:
        add("Count error (ต่ำ = ดี)", lab[s], v[s][0]["count"], v[s][1]["count"], f2, "low")
    return pd.DataFrame(rows)


def show(real_only=False):
    """Styled for a notebook: one block per metric, changed cells highlighted (real_only: real videos only)."""
    df = table()
    if real_only:
        df = df[~df["ชุดทดสอบ"].str.startswith("TSL51") & ~df["ตัววัด"].str.startswith("Segment")].reset_index(drop=True)
    col = {"ดีขึ้น": "background-color:#dcfce7;font-weight:bold", "เพิ่มขึ้น": "background-color:#dcfce7;font-weight:bold",
           "แย่ลง": "background-color:#fee2e2;font-weight:bold"}
    first = ~df["ตัววัด"].duplicated()
    view = df.assign(ตัววัด=df["ตัววัด"].where(first, ""))
    return (view.style.hide(axis="index").map(lambda x: col.get(x, ""), subset=["ผล"])
            .apply(lambda r: ["border-top:2px solid #94a3b8" if first[r.name] else "" for _ in r], axis=1)
            .set_properties(**{"text-align": "left"}))
