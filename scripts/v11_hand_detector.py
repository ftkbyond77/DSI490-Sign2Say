"""v11 extractor arm X1 (prompt-opt-v11 §2E / §3D): a pose-independent HAND DETECTOR trained on human-labelled boxes.

  python main.py v11-detector prep      # HaGRID 30k sample (cj-mills/hagrid-sample-30k-384p, CC BY-SA 4.0, human boxes) → YOLO 1-class 'hand'
                                        #   split by SUBJECT (user_id): train / val (early stopping) / test (independent mAP)
  python main.py v11-detector train [--hours 3]   # YOLOv8n, 640 px, early stopping (patience 20), local GPU
  python main.py v11-detector test      # mAP50 / mAP50-95 on the independent subject split → artifacts/reports/hand_detector_v11.json

Licence: Ultralytics is AGPL-3.0 → the detector is a research / server opt-in only (THAISLM_HAND_DETECTOR=1), never the default and never
in the browser. Never trained on MediaPipe-made labels (prompt rule C11.7).
Runs in an isolated environment that has `ultralytics` (the main 490-sign env is not modified): set THAISLM_YOLO_PY to its python.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "cache" / "hagrid_v11"
DS = SRC / "yolo"
OUT = ROOT / "artifacts" / "runs" / "v11-hand-detector"
REP = ROOT / "artifacts" / "reports" / "hand_detector_v11.json"


def cmd_prep(a):
    z = SRC / "hagrid-sample-30k-384p.zip"
    ext = SRC / "raw"
    if not ext.exists():
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():                                   # untrusted archive: no absolute paths / parent references
                if m.startswith("/") or ".." in Path(m).parts:
                    raise SystemExit(f"unsafe path in archive: {m}")
            f.extractall(ext)
    anns = list(ext.rglob("*.json"))
    imgs = {p.stem: p for p in ext.rglob("*.jpg")}
    print(f"{len(anns)} annotation files, {len(imgs)} images")
    recs = []
    for af in anns:
        d = json.loads(af.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "bboxes" in v and k in imgs:
                    recs.append((k, v))
    users = sorted({v.get("user_id", k) for k, v in recs})
    rnd = random.Random(0); rnd.shuffle(users)
    n = len(users)
    role = {u: ("test" if i < 0.15 * n else "val" if i < 0.25 * n else "train") for i, u in enumerate(users)}
    if DS.exists():
        shutil.rmtree(DS)
    cnt = {"train": 0, "val": 0, "test": 0}
    for k, v in recs:
        r = role[v.get("user_id", k)]
        (DS / r / "images").mkdir(parents=True, exist_ok=True); (DS / r / "labels").mkdir(parents=True, exist_ok=True)
        dst = DS / r / "images" / f"{k}.jpg"
        shutil.copyfile(imgs[k], dst)
        lines = []
        for (x, y, w, h), lab in zip(v["bboxes"], v.get("labels", ["hand"] * len(v["bboxes"]))):
            lines.append(f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")      # every labelled hand box (gesture + 'no_gesture')
        (DS / r / "labels" / f"{k}.txt").write_text("\n".join(lines))
        cnt[r] += 1
    (DS / "data.yaml").write_text(f"path: {DS.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: hand\n")
    json.dump(dict(images=cnt, subjects={k: sum(1 for u in users if role[u] == k) for k in cnt}, source="cj-mills/hagrid-sample-30k-384p",
                   licence="CC BY-SA 4.0", split="by user_id (subject-independent)"), open(DS / "split.json", "w"), indent=1)
    print("prepared", cnt)


def cmd_train(a):
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")
    m.train(data=str(DS / "data.yaml"), imgsz=640, epochs=200, patience=20, batch=16, time=a.hours, project=str(OUT.parent), name=OUT.name,
            exist_ok=True, workers=4, device=0, seed=0, plots=False, verbose=False)


def cmd_test(a):
    from ultralytics import YOLO
    m = YOLO(str(OUT / "weights" / "best.pt"))
    r = m.val(data=str(DS / "data.yaml"), split="test", imgsz=640, batch=16, device=0, plots=False, verbose=False)
    res = dict(mAP50=float(r.box.map50), mAP50_95=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr),
               split=json.load(open(DS / "split.json")), weights=str((OUT / "weights" / "best.pt").relative_to(ROOT)))
    REP.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep"); t = sub.add_parser("train"); t.add_argument("--hours", type=float, default=3.0); sub.add_parser("test")
    a = ap.parse_args()
    dict(prep=cmd_prep, train=cmd_train, test=cmd_test)[a.cmd](a)


if __name__ == "__main__":
    main()
