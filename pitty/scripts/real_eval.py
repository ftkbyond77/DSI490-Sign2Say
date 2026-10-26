"""pitty: real videos (her clips 5 + 2 new, data_test 5) — ordered scoring, same alignment as WER (pitty/scripts/wer_ops.py).
  python pitty/scripts/real_eval.py [name=<decoder override json> ...]   ("current" = models/spotting.json is always run)
Per set: WER, words right / wrong word / missing / extra, and top-5 hit = the aligned segment's top-5 holds the right word
(replaces the old order-free R@1 / R@5, which counted a word as found anywhere in the output). Prod bank, prod tagger."""
import sys, csv, json, logging, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.WARNING)
import my_eval
from wer_ops import align
from modules.mediapipe_pose import load_raw
from modules.pipeline import SignPipeline
from modules.decode import wer
from modules.lexicon import DATA_TEST_REFS
canon = my_eval.canon_fn(); E = my_eval.OUT
pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)
r1 = {Path(r["file"]).stem: r["ref"].split() for r in csv.DictReader(open(ROOT / "pitty/my_clips/refs_round1.csv", encoding="utf-8-sig"))}
sets = {"mine untrimmed (5)": [(E / "mp" / f"mine__{k}.npz", v) for k, v in r1.items()],
        "mine new (2)": [(E / "mp" / f"mine2__{k}.npz", v) for k, v in {"ฉันรักเพื่อน": "ฉัน รัก เพื่อน".split(), "วันหยุดพ่อชอบไปตลาด": "วันหยุด พ่อ ชอบ ไป ตลาด".split()}.items()],
        "data_test (5)": [(ROOT / "cache/data_test_mp" / f"{k}.npz", v) for k, v in DATA_TEST_REFS.items()]}
cfgs = {"current": None}
for extra in sys.argv[1:]:
    n, p = extra.split("=", 1); cfgs[n] = json.loads(p)
spot0 = dict(pipe.spot); out = {}
for tn, ov in cfgs.items():
    ov = dict(ov or {}); post = dict(spot0.get("post", {}), **ov.pop("post", {})); pipe.spot = dict(spot0, **ov, post=post)
    for sn, items in sets.items():
        W, n = [], dict(C=0, S=0, D=0, I=0, top5=0, ref=0)
        for f, ref in items:
            ref = [canon(w) for w in ref]
            r = pipe.run_landmarks(load_raw(f), alphabet="en" if sn.startswith("mine") else "off")
            segs = [s for s in r["segments"] if s["status"] != "transition"]; hyp = [s["nearest"] for s in segs]
            W.append(wer(ref, hyp)); n["ref"] += len(ref)
            for op, rw, hw, j, _ in align(ref, hyp):
                n[op] += 1
                if op in "CS" and rw in [c["word"] for c in segs[j]["candidates"][:5]]:
                    n["top5"] += 1
        out[f"{tn} | {sn}"] = dict(WER=float(np.mean(W)), **n)
        print(f"{tn:11s} {sn:20s} WER {np.mean(W):.3f}  right {n['C']}/{n['ref']}  wrong {n['S']}  missing {n['D']}  extra {n['I']}  top-5 hit {n['top5']}/{n['ref']}")
(ROOT / "cache/pitty_tune/real_eval_ordered.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
