"""EA-for-v12: v10 / v11 on the two NEW sources (never seen signers) — in-vocabulary isolated words + out-of-vocabulary words.
stage 1 (extract): MediaPipe Holistic 25 fps → scratch npz. stage 2 (run <model>): pipeline → per-clip result json."""
import sys, json, time, re, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(r"D:/SLM-labs/SLM-Labs-SignDINO"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.disable(logging.WARNING)
S = Path(sys.argv[1]); stage = sys.argv[2]
OUT = S / "ood_raw"; OUT.mkdir(exist_ok=True)
T = json.load(open(ROOT / "cloud_s3/data_prep/manifest/tiers.json", encoding="utf-8")); W = T["words"]
R = pd.read_csv(ROOT / "cloud_s3/sign-ราชบัณฑิต/manifest.csv"); E = pd.read_csv(ROOT / "cloud_s3/sign-etsl-chula/manifest.csv")
C = pd.read_parquet(ROOT / "cloud_s3/data_prep/shards/clips.parquet"); allc = set(C.concept.dropna().unique())
rng = np.random.RandomState(12)
sel = []
for r in R[(R.kind == "word") & (R.status == "new") & R.word.isin(W)].itertuples():
    sel.append(dict(src="rid", word=r.word, kind="in", video=str(ROOT / "cloud_s3/sign-ราชบัณฑิต" / r.file)))
Ew = E[E.type.isin(["word", "phrase"]) & (E.n_signs == 1)]
inv = Ew[Ew.word_th.isin(W)].drop_duplicates("word_th")
focus = ["กิน", "คุณ", "ยาย", "ไป", "อะไร", "เหงา", "น้อง", "น้า", "ข้าว", "อา", "ย่า", "ญาติ", "ดู", "ส่ง", "ลูก", "ปลา", "น้ำ", "แฟน"]
a = inv[inv.word_th.isin(focus)]; b = inv[~inv.word_th.isin(focus)].sample(frac=1.0, random_state=12).head(125 - len(a))
for r in pd.concat([a, b]).itertuples():
    sel.append(dict(src="etsl", word=r.word_th, kind="in", video=str(ROOT / "cloud_s3/sign-etsl-chula" / r.file)))
oov = Ew[~Ew.word_th.isin(allc) & ~Ew.word_th.isin(W) & (Ew.type == "word")].sample(frac=1.0, random_state=12).head(40)
for r in oov.itertuples():
    sel.append(dict(src="etsl", word=r.word_th, kind="oov", video=str(ROOT / "cloud_s3/sign-etsl-chula" / r.file)))
for i, s in enumerate(sel):
    s["key"] = f"{s['src']}_{s['kind']}_{i:03d}"; s["tier"] = W.get(s["word"], {}).get("tier", "-"); s["signers"] = W.get(s["word"], {}).get("signers", 0)
    s["eff"] = W.get(s["word"], {}).get("eff_signers", 0)
json.dump(sel, open(S / "ood_sel.json", "w"), ensure_ascii=False)
print("selected", len(sel), pd.DataFrame(sel).groupby(["src", "kind"]).size().to_dict(), flush=True)

if stage == "extract":
    from modules.mediapipe_pose import extract, save_raw
    t0 = time.time()
    for i, s in enumerate(sel):
        f = OUT / f"{s['key']}.npz"
        if f.exists() or not Path(s["video"]).exists():
            continue
        try:
            r = extract(s["video"], fps=25.0, normalize=False)
            if r is not None:
                save_raw(f, r)
        except Exception as e:
            print("ERR", s["key"], repr(e)[:100], flush=True)
        if i % 10 == 0:
            print(i, len(sel), round(time.time() - t0), flush=True)
    print("done", len(list(OUT.glob("*.npz"))), flush=True)
else:
    from modules.mediapipe_pose import load_raw
    from modules.pipeline import SignPipeline
    md = sys.argv[3]
    pipe = SignPipeline(model_dir=ROOT / "models" / md, backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)
    res = []
    for i, s in enumerate(sel):
        f = OUT / f"{s['key']}.npz"
        if not f.exists():
            continue
        raw = load_raw(f); hands = float(np.mean((np.abs(raw["lh"]).sum((1, 2)) > 0) | (np.abs(raw["rh"]).sum((1, 2)) > 0)))
        segs = [x for x in pipe.run_landmarks(raw)["segments"] if x["status"] != "transition"]
        res.append(dict(s, hands=hands, n_seg=len(segs), top=[x["nearest"] for x in segs], status=[x["status"] for x in segs],
                        p=[float(x.get("p_correct", 0) or 0) for x in segs],
                        cands=[[c["word"] for c in (x.get("candidates") or [])[:5]] for x in segs]))
        if i % 20 == 0:
            print(i, flush=True)
    json.dump(res, open(S / f"ood_res_{md}.json", "w"), ensure_ascii=False)
    D = pd.DataFrame(res)
    D["hit1"] = [r["word"] in r["top"] for r in res]; D["hit5"] = [any(r["word"] in c for c in r["cands"]) or r["word"] in r["top"] for r in res]
    D["shown_ok"] = [any(w == r["word"] and st == "accepted" for w, st in zip(r["top"], r["status"])) for r in res]
    D["shown_any"] = [any(st == "accepted" for st in r["status"]) for r in res]
    D["shown_wrong"] = [any(w != r["word"] and st == "accepted" for w, st in zip(r["top"], r["status"])) for r in res]
    D["noseg"] = D.n_seg == 0
    print(md); print(D.groupby(["src", "kind"])[["hit1", "hit5", "shown_ok", "shown_any", "shown_wrong", "noseg", "hands", "n_seg"]].mean().round(3).assign(n=D.groupby(["src", "kind"]).size()).to_string())
    I = D[D.kind == "in"]
    print(I.groupby(pd.cut(I.signers, [-1, 1, 2, 4, 9, 99]))[["hit1", "hit5", "shown_ok"]].mean().round(3).assign(n=I.groupby(pd.cut(I.signers, [-1, 1, 2, 4, 9, 99])).size()).to_string())
    print(I.groupby("tier")[["hit1", "hit5", "shown_ok"]].mean().round(3).to_string())
