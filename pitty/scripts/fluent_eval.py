"""pitty: fluent-signing evaluation set from held-out isolated clips + sentence WER through the production SignPipeline.

    python pitty/scripts/fluent_eval.py build                     # → cache/fluent_eval/{same,diff}.pkl (fixed seed, tune/test halves)
    python pitty/scripts/fluent_eval.py run <out.json> [tagger.onnx] [--split tune]

Two synthesis styles (signers = val/test only, conversation vocabulary, 3–5 words, one signer per sentence):
  same  trim 15–30 % of each sign core + Hermite / 0–4 frame cross-fade  (the style tagger A was trained on — favours A)
  diff  trim 30 % + 3–8 frame cross-fade, faster (1.2–1.5×)             (a different setting of the same idea)
Neither is real fluent signing: TSL51 (no regression) and data_test (once) remain the independent checks.
bank = artifacts/eval/bank_eval.npz (training signers only), decoder settings = models/spotting.json (unchanged)."""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pitty" / "scripts"))
OUT = ROOT / "cache" / "fluent_eval"


def build():
    from fluent_synth import FluentSynth, heldout_pool
    from modules.sequence import resample
    store, q = heldout_pool()
    groups = [g.drop_duplicates("concept") for _, g in q.groupby("signer")]
    groups = [g for g in groups if len(g) >= 4]
    OUT.mkdir(parents=True, exist_ok=True)
    for style, kw, spd in (("same", dict(mode="fluent+blend", trim=None, overlap=(0, 4)), (0.9, 1.3)),
                           ("diff", dict(mode="fluent+blend", trim=0.3, overlap=(3, 8)), (1.2, 1.5))):
        rng = np.random.RandomState(7 if style == "same" else 8)
        fs = FluentSynth(store, q, seed=11 if style == "same" else 12)
        items = []
        for i in range(120):
            g = groups[i % len(groups)]
            picks = g.sample(min(len(g), rng.randint(3, 6)), random_state=rng.randint(1e9))
            trim = kw["trim"] if kw["trim"] is not None else rng.uniform(0.15, 0.3)
            kp, sc, bounds, words = fs.sentence(picks, mode=kw["mode"], trim=trim, overlap=kw["overlap"])
            kp, sc = resample(kp, sc, rng.uniform(*spd))
            items.append(dict(id=f"{style}{i:03d}", split="tune" if i % 2 == 0 else "test", words=words, signer=picks.signer.iloc[0],
                              kp=kp.astype(np.float16), sc=sc.astype(np.float16)))
        with open(OUT / f"{style}.pkl", "wb") as f:
            pickle.dump(items, f)
        print(style, len(items), "sentences · words", sum(len(x["words"]) for x in items), "· signers", len({x["signer"] for x in items}))


def run(out_f, tagger=None, split=None):
    import logging
    logging.disable(logging.WARNING)
    from modules.decode import wer
    from modules.pipeline import SignPipeline
    from modules.schema import canonical_hands
    pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False, bank_file=Path(os.environ.get("BANK_FILE", ROOT / "artifacts/eval/bank_eval.npz")))
    if os.environ.get("SPOT_OVERRIDE"):    # pitty: decoder settings override (JSON, "post" merged) — models/spotting.json untouched
        ov = json.loads(os.environ["SPOT_OVERRIDE"]); post = dict(pipe.spot.get("post", {}), **ov.pop("post", {})); pipe.spot = dict(pipe.spot, **ov, post=post)
    if os.environ.get("SIGNER_NORM"):      # pitty: query-side signer normalisation (pitty/scripts/signer_norm.py)
        import signer_norm; print(signer_norm.install(os.environ["SIGNER_NORM"]))
    if tagger:
        import onnxruntime as ort
        from tagger_ensemble import session; pipe.be.tagger = session(tagger)       # one .onnx, or an ensemble "a.onnx+b.onnx"
    cur = {}

    def standardise(raw):
        kp, sc = cur["kp"], cur["sc"]
        present = sc[:, [0, 5, 6]].mean(1) > 0.3
        kp, sc, m = canonical_hands(kp, sc, (1, 1))
        return kp, sc, (1, 1), present, m
    pipe.standardise = standardise
    res, t0 = [], time.time()
    for style in ("same", "diff"):
        with open(OUT / f"{style}.pkl", "rb") as f:
            items = pickle.load(f)
        for it in items:
            if split and it["split"] != split:
                continue
            cur.update(kp=it["kp"].astype(np.float32), sc=it["sc"].astype(np.float32))
            r = pipe.run_landmarks(None)
            if os.environ.get("PAIR_CLF"):     # pitty: pair classifiers (pitty/scripts/pair_clf.py), "<clf.npz>:<gap>"
                import pair_clf as _pc; _pf, _, _pg = os.environ["PAIR_CLF"].rpartition(":"); _pc.apply(r["segments"], _pc.load(_pf), float(_pg))
            top1 = [s["nearest"] for s in r["segments"] if s["status"] != "transition"]
            res.append(dict(id=it["id"], style=style, split=it["split"], ref=it["words"], top1=top1, WER=wer(it["words"], top1),
                            recall1=float(np.mean([w in top1 for w in it["words"]])), n_pred=len(top1), n_ref=len(it["words"])))
    Path(out_f).write_text(json.dumps(dict(tagger=str(tagger or "models/onnx/tagger.onnx"), results=res), ensure_ascii=False), encoding="utf-8")
    import pandas as pd
    df = pd.DataFrame(res)
    print(out_f, round(time.time() - t0), "s")
    print(df.groupby(["style", "split"])[["WER", "recall1", "n_pred", "n_ref"]].mean().round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        args = [x for x in sys.argv[2:] if not x.startswith("--")]
        sp = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else None
        args = [x for x in args if x != sp]
        run(args[0], args[1] if len(args) > 1 else None, sp)
