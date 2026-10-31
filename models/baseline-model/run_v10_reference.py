"""Run the shipped model (models/model-v10, no LLM) on the same videos with the same scorer → results/v10_reference.json.
Only needed when the reference changes (a newer version): python models/baseline-model/run_v10_reference.py [model_dir]"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from baseline import canon_fn, score_video
from evalsets import VIDEOS, team
from modules.mediapipe_pose import load_raw
from modules.pipeline import SignPipeline

model = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "model-v10"
out = {}
for vocab in ("conversation", "full"):
    pipe = SignPipeline(model_dir=model, backend="onnx", device="cpu", vocab=vocab, use_llm=False)
    canon, bank_vocab = canon_fn(), {pipe.concepts[c] for c in pipe.bank.concepts_present}
    res = {}
    for name, (npz, ref, *_) in {**VIDEOS, **{"Jack/" + k.split("/")[-1]: v for k, v in team("Jack").items()}}.items():
        r = pipe.run_landmarks(load_raw(npz), title=name)
        segs = [dict(top=[(c["word"], c.get("score", 0.0)) for c in s["candidates"]] or [(s["nearest"], 0.0)]) for s in r["segments"] if s["status"] != "transition"]
        for s, t in zip(segs, [s for s in r["segments"] if s["status"] != "transition"]):
            s["top"][0] = (t["nearest"], s["top"][0][1])
        res[name] = score_video(segs, ref, canon, bank_vocab)
    out[vocab] = {k: v for k, v in res.items() if not k.startswith("Jack/")}
    out.setdefault("jack_" + vocab, {k: v for k, v in res.items() if k.startswith("Jack/")})
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "v10_reference.json").write_text(json.dumps(dict(model=model.name, results=out), ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote results/v10_reference.json")
