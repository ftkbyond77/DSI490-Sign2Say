"""v10 (models/model-v10, no LLM) on the slide's 6 clips (3 TSL51 + 3 Jack) -> results/v10_slide.json  (segments with times for the comparison page)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from modules.pipeline import SignPipeline
from modules.mediapipe_pose import load_raw
from evalsets import team
sel = json.loads((HERE / "results" / "slide_selection.json").read_text(encoding="utf-8"))
pipe = SignPipeline(model_dir=ROOT / "models" / "model-v10", backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)
def segs(r):
    return [dict(t0=s["t0"], t1=s["t1"], word=s["nearest"], status=s["status"], top5=[c["word"] for c in s["candidates"][:5]]) for s in r["segments"] if s["status"] != "transition"]
out = {}
jack = team("Jack")
for k in sel["jack"]:
    out[k] = segs(pipe.run_landmarks(load_raw(jack[k][0]), alphabet="en"))
for cid in sel["tsl51"]:
    z = np.load(ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz")
    raw = dict(kp=z["kp"].astype(np.float32), sc=z["sc"].astype(np.float32), hw=np.array([int(v) for v in z["hw"]]))      # prepared Standard Schema landmarks
    out[cid] = segs(pipe.run_landmarks(raw, alphabet="off"))
(HERE / "results" / "v10_slide.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in out.items(): print(k[-40:], [s["word"] for s in v])
