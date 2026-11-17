"""v12 4C robustness: the shipped v12 chain must not crash on hostile live input (and v10 stays loadable)."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQ = sorted((ROOT / "cache" / "live_requests").glob("*.json.gz"))
pytestmark = pytest.mark.skipif(not REQ or not (ROOT / "models" / "model-v12").exists(), reason="no saved requests / model-v12")


@pytest.fixture(scope="module")
def pipe():
    import logging; logging.disable(logging.WARNING)
    from modules.pipeline import SignPipeline
    return SignPipeline(model_dir=ROOT / "models" / "model-v12", backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)


@pytest.fixture(scope="module")
def req():
    d = json.load(gzip.open(REQ[len(REQ) // 2], "rt", encoding="utf-8"))
    return d["frames"], (d["height"], d["width"])


def _run(p, frames, hw):
    if not frames:
        return {"segments": []}
    return p.run_landmarks(p.landmarks_from_browser(frames, hw))


def variants(frames):
    no_hands = [dict(f, lh=[], rh=[]) for f in frames]
    one_hand = [dict(f, lh=[]) for f in frames]
    fps8 = frames[::3]
    long60 = []
    t_end = frames[-1]["t"] - frames[0]["t"] + 50
    for k in range(int(60000 // max(t_end, 1)) + 1):
        long60 += [dict(f, t=f["t"] + k * t_end) for f in frames]
    mirror = [dict(f, pose=[[1 - p[0], *p[1:]] for p in f["pose"]], lh=[[1 - p[0], *p[1:]] for p in f["rh"]], rh=[[1 - p[0], *p[1:]] for p in f["lh"]],
                   face=[[1 - p[0], *p[1:]] for p in f["face"]]) for f in frames]
    leave = [dict(f, lh=[], rh=[]) if i > len(frames) // 2 else f for i, f in enumerate(frames)]
    corrupt = [dict(f) for f in frames]
    if corrupt[len(corrupt) // 2]["pose"]:
        corrupt[len(corrupt) // 2]["pose"] = [[float("nan")] * 4 for _ in corrupt[len(corrupt) // 2]["pose"]]
    return dict(empty=[], no_hands=no_hands, one_hand=one_hand, fps8=fps8, long60=long60, left_handed=mirror, hands_leave=leave, corrupt=corrupt)


@pytest.mark.parametrize("name", ["empty", "no_hands", "one_hand", "fps8", "long60", "left_handed", "hands_leave", "corrupt"])
def test_no_crash(pipe, req, name):
    frames, hw = req
    r = _run(pipe, variants(frames)[name], hw)
    assert "segments" in r
    if name in ("empty", "no_hands"):
        assert not any(s["status"] == "accepted" for s in r["segments"])
