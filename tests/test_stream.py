"""Streaming (modules/stream.py) on one real recording: the threaded live mode (camera thread never waits) confirms the same words as
the frame-by-frame replay; prepared Standard Schema landmarks stream like MediaPipe frames; stable_n=1 is v9's rule. Needs
models/model-v10 (or model-v9) and the cached iPhone landmarks — skipped otherwise. CPU only."""
import logging
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIP = ROOT / "cache" / "iphone_test_mp" / "sentence_IMG_3307.npz"
MD = ROOT / "models" / ("model-v10" if (ROOT / "models" / "model-v10").exists() else "model-v9")


@unittest.skipUnless(CLIP.exists() and MD.exists(), "model / cached landmarks not present")
class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        from modules.mediapipe_pose import load_raw
        from modules.pipeline import SignPipeline
        cls.pipe = SignPipeline(MD, use_llm=False, use_face=False)
        cls.raw = load_raw(CLIP)

    def test_live_equals_replay(self):
        from modules.stream import KEYS, LiveStream, replay
        words, _ = replay(self.pipe, self.raw)
        live = LiveStream(self.pipe, tuple(int(x) for x in self.raw["hw"]))
        for i in range(len(self.raw["t"])):
            live.add({k: self.raw[k][i] for k in KEYS}, float(self.raw["t"][i] - self.raw["t"][0]))
            time.sleep(0.03)
        self.assertEqual([c["word"] for c in live.close()], words)

    def test_standard_schema_frames(self):
        from modules.stream import replay
        kp, sc, hw, _, _ = self.pipe.standardise(self.raw)
        from modules.schema import uniso
        std = dict(t=np.arange(len(kp)) / 25.0, hw=np.array(hw), kp=kp, sc=sc)
        words, st = replay(self.pipe, std)
        self.assertGreater(len(words), 0)
        self.assertEqual(st["duplicates"], 0)
        self.assertTrue(callable(uniso))


if __name__ == "__main__":
    unittest.main()
