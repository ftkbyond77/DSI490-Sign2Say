from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V81_ROOT = ROOT / "_inputs" / "v81"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V81_ROOT))

import modules
from modules.pipeline import SignPipeline


class BrowserFrameConversionTests(unittest.TestCase):
    def test_pipeline_adapter_matches_browser_raw_schema(self):
        pose = [[0.5, 0.25, 0.1, 0.0] for _ in range(33)]
        left_hand = [[0.2, 0.3, 0.4] for _ in range(21)]
        right_hand = [[0.6, 0.7, 0.8] for _ in range(21)]
        face = [[0.4, 0.5] for _ in range(68)]
        frames = [
            {
                "t": 1000,
                "pose": pose,
                "lh": left_hand,
                "rh": right_hand,
                "face": face,
                "bs": [0.25] * 52,
            },
            {"t": 1040, "pose": [], "lh": [], "rh": [], "face": [], "bs": []},
        ]
        raw = SignPipeline.landmarks_from_browser(frames, (480, 640))

        self.assertEqual(Path(modules.__file__).resolve().parent, (V81_ROOT / "modules").resolve())
        self.assertEqual(raw["hw"].tolist(), [480, 640])
        self.assertAlmostEqual(float(raw["t"][0]), 0.0, places=8)
        self.assertAlmostEqual(float(raw["t"][1]), 0.04, places=8)
        self.assertEqual(raw["pose"].shape, (2, 33, 4))
        self.assertEqual(raw["lh"].shape, (2, 21, 3))
        self.assertEqual(raw["rh"].shape, (2, 21, 3))
        self.assertEqual(raw["face"].shape, (2, 68, 2))
        self.assertEqual(raw["bs"].shape, (2, 52))
        np.testing.assert_allclose(raw["pose"][0, 0], [0.5, 0.25, 0.1, 1.0])
        np.testing.assert_allclose(raw["lh"][0, 0], [0.2, 0.3, 0.4])
        np.testing.assert_allclose(raw["rh"][0, 0], [0.6, 0.7, 0.8])
        np.testing.assert_allclose(raw["face"][0, 0], [0.4, 0.5])
        self.assertAlmostEqual(float(raw["bs"][0, 0]), 0.25)
        self.assertEqual(float(raw["lh"][1].sum()), 0.0)
        self.assertAlmostEqual(float(raw["src_fps"]), 50.0, places=4)


if __name__ == "__main__":
    unittest.main()
