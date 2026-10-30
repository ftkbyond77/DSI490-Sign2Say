"""Landmark quality proxies (modules/posequality.py) on synthetic hands: a rigid hand has no bone-length spread and no jitter after
smoothing a straight motion; dropped frames lower `seen` and set the longest gap; a hand at the other wrist counts as a swap."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.posequality import agreement, one_euro, segment_quality  # noqa: E402
from modules.schema import LH, RH, SH_L, SH_R, WR_L, WR_R  # noqa: E402


def body(T=40):
    kp = np.zeros((T, 133, 2), np.float32); sc = np.zeros((T, 133), np.float32)
    kp[:, SH_L] = (0.6, 0.5); kp[:, SH_R] = (0.4, 0.5); sc[:, [SH_L, SH_R]] = 1.0
    kp[:, WR_R] = (0.35, 0.6); kp[:, WR_L] = (0.65, 0.6); sc[:, [WR_L, WR_R]] = 1.0
    rng = np.random.RandomState(0)
    shape = rng.uniform(-0.02, 0.02, (21, 2)).astype(np.float32); shape[0] = 0; shape[9] = (0, -0.03)
    for t in range(T):                                   # right hand moves slowly to the right, rigid shape
        kp[t, RH] = kp[t, WR_R] + shape + (0.002 * t, 0)
    sc[:, RH] = 1.0
    return kp, sc


class PoseQualityTests(unittest.TestCase):
    def test_rigid_hand(self):
        kp, sc = body()
        q = segment_quality(kp, sc)["R"]
        self.assertAlmostEqual(q["seen"], 1.0)
        self.assertLess(q["bone_cv"], 1e-4)
        self.assertLess(q["jitter"], 0.05)
        self.assertEqual(q["swap"], 0.0)

    def test_dropped_frames(self):
        kp, sc = body()
        sc[10:20, RH] = 0.0
        q = segment_quality(kp, sc)["R"]
        self.assertAlmostEqual(q["seen"], 0.75)
        self.assertAlmostEqual(q["gap_s"], 10 / 25.0)

    def test_swap(self):
        kp, sc = body()
        kp[:, RH] += kp[:, WR_L][:, None] - kp[:, WR_R][:, None]     # the "right" hand sits at the left wrist
        self.assertEqual(segment_quality(kp, sc)["R"]["swap"], 1.0)

    def test_agreement_identical(self):
        kp, sc = body()
        a = agreement(kp, sc, kp, sc)["R"]
        self.assertAlmostEqual(a["dist"], 0.0); self.assertAlmostEqual(a["both"], 1.0)

    def test_one_euro_keeps_constant(self):
        x = np.ones((30, 3))
        self.assertTrue(np.allclose(one_euro(x), 1.0))
        self.assertEqual(segment_quality(*body())["L"]["seen"], 0.0)       # no left hand at all
        self.assertTrue(np.isnan(segment_quality(*body())["L"]["jitter"]))
        self.assertEqual(len(LH), 21)


if __name__ == "__main__":
    unittest.main()
