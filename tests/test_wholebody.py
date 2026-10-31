"""Tier-A extractor (modules/wholebody.py): the 133-point frame maps onto MediaPipe's raw layout, a low-score hand is written as
missing (MediaPipe's convention), the tier-A raw is harmonised like the RTMW corpus, and the vendored top-down crop equals rtmlib's
(when rtmlib is installed). No model files needed except the optional corpus parity check (skipped when absent)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import wholebody  # noqa: E402
from modules.mediapipe_pose import to_standard  # noqa: E402


def frame(hand_score=0.9):
    rng = np.random.RandomState(0)
    kp = rng.uniform(0.3, 0.7, (133, 2)).astype(np.float32)
    sc = np.full(133, 0.9, np.float32); sc[91:112] = hand_score
    return kp, sc


class WholeBodyTests(unittest.TestCase):
    def test_mp_layout(self):
        kp, sc = frame()
        pose, lh, rh, face, bs = wholebody.to_mp_raw_frame(kp, sc)
        self.assertTrue(np.allclose(pose[11, :2], kp[5])); self.assertTrue(np.allclose(pose[16, :2], kp[10]))
        self.assertTrue(np.allclose(lh[:, :2], kp[91:112])); self.assertTrue(np.allclose(rh[:, :2], kp[112:133]))
        self.assertEqual(face.shape, (68, 2)); self.assertEqual(bs.sum(), 0)

    def test_low_score_hand_is_missing(self):
        kp, sc = frame(hand_score=0.1)
        _, lh, rh, _, _ = wholebody.to_mp_raw_frame(kp, sc)
        self.assertEqual(np.abs(lh).sum(), 0); self.assertGreater(np.abs(rh).sum(), 0)

    def test_tier_a_raw_to_standard(self):
        frames = []
        for _ in range(30):
            kp, sc = frame()
            pose, lh, rh, face, bs = wholebody.to_mp_raw_frame(kp, sc)
            frames.append(dict(pose=pose, lh=lh, rh=rh, face=face, bs=bs, kp133=kp, sc133=sc))
        raw = wholebody.stack_frames(frames, np.arange(30) / 25.0, (480, 640))
        d = to_standard(raw)
        self.assertEqual(d["extractor"], "rtmw")
        self.assertEqual(d["kp"].shape[1:], (133, 2))
        self.assertGreater(d["sc"][:, 112:133].mean(), 0.5)

    def test_crop_matches_rtmlib(self):
        try:
            from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
        except ImportError:
            self.skipTest("rtmlib not installed")
        rng = np.random.RandomState(1)
        img = rng.randint(0, 255, (480, 640, 3)).astype(np.uint8)
        box = np.array([120, 40, 420, 470], np.float32)
        wb = wholebody.WholeBody.__new__(wholebody.WholeBody); wb.in_wh = (192, 256)
        x, center, scale = wb._crop(img, box)
        c2, s2 = bbox_xyxy2cs(box, padding=1.25)
        ref, s2 = top_down_affine((192, 256), s2, c2, img)
        ref = ((ref.astype(np.float32) - wholebody.MEAN) / wholebody.STD).transpose(2, 0, 1)[None]
        self.assertLess(np.abs(x - ref).mean(), 1e-3)
        self.assertTrue(np.allclose(scale, s2, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
