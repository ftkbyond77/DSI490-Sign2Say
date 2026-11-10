"""Face branch (modules/face_fer.py): unit tests + the non-interference guarantees that need no sign model.
Run: python tests/test_face_branch.py -v   (the sign-pipeline checks are in Face-Lab/non_interference.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.face_fer import (CLASSES, ExpressionSmoother, FaceBranch, crop48, crop_box, face_quality, hand_overlap, speech_affect)

MODEL = sorted(p for p in (Path(__file__).resolve().parents[1] / "models").glob("face-model-*") if (p / "config.json").exists())


def face_pts(cx=0.5, cy=0.45, w=0.2, h=0.28):
    a = np.zeros((68, 2), np.float32)
    a[:, 0] = np.linspace(cx - w / 2, cx + w / 2, 68); a[:, 1] = np.linspace(cy - h / 2, cy + h / 2, 68)
    return a


def _r(t, c, p=0.9, valid=True):
    pr = {k: (1 - p) / 6 for k in CLASSES}; pr[c] = p
    return dict(t=t, valid=valid, probs=pr, expression=c if valid else "unknown")


class FaceBranchTests(unittest.TestCase):
    def test_crop_geometry_and_shape(self):
        box = crop_box(face_pts(), 640, 480)
        self.assertAlmostEqual(float(box[2]), 0.28 * 480 * 1.22, delta=0.5)
        g = crop48(np.random.randint(0, 255, (480, 640), np.uint8), box)
        assert g.shape == (48, 48) and g.dtype == np.uint8
        assert crop48(np.zeros((480, 640, 3), np.uint8), (-50, -50, 120)).shape == (48, 48)     # box partly outside the picture


    def test_quality_gate(self):
        good = (np.random.RandomState(0).rand(48, 48) * 255).astype(np.uint8)
        assert face_quality(good, 100, 0.0) > face_quality(np.full((48, 48), 5, np.uint8), 100, 0.0)       # dark flat crop is worse
        assert face_quality(good, 100, 0.0) > face_quality(good, 100, 0.8)                                 # hand over the face
        assert face_quality(good, 100, 0.0) > face_quality(good, 20, 0.0)                                  # tiny face


    def test_hand_overlap(self):
        f = face_pts()
        assert hand_overlap(f, [np.tile([0.5, 0.45], (21, 1))]) == 1.0
        assert hand_overlap(f, [np.tile([0.1, 0.9], (21, 1))]) == 0.0
        assert hand_overlap(f, [np.zeros((21, 3))]) == 0.0


    def test_smoother_holds_label_against_a_blip_and_follows_a_real_change(self):
        s = ExpressionSmoother()
        for i in range(10):
            s.update(_r(i * 0.2, "neutral", 0.7))
        assert s.cur == "neutral"
        s.update(_r(2.0, "angry", 0.55))                                 # one noisy frame
        assert s.cur == "neutral"
        for i in range(12):
            s.update(_r(2.2 + i * 0.2, "happy", 0.85))
        assert s.cur == "happy"


    def test_smoother_unknown_after_loss_and_state_roundtrip(self):
        s = ExpressionSmoother()
        for i in range(5):
            s.update(_r(i * 0.2, "happy"))
        for i in range(12):
            s.update(_r(1.0 + i * 0.2, "happy", valid=False))
        assert s.cur == "unknown"
        s2 = ExpressionSmoother.from_state(ExpressionSmoother().state())
        assert s2.cur == "unknown"
        a = ExpressionSmoother()
        for i in range(6):
            a.update(_r(i * 0.2, "sad"))
        b = ExpressionSmoother.from_state(a.state())
        assert b.update(_r(1.2, "sad"))["expression"] == a.update(_r(1.2, "sad"))["expression"]


    def test_speech_affect_rules(self):
        ok = dict(expression="happy", valid_frac=0.9, probs={**{c: 0.02 for c in CLASSES}, "happy": 0.88})
        assert speech_affect(ok)[0] == "happy"
        assert speech_affect(dict(ok, expression="unknown"))[0] is None
        neu = dict(expression="neutral", valid_frac=0.9, probs={**{c: 0.02 for c in CLASSES}, "neutral": 0.88})
        assert speech_affect(neu)[0] is None
        sur = dict(expression="surprise", valid_frac=0.9, probs={**{c: 0.02 for c in CLASSES}, "surprise": 0.7, "neutral": 0.2})
        assert speech_affect(sur)[0] == "surprise"
        assert speech_affect(sur, grammar={"question_yesno": True})[0] is None                    # question brows are grammar, not surprise


    @unittest.skipUnless(MODEL, "models/face-model-*/ not present")
    def test_branch_never_raises_and_answers_unknown(self):
        fb = FaceBranch(MODEL[-1])
        assert fb.process([])["summary"]["expression"] == "unknown"
        bad = fb.process([dict(t=0.0, g=b"xx")])                           # wrong size -> caught inside, not raised
        assert bad["ok"] is False and bad["sign_modified"] is False
        live = fb.live([dict(t=0.0, g=np.zeros(2304, np.uint8))])          # black picture: quality gate -> unknown
        assert live["smoothed"]["expression"] == "unknown"


    @unittest.skipUnless(MODEL, "models/face-model-*/ not present")
    def test_branch_is_deterministic_and_does_not_mutate_inputs(self):
        fb = FaceBranch(MODEL[-1])
        g = (np.random.RandomState(1).rand(2304) * 255).astype(np.uint8)
        crops = [dict(t=i * 0.2, g=g.copy(), side_px=90) for i in range(8)]
        a, b = fb.process(crops, [dict(t0=0.0, t1=1.0)]), fb.process(crops, [dict(t0=0.0, t1=1.0)])
        assert a["summary"] == b["summary"] and len(a["per_segment"]) == 1
        assert all(np.array_equal(c["g"], g) for c in crops)


if __name__ == "__main__":
    unittest.main()
