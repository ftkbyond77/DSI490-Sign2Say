"""Face branch (modules/face_fer.py): unit tests + the non-interference guarantees that need no sign model.
Run: python tests/test_face_branch.py -v   (the sign-pipeline checks are in Face-Lab/non_interference.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.face_fer import (CLASSES, ExpressionSmoother, FaceBranch, crop48, crop_box, face_quality, hand_overlap, speech_affect)

MODEL = sorted(p for p in (Path(__file__).resolve().parents[1] / "models").glob("face-model-*") if (p / "config.json").exists())
ENSEMBLE = Path(__file__).resolve().parents[1] / "Face-Lab" / "candidates" / "face-model-ensemble-v2ferplus-b2"
RAVHEAD = Path(__file__).resolve().parents[1] / "models" / "face-model-ravhead-v1"


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
        ang = dict(expression="angry", valid_frac=0.9, probs={**{c: 0.05 for c in CLASSES}, "angry": 0.7})
        assert speech_affect(ang)[0] == "angry"
        assert speech_affect(ang, grammar={"question_wh": True})[0] is None                       # wh-question brows are grammar, not anger


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


    def test_crop96_geometry(self):
        box = crop_box(face_pts(), 640, 480)
        yy, xx = np.mgrid[0:480, 0:640]
        img = (127 + 100 * np.sin(xx / 23.0) * np.cos(yy / 31.0)).astype(np.uint8)       # smooth picture (noise does not survive any shrink)
        g96 = crop48(img, box, 96)
        assert g96.shape == (96, 96) and g96.dtype == np.uint8
        small = crop48(g96, (0, 0, 96), 48)                                # the 96 crop shrunk = (about) the 48 crop of the same box
        assert np.abs(small.astype(int) - crop48(img, box).astype(int)).mean() < 12


    @unittest.skipUnless(ENSEMBLE.exists(), "Face-Lab/candidates ensemble bundle not present")
    def test_ensemble_with_and_without_96px_crops(self):
        fb = FaceBranch(ENSEMBLE)
        assert fb.model.wants_96 and len(fb.model.members) == 2
        rng = np.random.RandomState(2)
        g, g96 = (rng.rand(2304) * 255).astype(np.uint8), (rng.rand(9216) * 255).astype(np.uint8)
        old = fb.process([dict(t=i * 0.2, g=g, side_px=90) for i in range(6)])                      # old client: 48 px only
        new = fb.process([dict(t=i * 0.2, g=g, g96=g96.tobytes(), side_px=90) for i in range(6)])     # new client: both
        bad = fb.process([dict(t=i * 0.2, g=g, g96=b"short", side_px=90) for i in range(6)])          # broken 96 px crop = ignored
        assert old["ok"] and new["ok"] and bad["ok"]
        assert bad["summary"] == old["summary"]
        p48 = fb.model.probs(np.stack([g.reshape(48, 48)] * 2), None)
        p_both = fb.model.probs(np.stack([g.reshape(48, 48)] * 2), [g96.reshape(96, 96), None])
        assert np.allclose(p_both[1], p48[1]) and not np.allclose(p_both[0], p48[0])                 # per crop: members that have their input
        assert np.allclose(p_both.sum(1), 1.0)


    @unittest.skipUnless(RAVHEAD.exists(), "Face-Lab/candidates RAVDESS-head bundle not present")
    def test_picture_head_window_and_live_state(self):
        fb = FaceBranch(RAVHEAD)
        h = fb.model.head
        assert h is not None and fb.model.wants_96
        rng = np.random.RandomState(3)
        V = np.c_[rng.randn(5, 7), rng.dirichlet(np.ones(7), 5), rng.dirichlet(np.ones(7), 5)]
        assert np.isclose(h.window_probs(V).sum(), 1.0)
        crops = [dict(t=i * 0.2, g=(rng.rand(2304) * 255).astype(np.uint8), g96=(rng.rand(9216) * 255).astype(np.uint8).tobytes(), side_px=90) for i in range(12)]
        a = fb.live(crops[:6]); b = fb.live(crops[6:], a["state"])                    # two calls carry the window ...
        one = fb.live(crops)                                                            # ... = one call with every crop
        assert a["ok"] and b["ok"] and one["ok"]
        assert len(b["state"]["head"]) <= 11 and all(r[0] > 2.2 - 2.0 for r in b["state"]["head"])   # only the last window_s is kept
        assert [f["top"] for f in b["frames"]] == [f["top"] for f in one["frames"][6:]]
        old = fb.live([dict(c, g96=None) for c in crops[:3]])                          # no 96 px crop: the ensemble answers, nothing carried
        assert old["ok"] and old["state"]["head"] == []
        assert fb.process(crops)["ok"]


    def test_brow_features_and_cue(self):
        from modules.face_fer import BrowCue, brow_features
        def pts(drop=0.0, roll=0.0):
            a = face_pts() * [640, 480]
            a[36:42] = [300, 200]; a[42:48] = [360, 200]; a[17:27, 1] = 180 + drop          # eyes 60 px apart, brows 20 px above
            c, s = np.cos(roll), np.sin(roll)
            return (a - 320) @ np.array([[c, -s], [s, c]]).T + 320
        f0 = brow_features(pts())
        assert abs(f0[1] - 20 / 60) < 1e-6
        assert np.allclose(brow_features(pts(roll=0.3)), f0)                                    # head roll does not change it
        assert brow_features(pts(drop=6))[1] < f0[1]                                              # lowered brows
        cue = BrowCue(dict(thr=2.0, span=1.5, w_max=0.5, unit=0.02, base_pct=85, min_base=6), CLASSES)
        judge = lambda p, q: dict(probs={c: float(p[i]) for i, c in enumerate(CLASSES)}, top=CLASSES[int(np.argmax(p))], quality=q)
        rest, frown = brow_features(pts()), brow_features(pts(drop=6))                            # 6 px = 0.1 eye distance = score 5
        hist = []
        fr = [dict(_r(i * 0.2, "neutral", 0.6), quality=1.0) for i in range(8)] + [dict(_r(2.0, "neutral", 0.6), quality=1.0), dict(_r(2.2, "happy", 0.9), quality=1.0)]
        cue.apply(fr, [rest] * 8 + [frown, frown], hist, judge)
        assert fr[8]["top"] == "angry" and fr[8]["brow"] > 2                                      # neutral + frown → angry
        assert fr[9]["probs"]["happy"] == 0.9                                                          # a confident non-neutral answer is left alone
        assert fr[0].get("brow") is None and len(hist) == 10                                      # no baseline yet for the first frames


if __name__ == "__main__":
    unittest.main()
