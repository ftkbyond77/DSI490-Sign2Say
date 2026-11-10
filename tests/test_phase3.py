"""Phase 3 (Error-Analysis-v2) logic that needs no model: the signing-gate veto, the head-shake counter, the duplicate collapse of the rules
composer. Pure NumPy / Python."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.face import _cycles  # noqa: E402
from modules.language import rules_compose  # noqa: E402
from modules.pipeline import gate_veto  # noqa: E402


def probs(labels):
    out = np.zeros((len(labels), 3), np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


class GateVeto(unittest.TestCase):
    def test_short_non_sign_runs_keep_the_tagger(self):
        tagger = probs([2] * 40)                       # the old tagger: inside a sign everywhere
        gate = probs([2] * 10 + [0] * 20 + [2] * 10)   # the gate: 20 frames (0.8 s) of non-sign
        self.assertTrue(np.array_equal(gate_veto(tagger, gate, 30), tagger))

    def test_long_non_sign_runs_are_vetoed(self):
        tagger = probs([2] * 80)
        gate = probs([2] * 10 + [0] * 60 + [2] * 10)   # 2.4 s still hold
        out = gate_veto(tagger, gate, 30)
        self.assertTrue((out[10:70].argmax(1) == 0).all())
        self.assertTrue(np.array_equal(out[:10], tagger[:10]) and np.array_equal(out[70:], tagger[70:]))

    def test_no_gate_change_when_gate_agrees(self):
        tagger = probs([0] * 5 + [1] + [2] * 20 + [0] * 5)
        self.assertTrue(np.array_equal(gate_veto(tagger, tagger.copy(), 30), tagger))


class HeadShake(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_a_real_shake_is_counted(self):
        for fps in (12.5, 25.0):
            t = np.arange(0, 5, 1 / fps)
            y = self.rng.normal(0, 0.01, len(t))
            m = (t > 2) & (t < 3.3)
            y[m] += 0.15 * np.sin(2 * np.pi * 2.5 * (t[m] - 2))
            self.assertGreaterEqual(_cycles(y, 0.06, t), 2, fps)

    def test_separate_head_turns_are_not_a_shake(self):
        t = np.arange(0, 8, 1 / 12.5)
        y = self.rng.normal(0, 0.01, len(t))
        for c, s in ((1.0, 1), (3.5, -1), (6.0, 1)):
            y += s * 0.12 * np.exp(-((t - c) / 0.25) ** 2)
        self.assertLess(_cycles(y, 0.06, t), 2)

    def test_clip_edges_are_not_swings(self):
        y = np.full(60, 0.4)                           # a head turned to one side the whole clip: the old zero-padded detrend saw swings at both ends
        self.assertEqual(_cycles(y, 0.06), 0)


class RulesCompose(unittest.TestCase):
    def test_dropped_unsure_sign_does_not_double_a_word(self):
        seg = lambda w, st: dict(status=st, candidates=[dict(word=w, score=0.9)], p_correct=0.9 if st == "accepted" else 0.4)
        out = rules_compose([seg("ฉัน", "accepted"), seg("สบายดี", "accepted"), seg("คุณ", "uncertain"), seg("สบายดี", "accepted")], {}, {})
        self.assertEqual(out["sentence_th"].count("สบายดี"), 1)


if __name__ == "__main__":
    unittest.main()
