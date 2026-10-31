"""Release gate (scripts/decision.py gate_check): a report that is worse beyond the frozen tolerance must fail, an equal or better
one must pass. Pure dictionaries — no models, no cache."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from decision import gate_check  # noqa: E402

BASE = dict(WER=0.296, R1=0.720, R5=0.856, macro_F1=0.795, seg_F1=0.873, boundary_F1=0.60, count_err=0.389, acc_prec=0.973,
            acc_cover=0.400, false_accept_rate=0.10, ECE=0.034, cov_at_p90=0.55)
TOL = {"t51_test_x1.0": {k: 0.005 for k in BASE}}


def report(new):
    return dict(sets={"t51_test_x1.0": {"v9": dict(BASE), "v10": new}},
                enrolled={"t51_test_x1.0": {"v9": {"WER": 0.119}, "v10": {"WER": 0.119}}},
                latency={"v9": dict(model_ms_per_video_second=100.0, rss_mb=800.0), "v10": dict(model_ms_per_video_second=105.0, rss_mb=820.0)})


class GateTests(unittest.TestCase):
    def test_identical_report_passes(self):
        _, fails = gate_check(report(dict(BASE)), "v9", "v10", TOL)
        self.assertEqual(fails, [])

    def test_better_report_passes(self):
        _, fails = gate_check(report(dict(BASE, WER=0.25, macro_F1=0.82)), "v9", "v10", TOL)
        self.assertEqual(fails, [])

    def test_worse_wer_fails(self):
        _, fails = gate_check(report(dict(BASE, WER=0.31)), "v9", "v10", TOL)
        self.assertTrue(any("WER" in f for f in fails))

    def test_worse_within_tolerance_passes(self):
        _, fails = gate_check(report(dict(BASE, WER=0.299)), "v9", "v10", TOL)
        self.assertEqual(fails, [])

    def test_lower_macro_f1_fails(self):
        _, fails = gate_check(report(dict(BASE, macro_F1=0.70)), "v9", "v10", TOL)
        self.assertTrue(any("macro_F1" in f for f in fails))

    def test_precision_floor(self):
        _, fails = gate_check(report(dict(BASE, acc_prec=0.85)), "v9", "v10", {"t51_test_x1.0": {k: 1.0 for k in BASE}})
        self.assertTrue(any("< 0.90" in f for f in fails))

    def test_latency_budget(self):
        r = report(dict(BASE))
        r["latency"]["v10"]["model_ms_per_video_second"] = 125.0
        _, fails = gate_check(r, "v9", "v10", TOL)
        self.assertTrue(any("model_ms_per_video_second" in f for f in fails))


if __name__ == "__main__":
    unittest.main()
