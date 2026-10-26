from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from capture_gap import compare_wer
from live_hand_body_diag import (
    ROOT as DIAG_ROOT,
    _find_repo_root,
    diagnose_file,
    diagnose_record,
    run as run_hand_body_diag,
)
from live_manifest import SCRIPT, build_manifest
from live_metrics import aggregate, score_row, selection_gates
from live_replay import apply_case, completed_keys, hand_fill_metrics, replay_rows
from modules import pipeline as pipeline_module
from modules import schema as schema_module
from modules.schema import fill_hand_gaps, finish, resample


class SyntheticFixtureTests(unittest.TestCase):
    @staticmethod
    def _diagnostic_frame(timestamp, left_visible=True, right_visible=True):
        pose = [[0.1, 0.2, 0.0, 1.0] for _ in range(33)]
        left = [[0.2, 0.3, 0.1] for _ in range(21)] if left_visible else [[0.0, 0.0, 0.0] for _ in range(21)]
        right = [[0.3, 0.4, 0.1] for _ in range(21)] if right_visible else [[0.0, 0.0, 0.0] for _ in range(21)]
        return {"t": timestamp, "pose": pose, "lh": left, "rh": right}

    def test_read_only_hand_body_diagnostic_reports_gaps_body_and_handedness(self):
        frames = []
        for index in range(20):
            left_visible = not (2 <= index < 5 or 8 <= index < 15)
            right_visible = not (4 <= index < 10)
            frame = self._diagnostic_frame(index * 100, left_visible, right_visible)
            if index == 0:
                frame["pose"][0][3] = 0.0
                frame["pose"][0][:2] = [0.0, 0.0]
                frame["pose"][11][3] = 0.0
                frame["pose"][11][:2] = [0.0, 0.0]
            if index == 1:
                frame["pose"][11][3] = 0.0
                frame["pose"][11][:2] = [0.0, 0.0]
            frames.append(frame)
        handedness = ["left"] * 9 + ["right"] * 3 + ["left"] * 8
        record = {
            "frames": frames,
            "handedness": handedness,
            "result": {"mirrored_to_right_hand": True},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(record, fh)
            result = diagnose_file(path)

        self.assertAlmostEqual(result["fps_actual"], 10.0)
        self.assertAlmostEqual(result["left_hand_visible_rate"], 0.5)
        self.assertAlmostEqual(result["left_hand_full_rate"], 0.5)
        self.assertAlmostEqual(result["right_hand_visible_rate"], 0.7)
        self.assertAlmostEqual(result["right_hand_full_rate"], 0.7)
        self.assertAlmostEqual(result["nose_visible_rate"], 0.95)
        self.assertAlmostEqual(result["left_shoulder_visible_rate"], 0.9)
        self.assertAlmostEqual(result["right_shoulder_visible_rate"], 1.0)
        self.assertAlmostEqual(result["cleared_by_present_rate"], 0.05)
        self.assertEqual(result["left_hand_gap_count"], 2)
        self.assertEqual(result["left_hand_gaps_longer_than_threshold"], 1)
        self.assertEqual(result["left_hand_gaps_at_or_shorter_than_threshold"], 1)
        self.assertAlmostEqual(result["left_hand_gap_fraction_longer_than_threshold"], 0.5)
        self.assertAlmostEqual(result["left_hand_gap_fraction_at_or_shorter_than_threshold"], 0.5)
        self.assertEqual(json.loads(result["left_hand_gap_frames_histogram_json"]), {"3": 1, "7": 1})
        self.assertEqual(json.loads(result["left_hand_gap_duration_ms_histogram_json"]), {"300-400": 1, "700-800": 1})
        self.assertEqual(result["right_hand_gaps_longer_than_threshold"], 1)
        self.assertEqual(result["handedness_data_status"], "มีข้อมูล")
        self.assertEqual(result["handedness_switch_count"], 2)
        self.assertEqual(result["result_flip_data_status"], "มีข้อมูล")
        self.assertEqual(result["result_flip_flag_count"], 1)
        self.assertEqual(result["result_flip_positive_flag_count"], 1)
        self.assertTrue(result["request_flipped"])
        self.assertEqual(result["request_flip_fraction"], 1.0)
        self.assertNotIn("pose", result)
        self.assertNotIn("lh", result)

    def test_partial_hand_is_diagnostic_visible_but_not_pipeline_full(self):
        frame = self._diagnostic_frame(0)
        frame["lh"][7:] = [[0.0, 0.0, 0.0] for _ in range(14)]
        result = diagnose_record({"frames": [frame]}, "partial-hand")
        self.assertEqual(result["left_hand_visible_rate"], 1.0)
        self.assertEqual(result["left_hand_full_rate"], 0.0)
        self.assertEqual(result["left_hand_gap_count"], 1)
        self.assertEqual(result["left_hand_gap_edge_count"], 1)

    def test_root_resolution_targets_repository_root(self):
        self.assertEqual(DIAG_ROOT, ROOT)
        self.assertEqual(_find_repo_root(HERE / "live_hand_body_diag.py"), ROOT)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "cannot locate repository root"):
                _find_repo_root(Path(directory) / "unrelated.py")

    def test_read_only_hand_body_diagnostic_writes_metrics_only_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            output_dir = root / "output"
            cache_dir.mkdir()
            for filename, result in (
                ("synthetic-flipped.json.gz", {"mirrored_to_right_hand": True}),
                ("synthetic-no-flip.json.gz", {}),
            ):
                with gzip.open(cache_dir / filename, "wt", encoding="utf-8") as fh:
                    json.dump({"frames": [self._diagnostic_frame(0)], "result": result}, fh)
            rows = run_hand_body_diag(cache_dir, output_dir)
            self.assertEqual(len(rows), 2)
            csv_text = (output_dir / "synthetic-flipped.csv").read_text(encoding="utf-8")
            markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("fps_actual", csv_text)
            self.assertIn("handedness_data_status", csv_text)
            self.assertIn("left_hand_full_rate", csv_text)
            self.assertIn("request_flip_fraction", csv_text)
            self.assertNotIn("pose", csv_text)
            self.assertNotIn("face", csv_text)
            self.assertIn("ไม่มีข้อมูล", csv_text)
            self.assertIn("Live hand/body diagnostic summary", markdown)
            self.assertIn("Requests with saved flip metadata: 1 / 2", markdown)
            self.assertIn("Requests recorded as flipped: 1", markdown)
            self.assertIn("Flip fraction among requests with flip metadata: 1.0000", markdown)

    def test_read_only_hand_body_diagnostic_reports_missing_handedness(self):
        result = diagnose_record({"frames": [self._diagnostic_frame(0)]}, "synthetic")
        self.assertEqual(result["handedness_data_status"], "ไม่มีข้อมูล")
        self.assertEqual(result["handedness_switch_count"], "")
        self.assertEqual(result["result_flip_data_status"], "ไม่มีข้อมูล")
        self.assertEqual(result["request_flip_fraction"], "")

    def test_saved_false_flip_flag_counts_as_known_not_flipped(self):
        result = diagnose_record(
            {"frames": [self._diagnostic_frame(0)], "result": {"mirrored_to_right_hand": False}},
            "not-flipped",
        )
        self.assertEqual(result["result_flip_data_status"], "มีข้อมูล")
        self.assertEqual(result["result_flip_flag_count"], 1)
        self.assertEqual(result["result_flip_positive_flag_count"], 0)
        self.assertFalse(result["request_flipped"])
        self.assertEqual(result["request_flip_fraction"], 0.0)

    def test_cache_saved_canonicalization_and_capture_metadata_are_diagnosed(self):
        from webapp.api.request_cache import save_request

        class CachedFrame:
            def __init__(self, t):
                self.t = t

            def model_dump(self):
                return {"t": self.t, "pose": [], "lh": [], "rh": [], "face": [], "bs": []}

        metadata = {
            "dominant_hand": "L",
            "dominant_hand_share": 0.71,
            "flip_margin": 0.62,
            "decision_rule": "mirror when dominant_hand is L and dominant_hand_share >= flip_margin",
            "flip_reason": "left_dominant_share_meets_margin",
        }
        frames = [self._diagnostic_frame(t) for t in (0, 100, 200)]

        with tempfile.TemporaryDirectory() as directory:
            for flipped in (True, False):
                case_dir = Path(directory) / str(flipped)
                case_dir.mkdir()
                request = SimpleNamespace(
                    frames=[CachedFrame(t) for t in (0, 100, 200)],
                    width=640,
                    height=480,
                    vocab="conversation",
                    alphabet="en",
                    user_bank=[],
                )
                output = {"mirrored_to_right_hand": flipped, "n_frames": 3}
                save_request(str(case_dir), request, output, metadata, 0.5)
                cache_path = next(case_dir.glob("*.json.gz"))
                with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
                    record = json.load(fh)
                record["frames"] = frames
                result = diagnose_record(record, cache_path.name)
                self.assertEqual(result["result_flip_data_status"], "มีข้อมูล")
                self.assertEqual(result["request_flipped"], flipped)
                self.assertEqual(result["flip_dominant_hand"], "L")
                self.assertAlmostEqual(result["flip_dominant_hand_share"], 0.71)
                self.assertEqual(result["flip_reason"], metadata["flip_reason"])
                self.assertEqual(result["hand_gap_s_recorded"], 0.5)
                self.assertEqual(result["n_frames_recorded"], 3)
                self.assertAlmostEqual(result["fps_actual_recorded"], 10.0)
                self.assertEqual(record["result"]["mirrored_to_right_hand"], flipped)
                self.assertNotIn("flipped", record["result"]["hand_canonicalization"])
                self.assertEqual(len(record["frames"]), 3)
                self.assertNotIn("hand_canonicalization", output)

    def test_canonical_hands_metadata_preserves_existing_decision(self):
        kp = np.zeros((2, 133, 2), dtype=np.float32)
        sc = np.ones((2, 133), dtype=np.float32)
        with patch.object(schema_module, "dominant_hand", return_value=("L", 0.73)):
            canonical_kp, canonical_sc, flipped, metadata = schema_module.canonical_hands_with_metadata(
                kp, sc, (480, 640)
            )
            legacy_kp, legacy_sc, legacy_flipped = schema_module.canonical_hands(kp, sc, (480, 640))
        self.assertTrue(flipped)
        self.assertTrue(legacy_flipped)
        self.assertEqual(metadata["dominant_hand"], "L")
        self.assertEqual(metadata["dominant_hand_share"], 0.73)
        self.assertEqual(metadata["flip_margin"], 0.62)
        self.assertEqual(metadata["flip_reason"], "left_dominant_share_meets_margin")
        expected_kp, expected_sc = schema_module.mirror(kp, sc)
        np.testing.assert_array_equal(canonical_kp, expected_kp)
        np.testing.assert_array_equal(canonical_sc, expected_sc)
        np.testing.assert_array_equal(legacy_kp, canonical_kp)
        np.testing.assert_array_equal(legacy_sc, canonical_sc)

    def test_read_only_diagnostic_fails_when_cache_directory_missing_or_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "live cache directory not found"):
                run_hand_body_diag(missing, root / "output")
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "no \\*\\.json\\.gz request files"):
                run_hand_body_diag(empty, root / "output")
            self.assertFalse((root / "output").exists())

    def test_edge_gaps_are_never_fillable_and_leading_duration_is_capped(self):
        frames = [
            self._diagnostic_frame(0, left_visible=False),
            self._diagnostic_frame(100, left_visible=False),
            self._diagnostic_frame(200),
            self._diagnostic_frame(300),
            self._diagnostic_frame(400, left_visible=False),
            self._diagnostic_frame(500, left_visible=False),
        ]
        result = diagnose_record({"frames": frames}, "edge-gaps", hand_gap_s=0.5)
        self.assertEqual(result["left_hand_gap_count"], 2)
        self.assertEqual(result["left_hand_gap_edge_count"], 2)
        self.assertEqual(result["left_hand_gap_fillable_count"], 0)
        self.assertEqual(result["left_hand_gap_not_fillable_count"], 2)
        self.assertEqual(result["left_hand_gap_too_long_count"], 0)
        self.assertEqual(result["left_hand_gaps_at_or_shorter_than_threshold"], 2)
        self.assertEqual(json.loads(result["left_hand_gap_duration_ms_histogram_json"]), {"100-200": 1, "200-300": 1})


    def test_body_clear_uses_mean_raw_visibility_scores(self):
        frame = self._diagnostic_frame(0)
        frame["pose"][0][3] = 0.0
        frame["pose"][11][3] = 0.6
        frame["pose"][12][3] = 0.0
        frame["pose"][12][:2] = [0.0, 0.0]
        result = diagnose_record({"frames": [frame]}, "synthetic")
        self.assertEqual(result["nose_visible_rate"], 1.0)
        self.assertEqual(result["left_shoulder_visible_rate"], 1.0)
        self.assertEqual(result["right_shoulder_visible_rate"], 0.0)
        self.assertEqual(result["cleared_by_present_rate"], 0.0)

    def test_clearing_uses_pipeline_binary_confidence_not_raw_mean(self):
        frame = self._diagnostic_frame(0)
        frame["pose"][0][3] = 1.0
        frame["pose"][11][3] = 0.4
        frame["pose"][12][3] = 0.4
        result = diagnose_record({"frames": [frame]}, "synthetic")
        self.assertEqual(result["nose_visible_rate"], 1.0)
        self.assertEqual(result["left_shoulder_visible_rate"], 0.0)
        self.assertEqual(result["right_shoulder_visible_rate"], 0.0)
        self.assertEqual(result["cleared_by_present_rate"], 1.0)

    def test_read_only_hand_body_diagnostic_rejects_unexpected_cache_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump({"frames": [{"t": 0, "pose": [], "lh": []}]}, fh)
            with self.assertRaisesRegex(ValueError, "missing required landmark fields: rh"):
                diagnose_file(path)

    def test_timestamp_resampling_uses_irregular_time_axis(self):
        times = np.array([0.0, 0.1, 0.4])
        keypoints = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[4.0, 0.0]]], dtype=np.float32)
        scores = np.ones((3, 1), dtype=np.float32)
        sampled, sampled_scores = resample(keypoints, scores, times, fps=10.0)
        self.assertEqual(len(sampled), 5)
        np.testing.assert_allclose(sampled[:, 0, 0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(sampled_scores, np.ones((5, 1), dtype=np.float32))

    def test_production_hand_gap_fill_interpolates_short_dropout(self):
        keypoints = np.zeros((3, 133, 2), dtype=np.float32)
        scores = np.zeros((3, 133), dtype=np.float32)
        keypoints[0, 91:112] = 0.0
        keypoints[2, 91:112] = 2.0
        scores[0, 91:112] = 1.0
        scores[2, 91:112] = 1.0
        filled, filled_scores = fill_hand_gaps(keypoints, scores, max_gap=1)
        np.testing.assert_allclose(filled[1, 91:112], 1.0)
        np.testing.assert_allclose(filled_scores[1, 91:112], 1.0)

    def test_hand_gap_cases_change_production_fill_max_gap_at_irregular_10fps(self):
        times = np.arange(12, dtype=np.float64) / 10.0
        keypoints = np.zeros((len(times), 133, 2), dtype=np.float32)
        scores = np.zeros((len(times), 133), dtype=np.float32)
        scores[:, 0] = scores[:, 5] = scores[:, 6] = 1.0
        keypoints[:, 5, 0] = 0.25
        keypoints[:, 6, 0] = 0.75
        left_points = np.full((len(times), 21, 3), 0.2, dtype=np.float32)
        left_points[3:9] = 0.0
        right_points = np.full((len(times), 21, 3), 0.3, dtype=np.float32)
        for frame in range(len(times)):
            keypoints[frame, 112:133] = right_points[frame, :, :2]
            scores[frame, 112:133] = 1.0
            if frame not in range(3, 9):
                keypoints[frame, 91:112] = left_points[frame, :, :2]
                scores[frame, 91:112] = 1.0
        raw = {
            "t": times,
            "kp": keypoints,
            "sc": scores,
            "hw": (480, 640),
            "lh": left_points,
            "rh": right_points,
        }
        observed_gaps = []
        original_fill = schema_module.fill_hand_gaps

        def record_fill(kp, sc, max_gap):
            observed_gaps.append(max_gap)
            return original_fill(kp, sc, max_gap)

        def synthetic_to_standard(landmarks, gap_s):
            return finish(
                landmarks["kp"].copy(), landmarks["sc"].copy(), landmarks["t"],
                landmarks["hw"], "synthetic", "mp", set(), gap_s=gap_s,
            )

        pipe = SimpleNamespace(
            spot={},
            spot_all={"hand_gap_s": 0.5},
            cal={"tau_accept": 0.8},
        )
        output_metrics = []
        with (
            patch.object(pipeline_module, "to_standard", side_effect=synthetic_to_standard),
            patch.object(schema_module, "fill_hand_gaps", side_effect=record_fill),
        ):
            for gap_s in (0.3, 0.8):
                pipe.spot_all = {"hand_gap_s": 0.5}
                effective = apply_case(pipe, {"hand_gap_s": gap_s})
                self.assertEqual(effective["hand_gap_s"], gap_s)
                standardized = pipeline_module.SignPipeline.standardise(pipe, raw)
                output_metrics.append(hand_fill_metrics(raw, standardized[1], standardized[4]))

        self.assertEqual(observed_gaps, [3, 8])
        self.assertLess(
            output_metrics[0]["left_hand_after_fill_rate"],
            output_metrics[1]["left_hand_after_fill_rate"],
        )
        self.assertLess(
            output_metrics[0]["left_hand_imputed_frame_fraction"],
            output_metrics[1]["left_hand_imputed_frame_fraction"],
        )

    def test_manifest_joins_filename_metadata_and_timestamp_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            filename = "synthetic.json.gz"
            hands = [[0.2, 0.3, 0.1] for _ in range(21)]
            record = {
                "time": "synthetic",
                "width": 640,
                "height": 480,
                "vocab": "conversation",
                "alphabet": "off",
                "frames": [
                    {"t": 0, "lh": hands, "rh": []},
                    {"t": 40, "lh": hands, "rh": hands},
                    {"t": 100, "lh": [], "rh": hands},
                ],
                "result": {"sentence": "synthetic only"},
            }
            with gzip.open(cache / filename, "wt", encoding="utf-8") as fh:
                json.dump(record, fh)
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=[
                    "request_id", "session_id", "signer_id", "script_index", "split",
                    "device", "browser", "advertised_fps",
                ])
                writer.writeheader()
                writer.writerow({
                    "request_id": filename, "session_id": "synth-session",
                    "signer_id": "synth-signer", "script_index": "1", "split": "tune",
                    "device": "synthetic", "browser": "synthetic", "advertised_fps": "30",
                })
            rows = build_manifest(cache, metadata)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reference"], list(SCRIPT[1]))
            self.assertEqual(rows[0]["fps_actual"], 20.0)
            self.assertAlmostEqual(rows[0]["duration_s"], 0.1)
            self.assertAlmostEqual(rows[0]["left_hand_frame_fraction"], 2 / 3)
            self.assertAlmostEqual(rows[0]["right_hand_frame_fraction"], 2 / 3)

    def test_metric_counts_wer_precision_false_accept_and_count_direction(self):
        row = {
            "request_id": "synthetic.json.gz", "session_id": "synth-session",
            "signer_id": "synth-signer", "split": "tune", "case": "baseline",
            "reference": ["A", "B"],
            "segments": [
                {"nearest": "A", "status": "accepted"},
                {"nearest": "X", "status": "accepted"},
                {"nearest": "Y", "status": "uncertain"},
            ],
            "left_hand_frame_fraction": 0.75, "right_hand_frame_fraction": 0.5,
            "hand_fill_metrics": {
                "left_hand_after_fill_rate": 0.8,
                "right_hand_after_fill_rate": 0.9,
                "left_hand_imputed_frame_fraction": 0.1,
                "right_hand_imputed_frame_fraction": 0.0,
            },
            "fps_actual": 24.0, "latency_ms": 120.0,
        }
        scored = score_row(row)
        self.assertEqual((scored["S"], scored["D"], scored["I"]), (1, 0, 1))
        self.assertEqual(scored["utterance_wer"], 1.0)
        self.assertEqual((scored["count_over"], scored["count_under"]), (1, 0))
        self.assertEqual(scored["accepted_precision"], 0.5)
        self.assertEqual(scored["false_accept_count"], 1)
        self.assertEqual(scored["left_hand_detection_rate"], 0.75)
        case_rows, session_rows = aggregate([row])
        self.assertEqual(case_rows[0]["macro_wer"], 1.0)
        self.assertEqual(case_rows[0]["corpus_wer"], 1.0)
        self.assertEqual(case_rows[0]["mean_left_hand_after_fill_rate"], 0.8)
        self.assertEqual(case_rows[0]["mean_left_hand_imputed_frame_fraction"], 0.1)
        self.assertEqual(session_rows[0]["D"], 0)

    def test_capture_gap_reports_offline_minus_live_wer(self):
        live = {
            "request_id": "synthetic.json.gz", "session_id": "synth-session",
            "signer_id": "synth-signer", "split": "eval", "case": "baseline",
            "reference": ["A"], "segments": [{"nearest": "A", "status": "accepted"}],
        }
        compared = compare_wer(live, [{"nearest": "B", "status": "accepted"}])
        self.assertEqual(compared["live_wer"], 0.0)
        self.assertEqual(compared["offline_wer"], 1.0)
        self.assertEqual(compared["offline_minus_live_wer"], 1.0)

    def test_selection_gate_uses_tune_only_and_requires_heldout_for_promotion(self):
        sessions = []
        for index in range(3):
            sessions.extend([
                {"split": "tune", "session_id": f"tune-{index}", "signer_id": f"signer-{index}",
                 "case": "baseline", "macro_wer": 0.1},
                {"split": "tune", "session_id": f"tune-{index}", "signer_id": f"signer-{index}",
                 "case": "candidate", "macro_wer": 0.097},
            ])
        summary = [{"split": "tune", "case": "candidate", "accepted_precision": 0.95}]
        gate = selection_gates(sessions, summary)[0]
        self.assertTrue(gate["pass_gate"])
        self.assertFalse(gate["eval_separation_valid"])
        self.assertFalse(gate["promotion_eligible"])
        sessions.extend([
            {"split": "eval", "session_id": "eval-1", "signer_id": "heldout",
             "case": "baseline", "macro_wer": 0.2},
        ])
        gate = selection_gates(sessions, summary)[0]
        self.assertTrue(gate["pass_gate"])
        self.assertTrue(gate["promotion_eligible"])

    def test_replay_resume_skips_completed_request_case_without_duplicate(self):
        class SyntheticPipeline:
            spot = {"act_thr": 0.5, "post": {"p_sign_min": 0.6}}
            spot_all = {"hand_gap_s": 0.5}
            cal = {"tau_accept": 0.8}

            @staticmethod
            def landmarks_from_browser(frames, hw):
                count = len(frames)
                return {
                    "frames": frames,
                    "hw": hw,
                    "t": np.array([frame["t"] / 1000 for frame in frames], dtype=np.float64),
                    "lh": np.zeros((count, 21, 3), dtype=np.float32),
                    "rh": np.zeros((count, 21, 3), dtype=np.float32),
                }

            @staticmethod
            def standardise(raw):
                return (
                    None,
                    np.zeros((len(raw["t"]), 133), dtype=np.float32),
                    None, None, None,
                )

            def run_landmarks(self, raw, **kwargs):
                self.standardise(raw)
                return {
                    "sentence": "synthetic",
                    "segments": [{"nearest": "A", "status": "accepted"}],
                    "duration_s": 0.08,
                    "signer_visible_frac": 1.0,
                    "timings_ms": {"total_ms": 1.0},
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            filename = "synthetic.json.gz"
            record = {
                "height": 480, "width": 640, "alphabet": "off",
                "vocab": "conversation",
                "frames": [{"t": 0}, {"t": 40}],
            }
            with gzip.open(cache / filename, "wt", encoding="utf-8") as fh:
                json.dump(record, fh)
            output = root / "replay.jsonl"
            manifest = [{
                "request_id": filename, "session_id": "synth-session",
                "signer_id": "synth-signer", "split": "tune", "reference": ["A"],
            }]
            cases = {
                "baseline": {},
                "short_gap": {"hand_gap_s": 0.3},
                "baseline_again": {},
                "lower_tau": {"tau_accept": 0.7},
            }
            factory = SyntheticPipeline
            self.assertEqual(replay_rows(manifest, cache, output, cases, factory), 4)
            self.assertEqual(replay_rows(manifest, cache, output, cases, factory), 0)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 4)
            lowered = next(row for row in records if row["case"] == "lower_tau")
            self.assertEqual(lowered["effective_config"]["tau_accept"], 0.7)
            self.assertEqual(lowered["effective_config"]["hand_gap_s"], 0.5)
            self.assertEqual(lowered["effective_hand_gap_s"], 0.5)
            self.assertIn("left_hand_after_fill_rate", lowered["hand_fill_metrics"])
            short_gap = next(row for row in records if row["case"] == "short_gap")
            self.assertEqual(short_gap["effective_config"]["hand_gap_s"], 0.3)
            baseline_again = next(row for row in records if row["case"] == "baseline_again")
            self.assertEqual(baseline_again["effective_config"]["hand_gap_s"], 0.5)
            baseline = next(row for row in records if row["case"] == "baseline")
            self.assertEqual(baseline["effective_config"]["tau_accept"], 0.8)

    def test_replay_refuses_to_resume_rows_without_hand_gap_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "old-replay.jsonl"
            output.write_text(
                json.dumps({"request_id": "synthetic", "case": "baseline"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "predates hand-gap metrics"):
                completed_keys(output, require_hand_metrics=True)


if __name__ == "__main__":
    unittest.main()
