"""Read-only hand/body landmark diagnostics for sign-api gzip cache records.

Only derived rates, histograms and handedness transition counts are written;
landmark coordinates and face data are never emitted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _find_repo_root(script_path: Path) -> Path:
    for parent in script_path.resolve().parents:
        if (parent / ".git").exists() and (parent / "modules").is_dir():
            return parent
    raise RuntimeError(f"cannot locate repository root from {script_path}; expected .git and modules/")


ROOT = _find_repo_root(Path(__file__))
HAND_GAP_S = 0.5
HAND_POINTS = 21
POSE_POINTS = 33
POSE_SCORE_INDEX = 3
BODY_POINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}
HANDEDNESS_LABELS = {
    "l": "left",
    "left": "left",
    "r": "right",
    "right": "right",
}
HISTOGRAM_MS_WIDTH = 100


def _number(value: Any, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


def _validate_points(value: Any, count: int, dimensions: int, context: str) -> list[list[float]]:
    if value in (None, []):
        return []
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{context} must be empty or contain exactly {count} points")
    result = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) < dimensions:
            raise ValueError(f"{context}[{index}] must contain at least {dimensions} values")
        result.append([_number(v, f"{context}[{index}]") for v in point[:dimensions]])
    return result


def _normalize_handedness(value: Any, context: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a left/right string or null")
    normalized = HANDEDNESS_LABELS.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"{context} must be one of left, right, L, R, or null")
    return normalized


def _validate_record(
    record: Any,
    request_id: str,
) -> tuple[list[dict[str, Any]], list[float], list[str | None]]:
    if not isinstance(record, dict):
        raise ValueError(f"{request_id}: cache root must be a JSON object")
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{request_id}: frames must be a non-empty list")

    record_handedness = record.get("handedness")
    if record_handedness is not None:
        if not isinstance(record_handedness, list) or len(record_handedness) != len(frames):
            raise ValueError(f"{request_id}: top-level handedness must align with frames")

    timestamps: list[float] = []
    normalized_frames = []
    handedness: list[str | None] = []
    for index, frame in enumerate(frames):
        context = f"{request_id}: frame {index}"
        if not isinstance(frame, dict):
            raise ValueError(f"{context} must be an object")
        if "t" not in frame:
            raise ValueError(f"{context} is missing timestamp field 't'")
        missing = {"pose", "lh", "rh"} - set(frame)
        if missing:
            raise ValueError(f"{context} is missing required landmark fields: {', '.join(sorted(missing))}")
        timestamps.append(_number(frame["t"], f"{context} t"))
        pose = _validate_points(frame["pose"], POSE_POINTS, 4, f"{context} pose")
        left = _validate_points(frame["lh"], HAND_POINTS, 2, f"{context} lh")
        right = _validate_points(frame["rh"], HAND_POINTS, 2, f"{context} rh")
        if record_handedness is not None and frame.get("handedness") is not None:
            raise ValueError(f"{context}: provide handedness either per frame or at top level, not both")
        frame_label = frame.get("handedness") if record_handedness is None else record_handedness[index]
        handedness.append(_normalize_handedness(frame_label, f"{context} handedness"))
        normalized_frames.append({"pose": pose, "lh": left, "rh": right})

    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{request_id}: frame timestamps must be strictly increasing")
    return normalized_frames, timestamps, handedness


def _hand_score(points: list[list[float]]) -> float:
    if not points:
        return 0.0
    return sum((abs(point[0]) + abs(point[1])) > 0 for point in points) / HAND_POINTS


def _hand_full(points: list[list[float]]) -> bool:
    return len(points) == HAND_POINTS and all(
        abs(point[0]) + abs(point[1]) > 0 for point in points
    )


def _body_score(pose: list[list[float]], point_index: int) -> float:
    if not pose:
        return 0.0
    point = pose[point_index]
    score = point[POSE_SCORE_INDEX]
    if score == 0 and abs(point[0]) + abs(point[1]) > 0:
        score = 1.0
    return float(score > 0.5 and abs(point[0]) + abs(point[1]) > 0)


def _gap_durations_ms(
    start: int,
    stop: int,
    timestamps_ms: list[float],
    median_frame_ms: float,
) -> float:
    """Estimate missing-frame span; leading gaps are capped at one frame period."""
    duration = timestamps_ms[stop - 1] - timestamps_ms[start] + median_frame_ms
    if start == 0:
        duration = min(duration, median_frame_ms)
    return max(0.0, duration)


def _hand_gaps(
    visible: list[bool],
    timestamps_ms: list[float],
    median_frame_ms: float,
) -> list[dict[str, float | int]]:
    gaps = []
    index = 0
    while index < len(visible):
        if visible[index]:
            index += 1
            continue
        start = index
        while index < len(visible) and not visible[index]:
            index += 1
        stop = index
        length_frames = stop - start
        duration_ms = _gap_durations_ms(start, stop, timestamps_ms, median_frame_ms)
        at_clip_edge = start == 0 or stop == len(visible)
        gaps.append({
            "frames": length_frames,
            "duration_ms": duration_ms,
            "at_clip_edge": at_clip_edge,
        })
    return gaps


def _flip_metrics(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    flags: list[bool] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                label = key.casefold()
                if ("mirror" in label or "flip" in label) and isinstance(nested, bool):
                    flags.append(nested)
                else:
                    visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    if result is not None:
        if not isinstance(result, dict):
            raise ValueError("result must be a JSON object when present")
        visit(result)
    if not flags:
        return {
            "result_flip_data_status": "ไม่มีข้อมูล",
            "result_flip_flag_count": 0,
            "result_flip_positive_flag_count": "",
            "request_flipped": "",
            "request_flip_fraction": "",
        }
    flipped = any(flags)
    return {
        "result_flip_data_status": "มีข้อมูล",
        "result_flip_flag_count": len(flags),
        "result_flip_positive_flag_count": sum(flags),
        "request_flipped": flipped,
        "request_flip_fraction": 1.0 if flipped else 0.0,
    }


def _histograms(gaps: list[dict[str, float | int]]) -> tuple[dict[str, int], dict[str, int]]:
    frame_hist = Counter(str(gap["frames"]) for gap in gaps)
    duration_hist = Counter(
        f"{int(float(gap['duration_ms']) // HISTOGRAM_MS_WIDTH) * HISTOGRAM_MS_WIDTH}-"
        f"{(int(float(gap['duration_ms']) // HISTOGRAM_MS_WIDTH) + 1) * HISTOGRAM_MS_WIDTH}"
        for gap in gaps
    )
    return (
        dict(sorted(frame_hist.items(), key=lambda pair: int(pair[0]))),
        dict(sorted(duration_hist.items(), key=lambda pair: int(pair[0].split("-", 1)[0]))),
    )


def diagnose_record(record: Any, request_id: str, hand_gap_s: float = HAND_GAP_S) -> dict[str, Any]:
    """Validate one cache record and return metrics without returning source data."""
    if not math.isfinite(hand_gap_s) or hand_gap_s < 0:
        raise ValueError("hand_gap_s must be a finite non-negative number")
    frames, timestamps, handedness = _validate_record(record, request_id)
    n_frames = len(frames)
    duration_s = (timestamps[-1] - timestamps[0]) / 1000 if n_frames > 1 else 0.0
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
    median_frame_ms = median(intervals) if intervals else 1000 / 25
    actual_fps = (n_frames - 1) / duration_s if duration_s > 0 else 0.0

    hand_visible = {
        side: [_hand_score(frame[key]) > 0.3 for frame in frames]
        for side, key in (("left", "lh"), ("right", "rh"))
    }
    hand_full = {
        side: [_hand_full(frame[key]) for frame in frames]
        for side, key in (("left", "lh"), ("right", "rh"))
    }
    body_visible = {
        name: [
            _body_score(frame["pose"], point_index) > 0.5
            for frame in frames
        ]
        for name, point_index in BODY_POINTS.items()
    }
    present = [
        sum(_body_score(frame["pose"], idx) for idx in (0, 11, 12)) / 3 > 0.5
        for frame in frames
    ]

    result: dict[str, Any] = {
        "request_id": request_id,
        "n_frames": n_frames,
        "duration_s": duration_s,
        "fps_actual": actual_fps,
        "hand_gap_s_threshold": hand_gap_s,
    }
    for side, visibility in hand_visible.items():
        gaps = _hand_gaps(hand_full[side], timestamps, median_frame_ms)
        frame_hist, duration_hist = _histograms(gaps)
        within_threshold = [
            float(gap["duration_ms"]) <= hand_gap_s * 1000
            for gap in gaps
        ]
        longer = sum(not is_short for is_short in within_threshold)
        shorter = sum(within_threshold)
        gap_count = len(gaps)
        result[f"{side}_hand_visible_rate"] = sum(visibility) / n_frames
        result[f"{side}_hand_full_rate"] = sum(hand_full[side]) / n_frames
        result[f"{side}_hand_gap_count"] = gap_count
        result[f"{side}_hand_gaps_longer_than_threshold"] = longer
        result[f"{side}_hand_gaps_at_or_shorter_than_threshold"] = shorter
        result[f"{side}_hand_gap_fraction_longer_than_threshold"] = longer / gap_count if gap_count else ""
        result[f"{side}_hand_gap_fraction_at_or_shorter_than_threshold"] = shorter / gap_count if gap_count else ""
        fillable = [
            not gap["at_clip_edge"] and is_short
            for gap, is_short in zip(gaps, within_threshold)
        ]
        result[f"{side}_hand_gap_fillable_count"] = sum(fillable)
        result[f"{side}_hand_gap_not_fillable_count"] = gap_count - sum(fillable)
        result[f"{side}_hand_gap_edge_count"] = sum(gap["at_clip_edge"] for gap in gaps)
        result[f"{side}_hand_gap_too_long_count"] = sum(
            not gap["at_clip_edge"] and not is_short
            for gap, is_short in zip(gaps, within_threshold)
        )
        result[f"{side}_hand_gap_frames_histogram_json"] = json.dumps(frame_hist, sort_keys=True)
        result[f"{side}_hand_gap_duration_ms_histogram_json"] = json.dumps(duration_hist, sort_keys=True)
    for name, visibility in body_visible.items():
        result[f"{name}_visible_rate"] = sum(visibility) / n_frames
    result["cleared_by_present_rate"] = sum(not is_present for is_present in present) / n_frames

    known = [label for label in handedness if label is not None]
    if not known:
        result["handedness_data_status"] = "ไม่มีข้อมูล"
        result["handedness_switch_count"] = ""
    else:
        result["handedness_data_status"] = "มีข้อมูล"
        result["handedness_switch_count"] = sum(
            a != b
            for a, b in zip(handedness, handedness[1:])
            if a is not None and b is not None
        )
    result.update(_flip_metrics(record))
    return result


def diagnose_file(path: Path, hand_gap_s: float = HAND_GAP_S) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name}: unable to read valid gzip JSON cache: {exc}") from exc
    return diagnose_record(record, path.name, hand_gap_s)


def _write_summary(path: Path, rows: list[dict[str, Any]], hand_gap_s: float) -> None:
    metrics = [
        "fps_actual",
        "left_hand_visible_rate",
        "right_hand_visible_rate",
        "left_hand_full_rate",
        "right_hand_full_rate",
        "nose_visible_rate",
        "left_shoulder_visible_rate",
        "right_shoulder_visible_rate",
        "left_elbow_visible_rate",
        "right_elbow_visible_rate",
        "left_wrist_visible_rate",
        "right_wrist_visible_rate",
        "cleared_by_present_rate",
        "left_hand_gap_fraction_longer_than_threshold",
        "right_hand_gap_fraction_longer_than_threshold",
    ]
    flip_data_rows = [row for row in rows if row["result_flip_data_status"] == "มีข้อมูล"]
    flipped_rows = [row for row in flip_data_rows if row["request_flipped"]]
    lines = [
        "# Live hand/body diagnostic summary",
        "",
        f"- Requests analyzed: {len(rows)}",
        f"- Hand dropout threshold: {hand_gap_s:.3f} s",
        f"- Requests with saved flip metadata: {len(flip_data_rows)} / {len(rows)}",
        f"- Requests recorded as flipped: {len(flipped_rows)}",
        f"- Flip fraction among requests with flip metadata: "
        f"{len(flipped_rows) / len(flip_data_rows):.4f}" if flip_data_rows else
        "- Flip fraction among requests with flip metadata: ไม่มีข้อมูล",
        "- No landmark coordinates, face fields, or face data are included.",
        "- Hand gap durations are estimated between bounding visible frames; edge gaps use median frame cadence.",
        f"- Millisecond histogram buckets are {HISTOGRAM_MS_WIDTH} ms wide.",
        "",
        "| Request | " + " | ".join(metrics) + " | Handedness | Switches | Flip data | Flipped |",
        "|---|" + "|".join("---:" for _ in metrics) + "|---|---:|---|---|",
    ]
    for row in rows:
        values = [
            f"{row[name]:.4f}" if isinstance(row[name], (int, float)) else "n/a"
            for name in metrics
        ]
        lines.append(
            "| " + str(row["request_id"]) + " | " + " | ".join(values)
            + f" | {row['handedness_data_status']} | {row['handedness_switch_count']}"
            + f" | {row['result_flip_data_status']} | {row['request_flipped']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cache_dir: Path, output_dir: Path, hand_gap_s: float = HAND_GAP_S) -> list[dict[str, Any]]:
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"live cache directory not found: {cache_dir}")
    cache_files = sorted(cache_dir.glob("*.json.gz"))
    if not cache_files:
        raise FileNotFoundError(f"no *.json.gz request files found in live cache directory: {cache_dir}")
    rows = [diagnose_file(path, hand_gap_s) for path in cache_files]
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = Path(row["request_id"]).name.removesuffix(".json.gz") + ".csv"
        output_path = output_dir / filename
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    _write_summary(output_dir / "summary.md", rows, hand_gap_s)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/live_requests"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/phase2.2/hand_body_diag"))
    parser.add_argument("--hand-gap-s", type=float, default=HAND_GAP_S)
    args = parser.parse_args()
    if args.cache_dir.is_absolute() or ".." in args.cache_dir.parts:
        parser.error("--cache-dir must be relative to the checkout root and cannot contain '..'")
    if args.output_dir.is_absolute() or ".." in args.output_dir.parts:
        parser.error("--output-dir must be relative to the checkout root and cannot contain '..'")
    cache_dir, output_dir = ROOT / args.cache_dir, ROOT / args.output_dir
    try:
        rows = run(cache_dir, output_dir, args.hand_gap_s)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    print(f"Analyzed {len(rows)} request(s); wrote per-request CSV and summary to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
