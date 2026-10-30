"""Compare saved live landmarks with offline landmarks from the same video.

This CLI reads video only when explicitly run by the user. It is not used by
the synthetic tests or by live_manifest/live_replay.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from live_replay import append_result, completed_keys


def checkout_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be checkout-relative and cannot use '..': {path}")
    return ROOT / path


def load_video_metadata(path: Path) -> dict[str, str]:
    required = {"request_id", "offline_video"}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata CSV missing columns: {', '.join(sorted(missing))}")
        rows = {}
        for row in reader:
            request_id = (row.get("request_id") or "").strip()
            video = (row.get("offline_video") or "").strip()
            if not request_id or request_id in rows:
                raise ValueError(f"blank or duplicate request_id in metadata: {request_id!r}")
            if video:
                path_value = Path(video)
                if path_value.is_absolute() or ".." in path_value.parts:
                    raise ValueError(f"{request_id}: offline_video must be checkout-relative and cannot use '..'")
                rows[request_id] = video
    return rows


def compare_wer(live_row: dict[str, Any], offline_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Return paired WERs; positive difference means offline has higher WER."""
    from live_metrics import score_row

    offline_row = {
        **live_row,
        "segments": offline_segments,
        "latency_ms": None,
    }
    live_wer = score_row(live_row)["utterance_wer"]
    offline_wer = score_row(offline_row)["utterance_wer"]
    return {
        "request_id": live_row["request_id"],
        "session_id": live_row["session_id"],
        "signer_id": live_row["signer_id"],
        "split": live_row["split"],
        "case": live_row["case"],
        "live_wer": live_wer,
        "offline_wer": offline_wer,
        "offline_minus_live_wer": offline_wer - live_wer,
    }


def run_capture_gap(
    manifest: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    metadata_videos: dict[str, str],
    pipeline_factory: Callable[[], Any],
    output: Path,
) -> int:
    pipe = pipeline_factory()
    base_spot = copy.deepcopy(pipe.spot)
    base_tau = float(pipe.cal["tau_accept"])
    items = {row["request_id"]: row for row in manifest}
    predictions = {(row["request_id"], row["case"]): row for row in replay_rows}
    added = 0
    for (request_id, case), live in sorted(predictions.items()):
        if (request_id, case) in completed_keys(output):
            continue
        if request_id not in items or request_id not in metadata_videos:
            continue
        video_path = (ROOT / metadata_videos[request_id]).resolve()
        if not video_path.is_relative_to(ROOT):
            raise ValueError(f"{request_id}: offline_video resolves outside the checkout")
        if not video_path.is_file():
            raise FileNotFoundError(f"{request_id}: offline video not found: {metadata_videos[request_id]}")
        video_raw = pipe.landmarks_from_video(str(video_path))
        if video_raw is None:
            raise RuntimeError(f"{request_id}: production landmark extraction returned no frames")
        pipe.spot = copy.deepcopy(base_spot)
        pipe.cal["tau_accept"] = base_tau
        effective = live.get("effective_config") or {}
        pipe.spot = copy.deepcopy(effective.get("spot", pipe.spot))
        if effective.get("tau_accept") is not None:
            pipe.cal["tau_accept"] = float(effective["tau_accept"])
        result = pipe.run_landmarks(video_raw, title=request_id, alphabet=live.get("alphabet", "off"))
        segments = []
        for segment in result.get("segments", []):
            clean_segment = dict(segment)
            clean_segment.pop("emb", None)
            segments.append(clean_segment)
        gap = compare_wer(live, segments)
        gap["offline_video"] = metadata_videos[request_id]
        gap["live_fps_actual"] = live.get("fps_actual")
        gap["offline_duration_s"] = result.get("duration_s")
        if append_result(output, gap):
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/phase2.2/live_manifest.jsonl"))
    parser.add_argument("--replay", type=Path, default=Path("cache/phase2.2/replay.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("Boom/phase2.2/capture_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("cache/phase2.2/capture_gap.jsonl"))
    args = parser.parse_args()
    manifest_path, replay_path = checkout_path(args.manifest), checkout_path(args.replay)
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    replay = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines() if line]
    videos = load_video_metadata(checkout_path(args.metadata))

    def make_pipeline():
        from modules.pipeline import SignPipeline
        return SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)

    output = checkout_path(args.output)
    count = run_capture_gap(manifest, replay, videos, make_pipeline, output)
    print(f"Appended {count} paired capture-gap rows to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
