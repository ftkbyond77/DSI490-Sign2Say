"""Replay local diagnostic landmarks through the production SignPipeline.

Replay receives irregular browser timestamps unchanged; the production
landmarks_from_browser -> run_landmarks path performs the timestamp-aware
standardisation and 25-fps resampling used by live inference.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def checkout_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be checkout-relative and cannot use '..': {path}")
    return ROOT / path


def apply_case(pipe: Any, patch: dict[str, Any]) -> dict[str, Any]:
    """Apply only declared tuning keys and return an explicit effective config."""
    for key, value in patch.items():
        if key == "tau_accept":
            pipe.cal["tau_accept"] = float(value)
        elif key.startswith("spot."):
            path = key.split(".")[1:]
            target = pipe.spot
            for part in path[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise ValueError(f"unknown spotting config path: {key}")
                target = target[part]
            if path[-1] not in target:
                raise ValueError(f"unknown spotting config key: {key}")
            target[path[-1]] = value
        else:
            raise ValueError(f"unsupported one-variable config key: {key}")
    return {"tau_accept": float(pipe.cal["tau_accept"]), "spot": copy.deepcopy(pipe.spot)}


def completed_keys(output: Path) -> set[tuple[str, str]]:
    if not output.exists():
        return set()
    keys = set()
    with output.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed replay JSONL at line {line_number}; refusing unsafe resume") from exc
            keys.add((row["request_id"], row["case"]))
    return keys


def append_result(output: Path, row: dict[str, Any]) -> bool:
    """Append one result if not already present; return False for a resumed row."""
    key = (row["request_id"], row["case"])
    if key in completed_keys(output):
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        fh.flush()
    return True


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def replay_rows(
    manifest: list[dict[str, Any]],
    cache_dir: Path,
    output: Path,
    cases: dict[str, dict[str, Any]],
    pipeline_factory: Callable[[], Any],
) -> int:
    """Replay manifest rows. pipeline_factory is injectable for synthetic tests."""
    pipe = pipeline_factory()
    baseline_spot = copy.deepcopy(pipe.spot)
    baseline_tau = float(pipe.cal["tau_accept"])
    count = 0
    for item in manifest:
        cache_path = cache_dir / item["request_id"]
        with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
            capture = json.load(fh)
        frames = capture.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{item['request_id']}: cache record has no frames")
        if capture.get("vocab") != "conversation":
            raise ValueError(f"{item['request_id']}: script references require conversation vocabulary")
        if capture.get("n_user_signs", 0):
            raise ValueError(
                f"{item['request_id']}: cache records only the count of personal signs, not their embeddings; "
                "disable personal sign memory for this protocol capture"
            )
        for case_name, patch in cases.items():
            if (item["request_id"], case_name) in completed_keys(output):
                continue
            pipe.spot = copy.deepcopy(baseline_spot)
            pipe.cal["tau_accept"] = baseline_tau
            effective = apply_case(pipe, patch)
            started = time.perf_counter()
            raw = pipe.landmarks_from_browser(frames, (capture["height"], capture["width"]))
            result = pipe.run_landmarks(raw, title=item["request_id"], alphabet=capture.get("alphabet", "off"))
            elapsed_ms = (time.perf_counter() - started) * 1000
            segments = []
            for segment in result.get("segments", []):
                segment = dict(segment)
                segment.pop("emb", None)
                segments.append(segment)
            record = {
                **item,
                "case": case_name,
                "effective_config": effective,
                "replay_assumptions": {
                    "vocab": "conversation",
                    "llm": False,
                    "face": False,
                    "spell_only": False,
                    "personal_sign_memory": False,
                    "alphabet": capture.get("alphabet", "off"),
                },
                "latency_ms": elapsed_ms,
                "prediction": result.get("sentence", ""),
                "segments": segments,
                "result_metrics": {
                    "duration_s": result.get("duration_s"),
                    "signer_visible_frac": result.get("signer_visible_frac"),
                    "timings_ms": result.get("timings_ms"),
                },
            }
            if append_result(output, record):
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/phase2.2/live_manifest.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/live_requests"))
    parser.add_argument("--plan", type=Path, default=Path("Boom/phase2.2/plan.json"))
    parser.add_argument("--output", type=Path, default=Path("cache/phase2.2/replay.jsonl"))
    args = parser.parse_args()
    manifest_path, cache_dir, plan_path, output = tuple(
        checkout_path(path) for path in (args.manifest, args.cache_dir, args.plan, args.output)
    )
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if any(row["split"] not in ("tune", "eval") for row in manifest):
        raise ValueError("manifest has an invalid split")

    def make_pipeline():
        from modules.pipeline import SignPipeline
        pipeline = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)
        return pipeline

    count = replay_rows(manifest, cache_dir, output, plan["cases"], make_pipeline)
    print(f"Appended {count} new replay rows to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
