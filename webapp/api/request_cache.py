"""Local gzip cache writer for summarized live-request diagnostics."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any


def save_request(
    save_dir: str,
    req: Any,
    out: dict,
    hand_canonicalization: dict | None,
    hand_gap_s: float,
) -> None:
    if not save_dir:
        return
    try:
        timestamps = [float(frame.t) for frame in req.frames]
        duration_ms = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        fps_actual = (len(timestamps) - 1) * 1000 / duration_ms if duration_ms > 0 else None
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(
            time=time.strftime("%Y-%m-%d %H:%M:%S"),
            width=req.width,
            height=req.height,
            vocab=req.vocab,
            alphabet=req.alphabet,
            n_user_signs=len(req.user_bank),
            n_frames=len(req.frames),
            fps_actual=fps_actual,
            hand_gap_s=hand_gap_s,
            frames=[frame.model_dump() for frame in req.frames],
            result={
                key: out.get(key)
                for key in ("sentence", "language", "face", "timings_ms", "duration_s", "mirrored_to_right_hand", "model_version")
            },
        )
        perf_mode = getattr(req, "perf_mode", None)
        client_fps = getattr(req, "client_fps", None)
        if perf_mode is not None:
            record["perf_mode"] = perf_mode
        if client_fps is not None:
            record["client_fps"] = client_fps
        if hand_canonicalization is not None:
            record["result"]["hand_canonicalization"] = hand_canonicalization
        # Keep embeddings out of the cache; the response still carries them for user teaching.
        record["result"]["segments"] = [
            {key: value for key, value in segment.items() if key != "emb"}
            for segment in out.get("segments") or []
        ]
        filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.json.gz"
        with gzip.open(directory / filename, "wt", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
    except Exception:  # noqa: BLE001 (diagnostic caching must never break a request)
        pass
