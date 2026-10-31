"""Build a metadata-joined manifest from local sign-api diagnostic captures.

The module never guesses session or signer identity: those values must be
provided in the companion capture metadata CSV.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE_DEFAULT = Path("cache/live_requests")
SCRIPT = {
    1: ("ฉัน", "อยาก", "กิน", "ข้าว"),
    2: ("วันนี้", "ฉัน", "ดื่ม", "กาแฟ"),
    3: ("พรุ่งนี้", "เรา", "ไป", "โรงเรียน"),
    4: ("คุณ", "ชอบ", "อาหาร", "อะไร"),
    5: ("ฉัน", "หิว", "อยาก", "ขนมปัง"),
    6: ("ครอบครัว", "ฉัน", "มีความสุข"),
    7: ("ฉัน", "อยาก", "ไป", "ตลาด"),
    8: ("โรงพยาบาล", "อยู่", "ที่ไหน"),
    9: ("พรุ่งนี้", "ฉัน", "เจอ", "เพื่อน"),
    10: ("ช่วย", "ฉัน", "ด้วย"),
    11: ("ฉัน", "เหนื่อย", "อยาก", "พัก"),
    12: ("วันนี้", "หนาว"),
    13: ("ฉัน", "ชอบ", "อ่าน", "หนังสือ"),
    14: ("ห้องน้ำ", "อยู่", "ที่ไหน"),
    15: ("ฉัน", "กิน", "ผลไม้"),
    16: ("ขอบคุณ", "ลาก่อน"),
    17: ("น้องสาว", "อยู่บ้าน"),
    18: ("ฉัน", "อยาก", "ซื้อ", "น้ำ"),
    19: ("เรา", "เจอ", "เมื่อไร"),
    20: ("วันนี้", "ฉัน", "รู้สึก", "ดี"),
}
REQUIRED_METADATA = (
    "request_id", "session_id", "signer_id", "script_index", "split",
    "device", "browser", "advertised_fps",
)


def checkout_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be checkout-relative and cannot use '..': {path}")
    return ROOT / path


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _present_hand(frame: dict[str, Any], key: str) -> bool:
    points = frame.get(key) or []
    if not isinstance(points, list) or len(points) != 21:
        return False
    return any(
        isinstance(point, list) and len(point) >= 2
        and any(_finite_number(v, f"{key} coordinate") != 0 for v in point[:2])
        for point in points
    )


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = set(REQUIRED_METADATA) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata CSV missing columns: {', '.join(sorted(missing))}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            request_id = (row.get("request_id") or "").strip()
            if not request_id or request_id in rows:
                raise ValueError(f"metadata request_id is blank or duplicated: {request_id!r}")
            for column in REQUIRED_METADATA:
                if not (row.get(column) or "").strip():
                    raise ValueError(f"metadata {request_id!r} has blank {column}")
            if row["split"] not in ("tune", "eval"):
                raise ValueError(f"metadata {request_id!r} split must be tune or eval")
            try:
                index = int(row["script_index"])
            except ValueError as exc:
                raise ValueError(f"metadata {request_id!r} script_index must be an integer") from exc
            if index not in SCRIPT:
                raise ValueError(f"metadata {request_id!r} script_index must be between 1 and 20")
            advertised_fps = _finite_number(row["advertised_fps"], "advertised_fps")
            if advertised_fps <= 0:
                raise ValueError(f"metadata {request_id!r} advertised_fps must be positive")
            rows[request_id] = {**row, "script_index": str(index)}
    return rows


def _capture_metrics(record: dict[str, Any], request_id: str) -> dict[str, Any]:
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{request_id}: cache record has no frames")
    timestamps = [_finite_number(frame.get("t"), f"{request_id} frame t") for frame in frames]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{request_id}: frame timestamps must be strictly increasing")
    duration_s = (timestamps[-1] - timestamps[0]) / 1000 if len(timestamps) > 1 else 0.0
    fps = (len(timestamps) - 1) / duration_s if duration_s > 0 else 0.0
    return {
        "duration_s": duration_s,
        "fps_actual": fps,
        "n_frames": len(frames),
        "left_hand_frame_fraction": sum(_present_hand(f, "lh") for f in frames) / len(frames),
        "right_hand_frame_fraction": sum(_present_hand(f, "rh") for f in frames) / len(frames),
        "width": record.get("width"),
        "height": record.get("height"),
        "vocab": record.get("vocab"),
        "alphabet": record.get("alphabet"),
        "n_user_signs": record.get("n_user_signs", 0),
        "captured_at": record.get("time"),
        "old_result": record.get("result", {}),
    }


def build_manifest(cache_dir: Path, metadata_csv: Path) -> list[dict[str, Any]]:
    """Return manifest rows for every metadata row, requiring a matching cache."""
    metadata = _load_metadata(metadata_csv)
    cached_ids = {path.name for path in cache_dir.glob("*.json.gz")}
    unlisted = cached_ids - set(metadata)
    if unlisted:
        raise ValueError(f"cache files missing metadata rows: {', '.join(sorted(unlisted))}")
    rows = []
    for request_id, meta in metadata.items():
        capture_path = cache_dir / request_id
        if capture_path.name != request_id or not capture_path.is_file():
            raise FileNotFoundError(f"cache file not found for request_id {request_id!r}: {capture_path}")
        if not request_id.endswith(".json.gz"):
            raise ValueError(f"{request_id}: expected sign-api .json.gz cache filename")
        with gzip.open(capture_path, "rt", encoding="utf-8") as fh:
            record = json.load(fh)
        if record.get("vocab") != "conversation":
            raise ValueError(f"{request_id}: the fixed reference script requires conversation vocabulary")
        script_index = int(meta["script_index"])
        row = {
            "request_id": request_id,
            "session_id": meta["session_id"],
            "signer_id": meta["signer_id"],
            "script_index": script_index,
            "split": meta["split"],
            "reference": list(SCRIPT[script_index]),
            "device": meta["device"],
            "browser": meta["browser"],
            "advertised_fps": float(meta["advertised_fps"]),
            "offline_video": (meta.get("offline_video") or "").strip() or None,
        }
        row.update(_capture_metrics(record, request_id))
        rows.append(row)
    return sorted(rows, key=lambda row: (row["session_id"], row["script_index"], row["request_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--metadata", type=Path, default=Path("Boom/phase2.2/capture_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("cache/phase2.2/live_manifest.jsonl"))
    args = parser.parse_args()
    rows = build_manifest(checkout_path(args.cache_dir), checkout_path(args.metadata))
    output = checkout_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} manifest rows to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
