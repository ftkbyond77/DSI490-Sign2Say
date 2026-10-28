#!/usr/bin/env python3
"""Replay the staged live landmark cache through the isolated v8.1 pipeline."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "_inputs"
CACHE_DIR = INPUTS / "cache" / "live_requests"
V81_ROOT = INPUTS / "v81"
MODEL_DIR = V81_ROOT / "models" / "Jack-model"
OUTPUT_CSV = HERE / "outputs" / "per_file.csv"
REPORT = HERE / "REPORT.md"
EXPECTED_COMMIT = "99e99eee8dd6eda84c21d4e289721153dd80aa3a"
EXPECTED_COUNTS = {"live_request": 53, "module_at_commit": 22, "model_asset": 26}
TORCH_ATTEMPT_TRACEBACK = """Traceback (most recent call last):
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/replay_v81.py", line 482, in main
    rows, _ = run_replay(args.threads)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/replay_v81.py", line 438, in run_replay
    pipeline = SignPipeline(
               ^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/pipeline.py", line 98, in __init__
    self.be = make_backend(md, backend, device, **({} if backend == "onnx" else {"tagger_arch": self.cfg.get("tagger_arch", "gru")}))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/runtime.py", line 115, in make_backend
    return OnnxBackend(model_dir, device, **kw) if backend == "onnx" else TorchBackend(model_dir, device, **kw)
                                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/runtime.py", line 90, in __init__
    from .encoder import load_encoder
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/encoder.py", line 29, in <module>
    from .unisign import UniSignPose
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/unisign.py", line 20, in <module>
    from stgcn_layers import Graph, get_stgcn_chain  # noqa: E402
ModuleNotFoundError: No module named 'stgcn_layers'
"""
STATUSES = ("accepted", "uncertain", "unknown", "transition")
WORDS_OF_INTEREST = ("สวัสดี", "ขอบคุณ")
TARGET_CLUES = ("ขอบคุณ", "กรุงเทพ")
TIME_START = "20261027_074900"
TIME_END = "20261027_075399"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_staging() -> dict[str, Any]:
    ready_path = INPUTS / "STAGING_READY.json"
    manifest_path = INPUTS / "STAGING_MANIFEST.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    if not ready.get("ready") or not ready.get("all_sha256_verified"):
        raise ValueError("STAGING_READY.json does not confirm a complete SHA-verified staging set")
    if _sha256(manifest_path) != ready.get("manifest_sha256"):
        raise ValueError("STAGING_MANIFEST.json SHA-256 does not match STAGING_READY.json")
    manifest = json.loads(manifest_bytes)
    if manifest.get("commit") != EXPECTED_COMMIT:
        raise ValueError(f"staged v8.1 commit mismatch: {manifest.get('commit')!r}")

    counts = Counter(item.get("kind") for item in manifest.get("files", []))
    for kind, expected in EXPECTED_COUNTS.items():
        if counts[kind] != expected or ready.get("counts", {}).get(kind) != expected:
            raise ValueError(f"staged {kind} count mismatch: {counts[kind]} (expected {expected})")

    root = INPUTS.resolve()
    seen: set[str] = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in seen:
            raise ValueError(f"unsafe or duplicate staged manifest path: {item['path']!r}")
        seen.add(item["path"])
        path = (INPUTS / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"staged path escapes _inputs: {item['path']!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"staged input is missing: {item['path']}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"staged input SHA-256 mismatch: {item['path']}")
    return manifest


def configure_v81_imports():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 is required; found {platform.python_version()}")
    if not V81_ROOT.is_dir() or not (V81_ROOT / "modules" / "__init__.py").is_file():
        raise FileNotFoundError(f"staged v8.1 modules not found under {V81_ROOT}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"staged v8.1 model not found under {MODEL_DIR}")

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["THAISLM_MODELS"] = str(MODEL_DIR)
    sys.dont_write_bytecode = True
    repo_root = HERE.parents[1].resolve()

    def is_repo_root(entry: str) -> bool:
        try:
            return Path(entry or os.getcwd()).resolve() == repo_root
        except OSError:
            return False

    sys.path[:] = [entry for entry in sys.path if not is_repo_root(entry)]
    sys.path.insert(0, str(V81_ROOT.resolve()))

    import modules
    from modules.pipeline import SignPipeline

    package_dir = Path(modules.__file__).resolve().parent
    expected_modules = (V81_ROOT / "modules").resolve()
    if package_dir != expected_modules:
        raise ImportError(f"modules resolved outside staged v8.1 tree: {package_dir}")
    return SignPipeline


def _load_capture(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError(f"{path.name}: cache root must be a JSON object")
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path.name}: cache has no frames")
    if not isinstance(record.get("result"), dict):
        raise ValueError(f"{path.name}: cache has no result object")
    if int(record.get("width", 0)) <= 0 or int(record.get("height", 0)) <= 0:
        raise ValueError(f"{path.name}: cache width/height must be positive")
    if record.get("n_user_signs", 0):
        raise ValueError(
            f"{path.name}: cache records personal-sign count but not embeddings; "
            "cannot reproduce the user bank without inventing data"
        )
    timestamps = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or "t" not in frame:
            raise ValueError(f"{path.name}: frame {index} is missing the browser timestamp")
        try:
            timestamp = float(frame["t"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: frame {index} timestamp is not numeric") from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"{path.name}: frame {index} timestamp is not finite")
        timestamps.append(timestamp)
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{path.name}: frame timestamps are not strictly increasing")
    if record.get("vocab") != "conversation":
        raise ValueError(f"{path.name}: expected conversation vocabulary, found {record.get('vocab')!r}")
    return record


def _normalise_segment(segment: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for candidate in (segment.get("candidates") or [])[:5]:
        candidates.append(
            {
                key: candidate[key]
                for key in ("word", "score", "support", "ai_prob", "pose_score", "fused_score")
                if key in candidate
            }
        )
    return {
        "word": segment.get("nearest", segment.get("word", "")),
        "status": segment.get("status", "unknown"),
        "p": segment.get("p_correct", segment.get("p")),
        "kind": segment.get("kind", "sign"),
        "top5": candidates,
        "t0": segment.get("t0"),
        "t1": segment.get("t1"),
    }


def _rank1(segment: dict[str, Any]) -> str:
    candidates = segment.get("candidates") or segment.get("top5") or []
    if candidates:
        return str(candidates[0].get("word", ""))
    return str(segment.get("nearest", segment.get("word", "")))


def _non_transition(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [segment for segment in segments if segment.get("status") != "transition"]


def _tail_text(segments: list[dict[str, Any]]) -> str:
    tail = _non_transition(segments)[-2:]
    if not tail:
        return "—"
    chunks = []
    for segment in tail:
        word = segment.get("nearest", segment.get("word", ""))
        status = segment.get("status", "?")
        p = segment.get("p_correct", segment.get("p"))
        p_text = "?" if p is None else f"{float(p):.3f}"
        top5 = segment.get("candidates") or segment.get("top5") or []
        top5_items = []
        for item in top5[:5]:
            candidate = str(item.get("word", ""))
            metrics = []
            for key in ("score", "support"):
                value = item.get(key)
                if value is not None:
                    metrics.append(f"{key}={float(value):.3f}")
            top5_items.append(f"{candidate}({', '.join(metrics)})" if metrics else candidate)
        chunks.append(f"{word}/{status}/p={p_text} [Top5: {', '.join(top5_items) or '—'}]")
    return "<br>".join(chunks)


def _target_membership(filename: str, old_segments: list[dict[str, Any]]) -> tuple[bool, bool]:
    timestamp = filename[:15]
    in_window = TIME_START <= timestamp <= TIME_END
    tail_words = {
        str(segment.get("word", ""))
        for segment in _non_transition(old_segments)[-2:]
    }
    has_clue = bool(tail_words.intersection(TARGET_CLUES))
    return in_window, in_window and has_clue


def _make_row(path: Path, capture: dict[str, Any], result: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    old_result = capture["result"]
    old_segments = [_normalise_segment(segment) for segment in old_result.get("segments") or []]
    new_segments = [_normalise_segment(segment) for segment in result.get("segments") or []]
    in_window, target = _target_membership(path.name, old_segments)
    if target:
        cohort = "time_and_tail_clue"
    elif in_window:
        cohort = "time_only_no_tail_clue"
    else:
        cohort = ""
    old_latency = (old_result.get("timings_ms") or {}).get("total_ms")
    return {
        "file": path.name,
        "captured_at": capture.get("time", ""),
        "vocab": capture.get("vocab", ""),
        "alphabet": capture.get("alphabet", ""),
        "target_window": in_window,
        "target_clue_match": target,
        "target_cohort": cohort,
        "frame_count": len(capture.get("frames") or []),
        "old_sentence": old_result.get("sentence") or "",
        "new_sentence": result.get("sentence") or "",
        "old_latency_ms": old_latency,
        "new_latency_ms": round(elapsed_ms, 1),
        "old_segments_json": json.dumps(old_segments, ensure_ascii=False, separators=(",", ":")),
        "new_segments_json": json.dumps(new_segments, ensure_ascii=False, separators=(",", ":")),
        "old_timings_json": json.dumps(old_result.get("timings_ms") or {}, ensure_ascii=False, separators=(",", ":")),
        "new_timings_json": json.dumps(result.get("timings_ms") or {}, ensure_ascii=False, separators=(",", ":")),
    }


def _counts(rows: list[dict[str, Any]], version: str) -> Counter:
    status_counts = Counter()
    for row in rows:
        segments = json.loads(row[f"{version}_segments_json"])
        status_counts.update(segment["status"] for segment in segments)
    return status_counts


def _latencies(rows: list[dict[str, Any]], version: str) -> list[float]:
    values = []
    for row in rows:
        value = row[f"{version}_latency_ms"]
        if value is not None and value != "":
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append(number)
    return values


def _word_counts(rows: list[dict[str, Any]], version: str, cohort: str, word: str) -> tuple[int, int]:
    selected = [row for row in rows if row["target_window"] and (cohort == "time_window" or row["target_clue_match"])]
    rank1_files, accepted_files = 0, 0
    for row in selected:
        segments = json.loads(row[f"{version}_segments_json"])
        rank1 = any(_rank1(segment) == word for segment in segments)
        accepted = any(
            _rank1(segment) == word and segment.get("status") == "accepted"
            for segment in segments
        )
        rank1_files += int(rank1)
        accepted_files += int(accepted)
    return rank1_files, accepted_files


def _format_counts(counter: Counter) -> str:
    return ", ".join(f"{status} {counter.get(status, 0)}" for status in STATUSES)


def _format_mean(values: list[float]) -> str:
    return "ไม่มีข้อมูล" if not values else f"{statistics.mean(values):.1f} ms (n={len(values)})"


def write_report(rows: list[dict[str, Any]], manifest: dict[str, Any], threads: int) -> None:
    old_counts = _counts(rows, "old")
    new_counts = _counts(rows, "new")
    old_lat = _latencies(rows, "old")
    new_lat = _latencies(rows, "new")
    time_rows = [row for row in rows if row["target_window"]]
    strict_rows = [row for row in rows if row["target_clue_match"]]
    no_clue_rows = [row for row in time_rows if not row["target_clue_match"]]
    exact_phrase = sum(row["old_sentence"] == "สวัสดี ขอบคุณ" for row in time_rows)

    lines = [
        "# สังเกตการณ์ replay live landmarks: v8.1 เทียบ cache v7.2",
        "",
        "การ replay นี้เป็นการสังเกตผลของโมเดล ไม่ใช่การประเมินความถูกต้อง: ไม่มีเฉลยครบ จึงไม่คำนวณ WER หรือกล่าวอ้าง accuracy. ตัวอย่างทั้งหมดเป็นผู้ลงนามคนเดียวตามข้อมูลที่ผู้ใช้ให้มา.",
        "",
        "## วิธีและขอบเขต",
        "",
        f"- ใช้ cache จริง {len(rows)} ไฟล์ ({sum(int(row.get('frame_count', 0)) for row in rows)} frames), source modules จาก commit `{manifest['commit']}` และโมเดล `Jack-model` ใน `_inputs/` เท่านั้น (config v8.1: `hand_w=1.2`, `accept_min=0.8`); manifest และ SHA-256 ของไฟล์ staged {len(manifest['files'])} รายการตรวจผ่าน.",
        f"- แปลงเฟรมด้วย `SignPipeline.landmarks_from_browser(frames, (height, width))` ของ v8.1 ซึ่ง delegate ไป `modules.mediapipe_pose.from_browser` แบบเดียวกับ `_frames_job` ของ API; ไม่อ่าน/ถอดรหัสวิดีโอหรือภาพใบหน้า.",
        f"- ใช้ ONNX Runtime CPU (`onnxruntime {importlib.metadata.version('onnxruntime')}`), conversation vocab, face analysis เปิด, alphabet จาก cache, LLM ปิดเพื่อไม่เรียก network; warm-up แบบสังเคราะห์ก่อนจับเวลา. ORT threads={threads}.",
        f"- Python ที่ใช้จริง: {platform.python_version()} จาก host environment. พาธ `Boom/venv-sign2say/bin/python` ที่ระบุไม่มีอยู่ใน worktree นี้; จึงใช้ Python 3.11.8 ที่มีอยู่แทน. รันด้วย `python3 -B Boom/v81_replay/replay_v81.py --threads 4` และ session-local `onnxruntime==1.23.2` ใน `PYTHONPATH` (ใน venv ที่มีแพ็กเกจนี้แล้ว ไม่ต้องเติม target path).",
        "- Cache ไม่มี request option `llm`, `face_question`, `face_baseline`, `spell_only` หรือ user-bank embeddings (มีเพียง `n_user_signs`; ทั้ง 53 ไฟล์เป็น 0). สมมติ `face_question=false`, `face_baseline=null`, `spell_only=false`, ไม่มี personal signs; ปิด LLM แบบตั้งใจเพื่อไม่ใช้บริการภายนอก. ไม่ประมาณข้อมูล landmark; ใช้ frame fields ที่ cache มี และ adapter จะเติมศูนย์ให้ fields ที่ว่างตาม implementation.",
        "- latency v8.1 วัด wall-clock ต่อรายการตั้งแต่แปลง browser frames จน `run_landmarks` คืนผล ไม่รวมสร้าง pipeline/warm-up/เขียน CSV; ค่า v7.2 นำจาก `result.timings_ms.total_ms` ใน cache.",
        "",
        "## กลุ่มเวลา 20261027 07:49–07:53",
        "",
        f"มี {len(time_rows)} ไฟล์ในช่วงเวลาที่กำหนด; {len(strict_rows)} ไฟล์ผ่านเงื่อนไขเพิ่มว่า segment หนึ่งใน 2 segment ที่ไม่ใช่ transition ตอนท้ายของผล v7.2 มีคำ `ขอบคุณ` หรือ `กรุงเทพ`. ค่า `result.sentence` เก่าที่ตรงข้อความ `สวัสดี ขอบคุณ` มี {exact_phrase}/{len(time_rows)} ไฟล์—ดังนั้นชื่อวลีนี้ไม่ถือเป็น label จริง.",
        "",
        "| กลุ่ม | คำ | v7.2 Top-1 อย่างน้อย 1 segment | v7.2 Accepted | v8.1 Top-1 อย่างน้อย 1 segment | v8.1 Accepted |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cohort, label, subset in (
        ("time_window", "ทั้งหมดในช่วงเวลา", time_rows),
        ("strict", "ช่วงเวลา + clue ท้าย", strict_rows),
    ):
        for word in WORDS_OF_INTEREST:
            old_rank, old_accept = _word_counts(rows, "old", cohort, word)
            new_rank, new_accept = _word_counts(rows, "new", cohort, word)
            lines.append(
                f"| {label} (n={len(subset)}) | {word} | {old_rank}/{len(subset)} | {old_accept}/{len(subset)} | {new_rank}/{len(subset)} | {new_accept}/{len(subset)} |"
            )
    lines.extend(
        [
            "",
            "Top-1 นับจาก candidate อันดับแรกของแต่ละ segment; Accepted นับเฉพาะ segment ที่ status เป็น `accepted` และคำอันดับแรกตรงคำเป้าหมาย. นับเป็นจำนวนไฟล์ที่มีอย่างน้อยหนึ่ง segment เช่นนั้น ไม่ใช่จำนวน segment และไม่เท่ากับความถูกต้องเมื่อไม่มีเฉลย.",
            "ใน Top-5: `score` คือ candidate score จาก encoder; `support` คือ score หลัง fusion ที่เรียง candidate ใน v8.1 (v7.2 cache ไม่มี `support`).",
            "",
            "### รายชื่อและผลช่วงท้าย",
            "",
            "| ไฟล์ | v7.2 sentence | v7.2 สอง segment ท้าย (word/status/p; Top-5) | v8.1 sentence | v8.1 สอง segment ท้าย (word/status/p; Top-5) |",
            "|---|---|---|---|---|",
        ]
    )
    for row in time_rows:
        old_segments = json.loads(row["old_segments_json"])
        new_segments = json.loads(row["new_segments_json"])
        old_tail = _tail_text(old_segments)
        new_tail = _tail_text(new_segments)
        tail_label = "*" if row["target_clue_match"] else "†"
        lines.append(
            f"| {tail_label} `{row['file']}` | {row['old_sentence'] or '—'} | {old_tail} | {row['new_sentence'] or '—'} | {new_tail} |"
        )
    if no_clue_rows:
        excluded = ", ".join(
            f"`{row['file']}` (ท้าย v7.2: {_tail_text(json.loads(row['old_segments_json']))})"
            for row in no_clue_rows
        )
        lines.extend(["", f"`*` = time window + clue; `†` = อยู่ในช่วงเวลาแต่ไม่ผ่าน clue และไม่รวมในกลุ่ม strict: {excluded}."])
    lines.extend(
        [
            "",
            "## สรุป 53 ไฟล์",
            "",
            "| รุ่น | Segments: accepted | uncertain | unknown | transition | latency เฉลี่ย |",
            "|---|---:|---:|---:|---:|---:|",
            f"| v7.2 (cache) | {old_counts.get('accepted', 0)} | {old_counts.get('uncertain', 0)} | {old_counts.get('unknown', 0)} | {old_counts.get('transition', 0)} | {_format_mean(old_lat)} |",
            f"| v8.1 (replay) | {new_counts.get('accepted', 0)} | {new_counts.get('uncertain', 0)} | {new_counts.get('unknown', 0)} | {new_counts.get('transition', 0)} | {_format_mean(new_lat)} |",
            "",
            f"ผลต่อไฟล์และ Top-5 แบบเต็มอยู่ใน [`outputs/per_file.csv`](outputs/per_file.csv). จำนวน segments ใน CSV เป็นผลที่แต่ละ pipeline ส่งคืน; transition แสดงแยกและไม่นำไปนับเป็นคำ Top-1 ในรายงานช่วงท้าย.",
            "",
            "## ข้อจำกัด",
            "",
            "- เป็นการสังเกต cache จากผู้ลงนามคนเดียวตามที่แจ้ง ไม่ใช่ตัวอย่างอิสระหลายคน และไม่ได้ยืนยันว่า gesture ที่ทำจริงตรงกับ segment ใด.",
            "- cache เก็บผลเดิมและ landmarks แต่ไม่เก็บทุก option ที่ส่งใน request; เงื่อนไข offline ข้างต้นทำให้ sentence layer ของ v8.1 ไม่ใช่การ replay LLM ใน production.",
            "- ไม่มีการอ่านชุดข้อมูล TSL51 หรือ Eval-Test, ไม่มีการเทรน, Docker หรือการเรียกบริการ inference/LLM ภายนอก; ติดตั้ง ONNX Runtime ตามเวอร์ชันใน requirements ลง session-local package target นอก repo. staged modules/model/cache ไม่ถูกแก้ไข.",
            "",
            "### Traceback จากการลอง Torch backend ก่อนหน้า (แก้โดยใช้ ONNX backend ของ pipeline)",
            "",
            "Torch backend ต้องการ vendored `third_party/unisign/stgcn_layers.py` ซึ่งไม่ได้อยู่ในชุด staging ที่ SHA-verified; ไม่ได้นำโค้ดจาก repo หลักหรือ external checkout มาปน. เปลี่ยนมาใช้ ONNX backend ที่ pipeline รองรับและโมเดล Jack-model มีอยู่ครบ:",
            "",
            "```text",
            TORCH_ATTEMPT_TRACEBACK.rstrip(),
            "```",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _warm_pipeline(pipeline) -> None:
    import numpy as np
    from modules.parts import prep

    rng = np.random.RandomState(0)
    pipeline.be.embed(
        [
            prep(
                rng.uniform(0.3, 0.7, (30, 133, 2)).astype(np.float32),
                np.ones((30, 133), np.float32),
            )
        ]
    )


def _write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "captured_at",
        "vocab",
        "alphabet",
        "target_window",
        "target_clue_match",
        "target_cohort",
        "frame_count",
        "old_sentence",
        "new_sentence",
        "old_latency_ms",
        "new_latency_ms",
        "old_segments_json",
        "new_segments_json",
        "old_timings_json",
        "new_timings_json",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_replay(threads: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = verify_staging()
    SignPipeline = configure_v81_imports()
    os.environ["ORT_THREADS"] = str(threads)
    cache_files = sorted(CACHE_DIR.glob("*.json.gz"))
    if len(cache_files) != EXPECTED_COUNTS["live_request"]:
        raise ValueError(f"expected exactly 53 cache files, found {len(cache_files)}")
    if manifest.get("execution") != "Inputs only; do not import modules from external checkout. Use _inputs/v81 as PYTHONPATH. Do not write bytecode into staging.":
        raise ValueError("staging manifest execution boundary did not match the approved input-only instructions")

    vocabularies = set()
    captures = []
    for path in cache_files:
        capture = _load_capture(path)
        vocabularies.add(capture["vocab"])
        captures.append((path, capture))
    if vocabularies != {"conversation"}:
        raise ValueError(f"expected all captures to use conversation vocabulary, found {sorted(vocabularies)}")

    pipeline = SignPipeline(
        model_dir=MODEL_DIR,
        backend="onnx",
        device="cpu",
        vocab="conversation",
        use_llm=False,
        use_face=True,
    )
    _warm_pipeline(pipeline)

    rows = []
    for index, (path, capture) in enumerate(captures, 1):
        started = time.perf_counter()
        raw = pipeline.landmarks_from_browser(
            capture["frames"],
            (int(capture["height"]), int(capture["width"])),
        )
        result = pipeline.run_landmarks(
            raw,
            title=path.name,
            face_question=False,
            face_baseline=None,
            user_bank=None,
            alphabet=capture.get("alphabet", "en"),
            spell_only=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        rows.append(_make_row(path, capture, result, elapsed_ms))
        print(f"[{index:02d}/{len(captures)}] {path.name} {elapsed_ms:.1f} ms", flush=True)

    if len(rows) != EXPECTED_COUNTS["live_request"]:
        raise RuntimeError(f"replay produced {len(rows)} rows; expected 53")
    _write_csv(rows)
    write_report(rows, manifest, threads)
    return rows, manifest


def run_port(args):
    """Use PR6 parsing/row schema with the ported code; inputs remain read-only."""
    global OUTPUT_CSV
    root = HERE.parents[1].resolve()
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    os.environ['ORT_THREADS'] = str(args.threads)
    import modules.pipeline as module
    if Path(module.__file__).resolve() != root/'modules/pipeline.py':
        raise ImportError(f'Wrong port import: {module.__file__}')
    if args.backend == 'torch':
        import torch
        torch.set_num_threads(args.threads)
    paths = sorted(args.cache.glob('*.json.gz'))
    if len(paths) != 53:
        raise ValueError(f'Expected 53 frozen recordings, got {len(paths)}')
    if args.output.exists():
        raise FileExistsError('Refusing duplicate run into existing output directory')
    args.output.mkdir(parents=True)
    OUTPUT_CSV = args.output/'per_file.csv'
    pipeline = module.SignPipeline(model_dir=args.models, backend=args.backend, device='cpu',
                                  vocab='conversation', use_llm=False, use_face=True)
    manifest = dict(backend=args.backend, module=str(module.__file__), models=str(args.models),
                    python=sys.executable, threads=args.threads, version=pipeline.cfg.get('version'),
                    tau_accept=pipeline.cal.get('tau_accept'), hand_w=pipeline.spot.get('hand_w'),
                    input_hashes={p.name:_sha256(p) for p in paths},
                    model_hashes={str(p.relative_to(args.models)):_sha256(p) for p in args.models.rglob('*') if p.is_file()},
                    assumptions=dict(llm=False,use_face=True,face_question=False,face_baseline=None,user_bank=None,spell_only=False),
                    notes='No labels; observational replay. No synthetic warm-up. Latency excludes model load.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    rows=[]
    with (args.output/'metadata.jsonl').open('w',encoding='utf-8') as meta:
        for index,path in enumerate(paths,1):
            capture=_load_capture(path)
            start=time.perf_counter()
            raw=pipeline.landmarks_from_browser(capture['frames'],(int(capture['height']),int(capture['width'])))
            result=pipeline.run_landmarks(raw,title=path.name,face_question=False,face_baseline=None,user_bank=None,
                                          alphabet=capture.get('alphabet','en'),spell_only=False)
            elapsed=(time.perf_counter()-start)*1000
            for field in ('sentence','segments','face','timings_ms','hand_canonicalization','mirrored_to_right_hand','model_version'):
                if field not in result: raise ValueError(f'Response missing {field}')
            meta.write(json.dumps(dict(file=path.name,model_version=result['model_version'],
                       hand_canonicalization=result['hand_canonicalization'],mirrored_to_right_hand=result['mirrored_to_right_hand']),ensure_ascii=False)+'\n');meta.flush()
            rows.append(_make_row(path,capture,result,elapsed))
            _write_csv(rows)
            print(f'{args.backend} {index}/53 {path.name} {elapsed:.1f} ms',flush=True)
    for name,h in manifest['model_hashes'].items():
        if _sha256(args.models/name)!=h:raise RuntimeError(f'Model changed during run: {name}')
    (args.output/'DONE.json').write_text(json.dumps(dict(complete=True,requests=len(rows),backend=args.backend)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--ported', action='store_true', help='Use this worktree modules with PR6 input/row protocol')
    parser.add_argument('--backend', choices=['onnx','torch'], default='onnx')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.ported:
        if not all((args.models,args.cache,args.output)):
            parser.error('--ported requires --models, --cache, --output')
        if args.threads < 1: parser.error('--threads must be positive')
        run_port(args)
        return 0
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    try:
        rows, _ = run_replay(args.threads)
    except Exception:
        error = traceback.format_exc()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            "# v8.1 live replay ไม่สำเร็จ\n\n"
            "การรันหยุดก่อนสรุปผล; ไม่มีการสร้างผลสำเร็จปลอม. Traceback:\n\n"
            "```text\n"
            f"{error}```\n",
            encoding="utf-8",
        )
        print(error, file=sys.stderr, end="")
        return 1
    print(f"Completed {len(rows)} cache replays. Wrote {OUTPUT_CSV} and {REPORT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
