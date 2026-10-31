"""Compute live utterance, corpus, session and paired-bootstrap metrics."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def checkout_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be checkout-relative and cannot use '..': {path}")
    return ROOT / path


def align(reference: list[str], hypothesis: list[str]) -> list[tuple[int | None, int | None, str]]:
    """Minimum-edit alignment; diagonal transitions win ties, as in Phase 2.1."""
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    result = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            result.append((i - 1, j - 1, "C" if reference[i - 1] == hypothesis[j - 1] else "S"))
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            result.append((i - 1, None, "D"))
            i -= 1
        else:
            result.append((None, j - 1, "I"))
            j -= 1
    return list(reversed(result))


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    reference = list(row["reference"])
    segments = [s for s in row.get("segments", []) if s.get("status") != "transition"]
    hypothesis = [s.get("nearest") or s.get("word") or "" for s in segments]
    alignment = align(reference, hypothesis)
    counts = {op: sum(item[2] == op for item in alignment) for op in ("S", "D", "I", "C")}
    correct_pred = {j for i, j, op in alignment if j is not None and op == "C"}
    accepted_indices = {j for j, segment in enumerate(segments) if segment.get("status") == "accepted"}
    accepted_correct = len(accepted_indices & correct_pred)
    accepted = len(accepted_indices)
    n_ref, n_pred = len(reference), len(hypothesis)
    if n_ref == 0:
        raise ValueError(f"{row.get('request_id')}: reference must not be empty")
    latency = row.get("latency_ms")
    result_metrics = row.get("result_metrics") or {}
    hand_fill = row.get("hand_fill_metrics") or {}
    return {
        "request_id": row["request_id"],
        "session_id": row["session_id"],
        "signer_id": row["signer_id"],
        "split": row["split"],
        "case": row["case"],
        "S": counts["S"],
        "D": counts["D"],
        "I": counts["I"],
        "C": counts["C"],
        "n_ref": n_ref,
        "n_pred": n_pred,
        "utterance_wer": (counts["S"] + counts["D"] + counts["I"]) / n_ref,
        "count_over": max(0, n_pred - n_ref),
        "count_under": max(0, n_ref - n_pred),
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "accepted_precision": accepted_correct / accepted if accepted else None,
        "false_accept_count": accepted - accepted_correct,
        "false_accept_rate": (accepted - accepted_correct) / n_ref,
        "left_hand_detection_rate": row.get("left_hand_frame_fraction"),
        "right_hand_detection_rate": row.get("right_hand_frame_fraction"),
        "left_hand_after_fill_rate": hand_fill.get("left_hand_after_fill_rate"),
        "right_hand_after_fill_rate": hand_fill.get("right_hand_after_fill_rate"),
        "left_hand_imputed_frame_fraction": hand_fill.get("left_hand_imputed_frame_fraction"),
        "right_hand_imputed_frame_fraction": hand_fill.get("right_hand_imputed_frame_fraction"),
        "fps_actual": row.get("fps_actual"),
        "end_to_end_latency_ms": latency if latency is not None else (result_metrics.get("timings_ms") or {}).get("total_ms"),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return case-level macro/corpus metrics and per-session macro WER."""
    scored = [score_row(row) for row in rows]
    by_session: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_session[(row["split"], row["session_id"], row["case"])].append(row)
    session_rows = []
    for (split, session, case), group in sorted(by_session.items()):
        n_ref = sum(r["n_ref"] for r in group)
        session_rows.append({
            "split": split,
            "session_id": session,
            "signer_id": group[0]["signer_id"],
            "case": case,
            "utterances": len(group),
            "macro_wer": _mean([r["utterance_wer"] for r in group]),
            "S": sum(r["S"] for r in group),
            "D": sum(r["D"] for r in group),
            "I": sum(r["I"] for r in group),
            "n_ref": n_ref,
            "corpus_wer": (sum(r["S"] + r["D"] + r["I"] for r in group) / n_ref) if n_ref else None,
            "accepted": sum(r["accepted"] for r in group),
            "accepted_correct": sum(r["accepted_correct"] for r in group),
            "false_accept_count": sum(r["false_accept_count"] for r in group),
            "mean_left_hand_detection_rate": _mean([r["left_hand_detection_rate"] for r in group if r["left_hand_detection_rate"] is not None]),
            "mean_right_hand_detection_rate": _mean([r["right_hand_detection_rate"] for r in group if r["right_hand_detection_rate"] is not None]),
            "mean_left_hand_after_fill_rate": _mean([r["left_hand_after_fill_rate"] for r in group if r["left_hand_after_fill_rate"] is not None]),
            "mean_right_hand_after_fill_rate": _mean([r["right_hand_after_fill_rate"] for r in group if r["right_hand_after_fill_rate"] is not None]),
            "mean_left_hand_imputed_frame_fraction": _mean([r["left_hand_imputed_frame_fraction"] for r in group if r["left_hand_imputed_frame_fraction"] is not None]),
            "mean_right_hand_imputed_frame_fraction": _mean([r["right_hand_imputed_frame_fraction"] for r in group if r["right_hand_imputed_frame_fraction"] is not None]),
            "mean_fps_actual": _mean([r["fps_actual"] for r in group if r["fps_actual"] is not None]),
            "mean_end_to_end_latency_ms": _mean([r["end_to_end_latency_ms"] for r in group if r["end_to_end_latency_ms"] is not None]),
        })
    case_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        case_groups[(row["split"], row["case"])].append(row)
    case_rows = []
    for (split, case), group in sorted(case_groups.items()):
        n_ref = sum(r["n_ref"] for r in group)
        accepted = sum(r["accepted"] for r in group)
        correct = sum(r["accepted_correct"] for r in group)
        edits = sum(r["S"] + r["D"] + r["I"] for r in group)
        case_rows.append({
            "split": split,
            "case": case,
            "utterances": len(group),
            "macro_wer": _mean([r["utterance_wer"] for r in group]),
            "corpus_wer": edits / n_ref if n_ref else None,
            "S": sum(r["S"] for r in group),
            "D": sum(r["D"] for r in group),
            "I": sum(r["I"] for r in group),
            "n_ref": n_ref,
            "count_over": sum(r["count_over"] for r in group),
            "count_under": sum(r["count_under"] for r in group),
            "accepted": accepted,
            "accepted_correct": correct,
            "accepted_precision": correct / accepted if accepted else None,
            "false_accept_count": sum(r["false_accept_count"] for r in group),
            "false_accept_rate": sum(r["false_accept_count"] for r in group) / n_ref if n_ref else None,
            "mean_left_hand_detection_rate": _mean([r["left_hand_detection_rate"] for r in group if r["left_hand_detection_rate"] is not None]),
            "mean_right_hand_detection_rate": _mean([r["right_hand_detection_rate"] for r in group if r["right_hand_detection_rate"] is not None]),
            "mean_left_hand_after_fill_rate": _mean([r["left_hand_after_fill_rate"] for r in group if r["left_hand_after_fill_rate"] is not None]),
            "mean_right_hand_after_fill_rate": _mean([r["right_hand_after_fill_rate"] for r in group if r["right_hand_after_fill_rate"] is not None]),
            "mean_left_hand_imputed_frame_fraction": _mean([r["left_hand_imputed_frame_fraction"] for r in group if r["left_hand_imputed_frame_fraction"] is not None]),
            "mean_right_hand_imputed_frame_fraction": _mean([r["right_hand_imputed_frame_fraction"] for r in group if r["right_hand_imputed_frame_fraction"] is not None]),
            "mean_fps_actual": _mean([r["fps_actual"] for r in group if r["fps_actual"] is not None]),
            "mean_end_to_end_latency_ms": _mean([r["end_to_end_latency_ms"] for r in group if r["end_to_end_latency_ms"] is not None]),
        })
    return case_rows, session_rows


def paired_bootstrap(
    session_rows: list[dict[str, Any]],
    case: str,
    split: str = "tune",
    replicates: int = 10_000,
    seed: int = 490,
) -> dict[str, Any]:
    """Percentile CI for candidate-minus-baseline session macro-WER difference."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    selected = [r for r in session_rows if r["split"] == split and r["case"] in ("baseline", case)]
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in selected:
        paired[row["session_id"]][row["case"]] = float(row["macro_wer"])
    differences = [
        pair[case] - pair["baseline"]
        for pair in paired.values()
        if case in pair and "baseline" in pair
    ]
    if not differences:
        raise ValueError(f"no paired {split} sessions for baseline and {case}")
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(replicates)
    )
    return {
        "case": case,
        "split": split,
        "paired_sessions": len(differences),
        "mean_delta_macro_wer": _mean(differences),
        "ci_95_low": samples[int(0.025 * (replicates - 1))],
        "ci_95_high": samples[int(0.975 * (replicates - 1))],
        "bootstrap_replicates": replicates,
        "seed": seed,
    }


def selection_gates(session_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report tune-only gates and reject signer/session overlap with eval."""
    tune = [r for r in session_rows if r["split"] == "tune"]
    evaluation = [r for r in session_rows if r["split"] == "eval"]
    tune_sessions = {r["session_id"] for r in tune}
    eval_sessions = {r["session_id"] for r in evaluation}
    eval_signers = {r["signer_id"] for r in evaluation}
    gates = []
    baseline = {r["session_id"]: r for r in tune if r["case"] == "baseline"}
    for case in sorted({r["case"] for r in tune} - {"baseline"}):
        candidate = {r["session_id"]: r for r in tune if r["case"] == case}
        paired_ids = sorted(set(baseline) & set(candidate))
        deltas = [candidate[s]["macro_wer"] - baseline[s]["macro_wer"] for s in paired_ids]
        summary = next((r for r in case_rows if r["split"] == "tune" and r["case"] == case), {})
        precision = summary.get("accepted_precision")
        complete_tune_pairing = set(paired_ids) == tune_sessions
        distinct_holdout = bool(evaluation) and not (tune_sessions & eval_sessions) and not any(
            candidate[s]["signer_id"] in eval_signers for s in paired_ids
        )
        max_delta = max(deltas) if deltas else None
        mean_delta = _mean(deltas)
        gates.append({
            "case": case,
            "paired_tune_sessions": len(deltas),
            "complete_tune_pairing": complete_tune_pairing,
            "max_session_delta": max_delta,
            "mean_session_delta": mean_delta,
            "accepted_precision": precision,
            "eval_separation_valid": distinct_holdout,
            "pass_gate": bool(
                deltas and complete_tune_pairing and max_delta <= 0.003
                and mean_delta < -0.002 and precision is not None and precision >= 0.90
            ),
            "promotion_eligible": bool(
                deltas and complete_tune_pairing and distinct_holdout and max_delta <= 0.003
                and mean_delta < -0.002 and precision is not None and precision >= 0.90
            ),
        })
    return gates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("cache/phase2.2/replay.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/phase2.2/metrics"))
    args = parser.parse_args()
    input_path, output_dir = checkout_path(args.input), checkout_path(args.output_dir)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    cases, sessions = aggregate(rows)
    gates = selection_gates(sessions, cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "case_metrics.csv", cases)
    _write_csv(output_dir / "session_metrics.csv", sessions)
    _write_csv(output_dir / "selection_gates.csv", gates)
    intervals = []
    for case in sorted({row["case"] for row in sessions} - {"baseline"}):
        if any(row["split"] == "tune" and row["case"] == case for row in sessions):
            intervals.append(paired_bootstrap(sessions, case))
    (output_dir / "paired_bootstrap.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    print(f"Wrote metrics for {len(rows)} replay rows to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
