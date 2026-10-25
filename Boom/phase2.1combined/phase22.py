"""Inspect the actual Phase 2.2 baseline without inference or model mutation."""
from pathlib import Path
import json
import subprocess
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def read_json(path: Path) -> dict:
    """Read a UTF-8 configuration."""
    return json.loads(path.read_text(encoding="utf-8"))
def plan() -> dict:
    """Return the inactive experiment specification."""
    return read_json(HERE / "experiment_plan.json")
def baseline() -> dict:
    """Inspect real config and effective conversation acceptance threshold."""
    cfg = read_json(ROOT / "models/config.json")
    spotting = read_json(ROOT / "models/spotting.json")
    calibration = read_json(ROOT / "models/segment_calibration.json")
    cal = calibration.get("conversation", calibration)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"root": str(ROOT), "version": cfg.get("version"), "revision": revision,
            "working_tree_dirty": bool(dirty.strip()), "spotting": spotting.get("conversation", spotting),
            "effective_tau_accept": max(float(cal.get("tau_accept", 0)), float(spotting.get("accept_min", .8))),
            "inference_run": False}
if __name__ == "__main__":
    print(json.dumps(baseline(), ensure_ascii=False, indent=2))
