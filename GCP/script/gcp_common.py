"""Shared helpers for the ThaiSLM Vertex AI jobs (read .env, call gcloud, GCS layout, cost estimate).

Layout on GCS (bucket = .env GCS_BUCKET, region asia-southeast1):
  gs://<bucket>/thaislm/code/<tag>.tar.gz      source snapshot used by a job
  gs://<bucket>/thaislm/raw_data/              raw videos the pose job reads (same layout as s3://dsi490-lake-signdata/raw_data/)
  gs://<bucket>/thaislm/pose_raw/              RTMW-X output of the pose job (one npz per video; reruns skip finished videos)
  gs://<bucket>/thaislm/data/                  training shards built locally from data_prep/ (pose_store.npz, clips.parquet, concepts.json)
  gs://<bucket>/thaislm/models/                pretrained weights (Uni-Sign base, rtmlib ONNX checkpoints)
  gs://<bucket>/thaislm/runs/<run_id>/         everything a job writes (logs, checkpoints, reports)
Inside a Vertex custom job the bucket is mounted at /gcs/<bucket> (Cloud Storage FUSE), so the job reads/writes plain paths.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "GCP" / "result"
PREFIX = "thaislm"

# prebuilt Vertex AI training container: PyTorch 2.4 + CUDA 12.x (L4 / Ada supported), same region family as the job
IMAGE = "asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

# Hardware profiles. .env asks for 1×L4 on g2-standard-4, but project gpu-job has Vertex training quota 0 for L4 (on-demand and
# Spot, every region; checked 2026-09-14). Quota that exists: Spot A100 ×8 / T4 ×1 in asia-southeast1 → default = Spot A100.
PROFILES = {
    "l4": dict(machine="g2-standard-4", gpu_type="NVIDIA_L4", gpu_count=1, spot=False),
    "a100_spot": dict(machine="a2-highgpu-1g", gpu_type="NVIDIA_TESLA_A100", gpu_count=1, spot=True),
    "t4_spot": dict(machine="n1-standard-8", gpu_type="NVIDIA_TESLA_T4", gpu_count=1, spot=True),
}
DEFAULT_PROFILE = os.environ.get("VERTEX_PROFILE", "a100_spot")
# approximate USD/h in asia-southeast1 incl. Vertex training fee (Spot ≈ 30–40 % of on-demand) — estimate only
PRICE_PER_HOUR = {("g2-standard-4", "NVIDIA_L4", False): 1.07, ("a2-highgpu-1g", "NVIDIA_TESLA_A100", True): 1.55,
                  ("a2-highgpu-1g", "NVIDIA_TESLA_A100", False): 4.6, ("n1-standard-8", "NVIDIA_TESLA_T4", True): 0.35}


def load_env() -> dict:
    env = {}
    p = ROOT / ".env"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.split("#", 1)[0].strip()
    return env


def cfg() -> dict:
    e = load_env()
    project = e.get("GCP_PROJECT_ID", "gpu-job")
    sa = e.get("SERVICE_ACCOUNT", "")
    # GCP project ids are lowercase; the service-account domain carries the real id (…@gpu-job.iam.gserviceaccount.com)
    if sa and "@" in sa:
        project = sa.split("@", 1)[1].split(".", 1)[0]
    pname = os.environ.get("VERTEX_PROFILE", DEFAULT_PROFILE)
    prof = PROFILES[pname]
    return dict(project=project.lower(), region=os.environ.get("VERTEX_REGION") or e.get("GCP_REGION", "asia-southeast1"), bucket=e.get("GCS_BUCKET"),
                job_name=e.get("VERTEX_JOB_NAME", "thai-sign-training"), profile=pname, env_gpu=f"{e.get('GPU_TYPE')}@{e.get('MACHINE_TYPE')}",
                service_account=sa, **prof)


def gcloud_bin() -> str:
    for c in (shutil.which("gcloud"), shutil.which("gcloud.cmd"),
              str(Path.home() / "AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd")):
        if c and Path(c).exists():
            return c
    sys.exit("gcloud not found — install the Google Cloud SDK and run `gcloud auth login`")


def gcloud(*args, capture=True, check=True) -> str:
    c = cfg()
    cmd = [gcloud_bin(), *args, f"--project={c['project']}"]
    r = subprocess.run(cmd, capture_output=capture, text=True, encoding="utf-8", errors="replace", shell=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"gcloud {' '.join(args[:4])} failed ({r.returncode}):\n{(r.stderr or '')[-2000:]}")
    return (r.stdout or "").strip()


def gs(path: str = "") -> str:
    return f"gs://{cfg()['bucket']}/{PREFIX}/{path}".rstrip("/")


def gcs_mount(path: str = "") -> str:
    """Same object as gs(path), seen from inside a Vertex job."""
    return f"/gcs/{cfg()['bucket']}/{PREFIX}/{path}".rstrip("/")


def estimate_cost(hours: float) -> float:
    c = cfg()
    return round(hours * PRICE_PER_HOUR.get((c["machine"], c["gpu_type"], c["spot"]), 1.5), 3)


def write_result(name: str, obj: dict) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULT_DIR / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p
