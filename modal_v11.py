"""v11 GPU runs on Modal (prompt-opt-v11 C8): the project's own scripts in a container, data on a Volume, results back to artifacts/runs.

  modal volume create thaislm-v11
  modal volume put thaislm-v11 cache/shards_mm /shards_mm          # memory-mapped pose store (1.2 GB) + meta
  modal volume put thaislm-v11 cloud_s3/data_prep/shards /shards   # clips.parquet, transitions, daily concepts (pose_store.npz NOT needed)
  modal run modal_v11.py::encoder --run v11-R0 --preset R0 --clips 10e6 [--gpu L4] [--cpu 16] [--hours 1.5]
  modal volume get thaislm-v11 /runs/v11-R0 artifacts/runs/          # checkpoints, history, embeddings
Loader-bound (NumPy augmentation) → many CPU cores, the pose store in RAM (THAISLM_STORE_RAM=1, shared after fork); GPU = L4 by default
(the 3.4 M / 15 M models do not need an A100). Container timeout = time box + 15 min; the script closes at its best inside the box.
"""
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent
vol = modal.Volume.from_name("thaislm-v11", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.0", "numpy==1.26.4", "pandas==2.2.3", "pyarrow==17.0.0", "scikit-learn==1.6.1", "scipy==1.13.1", "onnx==1.16.2", "onnxruntime==1.19.2")
    .add_local_dir(LOCAL / "modules", "/app/modules", ignore=["__pycache__"])
    .add_local_dir(LOCAL / "scripts", "/app/scripts", ignore=["__pycache__", "collect"])
    .add_local_dir(LOCAL / "pitty" / "scripts", "/app/pitty/scripts", ignore=["__pycache__"])
    .add_local_dir(LOCAL / "models" / "unisign", "/app/models/unisign", ignore=["__pycache__"])
    .add_local_file(LOCAL / "artifacts" / "runs" / "seq-v7" / "tsl51_alignment.json", "/app/artifacts/runs/seq-v7/tsl51_alignment.json")
    .add_local_file(LOCAL / "main.py", "/app/main.py")
    .add_local_file(LOCAL / "models" / "model-v10" / "vocab_conversation.json", "/app/models/model-v10/vocab_conversation.json")
    .add_local_file(LOCAL / "artifacts" / "eval" / "blind_v11_manifest.json", "/app/artifacts/eval/blind_v11_manifest.json")
    .add_local_dir(LOCAL / "cloud_s3" / "data_prep" / "manifest", "/app/cloud_s3/data_prep/manifest",
                   ignore=["clips.parquet", "lexicon.parquet", "synonym_decisions.parquet", "concepts.json", "summary.csv"])
)
app = modal.App("thaislm-v11", image=image)
TIMEOUT = 4 * 3600 + 900


def _run(cmd, env=None):
    import os
    import subprocess
    import sys
    os.makedirs("/app/cache", exist_ok=True)
    for src, dst in (("/vol/shards_mm", "/app/cache/shards_mm"), ("/vol/shards", "/app/cloud_s3/data_prep/shards")):
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True); os.symlink(src, dst)
    os.makedirs("/app/artifacts/runs", exist_ok=True)
    os.makedirs("/vol/runs", exist_ok=True)
    for d in os.listdir("/vol/runs"):                   # finished runs on the Volume are visible as artifacts/runs/<name>
        if not os.path.exists(f"/app/artifacts/runs/{d}"):
            os.symlink(f"/vol/runs/{d}", f"/app/artifacts/runs/{d}")
    for d in (env or {}).get("THAISLM_NEW_RUNS", "").split(","):   # outputs of a script run go to the Volume
        if d and not os.path.exists(f"/app/artifacts/runs/{d}"):
            os.makedirs(f"/vol/runs/{d}", exist_ok=True); os.symlink(f"/vol/runs/{d}", f"/app/artifacts/runs/{d}")
    print("$", " ".join(cmd), flush=True)
    e = {**os.environ, "PYTHONUNBUFFERED": "1", "THAISLM_STORE_RAM": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
    p = subprocess.Popen(cmd, cwd="/app", stdout=sys.stdout, stderr=subprocess.STDOUT, env=e)
    import threading
    stop = threading.Event()

    def committer():                          # checkpoints reach the Volume while the run is going (resumable after a preemption)
        while not stop.wait(600):
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    rc = p.wait(); stop.set()
    vol.commit()
    if rc:
        raise SystemExit(f"exit {rc}")


def _train(run, preset, clips, workers, hours, extra):
    out = f"/vol/runs/{run}"
    # the pose store meta records the LOCAL npz mtime: point THAISLM_SHARDS at the volume copy and skip the rebuild (meta matches)
    _run(["python", "scripts/train_v11.py", "--run", run, "--preset", preset, "--clips", str(clips), "--workers", str(workers), "--out", out,
          "--budget_s", str(int(hours * 3600))] + (extra.split() if extra else []), env={"THAISLM_SKIP_MM_CHECK": "1"})


GPUS = ["A10G", "L4", "T4"]          # fallback order (L4 capacity was not available on 2026-11-13; A10 ≈ L4 for these small models)


@app.function(gpu=GPUS, cpu=16, memory=40 * 1024, timeout=TIMEOUT, volumes={"/vol": vol})
def train(run: str, preset: str, clips: float, workers: int, hours: float, extra: str = ""):
    _train(run, preset, clips, workers, hours, extra)


@app.function(gpu=GPUS, cpu=16, memory=32 * 1024, timeout=TIMEOUT, volumes={"/vol": vol})
def script(args: str, hours: float = 1.0, new_runs: str = ""):
    """any repo script on the GPU (segmentation heads, prototype-CTC, detector): args = 'scripts/x.py --a b'; new_runs = output run names"""
    _run(["python"] + args.split(), env={"THAISLM_SKIP_MM_CHECK": "1", "THAISLM_NEW_RUNS": new_runs})


@app.local_entrypoint()
def encoder(run: str, preset: str = "R0", clips: float = 10e6, workers: int = 14, hours: float = 1.5, extra: str = ""):
    train.remote(run, preset, clips, workers, hours, extra)


@app.local_entrypoint()
def run_script(args: str, hours: float = 1.0, new_runs: str = ""):
    script.remote(args, hours, new_runs)
