"""v12 GPU runs on Modal (prompt-opt-v12 2E, owner budget $6-7): the primary reader (Uni-Sign, enc-v7) re-fine-tuned on old + new-source data.

  bash cache/modalw.sh volume create thaislm-v12
  bash cache/modalw.sh volume put thaislm-v12 cache/v12/modal_models /models            # mt5_config.json, encoder_v7.pt, kd_new_concepts.json
  bash cache/modalw.sh volume put thaislm-v12 cloud_s3/data_prep/shards /shards          # old store (1.2 GB) + tables
  bash cache/modalw.sh volume put thaislm-v12 cloud_s3/data_prep/shards_v12 /shards_v12  # new-source signers (TRAIN + NEWSIG-dev)
  bash cache/modalw.sh run modal_v12.py::primary --run v12-P1 [--hours 1.5] [--extra "..."]
  bash cache/modalw.sh volume get thaislm-v12 /runs/v12-P1 artifacts/runs/

Selection: held-out-signer MRR (TTRS / TSL-ONE-S / th_sl / YouTube val) + NEWSIG-dev MRR, every 250 steps, early stop after >= 2,000 steps and
6 evaluations without a new best (max 6,000). BLIND-v11 / v12 rows never train (--drop_sealed). Container timeout = time box + 10 min.
"""
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent
vol = modal.Volume.from_name("thaislm-v12", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.0", "transformers==4.51.3", "numpy==1.26.4", "pandas==2.2.3", "pyarrow==17.0.0", "scipy==1.13.1")
    .add_local_dir(LOCAL / "modules", "/app/modules", ignore=["__pycache__"])
    .add_local_dir(LOCAL / "GCP" / "script", "/app/GCP/script", ignore=["__pycache__"])
    .add_local_dir(LOCAL / "models" / "unisign", "/app/models/unisign", ignore=["__pycache__"])
    .add_local_file(LOCAL / "artifacts" / "eval" / "blind_v11_manifest.json", "/app/artifacts/eval/blind_v11_manifest.json")
    .add_local_file(LOCAL / "artifacts" / "eval" / "blind_v12_manifest.json", "/app/artifacts/eval/blind_v12_manifest.json")
)
app = modal.App("thaislm-v12", image=image)

P1 = ("--init ckpt:/vol/models/encoder_v7.pt --lr_gcn 5e-5 --lr_mt5 1e-5 --lr_head 5e-4 --warmup 200 --steps 6000 --eval_every 250 "
      "--patience 6 --min_steps 2000 --kd 1.0 --kd_end 0.2 --kd_new_concepts /vol/models/kd_new_concepts.json --drop_sealed --val_newsig "
      "--signer_aug 0.5 --mirror 0.1 --ema 0.998 --ckpt_every 2")


def _run(cmd, hours):
    import os
    import subprocess
    import sys
    import threading
    print("$", " ".join(cmd), flush=True)
    e = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen(cmd, cwd="/app", stdout=sys.stdout, stderr=subprocess.STDOUT, env=e)
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    try:
        rc = p.wait(timeout=hours * 3600)
    except subprocess.TimeoutExpired:            # time box: the best checkpoint so far is already on the Volume
        p.terminate(); rc = 0
        print("time box reached — stopped; encoder_best.pt holds the best evaluation so far", flush=True)
    stop.set(); vol.commit()
    if rc:
        raise SystemExit(f"exit {rc}")


@app.function(gpu=["A10G", "L4"], cpu=12, memory=48 * 1024, timeout=4 * 3600, volumes={"/vol": vol})
def train_primary(run: str, hours: float, extra: str = ""):
    args = ["python", "GCP/script/job_entry.py", "--stage", "train_encoder", "--run_id", run, "--data", "/vol/shards", "--data2", "/vol/shards_v12",
            "--models", "/vol/models", "--out", f"/vol/runs/{run}"] + P1.split() + (extra.split() if extra else [])
    _run(args, hours)


@app.local_entrypoint()
def primary(run: str = "v12-P1", hours: float = 1.5, extra: str = ""):
    train_primary.remote(run, hours, extra)
