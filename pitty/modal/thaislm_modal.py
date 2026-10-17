"""Run the project's own GPU stages on Modal (pitty) — encoder fine-tune (A100) and the sequence stage (span head + tagger).

One-time upload of the data to a Modal Volume (from Main_demo/):
    python -m modal volume create thaislm
    python -m modal volume put thaislm cloud_s3/data_prep/shards /shards
    python -m modal volume put thaislm artifacts/runs/enc-v7-europe-west4 /enc-v7
Runs (each writes /runs/<name> on the volume):
    python -m modal run pitty/modal/thaislm_modal.py::encoder --name enc-ctrl --steps 1500
    python -m modal run pitty/modal/thaislm_modal.py::encoder --name enc-aug  --steps 1500 --yaw 0.5 --hand 0.5 --space 0.5
    python -m modal run pitty/modal/thaislm_modal.py::sequence --name seq-fluent05 --fluent 0.5   # tagger + span head (enc-v7)
Download:
    python -m modal volume get thaislm /runs/enc-aug cache/modal/
The code is the working tree (modules/, GCP/script/, scripts/, third_party/, main.py) mounted at run time; models/mt5_config.json too.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/app")  # the container has this file at /root/
vol = modal.Volume.from_name("thaislm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.7.0", "numpy==1.26.4", "pandas==2.2.3", "pyarrow>=14", "scikit-learn==1.6.1",
                 "transformers==4.51.3", "sentencepiece==0.2.0", "pyyaml", "pythainlp>=5.0")
    .add_local_dir(ROOT / "modules", "/app/modules")
    .add_local_dir(ROOT / "GCP" / "script", "/app/GCP/script")
    .add_local_dir(ROOT / "scripts", "/app/scripts")
    .add_local_dir(ROOT / "third_party", "/app/third_party")
    .add_local_file(ROOT / "main.py", "/app/main.py")
    .add_local_file(ROOT / "models" / "mt5_config.json", "/app/models/mt5_config.json")
)
app = modal.App("thaislm-pitty", image=image)


def _run(cmd):
    import subprocess
    import sys
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd="/app", stdout=sys.stdout, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    vol.commit()
    if r.returncode:
        raise SystemExit(f"exit {r.returncode}")


@app.function(gpu="A100-40GB", cpu=12, memory=48 * 1024, timeout=2 * 3600, volumes={"/vol": vol})
def train_encoder(name: str, steps: int, yaw: float, hand: float, space: float, eval_every: int):
    _run(["python", "GCP/script/job_entry.py", "--stage", "train_encoder", "--run_id", name, "--data", "/vol/shards", "--models", "/app/models",
          "--out", f"/vol/runs/{name}", "--init", "ckpt:/vol/enc-v7/encoder_best.pt", "--steps", str(steps), "--eval_every", str(eval_every),
          "--workers", "10", "--embed_bs", "512", "--patience", "3", "--aug_yaw", str(yaw), "--aug_hand", str(hand), "--aug_space", str(space)])


@app.function(gpu="L4", cpu=8, memory=32 * 1024, timeout=4 * 3600, volumes={"/vol": vol})
def train_sequence(enc: str, name: str, fluent: float, fluent_trim: str = "0.15,0.3", fluent_overlap: str = "0,4"):
    enc_dir = "/vol/enc-v7" if enc == "enc-v7" else f"/vol/runs/{enc}"
    _run(["python", "GCP/script/job_entry.py", "--stage", "sequence", "--run_id", name, "--data", "/vol/shards", "--models", "/app/models",
          "--out", f"/vol/runs/{name}", "--encoder_run", enc_dir, "--fluent", str(fluent), "--fluent_trim", fluent_trim, "--fluent_overlap", fluent_overlap])


@app.local_entrypoint()
def encoder(name: str, steps: int = 1500, yaw: float = 0.0, hand: float = 0.0, space: float = 0.0, eval_every: int = 300):
    train_encoder.remote(name, steps, yaw, hand, space, eval_every)


@app.local_entrypoint()
def sequence(name: str, enc: str = "enc-v7", fluent: float = 0.0, fluent_trim: str = "0.15,0.3", fluent_overlap: str = "0,4"):
    train_sequence.remote(enc, name, fluent, fluent_trim, fluent_overlap)
