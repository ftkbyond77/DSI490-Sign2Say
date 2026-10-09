"""Where things live. The reverse system reads the shared data lake mirror (repo-root cloud_s3/, pulled from S3) and writes only
under reverse_text2sign/:  data/ (derived, git-ignored) · models/ (deployable bundle) · result_reporting/ (reports)."""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
HERE = PKG.parent                                    # reverse_text2sign/
REPO = HERE.parent                                   # the forward project root (shared data, .env)
DATA_ROOT = Path(os.environ.get("T2S_DATA_ROOT", REPO / "cloud_s3"))
PREP = DATA_ROOT / "data_prep"
RAW = DATA_ROOT / "raw_data"
AGENT = DATA_ROOT / "agent_data"
MANIFEST = PREP / "manifest" / "clips.parquet"
MP_RAW = PREP / "pose_mp_raw"
HARVEST = AGENT / "reverse_harvest"                # targeted YouTube harvest of the reverse part (videos, MediaPipe, QC)                        # raw MediaPipe Holistic arrays per clip (pose 33 xyz+vis, hands 21 xyz, face 68, bs 52)

WORK = Path(os.environ.get("T2S_WORK", HERE / "data"))          # derived data (audit tables, contact sheets, eval renders)
MODELS = Path(os.environ.get("T2S_MODELS", HERE / "models"))    # deployable bundle read by the API
REPORT = HERE / "result_reporting"
AVATAR_SRC = HERE / "vrm" / "default.vrm"                 # the library avatar as delivered (VRoid AvatarSample_C)
AVATAR = MODELS / "avatar.vrm"                              # web-optimised copy (same skeleton) — what the API / browser use

def mp_raw_path(clip_id: str, dataset: str) -> Path:
    """manifest clip → its raw MediaPipe arrays (pose_mp_raw/<dataset>/<name>.npz)."""
    name = clip_id.split(":", 1)[1]
    return MP_RAW / dataset / (f"ttrs_{name}.npz" if dataset == "ttrs" else f"{name}.npz")


def video_path(clip_id: str, dataset: str, source_file: str) -> Path:
    """the video a clip was extracted from (TSL51 manifest rows point at the landmark folder)."""
    if dataset.startswith("tsl51_user_"):
        sub = "user_sign" if dataset == "tsl51_user_sign" else "user_sentence"
        return RAW / "TSL51" / "videos" / sub / (clip_id.split(":", 1)[1] + ".mp4")
    return DATA_ROOT / source_file


def load_env():
    """Secrets from the repo .env into os.environ (never printed)."""
    p = REPO / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip('"').strip("'")
        if v and not os.environ.get(k.strip()):
            os.environ[k.strip()] = v
