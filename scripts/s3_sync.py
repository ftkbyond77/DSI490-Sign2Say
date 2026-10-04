"""Sync the shared data lake (S3 = source of truth) with the local mirror cloud_s3/.

  python scripts/s3_sync.py pull data_prep            # s3://dsi490-lake-signdata/data_prep → cloud_s3/data_prep (what training needs)
  python scripts/s3_sync.py pull raw_data [--only word_level/th_sl_dictionary]
  python scripts/s3_sync.py push data_prep --dry-run  # show what would be uploaded (the data owner pushes)
  python scripts/s3_sync.py ls [prefix]

Uses the AWS CLI with the SSO profile named in .env (AWS_PROFILE). If the token expired:
  aws sso login --profile <AWS_PROFILE>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.env import load_env  # noqa: E402
from modules.utils import DATA  # noqa: E402

BUCKET = os.environ.get("THAISLM_S3_BUCKET", "dsi490-lake-signdata")


def aws(*args):
    load_env()
    prof = os.environ.get("AWS_PROFILE", "").strip('"')
    cmd = ["aws", *args] + (["--profile", prof] if prof else [])
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pull", "push", "ls"])
    ap.add_argument("prefix", nargs="?", default="")
    ap.add_argument("--only", default="", help="sub-folder inside the prefix")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = "/".join(x for x in (a.prefix, a.only) if x)
    remote, local = f"s3://{BUCKET}/{key}".rstrip("/"), DATA / key
    extra = ["--dryrun"] if a.dry_run else []
    if a.action == "ls":
        sys.exit(aws("s3", "ls", remote + "/"))
    if a.action == "pull":
        local.mkdir(parents=True, exist_ok=True)
        sys.exit(aws("s3", "sync", remote, str(local), *extra))
    sys.exit(aws("s3", "sync", str(local), remote, *extra))


if __name__ == "__main__":
    main()
