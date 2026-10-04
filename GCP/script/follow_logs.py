"""Stream the important lines of a running Vertex job (Cloud Logging poll) until the job ends.

  python GCP/script/follow_logs.py --run <run_id> [--pattern "eval step|tagger ep|done|Traceback|Error"]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcp_common import RESULT_DIR, cfg, gcloud, gcloud_bin  # noqa: E402

TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pattern", default=r"zero-shot|eval step|resumed|aligned|alignment|synthetic|tagger ep|stage done|done in|Traceback|Error|OutOfMemory|Killed")
    ap.add_argument("--poll", type=int, default=90)
    a = ap.parse_args()
    rec = json.loads((RESULT_DIR / f"{a.run}.json").read_text(encoding="utf-8"))
    jid = rec["job"].split("/")[-1]
    pat = re.compile(a.pattern)
    seen, since, state = set(), datetime.now(timezone.utc) - timedelta(hours=3), None
    while True:
        try:
            d = json.loads(gcloud("ai", "custom-jobs", "describe", rec["job"], f"--region={cfg()['region']}", "--format=json"))
            if d.get("state") != state:
                state = d.get("state"); print(f"[state] {state}", flush=True)
            r = subprocess.run([gcloud_bin(), "logging", "read", f'resource.type="ml_job" AND resource.labels.job_id="{jid}" AND timestamp>="{since.isoformat()}"',
                                "--limit=500", "--order=asc", "--format=json", f"--project={cfg()['project']}"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            for e in json.loads(r.stdout or "[]"):
                key = e.get("insertId")
                msg = (e.get("textPayload") or e.get("jsonPayload", {}).get("message") or "").strip()
                if key in seen or not msg:
                    continue
                seen.add(key)
                if pat.search(msg):
                    print(msg[:400], flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[poll error] {type(ex).__name__}", flush=True)
        if state in TERMINAL:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
