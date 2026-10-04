"""ThaiSLM — Vertex AI job launcher, cost-aware, with multi-region load balancing.

  python GCP/script/vertex.py smoke                                  # 1-minute job: GPU/quota/service-account/bucket check
  python GCP/script/vertex.py upload-code --tag <tag>                # snapshot modules/ scripts/ third_party GCP/script → GCS
  python GCP/script/vertex.py upload-models                          # artifacts/pretrained + rtmlib ONNX checkpoints → gs://…/thaislm/models
  python GCP/script/vertex.py upload-data                            # cloud_s3/data_prep/shards → gs://…/thaislm/data (training shards)
  python GCP/script/vertex.py launch --tag <tag> --stage <stage> --args "…" [--hours 3]
        # load balancing: the same job is submitted to every region with Spot A100 quota; the first one that gets a GPU wins and
        # the others are cancelled while still pending (pending jobs are not billed); then waits and fetches the outputs
  python GCP/script/vertex.py run --tag <tag> --stage <stage> --args "…" [--spot] [--hours 3]   # single region (VERTEX_REGION)
  python GCP/script/vertex.py wait --run <run_id>                    # poll until the job ends, fetch outputs, write GCP/result/<run_id>.json
  python GCP/script/vertex.py status | cancel --all
  python GCP/script/vertex.py cleanup                                # end of round: empty the bucket, delete every job, check nothing bills

Stages (GCP/script/job_entry.py): train_encoder (v7: the only cloud stage; poses, tagger and tuning run locally).

Identity: jobs run as the Vertex AI Custom Code Service Agent by default. The SERVICE_ACCOUNT in .env
(gpu-job-run-slm@…) made the container fail with INTERNAL before any log line (missing IAM roles, e.g. Storage Object Admin on
the bucket + Logs Writer, and iam.serviceAccountUser for the submitting user); set VERTEX_USE_SA=1 once those roles exist.

Cost controls: a job is a one-shot container (billing stops when the script exits), every job has a hard timeout
(`--hours`), `--spot` uses Spot capacity (≈60–70 % cheaper, preemptible; the stages checkpoint to GCS and resume),
and `wait` cancels the job if you press Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcp_common import ROOT, cfg, estimate_cost, gcloud, gcs_mount, gs, write_result  # noqa: E402

RACE_REGIONS = ("asia-southeast1", "us-central1", "europe-west4")   # regions where the project has Spot A100 quota (8 each)
ALL_REGIONS = RACE_REGIONS + ("asia-northeast1", "us-east1", "us-west1")   # + Spot T4 quota (1 each)


def image_for(region: str) -> str:
    """Prebuilt Vertex AI PyTorch 2.4 GPU training container from the registry nearest to the job's region."""
    host = "us" if region.startswith(("us-", "northamerica", "southamerica")) else ("europe" if region.startswith("europe") else "asia")
    return f"{host}-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"


def _region_of(job):
    return job.split("/locations/")[1].split("/")[0]

CODE_DIRS = ["modules", "scripts", "GCP/script", "third_party/unisign/stgcn_layers"]
CODE_SKIP = ("__pycache__", ".pyc", ".ipynb_checkpoints")
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def _now():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------------- uploads
def cmd_upload_code(a):
    tmp = Path(tempfile.mkdtemp()) / f"{a.tag}.tar.gz"
    with tarfile.open(tmp, "w:gz") as tf:
        for d in CODE_DIRS:
            for p in (ROOT / d).rglob("*"):
                if p.is_file() and not any(s in str(p) for s in CODE_SKIP):
                    tf.add(p, arcname=str(p.relative_to(ROOT)).replace("\\", "/"))
    gcloud("storage", "cp", str(tmp), gs(f"code/{a.tag}.tar.gz"))
    print(f"code → {gs(f'code/{a.tag}.tar.gz')} ({tmp.stat().st_size / 1e6:.2f} MB)")


def cmd_upload_data(a):
    src = Path(os.environ.get("THAISLM_DATA", ROOT / "cloud_s3")) / "data_prep" / "shards"
    gcloud("storage", "rsync", str(src), gs("data"), "--recursive", capture=False)
    print(f"data → {gs('data')}")


def cmd_upload_models(a):
    """Only what the training job loads: the Uni-Sign WLASL pose encoder + the mT5 config (v7 extracts poses locally with MediaPipe)."""
    for f in ("unisign_wlasl_enc.pt", "mt5_config.json"):
        gcloud("storage", "cp", str(ROOT / "artifacts" / "pretrained" / f), gs(f"models/{f}"), capture=False)
    print(f"models → {gs('models')}")


def _delete_job(name, region) -> int:
    """gcloud has no `ai custom-jobs delete` → Vertex REST DELETE; 1 if the API accepted the delete."""
    import urllib.request
    token = gcloud("auth", "print-access-token", check=False).strip()
    req = urllib.request.Request(f"https://{region}-aiplatform.googleapis.com/v1/{name}", method="DELETE", headers={"Authorization": f"Bearer {token}"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"WARNING could not delete {name}: {e}")
        return 0


def cmd_cleanup(a):
    """End of a GPU round — leave nothing that bills: every object in the bucket deleted (the bucket itself is kept), every Vertex AI
    custom job cancelled + deleted in every region, and a check that no VM / disk / IP / endpoint / deployed model exists."""
    c = cfg()
    bucket = f"gs://{c['bucket']}"
    n = gcloud("storage", "ls", "--recursive", f"{bucket}/**", check=False)
    if n.strip():
        gcloud("storage", "rm", "--recursive", "--all-versions", f"{bucket}/**", check=False, capture=False)
    left = gcloud("storage", "ls", "--recursive", "--all-versions", f"{bucket}/**", check=False)
    print(f"bucket {bucket}: {len([x for x in left.splitlines() if x.strip()])} objects left")
    print("soft delete:", gcloud("storage", "buckets", "describe", bucket, "--format=value(soft_delete_policy)", check=False))
    regions = sorted(set(ALL_REGIONS) | {"asia-east1", "asia-south1", "europe-west1", "us-east4", "us-west4"})
    deleted = 0
    for region in regions:
        jobs = json.loads(gcloud("ai", "custom-jobs", "list", f"--region={region}", "--format=json", check=False) or "[]")
        for j in jobs:
            if j.get("state") not in TERMINAL:
                gcloud("ai", "custom-jobs", "cancel", j["name"], f"--region={region}", check=False)
                for _ in range(30):
                    if json.loads(gcloud("ai", "custom-jobs", "describe", j["name"], f"--region={region}", "--format=json", check=False) or "{}").get("state") in TERMINAL:
                        break
                    time.sleep(10)
            deleted += _delete_job(j["name"], region)
        for kind in ("endpoints", "models"):
            items = json.loads(gcloud("ai", kind, "list", f"--region={region}", "--format=json", check=False) or "[]")
            if items:
                print(f"WARNING {region}: {len(items)} Vertex {kind} exist: {[i.get('displayName') for i in items]}")
    print(f"Vertex custom jobs deleted: {deleted}")
    time.sleep(5 if deleted else 0)
    left = sum(len(json.loads(gcloud("ai", "custom-jobs", "list", f"--region={r}", "--format=json", check=False) or "[]")) for r in regions)
    print(f"Vertex custom jobs remaining: {left}")
    for what in (("compute", "instances", "list"), ("compute", "disks", "list"), ("compute", "addresses", "list"), ("compute", "snapshots", "list"),
                 ("compute", "images", "list", "--no-standard-images")):
        out = gcloud(*what, "--format=value(name)", check=False)
        print(f"{' '.join(what[:2])}: {len([x for x in out.splitlines() if x.strip()])}" + (f" → {out.split()}" if out.strip() else ""))
    other = [b for b in gcloud("storage", "ls", check=False).split() if b.rstrip("/") != bucket]
    print("other buckets:", other or "none")


# ----------------------------------------------------------------------------- jobs
def job_spec(run_id, command: str, spot=False, hours=4.0):
    c = cfg()
    spec = {
        "workerPoolSpecs": [{
            "machineSpec": {"machineType": c["machine"], "acceleratorType": c["gpu_type"], "acceleratorCount": c["gpu_count"]},
            "replicaCount": 1,
            "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100},
            "containerSpec": {"imageUri": image_for(c["region"]), "command": ["bash", "-c"], "args": [command],
                              "env": [{"name": "RUN_ID", "value": run_id}, {"name": "PYTHONUNBUFFERED", "value": "1"}, {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
                                      {"name": "THAISLM_GCS", "value": gcs_mount()}]},
        }],
        "scheduling": {"timeout": f"{int(hours * 3600)}s"},
    }
    if spot or c["spot"]:
        spec["scheduling"]["strategy"] = "SPOT"
    if c["service_account"] and os.environ.get("VERTEX_USE_SA", "0") == "1":
        spec["serviceAccount"] = c["service_account"]
    return spec


def submit(run_id, command, display, spot=False, hours=4.0, meta=None):
    c = cfg()
    spec = job_spec(run_id, command, spot, hours)
    f = Path(tempfile.mkdtemp()) / "job.json"
    f.write_text(json.dumps(spec), encoding="utf-8")
    out = gcloud("ai", "custom-jobs", "create", f"--region={c['region']}", f"--display-name={display}", f"--config={f}", "--format=json")
    job = json.loads(out)
    rec = dict(run_id=run_id, job=job["name"], display_name=display, submitted=_now(), profile=c["profile"], machine=c["machine"],
               gpu=f"{c['gpu_count']}x{c['gpu_type']}", spot=bool(spot or c["spot"]), timeout_h=hours, env_requested=c["env_gpu"], region=c["region"], image=image_for(c["region"]), command=command, outputs=gs(f"runs/{run_id}"), **(meta or {}))
    write_result(run_id, rec)
    print(f"submitted {job['name']}  (run {run_id}, outputs {gs(f'runs/{run_id}')})")
    return rec


def cmd_smoke(a):
    c = cfg()
    run_id = f"smoke-{_now()}"
    mnt = gcs_mount(f"runs/{run_id}")
    command = ("set -euo pipefail; nvidia-smi; python -c \"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),"
               "torch.cuda.get_device_name(0))\"; mkdir -p " + mnt + "; nvidia-smi --query-gpu=name,memory.total --format=csv > " + mnt +
               "/gpu.txt; python -c \"import torch,time;x=torch.randn(4096,4096,device='cuda');t=time.time();[x@x for _ in range(50)];"
               "torch.cuda.synchronize();print('matmul50 s',time.time()-t)\" | tee " + mnt + "/bench.txt; ls /gcs/" + c["bucket"] + "; echo SMOKE_OK")
    submit(run_id, command, f"{c['job_name']}-smoke", spot=a.spot, hours=0.5, meta=dict(stage="smoke"))
    if a.wait:
        a.run = run_id
        cmd_wait(a)


def job_command(run_id, a):
    mnt = gcs_mount(f"runs/{run_id}")
    return (
        "set -euo pipefail; mkdir -p /workspace " + mnt + " && cd /workspace && "
        f"tar xzf {gcs_mount(f'code/{a.tag}.tar.gz')} && "
        "pip install -q --no-cache-dir -r GCP/script/requirements_job.txt 2>&1 | tail -2; "
        # stage the training shards on local SSD with the Cloud Storage client (the FUSE mount is slow for big files)
        "python -c \"import os,sys;from google.cloud import storage;b,p=os.environ['THAISLM_GCS'].split('/')[2],'/'.join(os.environ['THAISLM_GCS'].split('/')[3:]);"
        "os.makedirs('/tmp/data',exist_ok=True);os.makedirs('/tmp/models',exist_ok=True);"
        "[bl.download_to_filename('/tmp/data/'+bl.name.split('/')[-1]) for bl in storage.Client().list_blobs(b,prefix=p+'/data/')];"
        "[bl.download_to_filename('/tmp/models/'+bl.name.split('/')[-1]) for bl in storage.Client().list_blobs(b,prefix=p+'/models/') if bl.name.count('/')==p.count('/')+2]\" && "
        f"python GCP/script/job_entry.py --run_id {run_id} --stage {a.stage} --data /tmp/data --models /tmp/models {a.args} 2>&1 | tee /tmp/job.log; "
        f"status=${{PIPESTATUS[0]}}; cp /tmp/job.log {mnt}/job.log; exit $status")


def cmd_launch(a):
    """Load-balanced launch: same job in every region with quota; first to run wins, the rest are cancelled while pending."""
    if a.profile:
        os.environ["VERTEX_PROFILE"] = a.profile
    c = cfg()
    base = a.run_id or f"{a.stage}-{_now()}"
    ids = []
    for region in (a.regions.split(",") if a.regions else RACE_REGIONS):
        os.environ["VERTEX_REGION"] = region
        run_id = f"{base}-{region}"
        submit(run_id, job_command(run_id, a), f"{c['job_name']}-{a.stage}", spot=True, hours=a.hours,
               meta=dict(stage=a.stage, args=a.args, code_tag=a.tag, race_group=base))
        ids.append(run_id)
    os.environ.pop("VERTEX_REGION", None)
    a.runs = ",".join(ids)
    cmd_race(a)


def cmd_run(a):
    c = cfg()
    run_id = a.run_id or f"{a.stage}-{_now()}"
    command = job_command(run_id, a)
    rec = submit(run_id, command, f"{c['job_name']}-{a.stage}", spot=a.spot, hours=a.hours, meta=dict(stage=a.stage, args=a.args, code_tag=a.tag))
    if a.wait:
        a.run = rec["run_id"]
        cmd_wait(a)


def _describe(job):
    return json.loads(gcloud("ai", "custom-jobs", "describe", job, f"--region={_region_of(job)}", "--format=json"))


def cmd_wait(a):
    from gcp_common import RESULT_DIR
    rec = json.loads((RESULT_DIR / f"{a.run}.json").read_text(encoding="utf-8"))
    last = None
    try:
        while True:
            d = _describe(rec["job"])
            st = d.get("state")
            if st != last:
                print(f"[{_now()}] {rec['run_id']}: {st}", flush=True)
                last = st
                if st == "JOB_STATE_RUNNING" and not rec.get("observed_running_at"):
                    rec["observed_running_at"] = datetime.now(timezone.utc).isoformat()
                    write_result(rec["run_id"], rec)
            if st in TERMINAL:
                break
            time.sleep(a.poll)
    except KeyboardInterrupt:
        print("interrupted → cancelling the job so it stops billing")
        gcloud("ai", "custom-jobs", "cancel", rec["job"], f"--region={_region_of(rec['job'])}", check=False)
        d = _describe(rec["job"])
    fin(rec, d)


def cmd_race(a):
    """Same stage submitted in several regions: the first job to reach RUNNING wins, every other one is cancelled while still
    pending (pending jobs are not billed). Then wait for the winner and fetch its outputs."""
    from gcp_common import RESULT_DIR
    recs = [json.loads((RESULT_DIR / f"{r}.json").read_text(encoding="utf-8")) for r in a.runs.split(",")]
    winner = None
    while winner is None:
        live = []
        for rec in recs:
            d = _describe(rec["job"])
            st = d.get("state")
            if st == "JOB_STATE_RUNNING":
                winner = rec
                break
            if st not in TERMINAL:
                live.append(rec)
        if winner is None:
            if not live:
                sys.exit("all racers ended without running")
            print(f"[{_now()}] still pending: {', '.join(_region_of(r['job']) for r in live)}", flush=True)
            time.sleep(a.poll)
    print(f"[{_now()}] winner: {winner['run_id']} in {_region_of(winner['job'])}", flush=True)
    for rec in recs:
        if rec is not winner:
            gcloud("ai", "custom-jobs", "cancel", rec["job"], f"--region={_region_of(rec['job'])}", check=False)
            print(f"cancelled {rec['run_id']} ({_region_of(rec['job'])})", flush=True)
    for rec in recs:
        if rec is not winner:
            for _ in range(24):            # a loser may have obtained a GPU in the same minute: wait until it really stopped
                d = _describe(rec["job"])
                if d.get("state") in TERMINAL:
                    break
                time.sleep(10)
            fin(rec, d)
    a.run = winner["run_id"]
    cmd_wait(a)


def _first_container_log(job):
    """Timestamp of the first line the training container printed (Vertex sets startTime when it starts *provisioning*, so a job
    that waited for Spot capacity would otherwise look like it ran — and cost — during its whole queue time)."""
    import subprocess
    from gcp_common import gcloud_bin
    jid = job.split("/")[-1]
    r = subprocess.run([gcloud_bin(), "logging", "read", f'resource.type="ml_job" AND resource.labels.job_id="{jid}"', "--order=asc", "--limit=200",
                        "--format=json", f"--project={cfg()['project']}"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    for e in json.loads(r.stdout or "[]"):
        if "ml.googleapis.com/endpoint" in e.get("labels", {}):   # Vertex's own status lines ("Waiting for…", "Resources are insufficient…")
            continue
        ts = e["timestamp"].rstrip("Z")
        ts = ts.split(".")[0] + "." + ts.split(".")[1][:6] if "." in ts else ts
        return datetime.fromisoformat(ts + "+00:00")
    return None


def fin(rec, d):
    def ts(k):
        v = d.get(k)
        return datetime.fromisoformat(v.replace("Z", "+00:00")) if v else None
    provision, end, create = ts("startTime"), ts("endTime"), ts("createTime")
    try:
        start = _first_container_log(rec["job"])
    except Exception:  # noqa: BLE001
        start = None
    if start is None and rec.get("observed_running_at"):
        start = datetime.fromisoformat(rec["observed_running_at"])
    run_h = (end - start).total_seconds() / 3600 if start and end and end > start else 0.0
    rec["billable_basis"] = "first container log → end" if start else "never ran (no container output) → 0"
    rec.update(state=d.get("state"), create_time=str(create), provisioning_time=str(provision), start_time=str(start), end_time=str(end),
               queue_min=round((start - create).total_seconds() / 60, 1) if start and create else None,
               run_min=round(run_h * 60, 1), est_cost_usd=estimate_cost(run_h), error=d.get("error"))
    dst = ROOT / "artifacts" / "runs" / rec["run_id"]
    if not start:                                    # never-started racers have nothing to fetch
        p = write_result(rec["run_id"], rec)
        print(f"{rec['run_id']}: {rec['state']} · run {rec['run_min']} min · ≈${rec['est_cost_usd']} → {p}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    try:
        gcloud("storage", "rsync", gs(f"runs/{rec['run_id']}"), str(dst), "--recursive", capture=True)
        rec["fetched_to"] = str(dst.relative_to(ROOT))
        rec["files"] = sorted(str(p.relative_to(dst)).replace("\\", "/") for p in dst.rglob("*") if p.is_file())[:200]
    except Exception as e:  # noqa: BLE001
        rec["fetch_error"] = str(e)[-500:]
    p = write_result(rec["run_id"], rec)
    print(f"{rec['run_id']}: {rec['state']} · run {rec['run_min']} min · ≈${rec['est_cost_usd']} → {p}")


def cmd_status(a):
    for region in ALL_REGIONS:
        print(f"== {region}")
        print(gcloud("ai", "custom-jobs", "list", f"--region={region}", "--limit=8",
                     "--format=table(displayName,state,createTime,endTime,name.basename())"))


def cmd_cancel(a):
    jobs = [j for region in ALL_REGIONS for j in json.loads(gcloud("ai", "custom-jobs", "list", f"--region={region}", "--format=json") or "[]")]
    live = [j for j in jobs if j.get("state") in ("JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED")]
    for j in live:
        if a.all or j["name"].endswith(a.job or "-"):
            gcloud("ai", "custom-jobs", "cancel", j["name"], f"--region={_region_of(j['name'])}", check=False)
            print("cancelled", j["displayName"], j["name"])
    if not live:
        print("no running jobs")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--spot", action="store_true"); s.add_argument("--wait", action="store_true"); s.add_argument("--poll", type=int, default=20)
    u = sub.add_parser("upload-code"); u.add_argument("--tag", required=True)
    sub.add_parser("upload-data"); sub.add_parser("upload-models")
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True); r.add_argument("--stage", required=True); r.add_argument("--args", default="")
    r.add_argument("--spot", action="store_true"); r.add_argument("--hours", type=float, default=4.0); r.add_argument("--run_id")
    r.add_argument("--wait", action="store_true"); r.add_argument("--poll", type=int, default=60)
    w = sub.add_parser("wait"); w.add_argument("--run", required=True); w.add_argument("--poll", type=int, default=60)
    rc = sub.add_parser("race"); rc.add_argument("--runs", required=True, help="comma-separated run ids"); rc.add_argument("--poll", type=int, default=30)
    la = sub.add_parser("launch"); la.add_argument("--tag", required=True); la.add_argument("--stage", required=True); la.add_argument("--args", default="")
    la.add_argument("--hours", type=float, default=4.0); la.add_argument("--run_id"); la.add_argument("--poll", type=int, default=30)
    la.add_argument("--profile", default="", help="a100_spot (default) | t4_spot"); la.add_argument("--regions", default="", help="comma-separated; default = A100 regions")
    sub.add_parser("status"); sub.add_parser("cleanup")
    c = sub.add_parser("cancel"); c.add_argument("--all", action="store_true"); c.add_argument("--job")
    a = ap.parse_args()
    dict(smoke=cmd_smoke, run=cmd_run, wait=cmd_wait, status=cmd_status, cancel=cmd_cancel, race=cmd_race, launch=cmd_launch)[a.cmd](a) if a.cmd in ("smoke", "run", "wait", "status", "cancel", "race", "launch") \
        else {"upload-code": cmd_upload_code, "upload-data": cmd_upload_data, "upload-models": cmd_upload_models, "cleanup": cmd_cleanup}[a.cmd](a)


if __name__ == "__main__":
    main()
