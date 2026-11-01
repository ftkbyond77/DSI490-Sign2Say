"""One Spot-A100 job: fine-tune v3 -> v4, then generate the fixed evaluation sets + latency for BOTH models on the same GPU.  Reuses GCP/script helpers.

  python vertex_tts.py upload [--clips N_HOURS]    # code + subset of data + v3 model -> GCS (once)
  python vertex_tts.py launch [--minutes 50]       # race 3 regions, first GPU wins, waits, fetches data/run/
  python vertex_tts.py cleanup                     # empty bucket, delete every job, check nothing else bills

Cost guard: hard job timeout = minutes + 40 min; the estimate is printed before launch (Spot A100 ~ $1.55/h incl. Vertex fee) and the
launch refuses when the estimate exceeds --max_usd. The hard ceiling for the whole project is $4 (all services); target <= $3.
Fail-fast: every setup step exits non-zero on failure, so a broken image costs minutes, not the run.
"""
import argparse, os, shutil, sys, tarfile, tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "GCP" / "script"))
import vertex                                                                       # noqa: E402
from gcp_common import RESULT_DIR, estimate_cost, gcloud, gcs_mount, gs             # noqa: E402

GCP_DL = ("python -c \"import sys;from google.cloud import storage;b,p=sys.argv[1],sys.argv[2];"
          "storage.Client().bucket(b).blob(p).download_to_filename(sys.argv[3])\"")
V3, V4 = ROOT / "models" / "Speech-model-v3-TTS", ROOT / "models" / "Speech-model-v4-TTS"
PREFIX = "tts2"


def upload(a):
    import pandas as pd
    t = Path(tempfile.mkdtemp())
    m = pd.read_csv(HERE / "data" / "manifest.csv"); m = m[m.reason.isna() | (m.reason == "")]
    keep = pd.read_csv(HERE / "data" / "upload_set.csv")                           # clips chosen by the training plan (written by prep.py/plan)
    m = m[m.sample_id.isin(keep.sample_id)]
    m.to_csv(t / "manifest.csv", index=False)
    with tarfile.open(t / "code.tar.gz", "w:gz") as tf:
        for f in ("train.py", "tts.py", "evaluate.py"):
            tf.add(HERE / f, arcname=f)
        tf.add(HERE / "data" / "manifest_v1.csv", arcname="data/manifest_v1.csv")
        tf.add(V3, arcname="v3"); tf.add(V4 / "config.json", arcname="v4/config.json"); tf.add(V4 / "voices.json", arcname="v4/voices.json"); tf.add(V4 / "refs", arcname="v4/refs")
        tf.add(V4 / "vocab.txt", arcname="v4/vocab.txt")
    with tarfile.open(t / "data.tar", "w") as tf:                                     # flac is already compressed
        tf.add(t / "manifest.csv", arcname="manifest.csv")
        for p in m.path:
            tf.add(HERE / "data" / p, arcname=p)
    print(f"upload: {len(m)} clips, {m.duration.sum() / 3600:.1f} h, tar {(t / 'data.tar').stat().st_size / 1e9:.2f} GB", flush=True)
    for f in ("code.tar.gz", "data.tar"):
        gcloud("storage", "cp", str(t / f), gs(f"{PREFIX}/{f}"), capture=False)
    shutil.rmtree(t)


EVAL = (
        # evaluation on the same GPU, same seeds: v4 (default 16 steps) -> v3 (own voices, 32 steps) -> v3 with the new voices (separates reference effect from model effect) -> step sweep
        "python evaluate.py gen --model_dir v4_out --out out/gen_v4 --sets main,onset,short,emo --voices male,female,male_2,female_2 --emo_voices male,female 2>&1 | tail -2 | tee out/gen_v4.log; "
        "python evaluate.py gen --model_dir v3 --out out/gen_v3 --nfe 32 --sets main,onset,short --voices masculine_1,masculine_2,feminine_1,feminine_2 2>&1 | tail -2 | tee out/gen_v3.log; "
        "cp -r v3 v3y && cp v4/voices.json v3y/voices.json && cp v4/refs/* v3y/refs/; "
        "python evaluate.py gen --model_dir v3y --out out/gen_v3y_n32 --nfe 32 --sets onset,short --voices male,female --seeds 0 2>&1 | tail -1; "
        "python evaluate.py gen --model_dir v3y --out out/gen_v3y_n16 --nfe 16 --sets onset,short --voices male,female --seeds 0 2>&1 | tail -1; "
        "for n in 32 12 10 8; do python evaluate.py gen --model_dir v4_out --out out/gen_v4_n$n --nfe $n --sets onset,short --voices male,female --seeds 0 2>&1 | tail -1; done; "
        "python evaluate.py latency --model_dir v4_out --voice male --out out/latency_v4.json 2>&1 | tail -3; "
        "python evaluate.py latency --model_dir v3 --voice masculine_1 --nfe 32 --out out/latency_v3.json 2>&1 | tail -3; "
)


def command(run_id, minutes, evaluate=True):
    c = vertex.cfg(); b, pre = c["bucket"], f"thaislm/{PREFIX}"; mnt = gcs_mount(f"runs/{run_id}")
    return (
        f"mkdir -p /workspace/out {mnt}; cd /workspace; "
        f"{GCP_DL} {b} {pre}/code.tar.gz code.tar.gz && tar xzf code.tar.gz && "
        f"{GCP_DL} {b} {pre}/data.tar data.tar && mkdir -p data && tar xf data.tar -C data && rm data.tar && "
        "pip freeze | grep -E '^(torch|torchaudio|numpy)==' > /tmp/c.txt; pip uninstall -y -q torchvision; "
        "pip install -q --no-cache-dir --no-deps f5-tts==1.1.22 && pip install -q --no-cache-dir -c /tmp/c.txt accelerate click datasets ema_pytorch matplotlib pydub pypinyin rjieba "
        "torchdiffeq 'transformers>=4.45' vocos wandb x_transformers cached_path unidecode hydra-core tomli safetensors soundfile librosa pythainlp pandas pyarrow scipy praat-parselmouth 2>&1 | tail -3; "
        "pip install -q --no-cache-dir --no-deps --force-reinstall torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2; pip freeze | grep -iE '^(torch|numpy)' ; "
        "python -c 'import torch,torchaudio;print(torch.__version__,torchaudio.__version__,torch.cuda.is_available())' || exit 1; "
        "python -c \"import f5_tts.model, f5_tts.infer.utils_infer; print('import ok')\" || exit 1; "
        "cp -r v4 v4_out && cp v3/vocab.txt v4_out/vocab.txt; "
        # 2-batch gate (forward/backward on the GPU) before the long run: exit early if the image or the data pipeline is broken
        f"python train.py --data data --ckpt v3/model.safetensors --vocab v4/vocab.txt --config v4/config.json --out out/gate --minutes 1 --gate > out/gate.log 2>&1 || {{ tail -25 out/gate.log; cp out/gate.log {mnt}/; exit 1; }}; tail -2 out/gate.log; "
        f"python train.py --data data --ckpt v3/model.safetensors --vocab v4/vocab.txt --config v4/config.json --out out --minutes {minutes} 2>&1 | tee out/train.log; "
        f"cp out/model_best.safetensors {mnt}/model.safetensors; cp out/log.json out/train.log {mnt}/; "                                  # the weights first: the slow mounted copy of many wavs must never be able to lose them
        "cp out/model_best.safetensors v4_out/model.safetensors && cp v4/config.json v4_out/config.json && cp v4/voices.json v4_out/ && cp -r v4/refs v4_out/ && cp tts.py v4_out/ && "
        + (EVAL if evaluate else "") +
        f"for d in out/*/; do cp -r $d {mnt}/; done; nvidia-smi --query-gpu=name --format=csv,noheader > {mnt}/gpu.txt; ls -la {mnt}")


def launch(a):
    h = a.minutes / 60 + (0.30 if a.no_eval else 0.85)                  # the job timeout counts from PROVISIONING start (about 3 min before the first container log)
    est, worst = estimate_cost(a.minutes / 60 + (0.15 if a.no_eval else 0.58)), estimate_cost(h)
    print(f"est. cost ≈ ${est:.2f} (train {a.minutes:.0f} min + setup + eval + sweeps ≈ 35 min), hard timeout {h:.2f} h (≤ ${worst:.2f})", flush=True)
    if worst > a.max_usd:
        sys.exit(f"refused: worst case ${worst:.2f} > --max_usd {a.max_usd}")
    base = f"tts2-{vertex._now()}"; ids = []; cname = vertex.cfg()["job_name"]
    for region in (a.regions.split(",") if a.regions else vertex.RACE_REGIONS):
        os.environ["VERTEX_REGION"] = region; rid = f"{base}-{region}"
        vertex.submit(rid, command(rid, a.minutes, not a.no_eval), f"{cname}-tts", spot=True, hours=h, meta=dict(stage="tts2", race_group=base)); ids.append(rid)
    os.environ.pop("VERTEX_REGION", None)
    a.runs, a.run = ",".join(ids), None
    vertex.cmd_race(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["upload", "launch", "cleanup"])
    ap.add_argument("--minutes", type=float, default=50); ap.add_argument("--poll", type=int, default=45); ap.add_argument("--max_usd", type=float, default=3.0); ap.add_argument("--no_eval", action="store_true"); ap.add_argument("--regions", default="")
    a = ap.parse_args()
    {"upload": upload, "launch": launch, "cleanup": vertex.cmd_cleanup}[a.cmd](a)
