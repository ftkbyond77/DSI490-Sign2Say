"""ThaiSLM — Thai Sign Language video / camera → Thai words / sentence → speech.  One entry point for everything.

Use the model (needs only models/):
  python main.py infer --video path/to/video.mp4 [--out dir] [--vocab conversation|full] [--no_llm] [--tts] [--backend onnx|torch]
  python main.py test                          # data_test/ end-to-end from the raw videos + scores vs reference glosses
  docker compose up --build                    # web demo (camera, live mode, speech) → http://localhost:8080

Data (S3 = source of truth, cloud_s3/ = local mirror, see cloud_s3/data_prep/SCHEMA.md):
  python main.py data pull data_prep           # download the prepared data (what training needs)
  python main.py data pull raw_data            # download raw videos / landmarks (only to rebuild data_prep)
  python main.py prep <step>                   # rebuild data_prep: pose_items | ingest | face_emb | signers | lexicon | manifest | shards | docs
  python main.py extract [--datasets a,b]      # MediaPipe Holistic on every source video (the production extractor, local CPU)
  python main.py harvest <step>                # more signers for daily-conversation words: discover [--words a,b] | download | videos | signers | label
  python main.py letters <step>                # fingerspelling lessons → letter holds: discover | select | download | extract | label
  python main.py study-clips --sentences "…"   # reference clips of each word / letter of a sentence (for a person preparing a test)
  python main.py pretrained                    # rebuild artifacts/pretrained/ (Uni-Sign base, needed only for training)

Data quality (v8):
  python main.py audit-youtube measure|sheets|decide   # YouTube clip audit (cuts, people, framing, hands, label check + eye check)

Train (v8: everything local — the compact Thai-only encoder fits the laptop GPU):
  python main.py train-encoder --run <run> [--ema 0.998]   # landmark encoder from scratch (resumes after an interruption)
  python main.py train-encoder --embed_ref v7|zeroshot     # reference encoders on the same evaluation clips
  python main.py train-encoder --report <run>              # every protocol vs v7 / zero-shot → artifacts/reports/isolated_v8.json
  python main.py train-seq --encoder_run <dir> # tagger + span head + decoder tables on the local GPU
  python main.py cloud <vertex.py args>        # optional Vertex AI GPU jobs: upload / launch / wait / status / cleanup

Build, tune, evaluate (local):
  python main.py eval <step> [args]            # build | grammar | spot | isolated | infer | selftest | vocab | bank_add  (scripts/eval.py)
  python main.py train-letters                 # fingerspelling recogniser + letter bank (CPU, minutes) → models/fingerspell.npz, letters_bank.npz
  python main.py tune-spelling                 # letter settings on tune data (models/spotting.json "spell"), report on held-out data
  python main.py export-onnx                   # models/onnx/*.onnx for production (onnxruntime), with a parity check
  python main.py eval external                 # public videos of people in no source (artifacts/reports/external_test_list.tsv)
  python main.py diagnose <video|live request> [--intended "…"]   # why did my sign fail? (landmarks, segments, references)
  python main.py bench-tts                     # Thai voices: intelligibility (ASR round trip), speed, prosody
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {"prep": "scripts/data_prep.py", "eval": "scripts/eval.py", "data": "scripts/s3_sync.py", "cloud": "GCP/script/vertex.py",
           "export-onnx": "scripts/export_onnx.py", "infer": "scripts/infer.py", "pretrained": "scripts/pretrained.py",
           "extract": "scripts/extract_mp.py", "harvest": "scripts/harvest_words.py", "train-seq": "GCP/script/job_entry.py",
           "letters": "scripts/harvest_letters.py", "train-letters": "scripts/train_fingerspell.py", "tune-spelling": "scripts/tune_spelling.py",
           "study-clips": "scripts/study_clips.py", "audit-youtube": "scripts/audit_youtube.py",
           "train-encoder": "scripts/train_encoder.py", "bench-tts": "scripts/bench_tts.py", "diagnose": "scripts/diagnose.py"}


def main():
    for stream in (sys.stdout, sys.stderr):              # Thai text + symbols on Windows consoles (cp874 cannot print "→")
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "test":
        cmd, rest = "eval", ["infer", *rest]
    if cmd == "train-seq":
        rest = ["--stage", "sequence", "--run_id", "seq-local", "--data", str(ROOT / "cloud_s3" / "data_prep" / "shards"),
                "--models", str(ROOT / "artifacts" / "pretrained"), "--out", str(ROOT / "artifacts" / "runs" / "seq-v8"), *rest]
    if cmd not in SCRIPTS:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")
    sys.path.insert(0, str(ROOT))
    sys.argv = [SCRIPTS[cmd], *rest]
    runpy.run_path(str(ROOT / SCRIPTS[cmd]), run_name="__main__")


if __name__ == "__main__":
    main()
