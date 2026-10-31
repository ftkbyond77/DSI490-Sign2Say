#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
export MPLCONFIGDIR="$PWD/.matplotlib"
DEMO_PYTHON="/Users/thawaphorn/Documents/Codex/2026-09-17/good-id-word-category-video-url/work/demo-venv/bin/python"
"$DEMO_PYTHON" prepare.py
"$DEMO_PYTHON" train.py pretrain --epochs 40
"$DEMO_PYTHON" train.py finetune --epochs 40
exec "$DEMO_PYTHON" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
