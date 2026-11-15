#!/bin/bash
# OLD behaviour for comparison: plain v4 (tag + built-in PSOLA, no emotion-reference mix), port 8002.
cd "$(dirname "$0")/.."
source TTS-Lab/.venv-tts/bin/activate
python -c "import fastapi, uvicorn, httpx" 2>/dev/null || pip install -q fastapi uvicorn httpx
export SPEECH_DEVICE=cpu-v4 SPEECH_MIX=0
cd webapp/tts && exec uvicorn app:app --host 0.0.0.0 --port 8002 --workers 1
