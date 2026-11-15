#!/bin/bash
# tts service on this Mac (v4 on the CPU + the emotion mix), port 8001. First request of each new sentence takes ~20 s.
cd "$(dirname "$0")/.."
source TTS-Lab/.venv-tts/bin/activate
python -c "import fastapi, uvicorn, httpx" 2>/dev/null || pip install -q fastapi uvicorn httpx
export SPEECH_DEVICE=cpu-v4 SPEECH_MIX=1   # quality = same as the listening tests (16 steps). Faster but rougher: SPEECH_NFE=8 ./TTS-Lab/run_tts_native.sh
cd webapp/tts && exec uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
