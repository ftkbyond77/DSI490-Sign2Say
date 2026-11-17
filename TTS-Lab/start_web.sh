#!/bin/bash
# One command: our TTS (native, port 8001) + the whole web (docker, http://localhost:8080). Ctrl+C stops both.
cd "$(dirname "$0")/.."
lsof -ti :8001 | xargs kill 2>/dev/null; sleep 1
./TTS-Lab/run_tts_native.sh > /tmp/tts_native.log 2>&1 &
TTS_PID=$!
trap 'kill $TTS_PID 2>/dev/null; docker compose -f docker-compose.yml -f docker-compose.native-tts.yml down' EXIT
echo "waiting for the TTS (v4 load ~1 min) ... log: /tmp/tts_native.log"
until curl -s localhost:8001/v1/health | grep -q thonburian-v4; do sleep 3; done
echo "TTS ready: $(curl -s localhost:8001/v1/health)"
docker compose -f docker-compose.yml -f docker-compose.native-tts.yml up --build
