# Integration — APPLIED on branch boom-live-diag
Copied from the worktree sign2say-main (base ef4936b). The files in this folder are kept as a copy. Originals of webapp/tts (MMS-only) are in backup_orig_webapp_tts/ (also: git checkout -- webapp/tts).
Applied: webapp/tts/{app.py,text_emotion.py,Dockerfile,requirements.txt}, models/Speech-model-v4-TTS/ (incl. patched tts.py), docker-compose.native-tts.yml.
Not applied: webapp/api (affect_v2) — the branch already has its own changes there.
Run: python3 -m venv TTS-Lab/.venv-tts ; ./TTS-Lab/start_web.sh   (needs models/Speech-model-v4-TTS/refs/ from Chayut for plain v4; the emotion mix works without)
