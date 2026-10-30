# ThaiSLM tts service (CPU) — speech v2 plan (modules/speech.py: what to say, natural hesitation for unread signs, emotion) spoken by
# facebook/mms-tts-tha (VITS, offline after the first download) with the fillers of models/speech-model-v2 (mounted read-only).
# The F5-TTS-THAI voice (GPU) is the `gpu` compose profile: webapp/tts/Dockerfile.gpu.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/hf SPEECH_MODEL=/app/speech-model SPEECH_DEVICE=cpu
RUN apt-get update && apt-get install -y --no-install-recommends curl libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
COPY webapp/tts/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
# bake the voice into the image (no download at run time)
RUN python -c "from transformers import AutoTokenizer, VitsModel; AutoTokenizer.from_pretrained('facebook/mms-tts-tha'); VitsModel.from_pretrained('facebook/mms-tts-tha')"
COPY modules/__init__.py modules/tts.py modules/speech.py modules/utils.py /app/modules/
COPY webapp/tts/app.py /app/app.py
EXPOSE 8001
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s CMD curl -fs http://localhost:8001/v1/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
