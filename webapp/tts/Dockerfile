# ThaiSLM tts service — speech v4 (models/Speech-model-v4-TTS, mounted read-only; v3 works too): the fine-tuned ThonburianTTS (F5-TTS) voice on an NVIDIA GPU
# (docker-compose.yml reserves one), with MMS-TTS Thai (VITS, CPU, baked in) as the fallback voice when there is no GPU
# (docker compose -f docker-compose.yml -f docker-compose.cpu.yml up) or when a GPU request fails.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/hf SPEECH_MODEL=/app/speech-model
RUN apt-get update && apt-get install -y --no-install-recommends curl libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY webapp/tts/requirements.txt /app/requirements.txt
# torch / torchaudio / numpy stay as shipped in the image (constraints) — f5-tts' own dependency list would replace them
RUN pip freeze | grep -E '^(torch|torchaudio|numpy)==' > /tmp/c.txt \
 && pip install -c /tmp/c.txt -r /app/requirements.txt \
 && pip install --no-deps f5-tts==1.1.22 \
 && pip install -c /tmp/c.txt accelerate click datasets ema_pytorch matplotlib pydub pypinyin rjieba torchdiffeq vocos wandb x_transformers \
    cached_path unidecode hydra-core tomli safetensors librosa pythainlp
# voices baked into the image: Vocos vocoder and MMS-TTS Thai (fallback)
RUN python -c "from vocos import Vocos; Vocos.from_pretrained('charactr/vocos-mel-24khz')" \
 && python -c "from transformers import AutoTokenizer, VitsModel; AutoTokenizer.from_pretrained('facebook/mms-tts-tha'); VitsModel.from_pretrained('facebook/mms-tts-tha')"
COPY modules/__init__.py modules/tts.py modules/utils.py /app/modules/
COPY webapp/tts/app.py /app/app.py
EXPOSE 8001
HEALTHCHECK --interval=20s --timeout=5s --start-period=240s CMD curl -fs http://localhost:8001/v1/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
