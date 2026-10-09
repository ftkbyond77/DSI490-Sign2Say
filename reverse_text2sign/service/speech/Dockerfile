# reverse-speech: Thai / English speech recognition + English → Thai translation (CPU, int8 CTranslate2).
# Stage 1 converts the OFFICIAL checkpoints (biodatlab Thai Whisper-small, Meta NLLB-200 600M) once at build time; the runtime
# image carries only CTranslate2 + the converted weights (no PyTorch).
FROM python:3.11-slim-bookworm AS convert
ENV PIP_NO_CACHE_DIR=1
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install ctranslate2==4.5.0 transformers==4.46.3 sentencepiece==0.2.0 huggingface_hub==0.26.5
RUN ct2-transformers-converter --model biodatlab/whisper-th-small-combined --output_dir /models/whisper-th-small \
      --quantization int8 --copy_files tokenizer.json preprocessor_config.json \
 && ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir /models/nllb-600m --quantization int8 \
 && python -c "import shutil; from huggingface_hub import hf_hub_download, snapshot_download; \
shutil.copy(hf_hub_download('facebook/nllb-200-distilled-600M', 'sentencepiece.bpe.model'), '/models/nllb-600m/'); \
snapshot_download('Systran/faster-whisper-base.en', local_dir='/models/whisper-base-en')" \
 && rm -rf /root/.cache

FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY reverse_text2sign/service/speech/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY --from=convert /models /models
COPY reverse_text2sign/service/speech/app.py /app/app.py
ENV SPEECH_MODELS=/models SPEECH_THREADS=4
EXPOSE 8002
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s CMD curl -fs http://localhost:8002/health || exit 1
CMD ["uvicorn", "--app-dir", "/app", "app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
