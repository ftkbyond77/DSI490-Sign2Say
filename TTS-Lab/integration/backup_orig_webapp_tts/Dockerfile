# ThaiSLM tts service — facebook/mms-tts-tha (VITS, offline after the first download, CPU)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/hf
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
COPY webapp/tts/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
# bake the voice into the image (no download at run time)
RUN python -c "from transformers import AutoTokenizer, VitsModel; AutoTokenizer.from_pretrained('facebook/mms-tts-tha'); VitsModel.from_pretrained('facebook/mms-tts-tha')"
COPY modules/tts.py /app/modules/tts.py
COPY webapp/tts/app.py /app/app.py
RUN touch /app/modules/__init__.py
EXPOSE 8001
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8001/v1/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
