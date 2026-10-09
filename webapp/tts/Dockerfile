# ThaiSLM tts service (v8) — VachanaTTS Thai voices (VITS via ONNX Runtime, CPU, offline: voices baked into the image)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TTS_PROVIDER=vachana TTS_VOICE_DIR=/app/voices
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY webapp/tts/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY modules/tts.py /app/modules/tts.py
RUN touch /app/modules/__init__.py
# bake the four voices into the image (no download at run time); TTS_VOICE_VACHANA picks one
RUN python -c "import os; os.environ['TTS_VOICE_DIR']='/app/voices'; from modules.tts import VachanaTTS; [VachanaTTS(voice=v) for v in ('th_f_1','th_m_1','th_f_2','th_m_2')]"
COPY webapp/tts/app.py /app/app.py
EXPOSE 8001
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8001/v1/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
