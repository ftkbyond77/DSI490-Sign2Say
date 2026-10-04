# ThaiSLM sign-api (build context = repository root, see docker-compose.yml)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY webapp/api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
# MediaPipe Holistic model (same file as models/mediapipe/ and the browser) for server-side video extraction
ENV MP_HOLISTIC_MODEL=/app/models/mediapipe/holistic_landmarker.task
COPY modules /app/modules
COPY webapp/api/app.py /app/app.py
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/v1/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
