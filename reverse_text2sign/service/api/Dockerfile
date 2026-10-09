# reverse-api: text → IR → motion (build context = repository root, see docker-compose.yml). Models are mounted at /app/models.
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY reverse_text2sign/service/api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && python -c "from pythainlp.tokenize import word_tokenize; from pythainlp.tag import pos_tag; print(word_tokenize('ทดสอบระบบ'), pos_tag(['ฉัน', 'กิน'], engine='perceptron', corpus='orchid_ud'))"
COPY reverse_text2sign/t2s /app/t2s
COPY reverse_text2sign/service/api/app.py /app/service/api/app.py
ENV T2S_HOME=/app T2S_MODELS=/app/models
EXPOSE 8001
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8001/api/health || exit 1
CMD ["uvicorn", "--app-dir", "/app/service/api", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
