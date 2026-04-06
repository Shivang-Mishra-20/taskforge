# Task Forge AI — Dockerfile
# HF Spaces Docker SDK — HTTP server on port 7860 + CLI runner

FROM python:3.11-slim

LABEL maintainer="Task Forge Team"
LABEL description="Task Forge AI — SaaS Operations RL Environment"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for HF Spaces security
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY environment/ ./environment/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .
COPY app.py .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start HTTP API server (serves /health, /reset, /step, /state endpoints)
CMD ["python", "app.py"]
