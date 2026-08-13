# Use slim Python 3.10 to keep image small
FROM python:3.10-slim

# System libs needed by Pillow / OpenCV (libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY api/  ./api/
COPY src/   ./src/

# Directories expected at runtime
RUN mkdir -p artifacts/model logs

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/artifacts/model/model.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
