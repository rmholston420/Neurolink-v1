FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml .
COPY backend/src ./src

RUN pip install --no-cache-dir -e ".[dev]"

RUN mkdir -p /app/data

CMD ["uvicorn", "neurolink.main:app", "--host", "0.0.0.0", "--port", "8000"]
