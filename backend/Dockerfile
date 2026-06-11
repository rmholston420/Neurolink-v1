FROM python:3.12-slim

WORKDIR /app

# System deps for BLE (BlueZ) and build
RUN apt-get update && apt-get install -y --no-install-recommends \
    bluez \
    libglib2.0-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create data directory for SQLite
RUN mkdir -p /app/data

# Copy project metadata first (README required by hatchling before pip install)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

# Copy rest of project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "neurolink.main:app", "--host", "0.0.0.0", "--port", "8000"]
