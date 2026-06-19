# syntax=docker/dockerfile:1
# Compatible with both Docker BuildKit and Podman/Buildah.
# Podman builds this with: podman build -t neurolink-backend ./backend
# or via podman-compose which calls buildah under the hood.

FROM python:3.12-slim-bookworm

WORKDIR /app

# System deps for BLE (BlueZ + D-Bus) and build.
# bleak drives BlueZ via D-Bus; without dbus / libdbus-1-dev the
# BleakScanner backend raises a D-Bus connection-refused error at
# runtime even though `import bleak` succeeds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bluez \
    libglib2.0-dev \
    dbus \
    libdbus-1-dev \
    libdbus-glib-1-dev \
    gcc \
    curl \
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
