# Neurolink

EEG-based meditation and contemplative practice app.

Streams real-time EEG data from Muse S Gen 1 or Muse S Athena headsets, computes band powers, detects contemplative states via a dual-classifier system (v0.1 6-region S-space + v2 8-region alchemical), scores EA-1 multimodal eligibility, and exposes live state via REST + SSE.

## Quick Start

```bash
# Mock mode (no hardware needed)
export NEUROLINK_ADAPTER_TYPE=mock
export NEUROLINK_DB_PATH=data/neurolink.db
uvicorn neurolink.main:app --reload --host 0.0.0.0 --port 8000
```

## Stack

- **Backend:** FastAPI 0.115, Python 3.12, SQLAlchemy async, SQLite, Redis (optional)
- **Hardware:** Muse S Gen 1 (BLE/LSL), Muse S Athena (OpenMuse LSL), MockAdapter
- **Frontend:** React 18, TypeScript, Vite

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEUROLINK_ADAPTER_TYPE` | `mock` | `mock` / `ble` / `lsl` |
| `NEUROLINK_DEVICE_MODEL` | `muse_s_gen1` | `muse_s_gen1` / `muse_s_athena` |
| `NEUROLINK_MUSE_BLE_ADDRESS` | — | BLE MAC address (required for `ble` mode) |
| `NEUROLINK_DB_PATH` | `.data/neurolink.db` | SQLite file path |
| `NEUROLINK_REDIS_URL` | `redis://localhost:6379/0` | Leave empty to disable Redis |
| `NEUROLINK_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated CORS origins |

## Docker

```bash
docker-compose -f compose.dev.yml up --build
```

## Tests

```bash
pip install -e '.[dev]'
pytest -v --tb=short
pytest --cov=neurolink --cov-report=term-missing --cov-fail-under=85
```
