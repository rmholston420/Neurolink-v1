# Neurolink

**EEG-based meditation and contemplative practice app**

Neurolink streams real-time EEG data from a Muse S headset, computes band powers,
detects contemplative states using a dual-classifier system (6-region S-space v0.1 +
8-region alchemical v2), scores EA-1 multimodal eligibility, and exposes live state
via REST + SSE.

## Quick Start (Mock Mode — No Hardware)

```bash
cd backend
pip install -e ".[dev]"
export NEUROLINK_ADAPTER_TYPE=mock
export NEUROLINK_DB_PATH=:memory:
uvicorn neurolink.main:app --reload --port 8000
```

Then open: http://localhost:8000/health

## Docker Compose

```bash
docker-compose -f compose.dev.yml up --build
```

## Running Tests

```bash
cd backend
pytest -v
pytest --cov=neurolink --cov-fail-under=85
```

## Environment Variables

See `.env.template` for all configuration options.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/neurolink/connect` | Connect EEG adapter |
| POST | `/api/v1/neurolink/disconnect` | Disconnect |
| GET | `/api/v1/neurolink/state` | Current EEG state |
| GET | `/api/v1/neurolink/bands` | Band powers |
| GET | `/api/v1/neurolink/ea1` | EA-1 eligibility |
| GET | `/api/v1/neurolink/stream` | SSE stream (4 Hz) |
| POST | `/api/v1/neurolink/calibrate` | Start calibration |
| GET | `/api/v1/neurolink/sessions` | Session log |
| GET | `/api/v1/gate/status` | EEG gate status |

## Architecture

- **Hardware adapters:** MockAdapter, MuseSBleAdapter, MuseSLslAdapter, AthenaBlueAdapter
- **DSP pipeline:** bandpower, FAA, FMt, PPG/HRV, breathing, IMU, classifiers
- **Hub:** Thread-safe in-memory state store with SSE fan-out
- **EA-1 scorer:** 5-criterion multimodal eligibility gating
- **Focus/Fatigue:** Per-frame classifiers on hub.update()
- **Session log:** SQLite/SQLAlchemy async
- **Frontend:** React 18 + TypeScript + Vite (in `frontend/`)

## Hardware Support

| Device | Mode | Notes |
|--------|------|-------|
| Muse S Gen 1 | BLE | Direct bleak GATT, no muselsl required |
| Muse S Gen 1 | LSL | Requires `muselsl stream` running |
| Muse S Athena | LSL | Requires OpenMuse running |
| Mock | — | Deterministic sine-wave EEG, no hardware |
