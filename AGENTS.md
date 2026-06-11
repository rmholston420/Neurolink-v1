# Neurolink Agent Instructions

See `neurolink_spec.md` for the full specification.

## Quick Start (Mock Mode, No Hardware)

```bash
cd backend
pip install -e ".[dev]"
NEUROLINK_ADAPTER_TYPE=mock uvicorn neurolink.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing

```bash
cd backend
pytest -v --tb=short
pytest --cov=neurolink --cov-report=term-missing --cov-fail-under=85
ruff check . && ruff format --check .
mypy src/neurolink/
```

## Architecture

- `backend/src/neurolink/` — FastAPI application
- `frontend/` — React 18 + TypeScript + Vite dashboard
- `compose.dev.yml` — Docker Compose dev stack
- `compose.prod.yml` — Docker Compose prod stack

## Key Invariants

- BLE protocol constants in `ble_bridge.py` and `hardware/muse_s/ble_adapter.py` are IMMUTABLE
- Hub is process-global, thread-safe via `threading.Lock`
- All hardware imports are lazy (inside constructors)
- No business logic in routers — call `NeuroLinkService` only
- DSP functions are pure — no I/O, no globals
