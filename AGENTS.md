# Neurolink — Agent Instructions

> See `neurolink-app-spec.md` for full specification.

## Quick Start

```bash
cd backend
pip install -e ".[dev]"
pytest -v
NEUROLINK_ADAPTER_TYPE=mock uvicorn neurolink.main:app --reload --port 8000
```

## Key Decisions

- **Dual classifier:** v0.1 (6-region S-space) runs when `source == "muse_ble"`; v2 (8-region alchemical) always runs.
- **BLE protocol constants are IMMUTABLE.** See `hardware/muse_s/ble_adapter.py`.
- **Hub is process-global, thread-safe.** One `EEGHub` per process.
- **No business logic in routers.** Routers delegate to `NeuroLinkService` only.
- **Mock mode:** `NEUROLINK_ADAPTER_TYPE=mock` for CI/CD and dev without hardware.

## Running Tests

```bash
pytest tests/unit/ -v                    # fast unit tests only
pytest -v --tb=short                     # all tests
pytest --cov=neurolink --cov-fail-under=85  # with coverage
```

## BLE Hardware Setup (Muse S Gen 1)

```bash
NEUROLINK_ADAPTER_TYPE=ble \
NEUROLINK_MUSE_BLE_ADDRESS=<mac> \
uvicorn neurolink.main:app --port 8000
```

## LSL Hardware Setup (muselsl required externally)

```bash
muselsl stream --address <mac> &
NEUROLINK_ADAPTER_TYPE=lsl uvicorn neurolink.main:app --port 8000
```
