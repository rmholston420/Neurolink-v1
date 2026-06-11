# App Spec: Neurolink
> **Version:** 1.0 | **Status:** Draft | **Last Updated:** 2026-06-08
> **Agent Target:** Claude Code / Gemini CLI / GitHub Copilot / Cursor / Aider / OpenHands

---

## 0. Agent Instructions (Read First)

> This is a spec-driven development document. You are a senior engineer executing this spec.
> - **Plan in read-only mode first.** Analyze the codebase and ask clarifying questions before writing code.
> - **Never hallucinate APIs, libraries, or behaviors.** If uncertain, stop and ask.
> - **All hardware protocol constants (BLE UUIDs, command bytes, timing gaps) are FIXED.** Copy them exactly from the source inventory in Section 14.
> - **Follow the task breakdown in Section 10.** Do not skip steps or combine tasks.
> - **After each task, run the commands in Section 5 and self-verify against Section 8 before proceeding.**
> - **Prohibited completion phrases:** "tests should pass", "implementation looks correct", "follows best practices". You must cite specific test output and file paths as evidence.
> - **Hardware paths are never mocked in integration tests** — use the `MockAdapter` via `NEUROLINK_ADAPTER_TYPE=mock`.

---

## 1. Objective and Scope

### What We're Building

Neurolink is a **standalone EEG-based meditation and contemplative practice app** extracted from the Rigpa-v2 and Rigpa-v3 codebases. It is a self-contained FastAPI backend service (with a React/TypeScript frontend dashboard) that connects to either a **Muse S Gen 1** or a **Muse S Athena (Gen 2)** headset, streams raw EEG/PPG/IMU data, computes real-time band powers, detects contemplative states using a dual-classifier system (6-region S-space v0.1 + 8-region alchemical v2), scores EA-1 multimodal eligibility, and exposes the live state via REST + SSE. The app is designed for a single practitioner running locally on Linux (Kubuntu/Ubuntu).

### Success Looks Like

- [ ] User connects either Muse S Gen 1 or Muse S Athena via BLE or LSL and receives live EEG frames within 5 seconds
- [ ] Band powers (delta, theta, alpha, beta, gamma) stream at 4 Hz via SSE with < 50 ms publish latency
- [ ] Both the v0.1 6-region classifier and the v2 8-region alchemical classifier run in parallel on every frame
- [ ] EA-1 multimodal eligibility score is computed and included in every SSE frame
- [ ] FAA (Frontal Alpha Asymmetry) and FMt (Frontal Midline Theta) are included in every BLE frame
- [ ] PPG-derived HR, RMSSD, SDNN, pNN50, and Poincaré indices are computed and exposed
- [ ] Fused respiratory rate (PPG-FM + IMU accel-z) is computed and exposed
- [ ] Head orientation (pitch/roll) and motion_rms gating are computed from IMU
- [ ] Calibration baseline session sets per-subject alpha reference within 30 seconds
- [ ] Focus state (HIGH_FOCUS / MODERATE_FOCUS / LOW_FOCUS / DISTRACTED) is computed on every frame
- [ ] Fatigue score (rolling 30-sample theta/alpha ratio) is computed on every frame
- [ ] fNIRS oxy/deoxy values decoded for Muse Athena sessions
- [ ] Mock adapter allows full local development and CI/CD without hardware
- [ ] All tests pass with ≥ 85% coverage; lint + mypy clean
- [ ] Docker Compose brings up the full stack (backend + Redis + optional frontend)
- [ ] `/health` endpoint reports BLE/LSL/mock adapter status

### Not Included (Explicit Scope Boundary)

- Neo4j, Qdrant, PostgreSQL — Neurolink uses SQLite (session log) + Redis (live state cache) only
- MagiState, goal execution pipelines, Gnosis, Governance — Rigpa-v3 domains NOT ported here
- User authentication / JWT — single-user local app; auth is optional (env flag)
- OpenMuse subprocess management (Muse Athena BLE is managed externally by the user)
- muselsl subprocess management (Muse S Gen 1 LSL mode requires muselsl running externally)
- FPV drone control, depth camera integration
- Any Rigpa-v3 plugin that is not `plugins/neurolink/`
- Distributed tracing, rate limiting, multi-tenancy

---

## 2. Tech Stack and Versions

> **CRITICAL:** Use only these exact versions. Do not upgrade or substitute without asking first.

| Layer             | Technology                        | Version      |
|-------------------|-----------------------------------|--------------|
| Language          | Python                            | 3.12.x       |
| Web Framework     | FastAPI                           | 0.115.x      |
| ASGI Server       | Uvicorn                           | 0.30.x       |
| Data Validation   | Pydantic v2                       | 2.7.x        |
| Settings          | pydantic-settings                 | 2.x          |
| Live State Cache  | Redis                             | 7.x          |
| Session Log DB    | SQLite via aiosqlite              | 3.x / 0.20.x |
| ORM (session log) | SQLAlchemy async                  | 2.0.x        |
| BLE Driver        | bleak                             | 0.22.x       |
| LSL Consumer      | pylsl                             | 1.16.x       |
| Signal Processing | numpy                             | 1.26.x       |
| HRV / PPG         | neurokit2                         | 0.2.x        |
| Breathing / IMU   | scipy                             | 1.13.x       |
| Logging           | structlog                         | 24.x         |
| Testing           | pytest + pytest-asyncio           | 8.x / 0.23.x |
| HTTP Test Client  | httpx                             | 0.27.x       |
| Linting           | ruff                              | 0.4.x        |
| Type Checking     | mypy                              | 1.10.x       |
| Containerization  | Docker + docker-compose           | 27.x         |
| Frontend          | React 18 + TypeScript + Vite      | 18.x / 5.x   |
| Frontend Testing  | Vitest + React Testing Library    | 1.x / 14.x   |

### Key Dependencies (pyproject.toml)

```toml
[project]
name = "neurolink"
version = "1.0.0"
requires-python = ">=3.12"

dependencies = [
  "fastapi>=0.115,<0.116",
  "uvicorn[standard]>=0.30,<0.31",
  "pydantic>=2.7,<3",
  "pydantic-settings>=2.0,<3",
  "redis[hiredis]>=5.0",
  "aiosqlite>=0.20",
  "sqlalchemy[asyncio]>=2.0,<3",
  "bleak>=0.22,<0.23",
  "pylsl>=1.16",
  "numpy>=1.26,<2",
  "neurokit2>=0.2",
  "scipy>=1.13",
  "structlog>=24.0",
  "sse-starlette>=2.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "httpx>=0.27",
  "ruff>=0.4",
  "mypy>=1.10",
  "coverage[toml]>=7.0",
  "pytest-cov>=5.0",
]
```

---

## 3. Core Features and Requirements

### Feature List

1. **Hardware Adapter Layer** — Uniform `HardwareAdapter` ABC with three concrete implementations:
   - `MuseSBleAdapter` (Muse S Gen 1, direct bleak BLE, ported from Rigpa-v2 `ble_bridge.py`)
   - `MuseSLslAdapter` (Muse S Gen 1, pylsl consumer, ported from Rigpa-v3 `hardware/muse_s/lsl_adapter.py`)
   - `AthenaBlueAdapter` (Muse S Athena, OpenMuse LSL consumer, ported from Rigpa-v3 `hardware/muse_athena/ble_adapter.py`)
   - `MockAdapter` (no hardware, deterministic sine-wave EEG, ported from Rigpa-v2 `mock_stream.py` + Rigpa-v3 `hardware/mock.py`)

2. **Adapter Factory** — `create_adapter(adapter_type, device_model, address)` as in Rigpa-v3 `adapter_factory.py`. Reads defaults from env/settings.

3. **Ring Buffer Manager** — Per-session ring buffers for EEG (5ch × 4s @ 256 Hz), PPG (30s @ 64 Hz), accel/gyro (4s @ 52 Hz). Ported from Rigpa-v2 `muse_compute.make_buffers()`.

4. **Signal Processing Pipeline** — Pure-function DSP layer:
   - `bandpower(sig, lo, hi, fs)` — numpy rfft mean power
   - `derived_eeg(eeg_bufs)` — FAA + FMt
   - `classify_v01(alpha, theta, beta, delta, gamma, faa, fmt)` — 6-region S-space classifier (Rigpa-v2 `muse_compute.classify`)
   - `classify_v2(bands)` — 8-region alchemical classifier (Rigpa-v2 `classifier.py`)
   - `compute_ppg(ppg_ir, fs)` — NeuroKit2 Elgendi peak detector → HR, IBI, RMSSD, SDNN, pNN50, Poincaré
   - `compute_breathing(ibis_ms, accel_z)` — PPG-FM + IMU accel-z fused respiratory rate
   - `head_orientation(accel, gyro)` — gravity-referenced pitch/roll + gyro motion RMS
   - `decode_eeg(data)` / `decode_ppg(data)` / `decode_imu(data)` — raw GATT frame decoders (Rigpa-v2 `muse_decoders.py`)
   - `compute_all_bands(channel_samples)` — Welch-based band powers (Rigpa-v3 `hardware/muse_s/compute.py`)
   - `FNIRSDecoder.decode(raw_sample)` — Athena oxy/deoxy interleaved decoder (Rigpa-v3 `hardware/muse_athena/fnirs.py`)

5. **EA-1 Scorer** — `score(payload) → EA1Result` — 5-criterion multimodal eligibility scorer (Rigpa-v2 `ea1_scorer.py`). Criteria: alpha_power, theta_power, s_space gating, motion_rms gating, contact_quality.

6. **EEG Hub** — In-memory state store. Dual-path enrichment: v2 classifier always runs; v0.1 classifier gates on `source == "muse_ble"`. Thread-safe via `threading.Lock`. Ported from Rigpa-v2 `hub.py` + Rigpa-v3 `hub.py`. Public API:
   - `update(payload) → NeurolinkState`
   - `get_state() → NeurolinkState`
   - `get_ea1() → EA1Result`
   - `snapshot() → dict` (for service layer)
   - `get_latest() → EEGSample | None`
   - `reset()`

7. **EEG Pump** — Background asyncio task that reads from adapter at 4 Hz, builds `IngestPayload`, and calls `hub.update()`. Ported from Rigpa-v2 `eeg_pump.py` + Rigpa-v3 `eeg_pump.py`. Handles buffer-fill watchdog and rearm logic for BLE mode.

8. **BLE Bridge** — Full Muse S Gen 1/Gen 2 direct BLE session manager (Rigpa-v2 `ble_bridge.py`). Critical protocol constants preserved verbatim (see Section 14). Runs as background asyncio Task. Supports reconnect supervisor loop.

9. **Calibration Session** — 30-second baseline alpha capture using the adapter. Sets `hub.baseline_alpha`. Ported from Rigpa-v2 `calibration_router.py` + Rigpa-v3 `calibration.py`.

10. **EEG Gate Middleware + Router** — FastAPI middleware that blocks requests requiring an active EEG session. Ported from Rigpa-v2 `eeg_gate_middleware.py` + `eeg_gate_router.py`.

11. **Focus State Classifier** — `classify_focus(score) → FocusState` enum (HIGH_FOCUS / MODERATE_FOCUS / LOW_FOCUS / DISTRACTED). Thresholds from Rigpa-v3 `focus_state.py`. Runs on every hub frame.

12. **Fatigue Detector** — `FatigueDetector.update(theta, alpha) → float` rolling 30-sample window theta/alpha ratio. Ported from Rigpa-v3 `fatigue.py`.

13. **SSE Stream** — `/api/v1/neurolink/stream` Server-Sent Events endpoint. Fans out `NeurolinkState` JSON at 4 Hz to all connected clients. Ported from Rigpa-v2 `eeg_pump.py` SSE fanout.

14. **Session Log** — SQLite/SQLAlchemy async table records every EEG session: start time, device model, adapter type, frame count, final state. Lightweight; not in Rigpa source (new for standalone Neurolink).

15. **Mock Stream** — `MockAdapter` generates deterministic sine-wave EEG + PPG + IMU data. Ported from Rigpa-v2 `mock_stream.py` + Rigpa-v3 `hardware/mock.py`. Activated via `NEUROLINK_ADAPTER_TYPE=mock`.

16. **Health Endpoint** — `GET /health` returns adapter status, hub frame count, Redis ping, and SQLite reachability.

17. **React Dashboard (Frontend)** — Real-time visualization of band powers (bar chart), S-space region, alchemical stage, EA-1 score, HR/HRV, breathing rate, and focus/fatigue scores. SSE consumer. Vite + React 18 + TypeScript.

### User Stories

```
As a practitioner, I want to connect my Muse S headset by BLE MAC address so that
  I can receive live EEG data without running muselsl manually.
  Given NEUROLINK_ADAPTER_TYPE=ble, NEUROLINK_DEVICE_MODEL=muse_s_gen1
  When  POST /api/v1/neurolink/connect is called with {"address": "<mac>"}
  Then  BLE bridge starts, hub receives frames within 5 seconds, SSE stream emits data

As a practitioner, I want to see my current alchemical stage and S-space region
  so that I can track my contemplative depth during a session.
  Given SSE stream is active
  When  EEG frames arrive
  Then  /api/v1/neurolink/stream emits {"alchemical_stage": "Rubedo", "region": "E", ...}

As a practitioner, I want EA-1 eligibility scored on every frame
  so that I know when my state qualifies for advanced contemplative protocols.
  Given a frame with alpha >= 0.30, theta >= 0.15, motion_rms < 0.5
  When  hub.update() is called
  Then  ea1.eligible == True and ea1.score > 0.0

As a developer, I want to run the full app without hardware
  so that I can develop and test on any machine.
  Given NEUROLINK_ADAPTER_TYPE=mock
  When  docker-compose up
  Then  SSE stream emits mock EEG frames with plausible band powers

As a practitioner, I want to calibrate my personal alpha baseline
  so that focus scores are relative to my individual resting state.
  Given adapter connected
  When  POST /api/v1/neurolink/calibrate
  Then  30-second capture runs, hub.baseline_alpha is set, response {"status": "complete", "baseline_alpha": <float>}
```

### Non-Functional Requirements

| Concern         | Requirement                                                      |
|-----------------|------------------------------------------------------------------|
| Latency         | P99 SSE publish latency < 50 ms from frame receipt to fan-out   |
| Throughput      | 10 concurrent SSE clients without frame drops                    |
| Availability    | Single-process; crash-restart via Docker restart policy          |
| Data Retention  | Session log retained in SQLite; no automatic pruning in v1       |
| Security        | No secrets in code; all config via env; CORS restricted to localhost in prod |
| Observability   | structlog JSON; every BLE error logged at WARNING+; /health endpoint |
| Reliability     | BLE bridge reconnects automatically after link drop (≤ 20s wait) |
| Portability     | Works on Linux (BlueZ); macOS support best-effort                |

---

## 4. Architecture and Design

### Project Structure

```
neurolink/
├── backend/
│   ├── src/
│   │   └── neurolink/
│   │       ├── __init__.py
│   │       ├── main.py               # FastAPI app factory + lifespan
│   │       ├── config.py             # pydantic-settings (all env vars)
│   │       ├── dependencies.py       # Depends() providers (hub, adapter, service)
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   ├── eeg.py            # BandPowers, IngestPayload, NeurolinkState, EA1Result, etc.
│   │       │   └── session.py        # SQLAlchemy SessionLog ORM model
│   │       ├── hardware/
│   │       │   ├── __init__.py
│   │       │   ├── base.py           # HardwareAdapter ABC, EEGSample, DeviceModel enum
│   │       │   ├── mock.py           # MockAdapter (sine-wave EEG/PPG/IMU)
│   │       │   ├── muse_s/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── ble_adapter.py    # MuseSBleAdapter (direct bleak BLE)
│   │       │   │   ├── lsl_adapter.py    # MuseSLslAdapter (pylsl consumer)
│   │       │   │   └── compute.py        # compute_all_bands (Welch)
│   │       │   └── muse_athena/
│   │       │       ├── __init__.py
│   │       │       ├── ble_adapter.py    # AthenaBlueAdapter (OpenMuse LSL)
│   │       │       ├── compute.py        # Athena-specific band compute
│   │       │       └── fnirs.py          # FNIRSDecoder (oxy/deoxy)
│   │       ├── dsp/
│   │       │   ├── __init__.py
│   │       │   ├── bandpower.py          # bandpower(), make_buffers()
│   │       │   ├── derived_eeg.py        # derived_eeg() → FAA, FMt
│   │       │   ├── classifiers.py        # classify_v01(), classify_v2(), alchemical stage map
│   │       │   ├── ppg.py                # compute_ppg(), _poincare()
│   │       │   ├── breathing.py          # compute_breathing()
│   │       │   ├── imu.py                # head_orientation()
│   │       │   └── decoders.py           # decode_eeg(), decode_ppg(), decode_imu()
│   │       ├── hub.py                    # EEGHub (in-memory state, thread-safe)
│   │       ├── ea1_scorer.py             # score(payload) → EA1Result
│   │       ├── focus_state.py            # FocusState enum, classify_focus()
│   │       ├── fatigue.py                # FatigueDetector (rolling theta/alpha)
│   │       ├── calibration.py            # CalibrationSession (30s alpha capture)
│   │       ├── adapter_factory.py        # create_adapter() factory
│   │       ├── eeg_pump.py               # EEGPump background task
│   │       ├── ble_bridge.py             # BLE supervisor loop (Muse S BLE mode)
│   │       ├── service.py                # NeuroLinkService (async business logic)
│   │       ├── routers/
│   │       │   ├── __init__.py
│   │       │   ├── neurolink.py          # /api/v1/neurolink/* REST + SSE endpoints
│   │       │   ├── calibration.py        # /api/v1/neurolink/calibrate
│   │       │   └── eeg_gate.py           # EEG gate middleware + /gate/* endpoints
│   │       ├── db/
│   │       │   ├── __init__.py
│   │       │   ├── engine.py             # async SQLite engine + session factory
│   │       │   └── repository.py         # SessionLogRepository
│   │       └── utils/
│   │           └── timing.py             # monotonic clock helpers
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── unit/
│   │   │   ├── dsp/
│   │   │   │   ├── test_bandpower.py
│   │   │   │   ├── test_derived_eeg.py
│   │   │   │   ├── test_classifiers.py
│   │   │   │   ├── test_ppg.py
│   │   │   │   ├── test_breathing.py
│   │   │   │   └── test_imu.py
│   │   │   ├── test_ea1_scorer.py
│   │   │   ├── test_focus_state.py
│   │   │   ├── test_fatigue.py
│   │   │   ├── test_hub.py
│   │   │   ├── test_calibration.py
│   │   │   └── test_adapter_factory.py
│   │   └── integration/
│   │       ├── test_health.py
│   │       ├── test_neurolink_router.py
│   │       ├── test_sse_stream.py
│   │       └── test_session_log.py
│   ├── migrations/                       # Alembic for SQLite SessionLog
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── alembic.ini
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── BandPowerChart.tsx        # Real-time bar chart (5 bands)
│   │   │   ├── SSpaceDisplay.tsx         # S-space region + alchemical stage
│   │   │   ├── EA1Score.tsx              # EA-1 eligibility + score
│   │   │   ├── HRVPanel.tsx              # HR, RMSSD, breathing rate
│   │   │   ├── FocusFatigueGauge.tsx     # Focus state + fatigue score
│   │   │   └── ContactQuality.tsx        # Poor contact indicator
│   │   └── hooks/
│   │       └── useNeurolinkSSE.ts        # SSE consumer hook
│   ├── tests/
│   └── vite.config.ts
├── compose.dev.yml
├── compose.prod.yml
├── .env.template
└── AGENTS.md                             # (this file)
```

### Data Model

```python
# models/eeg.py — All Pydantic v2; ConfigDict(extra="ignore") on every model

class BandPowers(BaseModel):
    model_config = ConfigDict(extra="ignore")
    alpha: float = 0.0
    theta: float = 0.0
    beta:  float = 0.0
    delta: float = 0.0
    gamma: float = 0.0

class SSpaceCoords(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x: float = 0.0   # engagement index = beta / (alpha + theta)
    y: float = 0.0   # integration coverage = alpha / beta
    z: float = 0.0   # theta fraction (raw)

class PoincareIndices(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sd1: float = 0.0
    sd2: float = 0.0
    sd1_sd2_ratio: float = 0.0
    ellipse_area: float = 0.0

class PPGPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    hr_bpm: float = 0.0
    ibi_ms: list[float] = Field(default_factory=list)
    hrv_rmssd: float = 0.0
    hrv_sdnn: float = 0.0
    hrv_pnn50: float = 0.0
    poincare: PoincareIndices | None = None

class BreathingPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rr_bpm: float | None = None
    rr_ppg: float | None = None
    rr_accel: float | None = None

class IMUPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    motion_rms: float = 0.0

class IngestPayload(BaseModel):
    """Full multimodal ingest payload from any hardware adapter."""
    model_config = ConfigDict(extra="ignore")
    # Core EEG
    region: str = "A"
    alchemical_stage: str = "Nigredo"
    ea1_eligible: bool = False
    integration_coverage: float = 0.5
    engagement_index: float = 0.5
    bands: BandPowers = Field(default_factory=BandPowers)
    s_space: SSpaceCoords = Field(default_factory=SSpaceCoords)
    timestamp: float = 0.0
    source: str = "mock"               # "muse_ble" | "muse_lsl" | "athena_ble" | "mock"
    address: str = ""
    # Contact
    poor_contact: bool = False
    contact_quality: float | None = None
    # Derived EEG
    faa: float | None = None           # Frontal Alpha Asymmetry
    fmt: float | None = None           # Frontal Midline Theta
    # Optional multimodal
    ppg: PPGPayload | None = None
    breathing: BreathingPayload | None = None
    imu: IMUPayload | None = None
    # Athena-only
    fnirs_oxy: float | None = None
    fnirs_deoxy: float | None = None

class EA1Criterion(BaseModel):
    value: float | None = None
    threshold: float | None = None
    units: str = ""
    met: bool = False

class EA1Result(BaseModel):
    """EA-1 multimodal eligibility score."""
    eligible: bool = False
    score: float = 0.0
    criteria_met: int = 0
    criteria_total: int = 5
    label: str = "Ineligible"
    gates: dict[str, bool] = Field(default_factory=lambda: {"s_space": False, "motion": True})
    criteria: dict[str, Any] = Field(default_factory=dict)
    overlay_mode: str = "X0"
    alchemical_stage: str = "Nigredo"
    s_space_coords: SSpaceCoords = Field(default_factory=SSpaceCoords)
    s_space_region: str = "A"
    integration_coverage: float = 0.0

class NeurolinkState(BaseModel):
    """Current live state of the Neurolink hub."""
    connected: bool = False
    source: str = "none"
    region: str = "A"
    alchemical_stage: str = "Nigredo"
    integration_coverage: float = 0.0
    engagement_index: float = 0.0
    bands: BandPowers = Field(default_factory=BandPowers)
    s_space: SSpaceCoords = Field(default_factory=SSpaceCoords)
    ea1: EA1Result = Field(default_factory=EA1Result)
    last_ts: float = 0.0
    frame_count: int = 0
    poor_contact: bool = False
    # v0.1 6-region classifier (muse_ble only)
    region_v01: str = "A"
    alchemical_stage_v01: str = "Nigredo"
    # Extended multimodal
    faa: float | None = None
    fmt: float | None = None
    hr_bpm: float | None = None
    hrv_rmssd: float | None = None
    rr_bpm: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    motion_rms: float | None = None
    contact_quality: float | None = None
    # Focus + Fatigue (new in standalone Neurolink)
    focus_state: str = "unknown"       # FocusState enum value
    focus_score: float = 0.0
    fatigue_score: float = 0.0
    # Athena-only
    fnirs_oxy: float | None = None
    fnirs_deoxy: float | None = None

# models/session.py — SQLAlchemy ORM (SQLite)
class SessionLog(Base):
    __tablename__ = "session_logs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]
    device_model: Mapped[str]          # "muse_s_gen1" | "muse_s_athena" | "mock"
    adapter_type: Mapped[str]          # "ble" | "lsl" | "mock"
    address: Mapped[str | None]
    frame_count: Mapped[int] = mapped_column(default=0)
    final_region: Mapped[str | None]
    final_stage: Mapped[str | None]
    final_ea1_eligible: Mapped[bool] = mapped_column(default=False)
```

### API Contract

```python
# All request/response Pydantic v2 schemas — binding contracts

# POST /api/v1/neurolink/connect
class ConnectRequest(BaseModel):
    adapter_type: str = "ble"          # "ble" | "lsl" | "mock"
    device_model: str = "muse_s_gen1"  # "muse_s_gen1" | "muse_s_athena"
    address: str | None = None         # BLE MAC address (required for ble mode)

class ConnectResponse(BaseModel):
    ok: bool
    source: str
    message: str

# POST /api/v1/neurolink/disconnect
class DisconnectResponse(BaseModel):
    ok: bool

# GET /api/v1/neurolink/state
# → NeurolinkState (see models above)

# GET /api/v1/neurolink/stream
# → SSE stream; each event data: NeurolinkState JSON

# GET /api/v1/neurolink/bands?channel=AF7
class BandPowerResponse(BaseModel):
    channel: str
    alpha: float | None = None
    theta: float | None = None
    beta: float | None = None
    delta: float | None = None
    gamma: float | None = None
    error: str | None = None

# POST /api/v1/neurolink/calibrate
class CalibrateResponse(BaseModel):
    status: str                        # "started" | "complete" | "error"
    baseline_alpha: float | None = None

# GET /api/v1/neurolink/ea1
# → EA1Result

# GET /api/v1/neurolink/sessions  (list recent sessions)
class SessionSummary(BaseModel):
    id: int
    started_at: datetime
    ended_at: datetime | None
    device_model: str
    adapter_type: str
    frame_count: int
    final_ea1_eligible: bool

# GET /health
class HealthResponse(BaseModel):
    status: str                        # "ok" | "degraded"
    adapter_type: str
    adapter_connected: bool
    hub_frame_count: int
    redis: str                         # "connected" | "error"
    db: str                            # "connected" | "error"
```

### Key Architectural Decisions

- **Dual classifier tracks always coexist.** v0.1 (6-region S-space) runs when `source == "muse_ble"`. v2 (8-region alchemical) always runs. Both are stored in `NeurolinkState`. This is a hard requirement from Rigpa-v2 `hub.py`.
- **BLE protocol constants are IMMUTABLE.** The double-send `CMD_DATA` pattern with 250 ms gap, the keepalive at 30s, and the 20s reconnect wait are not configurable — they encode hard-won Muse S Gen 2 firmware behavior. See Section 14.
- **Hub is process-global, thread-safe.** One `EEGHub` instance per process. `threading.Lock` guards all writes. SSE fan-out reads via a separate asyncio queue.
- **No business logic in routers.** Routers call `NeuroLinkService` methods only.
- **Adapters are lazy-imported.** `bleak`, `pylsl`, `openmuse` are imported inside adapter constructors so mock mode never requires hardware drivers.
- **neurokit2 and scipy are optional at module level.** `compute_ppg` and `compute_breathing` return `{}` if not installed. They must never raise.
- **SSE fan-out via asyncio.Queue.** Hub publishes to a per-client queue; the SSE endpoint drains it. No Redis pub/sub needed for single-process.
- **Redis is used only for optional live state caching.** The hub's in-memory state is always authoritative. Redis is populated on each `hub.update()` as a convenience for external consumers.
- **SQLite is the session log only.** No EEG frame-by-frame storage — that would fill disk in minutes.
- **Focus and Fatigue computed inside hub.update().** Not in the router or service. Hub receives one `IngestPayload`, enriches it with both classifiers + EA1 + focus + fatigue, then emits `NeurolinkState`.

### Integration Points

| System           | Direction  | Protocol        | Auth     | Notes                                               |
|------------------|------------|-----------------|----------|-----------------------------------------------------|
| Muse S Gen 1 BLE | inbound    | bleak GATT      | none     | MAC address from env/request; keepalive 30s         |
| Muse S Gen 1 LSL | inbound    | pylsl           | none     | Requires muselsl running externally                 |
| Muse S Athena    | inbound    | OpenMuse LSL    | none     | Requires OpenMuse subprocess running externally     |
| Redis            | outbound   | redis-py async  | URL env  | Key: `neurolink:state`; TTL 10s; optional           |
| SQLite           | outbound   | aiosqlite       | file     | Path: `./data/neurolink.db`; configurable via env   |
| SSE clients      | outbound   | HTTP/SSE        | none     | 10 concurrent clients; per-client asyncio.Queue     |
| React frontend   | outbound   | HTTP + SSE      | CORS     | Vite dev proxy to :8000                             |

---

## 5. Commands

> The agent MUST run these commands in order after any code changes. Do not proceed if any fail.

```bash
# ── Setup ──────────────────────────────────────────────────────────────────
cd backend
pip install -e ".[dev]"

# ── Start services ──────────────────────────────────────────────────────────
docker-compose -f ../compose.dev.yml up -d redis

# ── Database migrations ─────────────────────────────────────────────────────
alembic upgrade head

# ── Run ALL tests (unit + integration) ─────────────────────────────────────
pytest -v --tb=short

# ── Unit tests only (no I/O, fast) ─────────────────────────────────────────
pytest tests/unit/ -v

# ── Coverage ────────────────────────────────────────────────────────────────
pytest --cov=neurolink --cov-report=term-missing --cov-fail-under=85

# ── Lint ────────────────────────────────────────────────────────────────────
ruff check . && ruff format --check .

# ── Type check ──────────────────────────────────────────────────────────────
mypy src/neurolink/

# ── Start dev server (mock mode, no hardware needed) ───────────────────────
NEUROLINK_ADAPTER_TYPE=mock uvicorn neurolink.main:app --reload --host 0.0.0.0 --port 8000

# ── Start dev server (BLE mode, Muse S Gen 1) ──────────────────────────────
NEUROLINK_ADAPTER_TYPE=ble \
NEUROLINK_DEVICE_MODEL=muse_s_gen1 \
NEUROLINK_MUSE_BLE_ADDRESS=<mac> \
uvicorn neurolink.main:app --reload --host 0.0.0.0 --port 8000

# ── Docker full stack ───────────────────────────────────────────────────────
docker-compose -f compose.dev.yml up --build

# ── Frontend dev ────────────────────────────────────────────────────────────
cd frontend && npm install && npm run dev

# ── Frontend tests ──────────────────────────────────────────────────────────
cd frontend && npm run test

# ── Smoke test health endpoint ──────────────────────────────────────────────
curl -s http://localhost:8000/health | python3 -m json.tool
```

---

## 6. Testing Strategy

### Framework and Conventions

- **pytest** with `pytest-asyncio` (mode = `asyncio` in `pyproject.toml`)
- **httpx.AsyncClient** with `app` for integration tests
- **No mocking of the DB** in integration tests — use an in-memory SQLite test DB
- **Hardware never imported in unit tests** — `MockAdapter` used everywhere via `NEUROLINK_ADAPTER_TYPE=mock`
- Test files mirror source: `neurolink/dsp/bandpower.py` → `tests/unit/dsp/test_bandpower.py`
- All test functions are `async def test_*`
- **DSP functions are pure** — test with numpy arrays, no fixtures needed
- **Hub tests** use `hub.reset()` in `@pytest.fixture(autouse=True)` to clear state between tests

### Coverage Requirements

| Module Type           | Minimum Coverage |
|-----------------------|-----------------|
| dsp/ (all DSP fns)    | 95%             |
| hub.py                | 92%             |
| ea1_scorer.py         | 90%             |
| focus_state.py        | 95%             |
| fatigue.py            | 95%             |
| calibration.py        | 85%             |
| service.py            | 88%             |
| routers/              | 85%             |
| hardware/mock.py      | 90%             |
| db/repository.py      | 80%             |
| Overall               | 85%             |

### Critical Test Cases (must exist)

```
# DSP
test_bandpower_returns_zero_for_short_signal
test_bandpower_alpha_peak_at_10hz
test_derived_eeg_faa_sign_convention        # log(AF8) - log(AF7) > 0 when AF8 > AF7
test_classify_v01_region_E_for_high_alpha_theta
test_classify_v01_region_F_for_delta_gt_50_pct
test_classify_v01_multiplicatio_escalation
test_classify_v02_rubedo_threshold
test_compute_ppg_returns_empty_for_short_buffer
test_compute_ppg_hr_in_valid_range          # 40-120 bpm
test_compute_breathing_fused_estimate
test_poincare_sd1_sd2_positive
test_head_orientation_pitch_roll_bounded    # ±90 degrees

# EA-1 Scorer
test_ea1_eligible_when_all_criteria_met
test_ea1_ineligible_poor_contact
test_ea1_motion_gate_blocks_eligibility
test_ea1_score_proportional_to_criteria_met

# Hub
test_hub_update_increments_frame_count
test_hub_dual_classifier_both_populated_for_muse_ble
test_hub_v01_not_run_for_mock_source
test_hub_reset_clears_state

# Focus + Fatigue
test_classify_focus_high_above_075
test_classify_focus_distracted_below_025
test_fatigue_detector_zero_when_empty
test_fatigue_detector_high_when_theta_dominates

# Integration
test_health_endpoint_ok_mock_mode
test_connect_mock_returns_ok
test_state_endpoint_returns_neurolink_state
test_sse_stream_emits_at_least_one_frame    # with asyncio timeout 5s
test_calibrate_starts_and_returns_started
test_session_log_created_on_connect

# fNIRS (Athena-only)
test_fnirs_decoder_averages_oxy_channels
test_fnirs_decoder_averages_deoxy_channels
```

### Test Fixtures Pattern

```python
# tests/conftest.py

import pytest
from httpx import AsyncClient, ASGITransport
from neurolink.main import create_app
from neurolink import hub as hub_module

@pytest.fixture(autouse=True)
def reset_hub():
    """Clear hub state before every test."""
    hub_module.reset()
    yield
    hub_module.reset()

@pytest.fixture
def app():
    import os
    os.environ["NEUROLINK_ADAPTER_TYPE"] = "mock"
    return create_app()

@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac
```

---

## 7. Code Style and Conventions

### Rules (binding)

- **Type annotations required** on all function signatures (enforced by mypy strict)
- **No `Any` type** without an inline comment explaining why
- **Dependency injection via `Depends()`** — never instantiate services in route handlers directly
- **Pydantic v2 style:** `model_config = ConfigDict(...)` not `class Config`
- **Error handling:** Custom exception classes in `neurolink/exceptions.py`; register handlers in `main.py`
- **Logging:** `structlog` JSON; never use `print()`
- **DSP functions are pure** — no I/O, no globals, no side effects
- **Hardware imports are lazy** — all `bleak` / `pylsl` imports inside adapter `__init__` or `connect()`
- **`neurokit2` and `scipy` imports are always try/except** — degrade gracefully to `{}` return

### Naming Conventions

| Item             | Convention          | Example                            |
|------------------|---------------------|------------------------------------|
| Files            | `snake_case.py`     | `ble_bridge.py`, `muse_compute.py` |
| Classes          | `PascalCase`        | `MuseSBleAdapter`, `EEGHub`        |
| Functions/vars   | `snake_case`        | `compute_ppg`, `band_powers`       |
| Constants        | `UPPER_SNAKE_CASE`  | `KEEPALIVE_SEC`, `EEG_FS`          |
| Enums            | `PascalCase` class  | `FocusState.HIGH_FOCUS`            |
| Route paths      | `kebab-case`        | `/api/v1/neurolink/band-powers`    |
| DB tables        | `snake_case plural` | `session_logs`                     |
| BLE UUIDs        | lowercase string    | `"273e0003-4c4d-454d-96be-..."` (exact) |

### Code Example (reference pattern)

```python
# ✅ Correct — thin router → service → hub
@router.get("/state", response_model=NeurolinkState)
async def get_state(
    service: NeuroLinkService = Depends(get_neurolink_service),
) -> NeurolinkState:
    return await service.get_current_state()

# ❌ Wrong — business logic + hub access in router
@router.get("/state")
async def get_state():
    from neurolink import hub
    s = hub.get_state()
    s.focus_score = compute_focus(s.bands)   # logic in router
    return s
```

---

## 8. Acceptance Criteria

| ID    | Description                                            | How to Verify                                                      | Pass Condition                                            |
|-------|--------------------------------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------|
| AC1   | `/health` returns 200 in mock mode                     | `curl http://localhost:8000/health`                                | `{"status":"ok","adapter_type":"mock",...}`               |
| AC2   | `POST /api/v1/neurolink/connect` starts mock adapter   | `pytest tests/integration/test_neurolink_router.py::test_connect_mock` | HTTP 200, `{"ok":true,"source":"mock"}`               |
| AC3   | `GET /api/v1/neurolink/state` returns NeurolinkState   | `pytest ...::test_state_endpoint`                                  | HTTP 200, all NeurolinkState fields present               |
| AC4   | SSE stream emits frames                                | `pytest ...::test_sse_stream_emits_at_least_one_frame`             | At least 1 SSE `data:` line received within 5s            |
| AC5   | v0.1 classifier runs for muse_ble source               | `pytest tests/unit/test_hub.py::test_hub_dual_classifier_muse_ble` | `region_v01` and `alchemical_stage_v01` populated         |
| AC6   | v0.1 classifier NOT run for mock source                | `pytest ...::test_hub_v01_not_run_for_mock_source`                 | `region_v01 == "A"` (default), no classify_v01 call       |
| AC7   | EA-1 eligible when criteria met                        | `pytest tests/unit/test_ea1_scorer.py::test_ea1_eligible_when_all_criteria_met` | `ea1.eligible == True`            |
| AC8   | Fatigue score rises with theta dominance               | `pytest ...::test_fatigue_detector_high_when_theta_dominates`      | `score > 0.8` after 30 samples with theta/alpha = 4.0     |
| AC9   | fNIRS decoder extracts oxy/deoxy for Athena            | `pytest tests/unit/dsp/test_fnirs.py`                              | `oxy == mean(evens)`, `deoxy == mean(odds)`                |
| AC10  | BLE bridge reconnects after link drop                  | `pytest ...::test_ble_bridge_reconnect` (mock BleakClient)         | Session restarts after `link_dropped.set()`               |
| AC11  | Calibration sets `baseline_alpha`                      | `pytest ...::test_calibrate_sets_baseline`                         | `hub.baseline_alpha` is float after 30-frame mock session |
| AC12  | Session log created on connect                         | `pytest tests/integration/test_session_log.py`                     | `SELECT COUNT(*) FROM session_logs` == 1                  |
| AC13  | All tests pass with ≥ 85% coverage                     | `pytest --cov=neurolink --cov-fail-under=85`                       | Exit code 0                                               |
| AC14  | No lint errors                                         | `ruff check .`                                                      | Exit code 0                                               |
| AC15  | No type errors                                         | `mypy src/neurolink/`                                               | "Success: no issues found"                                |
| AC16  | Docker Compose starts cleanly                          | `docker-compose -f compose.dev.yml up --build`                     | No FATAL/ERROR lines; `/health` returns 200 within 10s    |

---

## 9. Boundaries and Guardrails

### ✅ Always (no approval needed)
- Run `pytest` and `ruff check .` before reporting a task complete
- Use UTC for all datetimes; store as timezone-aware
- Log every unhandled exception with stack trace via `structlog`
- Use `async def` for all I/O-bound operations
- Add docstrings to all public service, DSP, and hardware methods
- Copy BLE protocol constants verbatim from Section 14 — do not recalculate

### ⚠️ Ask First (pause and confirm before doing)
- Adding any new Python dependency to `pyproject.toml`
- Changing the SQLite schema (alembic migrations)
- Modifying `compose.dev.yml` or `Dockerfile`
- Adding a new top-level module or package
- Changing any existing API response schema (breaking change)
- Implementing anything not explicitly described in Section 3
- Modifying the BLE protocol constants in Section 14

### 🚫 Never (hard stop — do not proceed)
- Commit secrets, API keys, BLE addresses, or tokens to any file
- Use `os.system()` or `subprocess` unless explicitly spec'd (muselsl/OpenMuse are external)
- Write blocking I/O in `async def` functions (use `asyncio.to_thread()` for bleak if needed)
- Delete or modify files in `migrations/` unless running `alembic revision`
- Modify BLE UUID constants or command bytes without explicit approval
- Store per-frame EEG samples in the database (only session metadata)
- Use `SELECT *` in repository queries
- Silence exceptions with bare `except: pass` — always log and return graceful fallback

---

## 10. Task Breakdown

> Execute tasks **in this exact order**. Each task is a single agent session.
> Do not begin a task until the previous one passes all relevant acceptance criteria.

### Phase 1: Foundation

- [ ] **Task 1.1** — Scaffold project structure: directories, `pyproject.toml`, `Dockerfile`, `compose.dev.yml`, `.env.template`, `AGENTS.md`
- [ ] **Task 1.2** — Implement `config.py` with all env vars (see Section 14 env table); verify with `mypy`
- [ ] **Task 1.3** — Implement `db/engine.py` + `models/session.py` + `db/repository.py`; run `alembic init`, create migration, run `alembic upgrade head` against `./data/test.db`
- [ ] **Task 1.4** — Implement `/health` endpoint (stub adapter/hub); verify AC1

### Phase 2: DSP Core

- [ ] **Task 2.1** — Implement `dsp/bandpower.py`: `bandpower()`, `make_buffers()`. Unit tests first. Verify `test_bandpower_alpha_peak_at_10hz`.
- [ ] **Task 2.2** — Implement `dsp/decoders.py`: `decode_eeg()`, `decode_ppg()`, `decode_imu()` (ported from Rigpa-v2 `muse_decoders.py`). Unit tests.
- [ ] **Task 2.3** — Implement `dsp/derived_eeg.py`: `derived_eeg()` → FAA + FMt. Unit tests. Verify sign convention AC (AC per test above).
- [ ] **Task 2.4** — Implement `dsp/classifiers.py`: both `classify_v01()` (6-region) and `classify_v2()` (8-region) + alchemical maps. Unit tests covering all 6 regions and Multiplicatio escalation.
- [ ] **Task 2.5** — Implement `dsp/ppg.py`: `compute_ppg()` + `_poincare()`. Unit tests. Verify graceful degradation without neurokit2.
- [ ] **Task 2.6** — Implement `dsp/breathing.py`: `compute_breathing()`. Unit tests. Verify graceful degradation without scipy.
- [ ] **Task 2.7** — Implement `dsp/imu.py`: `head_orientation()`. Unit tests. Verify pitch/roll bounds.
- [ ] **Task 2.8** — Implement `hardware/muse_s/compute.py`: `compute_all_bands()` (Welch). Unit tests.
- [ ] **Task 2.9** — Implement `hardware/muse_athena/fnirs.py`: `FNIRSDecoder`. Unit tests. Verify AC9.

### Phase 3: Classifiers and Scorers

- [ ] **Task 3.1** — Implement `ea1_scorer.py`: 5-criterion scorer, gate logic. Unit tests. Verify AC7.
- [ ] **Task 3.2** — Implement `focus_state.py`: `FocusState` enum, `classify_focus()`, `is_blocking()`. Unit tests.
- [ ] **Task 3.3** — Implement `fatigue.py`: `FatigueDetector`. Unit tests. Verify AC8.

### Phase 4: Hardware Adapters

- [ ] **Task 4.1** — Implement `hardware/base.py`: `HardwareAdapter` ABC, `EEGSample`, `DeviceModel` enum (ported from Rigpa-v3 `hardware/base.py`).
- [ ] **Task 4.2** — Implement `hardware/mock.py`: `MockAdapter` with deterministic sine-wave data at 4 Hz (ported from Rigpa-v2 `mock_stream.py` + Rigpa-v3 `hardware/mock.py`). Unit tests.
- [ ] **Task 4.3** — Implement `hardware/muse_s/ble_adapter.py`: `MuseSBleAdapter`. Port GATT UUID map, `_arm_stream()`, `_ctrl_data_double()`, `_connect_with_retry()`, `_keepalive_task()`. Protocol constants from Section 14 verbatim. Unit tests with mocked `BleakClient`.
- [ ] **Task 4.4** — Implement `hardware/muse_s/lsl_adapter.py`: `MuseSLslAdapter` (pylsl consumer). Unit tests with mocked pylsl.
- [ ] **Task 4.5** — Implement `hardware/muse_athena/ble_adapter.py`: `AthenaBlueAdapter` (OpenMuse LSL consumer + fNIRS). Unit tests with mocked pylsl.
- [ ] **Task 4.6** — Implement `adapter_factory.py`: `create_adapter()` factory (ported from Rigpa-v3). All lazy imports. Unit tests for all 4 combinations. Verify adapter factory test.

### Phase 5: Hub and Pipeline

- [ ] **Task 5.1** — Implement `hub.py`: `EEGHub` class with dual-classifier enrichment, EA-1 scoring, focus/fatigue state, thread-safe lock. Verify AC5, AC6. Unit tests for `update()`, `reset()`, `snapshot()`.
- [ ] **Task 5.2** — Implement `ble_bridge.py`: full BLE supervisor loop (ported from Rigpa-v2 `ble_bridge.py`). Protocol constants verbatim. Verify AC10.
- [ ] **Task 5.3** — Implement `eeg_pump.py`: `EEGPump` asyncio background task at 4 Hz. Buffer-fill watchdog. Unit tests with `MockAdapter`.
- [ ] **Task 5.4** — Implement `calibration.py`: `CalibrationSession` (30-second alpha capture). Unit tests. Verify AC11.

### Phase 6: Service and Routers

- [ ] **Task 6.1** — Implement `service.py`: `NeuroLinkService` with `get_current_state()`, `get_band_powers()`, `connect()`, `disconnect()`, `start_calibration()`. No business logic may be added to routers.
- [ ] **Task 6.2** — Implement `routers/neurolink.py`: `connect`, `disconnect`, `state`, `bands`, `ea1`, `sessions` endpoints. Verify AC2, AC3.
- [ ] **Task 6.3** — Implement `routers/neurolink.py` SSE: `GET /api/v1/neurolink/stream`. Asyncio queue fan-out. Verify AC4.
- [ ] **Task 6.4** — Implement `routers/calibration.py`: `POST /calibrate`. Verify AC11.
- [ ] **Task 6.5** — Implement `routers/eeg_gate.py` + `eeg_gate_middleware.py` (ported from Rigpa-v2). Unit tests.
- [ ] **Task 6.6** — Implement `dependencies.py`: all `Depends()` providers. Wire into `main.py`.

### Phase 7: Session Log Integration

- [ ] **Task 7.1** — Connect `SessionLogRepository` to `NeuroLinkService.connect()` and `disconnect()`. Verify AC12.
- [ ] **Task 7.2** — Wire Redis live state cache into `hub.update()` (best-effort; errors silently logged).

### Phase 8: Hardening

- [ ] **Task 8.1** — Ensure full coverage ≥ 85%; fill gaps with targeted tests. Verify AC13.
- [ ] **Task 8.2** — Fix all `ruff` and `mypy` issues. Verify AC14, AC15.
- [ ] **Task 8.3** — Docker Compose end-to-end smoke test. Verify AC16.

### Phase 9: Frontend

- [ ] **Task 9.1** — Scaffold React/TypeScript/Vite frontend: `vite.config.ts` with proxy to `:8000`, ESLint + Vitest config.
- [ ] **Task 9.2** — Implement `useNeurolinkSSE` hook: SSE consumer parsing `NeurolinkState` JSON.
- [ ] **Task 9.3** — Implement `BandPowerChart` component: 5-bar real-time chart (recharts or Chart.js).
- [ ] **Task 9.4** — Implement `SSpaceDisplay`, `EA1Score`, `HRVPanel`, `FocusFatigueGauge`, `ContactQuality` components.
- [ ] **Task 9.5** — Integrate all components into `App.tsx`. Write Vitest component tests. Ensure `npm run test` passes.

---

## 11. Self-Verification Checklist

> The agent must complete this checklist and return it filled in before the spec is considered done.
> Do NOT use "should", "looks like", or "probably". Cite actual command output.

```
[ ] pytest output: [paste last 5 lines of pytest run]
[ ] Coverage: [paste coverage summary line, e.g. "TOTAL  2341  1987  85%"]
[ ] ruff output: [paste output, must be "All checks passed."]
[ ] mypy output: [paste last line, must be "Success: no issues found in N source files"]
[ ] AC1 verified: [paste curl output]
[ ] AC2 verified: [paste pytest output line]
[ ] AC3 verified: [paste pytest output line]
[ ] AC4 verified: [paste pytest output line]
[ ] AC5 verified: [paste pytest output line]
[ ] AC7 verified: [paste pytest output line]
[ ] AC9 verified: [paste pytest output line]
[ ] AC13 verified: [paste "pytest --cov-fail-under=85" exit code]
[ ] AC16 verified: [paste first 3 startup lines from docker-compose up + health check output]
[ ] No secrets committed: [output of "git log --oneline -5" — confirm no .env or *.key files]
```

---

## 12. Performance Targets

| Endpoint / Operation               | Target Latency (P99) | Notes                          |
|------------------------------------|----------------------|--------------------------------|
| `GET /health`                       | < 10 ms             | No I/O                         |
| `GET /api/v1/neurolink/state`       | < 20 ms             | In-memory hub read             |
| `GET /api/v1/neurolink/bands`       | < 50 ms             | Single-channel FFT             |
| SSE publish latency                 | < 50 ms             | Frame receipt → client delivery|
| BLE frame → hub.update() complete   | < 5 ms              | Sync DSP pipeline              |
| `POST /api/v1/neurolink/connect`    | < 200 ms            | Adapter init; BLE scan is async|

### Error Handling Taxonomy

| Category              | HTTP Status | Error Code           | Retry? | Message                              |
|-----------------------|-------------|----------------------|--------|--------------------------------------|
| Validation error      | 422         | `VALIDATION_ERROR`   | No     | "Invalid input: {field detail}"      |
| Adapter not connected | 503         | `NOT_CONNECTED`      | Yes    | "EEG adapter not connected"          |
| BLE scan timeout      | 504         | `BLE_TIMEOUT`        | Yes    | "Muse S not found at {address}"      |
| Calibration in progress| 409        | `CALIBRATION_BUSY`   | No     | "Calibration already running"        |
| Device model unknown  | 400         | `UNKNOWN_DEVICE`     | No     | "Unknown device_model: {model}"      |
| No EEG data yet       | 202         | `NO_DATA`            | Yes    | "No EEG data available yet"          |

---

## 13. Open Questions

| #  | Question                                                          | Owner    | Resolution                         |
|----|-------------------------------------------------------------------|----------|------------------------------------|
| 1  | Should Neurolink expose a `POST /api/v1/neurolink/ingest` endpoint for external LSL push (vs. pull)? | Dev | Default: no; LSL adapters poll internally |
| 2  | Should the fNIRS oxy/deoxy values be stored in the session log?  | Dev      | No in v1; add in v2 if needed      |
| 3  | Should focus_score be computed from absolute alpha or calibration-normalized alpha? | Dev | Calibration-normalized when `baseline_alpha` is set; raw otherwise |
| 4  | Should the SSE stream publish at exactly 4 Hz or only on new frames? | Dev | On new frames (driven by EEG pump cadence ≈ 4 Hz) |
| 5  | What happens if both `ble_bridge.py` and `lsl_adapter.py` are started simultaneously? | Dev | Singleton hub; only one adapter active per process |

---

## 14. Glossary and Source Inventory

### Domain Glossary

| Term                | Definition                                                                                      |
|---------------------|-------------------------------------------------------------------------------------------------|
| S-space             | A 2D embedding of EEG state: x = engagement index (beta/(alpha+theta)), y = integration (alpha/beta) |
| FAA                 | Frontal Alpha Asymmetry = ln(α_AF8) − ln(α_AF7). Positive = approach, negative = withdrawal    |
| FMt                 | Frontal Midline Theta power at FPz/AUX. Validated meditation-depth indicator (Aftanas 2001)     |
| EA-1                | "Eligibility Assessment 1" — 5-criterion multimodal gate for advanced practice protocols        |
| Alchemical stage    | Contemplative depth label: Nigredo → Albedo → Citrinitas → Rubedo → Multiplicatio → Coagulatio |
| Region (v0.1)       | 6-region classifier: A (scattered), B (alerting), C (settling), D (flow), E (meditation), F (delta-contaminated) |
| Region (v2)         | 8-region classifier from `classifier.py` — band-ratio based, model-agnostic                    |
| p50                 | Muse S Gen 2 firmware multimodal preset: 4ch EEG @ 256 Hz + PPG @ 64 Hz + IMU @ 52 Hz        |
| poor_contact        | `delta > 0.50` — heuristic for electrode contact failure or drowsiness (delta contamination)   |
| Poincaré            | RR-interval scatterplot; SD1 = short-term HRV, SD2 = long-term HRV                             |
| BLE double-send     | Critical Muse S Gen 2 protocol: CMD_DATA must be sent TWICE with 250 ms gap                    |
| OpenMuse            | External process that streams Muse S Athena data as LSL outlet (fNIRS + EEG)                   |
| muselsl             | External process that streams Muse S Gen 1 EEG as LSL outlet                                   |

### BLE Protocol Constants (DO NOT MODIFY)

```python
# Source: Rigpa-v2 backend/src/rigpa/neurolink/ble_bridge.py
# These are firmware-level constants. Any change requires hardware re-validation.

_UUID_CONTROL = "273e0001-4c4d-454d-96be-f03bac821358"
_UUID_EEG_TP9 = "273e0003-4c4d-454d-96be-f03bac821358"
_UUID_EEG_AF7 = "273e0004-4c4d-454d-96be-f03bac821358"
_UUID_EEG_AF8 = "273e0005-4c4d-454d-96be-f03bac821358"
_UUID_EEG_TP10= "273e0006-4c4d-454d-96be-f03bac821358"
_UUID_EEG_AUX = "273e0007-4c4d-454d-96be-f03bac821358"
_UUID_PPG1    = "273e000f-4c4d-454d-96be-f03bac821358"
_UUID_PPG2    = "273e0010-4c4d-454d-96be-f03bac821358"
_UUID_PPG3    = "273e0011-4c4d-454d-96be-f03bac821358"
_UUID_ACC     = "273e000a-4c4d-454d-96be-f03bac821358"
_UUID_GYRO    = "273e0009-4c4d-454d-96be-f03bac821358"

_CMD_HALT   = bytes([0x02, 0x68, 0x0a])          # "h"
_CMD_PRESET = bytes([0x05, 0x70, 0x35, 0x30, 0x0a])  # "p50"
_CMD_START  = bytes([0x02, 0x73, 0x0a])           # "s"
_CMD_DATA   = bytes([0x02, 0x64, 0x0a])           # "d"

_DATA_DOUBLE_SEND_GAP: float = 0.25   # seconds between two CMD_DATA sends
KEEPALIVE_SEC: float = 30.0           # re-arm interval (Muse drops at ~50s idle)
RECONNECT_WAIT: float = 20.0          # wait after link drop before reconnect
PUBLISH_HZ: float = 4.0              # hub publish cadence

# Arming sequence: halt → 50ms → preset[p50] → 50ms → start → 50ms → data[1/2] → 250ms → data[2/2]
```

### v0.1 6-Region Classifier Map (DO NOT MODIFY)

```python
# Source: Rigpa-v2 backend/src/rigpa/neurolink/muse_compute.py

_REGION_TO_STAGE = {
    "A": "Nigredo",     # scattered/default (low alpha, low theta)
    "B": "Albedo",      # alerting/beta-driven (high beta, low alpha)
    "C": "Albedo",      # settling (rising alpha)
    "D": "Citrinitas",  # flow (alpha + beta balanced)
    "E": "Rubedo",      # meditation (high alpha, theta moderate)
    "F": "Coagulatio",  # delta-contaminated (poor contact / drowsy)
}

# Region assignment logic:
# F if delta > 0.50
# A if alpha < 0.15 and theta < 0.15
# B if beta > 0.35 and alpha < 0.20
# C if alpha >= 0.25 and theta < 0.20
# D if alpha >= 0.25 and theta >= 0.20 and beta >= 0.15
# E if alpha >= 0.30 and theta >= 0.15 and beta < 0.20
# else A

# Multiplicatio escalation: Rubedo → Multiplicatio
# when alpha >= 0.35 and theta >= 0.15 and (faa is None or faa >= -0.05)
```

### v2 8-Region Alchemical Classifier (source: Rigpa-v2 classifier.py)

The v2 classifier uses `compute_s_space(bands)`, `classify_region(s_space)`, and `classify_alchemical_stage(bands)`. These are pure functions operating on `BandPowers`. Port them verbatim from `backend/src/rigpa/neurolink/classifier.py`.

### EA-1 Scorer Criteria (source: Rigpa-v2 ea1_scorer.py)

5 criteria evaluated per frame:
1. `alpha_power >= threshold` (default 0.25)
2. `theta_power >= threshold` (default 0.15)
3. `s_space_gate`: region in {"C","D","E"} and not poor_contact
4. `motion_gate`: `motion_rms < 0.5` (or None → pass)
5. `contact_quality >= 0.5` (or None → pass)

Score = `criteria_met / criteria_total`. Eligible if `score >= 0.60` (3+ criteria).
`overlay_mode` = "X0" (0 met), "X1" (1 met), ... "X5" (5 met).

### Environment Variables

| Variable                    | Default         | Description                                             |
|-----------------------------|-----------------|----------------------------------------------------------|
| `NEUROLINK_ADAPTER_TYPE`    | `mock`          | `"mock"` \| `"ble"` \| `"lsl"`                          |
| `NEUROLINK_DEVICE_MODEL`    | `muse_s_gen1`   | `"muse_s_gen1"` \| `"muse_s_athena"`                    |
| `NEUROLINK_MUSE_BLE_ADDRESS`| `""`            | BLE MAC address (required for `ble` mode)               |
| `NEUROLINK_REDIS_URL`       | `redis://localhost:6379/0` | Redis URL (optional; set empty to disable)   |
| `NEUROLINK_DB_PATH`         | `./data/neurolink.db` | SQLite file path                                   |
| `NEUROLINK_CORS_ORIGINS`    | `http://localhost:5173` | Comma-separated CORS origins                     |
| `NEUROLINK_LOG_JSON`        | `false`         | `true` for production JSON logs                         |
| `NEUROLINK_EEG_MODE`        | `ble`           | Passed to startup for backward compat with Rigpa-v2     |
| `NEUROLINK_PUBLISH_HZ`      | `4.0`           | EEG pump cadence (do not change without DSP review)     |
| `NEUROLINK_PUBLISH_HZ`      | `4.0`           | EEG pump cadence (do not change without DSP review)     |

### Source File Mapping (Rigpa → Neurolink)

| Rigpa Source File                                       | Neurolink Destination                          | Notes                                |
|---------------------------------------------------------|------------------------------------------------|--------------------------------------|
| `rigpa-v2: neurolink/ble_bridge.py`                     | `hardware/muse_s/ble_adapter.py` + `ble_bridge.py` | Split BLE session from supervisor |
| `rigpa-v2: neurolink/muse_compute.py`                   | `dsp/bandpower.py`, `dsp/derived_eeg.py`, `dsp/classifiers.py`, `dsp/ppg.py`, `dsp/breathing.py`, `dsp/imu.py` | Decomposed by function |
| `rigpa-v2: neurolink/muse_decoders.py`                  | `dsp/decoders.py`                              | Direct port                          |
| `rigpa-v2: neurolink/models.py`                         | `models/eeg.py`                                | Extended with focus/fatigue/fnirs    |
| `rigpa-v2: neurolink/hub.py`                            | `hub.py`                                       | Merged with v3 hub; focus/fatigue added |
| `rigpa-v2: neurolink/ea1_scorer.py`                     | `ea1_scorer.py`                                | Direct port                          |
| `rigpa-v2: neurolink/classifier.py`                     | `dsp/classifiers.py` (v2 path)                 | Direct port                          |
| `rigpa-v2: neurolink/calibration_router.py`             | `calibration.py` + `routers/calibration.py`   | Separated session from router        |
| `rigpa-v2: neurolink/eeg_pump.py`                       | `eeg_pump.py`                                  | Direct port                          |
| `rigpa-v2: neurolink/mock_stream.py`                    | `hardware/mock.py`                             | Merged with v3 MockAdapter           |
| `rigpa-v2: neurolink/eeg_gate_middleware.py`            | `routers/eeg_gate.py`                          | Direct port                          |
| `rigpa-v2: neurolink/lsl_stream.py`                     | `hardware/muse_s/lsl_adapter.py`               | Merged with v3 lsl_adapter           |
| `rigpa-v3: plugins/neurolink/hardware/base.py`          | `hardware/base.py`                             | Direct port                          |
| `rigpa-v3: plugins/neurolink/hardware/mock.py`          | `hardware/mock.py`                             | Merged with v2 mock_stream           |
| `rigpa-v3: plugins/neurolink/hardware/muse_s/ble_adapter.py` | `hardware/muse_s/ble_adapter.py`          | Merged with v2 ble_bridge            |
| `rigpa-v3: plugins/neurolink/hardware/muse_s/compute.py`| `hardware/muse_s/compute.py`                  | Direct port                          |
| `rigpa-v3: plugins/neurolink/hardware/muse_s/lsl_adapter.py`| `hardware/muse_s/lsl_adapter.py`          | Direct port                          |
| `rigpa-v3: plugins/neurolink/hardware/muse_athena/ble_adapter.py`| `hardware/muse_athena/ble_adapter.py`| Direct port                          |
| `rigpa-v3: plugins/neurolink/hardware/muse_athena/compute.py` | `hardware/muse_athena/compute.py`         | Direct port                          |
| `rigpa-v3: plugins/neurolink/hardware/muse_athena/fnirs.py`| `hardware/muse_athena/fnirs.py`           | Direct port                          |
| `rigpa-v3: plugins/neurolink/adapter_factory.py`        | `adapter_factory.py`                           | Direct port                          |
| `rigpa-v3: plugins/neurolink/calibration.py`            | `calibration.py`                               | Merged with v2 calibration_router    |
| `rigpa-v3: plugins/neurolink/eeg_pump.py`               | `eeg_pump.py`                                  | Merged with v2 eeg_pump              |
| `rigpa-v3: plugins/neurolink/fatigue.py`                | `fatigue.py`                                   | Direct port                          |
| `rigpa-v3: plugins/neurolink/focus_state.py`            | `focus_state.py`                               | Direct port                          |
| `rigpa-v3: plugins/neurolink/hub.py`                    | `hub.py`                                       | Merged with v2 hub                   |
| `rigpa-v3: plugins/neurolink/router.py`                 | `routers/neurolink.py`                         | Merged with v2 gate router           |
| `rigpa-v3: plugins/neurolink/schemas.py`                | `models/eeg.py`                                | Merged into unified models           |
| `rigpa-v3: plugins/neurolink/service.py`                | `service.py`                                   | Merged with v2 service               |
