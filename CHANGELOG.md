# Changelog

All notable changes to Neurolink-v1 are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Dedicated unit tests for `artifact_detector`, `artifact_gate`, `ocular_regression`, `cardiac_regression`, `asr`, `bad_channels`, `spherical_spline`, `baseline`, and `filter_toggles` — closes all DSP coverage gaps identified in the multi-pass audit.
- `SECURITY.md` — vulnerability disclosure policy.
- `CHANGELOG.md` — this file.

---

## [0.1.0] — 2026-06-18

### Added
- Full DSP pipeline: `online_filter` → `artifact_detector` / `artifact_gate` → `bandpower` → `classifiers` → `ea1_scorer`.
- Multi-modal sensor stack: EEG, fNIRS, PPG, IMU, breathing.
- Artifact correction layers: `ocular_regression` (EOG), `cardiac_regression` (BCG/ECG).
- Adaptive spatial rejection: `asr.py`.
- Bad channel detection and spherical spline interpolation.
- Baseline normalization (rolling and fixed modes).
- Filter toggles with enable/disable/reset API.
- Stage 0 readiness gating: environment check, impedance check, IMU gate.
- Routers for stages 0, 1, 2, 3, 3b; BLE; calibration; filters; health; EEG gate.
- Hardware adapters: mock, Muse S, Muse Athena.
- Redis cache client and PostgreSQL DB repository with Alembic migrations.
- Docker Compose configs for dev (`compose.dev.yml`) and prod (`compose.prod.yml`).
- Living system specification (`neurolink_spec.md`).
- Unit test suite: 50+ files covering hub, EEG pump, calibration session, EA1 scorer, fNIRS, band power, classifiers, BLE bridge, service lifecycle, DB engine, DB repository, and Redis client.

---

[Unreleased]: https://github.com/rmholston420/Neurolink-v1/compare/HEAD...HEAD
[0.1.0]: https://github.com/rmholston420/Neurolink-v1/releases/tag/v0.1.0
