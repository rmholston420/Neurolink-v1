"""Unit tests for Pydantic models — serialisation, defaults, field contracts."""

from __future__ import annotations

from neurolink.models.eeg import (
    BandPowers,
    IngestPayload,
    NeurolinkState,
)


class TestBandPowers:
    def test_defaults_to_zero(self):
        bp = BandPowers()
        assert bp.alpha == bp.theta == bp.beta == bp.delta == bp.gamma == 0.0

    def test_round_trip_json(self):
        bp = BandPowers(alpha=0.3, theta=0.2, beta=0.25, delta=0.15, gamma=0.1)
        bp2 = BandPowers.model_validate_json(bp.model_dump_json())
        assert bp == bp2


class TestNeurolinkState:
    def test_eeg_samples_defaults_to_empty_list(self):
        state = NeurolinkState()
        assert state.eeg_samples == []

    def test_eeg_samples_round_trip(self):
        samples = [[float(i) for i in range(64)]] * 4
        state = NeurolinkState(eeg_samples=samples)
        dumped = state.model_dump()
        assert dumped["eeg_samples"] == samples

    def test_json_serialisation_includes_eeg_samples(self):
        samples = [[1.0, 2.0]] * 4
        state = NeurolinkState(eeg_samples=samples)
        json_str = state.model_dump_json()
        assert "eeg_samples" in json_str

    def test_connected_defaults_false(self):
        state = NeurolinkState()
        assert state.connected is False

    def test_frame_count_defaults_zero(self):
        state = NeurolinkState()
        assert state.frame_count == 0


class TestIngestPayload:
    def test_eeg_samples_defaults_to_empty(self):
        p = IngestPayload()
        assert p.eeg_samples == []

    def test_eeg_samples_stored_correctly(self):
        samples = [[0.1 * i for i in range(32)]] * 4
        p = IngestPayload(eeg_samples=samples)
        assert p.eeg_samples == samples

    def test_timestamp_auto_populated(self):
        p = IngestPayload()
        assert p.timestamp > 0
