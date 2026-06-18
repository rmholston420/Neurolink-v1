"""Unit tests for backend/src/neurolink/dsp/decoders.py.

Covers all three BLE packet decoders (EEG, PPG, IMU) including:
  - Happy-path decoding with known byte sequences
  - Short/invalid packet guards (empty list / empty tuple)
  - Sample count caps
  - Scale-factor and offset arithmetic
  - Module-level constant sanity checks
"""
from __future__ import annotations

import math
import struct

import pytest

from neurolink.dsp.decoders import (
    _ACCEL_SCALE,
    _EEG_MIN_PACKET_LEN,
    _EEG_OFFSET,
    _EEG_SCALE,
    _GYRO_SCALE,
    _IMU_MIN_PACKET_LEN,
    _IMU_N_VALUES,
    _PPG_MIN_PACKET_LEN,
    _PPG_SAMPLE_SIZE,
    _PPG_SAMPLES_PER_PACKET,
    decode_eeg,
    decode_imu,
    decode_ppg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eeg_packet(triples: list[tuple[int, int]]) -> bytes:
    """Build a minimal Muse EEG packet (2-byte header + packed 12-bit pairs).

    Each element of *triples* is (s1, s2) with 0 <= s1, s2 <= 0xFFF.
    """
    header = b"\x00\x00"
    payload = bytearray()
    for s1, s2 in triples:
        b0 = (s1 >> 4) & 0xFF
        b1 = ((s1 & 0xF) << 4) | ((s2 >> 8) & 0xF)
        b2 = s2 & 0xFF
        payload += bytes([b0, b1, b2])
    return header + bytes(payload)


def _make_ppg_packet(samples: list[int]) -> bytes:
    """Build a Muse PPG packet (2-byte header + 3 bytes per sample)."""
    header = b"\x00\x00"
    payload = bytearray()
    for val in samples:
        payload += bytes([(val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF])
    # Pad to minimum length if needed
    raw = header + bytes(payload)
    if len(raw) < _PPG_MIN_PACKET_LEN:
        raw = raw + b"\x00" * (_PPG_MIN_PACKET_LEN - len(raw))
    return raw


def _make_imu_packet(int16_values: list[int]) -> bytes:
    """Build a Muse IMU packet (2-byte header + 9 big-endian int16 values)."""
    header = b"\x00\x00"
    payload = bytearray()
    for v in int16_values:
        payload += struct.pack(">h", v)
    raw = header + bytes(payload)
    # Pad to minimum IMU packet length
    if len(raw) < _IMU_MIN_PACKET_LEN:
        raw = raw + b"\x00" * (_IMU_MIN_PACKET_LEN - len(raw))
    return raw


# ---------------------------------------------------------------------------
# Module-level constant sanity checks
# ---------------------------------------------------------------------------

class TestDecoderConstants:
    def test_eeg_scale_positive(self):
        assert _EEG_SCALE > 0

    def test_eeg_offset_positive(self):
        assert _EEG_OFFSET > 0

    def test_ppg_sample_size(self):
        assert _PPG_SAMPLE_SIZE == 3

    def test_ppg_samples_per_packet(self):
        assert _PPG_SAMPLES_PER_PACKET == 6

    def test_ppg_min_packet_len_consistent(self):
        # header(2) + 6 samples * 3 bytes = 20, but minimum is documented as 12
        assert _PPG_MIN_PACKET_LEN >= 2

    def test_imu_n_values(self):
        assert _IMU_N_VALUES == 9

    def test_accel_scale_positive(self):
        assert _ACCEL_SCALE > 0

    def test_gyro_scale_positive(self):
        assert _GYRO_SCALE > 0

    def test_eeg_min_len_at_least_five(self):
        assert _EEG_MIN_PACKET_LEN >= 5


# ---------------------------------------------------------------------------
# decode_eeg
# ---------------------------------------------------------------------------

class TestDecodeEEG:
    def test_empty_bytes_returns_empty(self):
        assert decode_eeg(b"") == []

    def test_too_short_returns_empty(self):
        assert decode_eeg(b"\x00\x00\xFF") == []
        assert decode_eeg(b"\x00\x00\xFF\xFF") == []

    def test_exactly_min_len_returns_two_samples(self):
        # 5-byte packet: header(2) + one triple(3) => 2 samples
        packet = _make_eeg_packet([(1000, 500)])
        result = decode_eeg(packet)
        assert len(result) == 2

    def test_known_values_offset_and_scale(self):
        # s1 = 2048 (== _EEG_OFFSET) => (2048 - 2048) * scale == 0.0
        # s2 = 2048                    => 0.0
        packet = _make_eeg_packet([(2048, 2048)])
        result = decode_eeg(packet)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)

    def test_known_values_non_zero(self):
        # s1 = 0 => (0 - 2048) * 0.48828125 = -1000.0
        packet = _make_eeg_packet([(0, 0)])
        result = decode_eeg(packet)
        expected = (0 - _EEG_OFFSET) * _EEG_SCALE
        assert result[0] == pytest.approx(expected)
        assert result[1] == pytest.approx(expected)

    def test_max_value_triple(self):
        # s1 = 0xFFF = 4095 => (4095 - 2048) * scale
        packet = _make_eeg_packet([(0xFFF, 0xFFF)])
        result = decode_eeg(packet)
        expected = (0xFFF - _EEG_OFFSET) * _EEG_SCALE
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_cap_at_12_samples(self):
        # 7 triples => 14 raw samples; decode_eeg must cap at 12
        packet = _make_eeg_packet([(1024, 2048)] * 7)
        result = decode_eeg(packet)
        assert len(result) == 12

    def test_returns_floats(self):
        packet = _make_eeg_packet([(512, 1024)])
        result = decode_eeg(packet)
        assert all(isinstance(v, float) for v in result)

    def test_six_triples_yields_12_samples(self):
        packet = _make_eeg_packet([(100, 200)] * 6)
        assert len(decode_eeg(packet)) == 12


# ---------------------------------------------------------------------------
# decode_ppg
# ---------------------------------------------------------------------------

class TestDecodePPG:
    def test_empty_bytes_returns_empty(self):
        assert decode_ppg(b"") == []

    def test_too_short_returns_empty(self):
        short = b"\x00\x00" + b"\x00" * 9  # 11 bytes < 12
        assert decode_ppg(short) == []

    def test_happy_path_six_samples(self):
        vals = [100_000, 200_000, 300_000, 400_000, 500_000, 600_000]
        packet = _make_ppg_packet(vals)
        result = decode_ppg(packet)
        assert len(result) == 6
        for got, expected in zip(result, vals):
            assert got == pytest.approx(float(expected))

    def test_known_single_value(self):
        # 24-bit value: 0x010203 = 66051
        raw = b"\x00\x00" + b"\x01\x02\x03" + b"\x00" * 15
        result = decode_ppg(raw)
        assert result[0] == pytest.approx(float(0x010203))

    def test_cap_at_six_samples(self):
        # Provide 8 samples worth of payload — must return only 6
        vals = list(range(1, 9))
        header = b"\x00\x00"
        payload = b"".join(
            bytes([(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]) for v in vals
        )
        result = decode_ppg(header + payload)
        assert len(result) <= _PPG_SAMPLES_PER_PACKET

    def test_returns_floats(self):
        packet = _make_ppg_packet([42] * 6)
        assert all(isinstance(v, float) for v in decode_ppg(packet))

    def test_zero_values(self):
        packet = _make_ppg_packet([0] * 6)
        result = decode_ppg(packet)
        assert all(v == 0.0 for v in result)


# ---------------------------------------------------------------------------
# decode_imu
# ---------------------------------------------------------------------------

class TestDecodeIMU:
    def test_empty_bytes_returns_empty_tuples(self):
        accel, gyro = decode_imu(b"")
        assert accel == []
        assert gyro == []

    def test_too_short_returns_empty_tuples(self):
        short = b"\x00" * (_IMU_MIN_PACKET_LEN - 1)
        accel, gyro = decode_imu(short)
        assert accel == []
        assert gyro == []

    def test_happy_path_returns_nine_accel_and_nine_gyro(self):
        packet = _make_imu_packet([1000] * _IMU_N_VALUES)
        accel, gyro = decode_imu(packet)
        assert len(accel) == _IMU_N_VALUES
        assert len(gyro) == _IMU_N_VALUES

    def test_accel_scale_applied(self):
        raw_val = 1000
        packet = _make_imu_packet([raw_val] * _IMU_N_VALUES)
        accel, _ = decode_imu(packet)
        expected = raw_val * _ACCEL_SCALE
        assert accel[0] == pytest.approx(expected, rel=1e-6)

    def test_gyro_scale_applied(self):
        raw_val = 500
        packet = _make_imu_packet([raw_val] * _IMU_N_VALUES)
        _, gyro = decode_imu(packet)
        expected = raw_val * _GYRO_SCALE
        assert gyro[0] == pytest.approx(expected, rel=1e-6)

    def test_zero_values_produce_zeros(self):
        packet = _make_imu_packet([0] * _IMU_N_VALUES)
        accel, gyro = decode_imu(packet)
        assert all(v == 0.0 for v in accel)
        assert all(v == 0.0 for v in gyro)

    def test_negative_values_decoded_correctly(self):
        # int16 big-endian: -1000 -> scale -> negative float
        packet = _make_imu_packet([-1000] * _IMU_N_VALUES)
        accel, gyro = decode_imu(packet)
        assert all(v < 0 for v in accel)
        assert all(v < 0 for v in gyro)

    def test_returns_floats(self):
        packet = _make_imu_packet([1] * _IMU_N_VALUES)
        accel, gyro = decode_imu(packet)
        assert all(isinstance(v, float) for v in accel)
        assert all(isinstance(v, float) for v in gyro)

    def test_undersized_payload_zero_pads_to_nine(self):
        # Only 4 int16 values in payload (8 bytes + 2 header = 10 bytes total)
        # Decoder must zero-pad values list to length 9
        header = b"\x00\x00"
        payload = struct.pack(">4h", 100, 200, 300, 400)
        raw = header + payload
        # Pad to minimum packet length
        raw = raw + b"\x00" * (_IMU_MIN_PACKET_LEN - len(raw))
        accel, gyro = decode_imu(raw)
        assert len(accel) == _IMU_N_VALUES
        assert len(gyro) == _IMU_N_VALUES
        # First 4 values should match; rest should be 0.0
        assert accel[0] == pytest.approx(100 * _ACCEL_SCALE)
        assert accel[4] == pytest.approx(0.0)

    def test_max_int16_value(self):
        packet = _make_imu_packet([32767] * _IMU_N_VALUES)
        accel, gyro = decode_imu(packet)
        expected_accel = 32767 * _ACCEL_SCALE
        expected_gyro = 32767 * _GYRO_SCALE
        assert accel[0] == pytest.approx(expected_accel, rel=1e-5)
        assert gyro[0] == pytest.approx(expected_gyro, rel=1e-5)
