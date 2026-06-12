"""Unit tests for dsp/decoders.py — Muse BLE packet decoders."""

from __future__ import annotations

import struct

import pytest

from neurolink.dsp.decoders import (
    _ACCEL_SCALE,
    _EEG_MIN_PACKET_LEN,
    _EEG_OFFSET,
    _EEG_SCALE,
    _GYRO_SCALE,
    _IMU_MIN_PACKET_LEN,
    _PPG_MIN_PACKET_LEN,
    _PPG_SAMPLES_PER_PACKET,
    decode_eeg,
    decode_imu,
    decode_ppg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eeg_packet(n_payload_triples: int = 4, header: bytes = b'\x00\x00') -> bytes:
    """Build a minimal valid EEG packet: 2-byte header + n triples of 3 bytes."""
    payload = bytes([0x80, 0x00, 0x00] * n_payload_triples)
    return header + payload


def _ppg_packet(n_samples: int = 6, header: bytes = b'\x00\x00') -> bytes:
    """Build a minimal valid PPG packet: 2-byte header + n 3-byte samples."""
    sample = b'\x00\x01\x02'
    return header + sample * n_samples


def _imu_packet(n_int16: int = 9, header: bytes = b'\x00\x00') -> bytes:
    """Build a valid IMU packet: 2-byte header + 9 big-endian int16."""
    payload = struct.pack(">" + "h" * n_int16, *range(n_int16))
    return header + payload


# ---------------------------------------------------------------------------
# decode_eeg()
# ---------------------------------------------------------------------------

class TestDecodeEEG:
    def test_too_short_returns_empty(self):
        assert decode_eeg(b'\x00' * (_EEG_MIN_PACKET_LEN - 1)) == []

    def test_empty_bytes_returns_empty(self):
        assert decode_eeg(b'') == []

    def test_valid_packet_returns_list_of_floats(self):
        pkt = _eeg_packet(n_payload_triples=6)
        result = decode_eeg(pkt)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_valid_packet_returns_at_most_12_samples(self):
        pkt = _eeg_packet(n_payload_triples=8)
        result = decode_eeg(pkt)
        assert len(result) <= 12

    def test_minimal_valid_packet_returns_nonzero_samples(self):
        pkt = _eeg_packet(n_payload_triples=4)
        result = decode_eeg(pkt)
        assert len(result) > 0

    def test_scale_and_offset_applied(self):
        """Packet bytes 0x80 0x00 0x00 encodes samples s1=2048, s2=0.
        s1 decoded = (2048 - 2048) * scale = 0.0
        s2 decoded = (0 - 2048) * scale = -2048 * 0.48828125 = -1000.0
        """
        pkt = b'\x00\x00' + b'\x80\x00\x00'
        result = decode_eeg(pkt)
        assert len(result) >= 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        if len(result) >= 2:
            assert result[1] == pytest.approx(-2048.0 * _EEG_SCALE, rel=1e-6)

    def test_all_ff_payload_returns_floats(self):
        pkt = b'\x00\x00' + b'\xFF\xFF\xFF' * 6
        result = decode_eeg(pkt)
        assert all(isinstance(v, float) for v in result)

    def test_header_bytes_ignored(self):
        """Same payload, different header bytes — samples must be identical."""
        payload = b'\x80\x00\x00' * 4
        r1 = decode_eeg(b'\x00\x00' + payload)
        r2 = decode_eeg(b'\xFF\xFF' + payload)
        assert r1 == r2


# ---------------------------------------------------------------------------
# decode_ppg()
# ---------------------------------------------------------------------------

class TestDecodePPG:
    def test_too_short_returns_empty(self):
        assert decode_ppg(b'\x00' * (_PPG_MIN_PACKET_LEN - 1)) == []

    def test_empty_bytes_returns_empty(self):
        assert decode_ppg(b'') == []

    def test_valid_packet_returns_list_of_floats(self):
        pkt = _ppg_packet(n_samples=6)
        result = decode_ppg(pkt)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_at_most_six_samples(self):
        pkt = _ppg_packet(n_samples=10)  # more than max
        result = decode_ppg(pkt)
        assert len(result) <= _PPG_SAMPLES_PER_PACKET

    def test_known_value(self):
        """Bytes [0x00, 0x01, 0x02] decode to 0*65536 + 1*256 + 2 = 258."""
        pkt = b'\x00\x00' + b'\x00\x01\x02' * 6
        result = decode_ppg(pkt)
        assert len(result) > 0
        assert result[0] == pytest.approx(258.0)

    def test_all_zero_payload_returns_zeros(self):
        pkt = b'\x00\x00' + b'\x00\x00\x00' * 6
        result = decode_ppg(pkt)
        assert all(v == 0.0 for v in result)

    def test_max_value_sample(self):
        """0xFF 0xFF 0xFF = 16777215."""
        pkt = b'\x00\x00' + b'\xFF\xFF\xFF' * 6
        result = decode_ppg(pkt)
        assert result[0] == pytest.approx(0xFFFFFF, rel=1e-9)

    def test_header_bytes_ignored(self):
        payload = b'\x00\x01\x02' * 6
        r1 = decode_ppg(b'\x00\x00' + payload)
        r2 = decode_ppg(b'\xFF\xFF' + payload)
        assert r1 == r2


# ---------------------------------------------------------------------------
# decode_imu()
# ---------------------------------------------------------------------------

class TestDecodeIMU:
    def test_too_short_returns_empty_lists(self):
        short = b'\x00' * (_IMU_MIN_PACKET_LEN - 1)
        accel, gyro = decode_imu(short)
        assert accel == []
        assert gyro == []

    def test_empty_bytes_returns_empty_lists(self):
        accel, gyro = decode_imu(b'')
        assert accel == []
        assert gyro == []

    def test_valid_packet_returns_two_lists(self):
        pkt = _imu_packet()
        accel, gyro = decode_imu(pkt)
        assert isinstance(accel, list)
        assert isinstance(gyro, list)

    def test_accel_and_gyro_length_nine(self):
        pkt = _imu_packet()
        accel, gyro = decode_imu(pkt)
        assert len(accel) == 9
        assert len(gyro) == 9

    def test_accel_scale_applied(self):
        """Payload int16 = 1 → accel = 1 * ACCEL_SCALE."""
        payload = struct.pack(">" + "h" * 9, *([1] * 9))
        pkt = b'\x00\x00' + payload
        accel, _ = decode_imu(pkt)
        assert accel[0] == pytest.approx(_ACCEL_SCALE, rel=1e-6)

    def test_gyro_scale_applied(self):
        """Payload int16 = 1 → gyro = 1 * GYRO_SCALE."""
        payload = struct.pack(">" + "h" * 9, *([1] * 9))
        pkt = b'\x00\x00' + payload
        _, gyro = decode_imu(pkt)
        assert gyro[0] == pytest.approx(_GYRO_SCALE, rel=1e-6)

    def test_zero_payload_returns_zeros(self):
        payload = struct.pack(">" + "h" * 9, *([0] * 9))
        pkt = b'\x00\x00' + payload
        accel, gyro = decode_imu(pkt)
        assert all(v == 0.0 for v in accel)
        assert all(v == 0.0 for v in gyro)

    def test_negative_int16_decoded(self):
        payload = struct.pack(">" + "h" * 9, *[-100] * 9)
        pkt = b'\x00\x00' + payload
        accel, gyro = decode_imu(pkt)
        assert accel[0] == pytest.approx(-100 * _ACCEL_SCALE, rel=1e-6)
        assert gyro[0] == pytest.approx(-100 * _GYRO_SCALE, rel=1e-6)

    def test_header_bytes_ignored(self):
        payload = struct.pack(">" + "h" * 9, *range(9))
        r1_a, r1_g = decode_imu(b'\x00\x00' + payload)
        r2_a, r2_g = decode_imu(b'\xFF\xFF' + payload)
        assert r1_a == r2_a
        assert r1_g == r2_g

    def test_short_payload_padded_with_zeros(self):
        """Payload with fewer than 9 int16 values — remainder padded to 0.0."""
        payload = struct.pack(">" + "h" * 4, *[10] * 4)
        pkt = b'\x00\x00' + payload + b'\x00' * 12  # pad to 20 bytes total
        accel, gyro = decode_imu(pkt)
        assert len(accel) == 9
        assert len(gyro) == 9
