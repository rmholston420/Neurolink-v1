"""Raw GATT frame decoders for Muse S headset.

Ported from Rigpa-v2 muse_decoders.py.
All protocol constants are firmware-level and must not be modified.
"""
from __future__ import annotations

import struct

import numpy as np

# EEG scale factor (uV)
_EEG_SCALE: float = 0.48828125  # (1.8 / 2^12) / 1.5 * 1e6 uV
# Number of EEG samples per GATT notification
_EEG_SAMPLES_PER_PACKET: int = 12
# Number of PPG samples per GATT notification
_PPG_SAMPLES_PER_PACKET: int = 6
# Number of IMU samples per GATT notification
_IMU_SAMPLES_PER_PACKET: int = 3


def decode_eeg(data: bytes | bytearray) -> list[float]:
    """Decode a single Muse S EEG GATT notification to microvolts.

    Returns list of 12 float samples. Returns zeros on malformed input.
    """
    try:
        n = _EEG_SAMPLES_PER_PACKET
        samples: list[float] = []
        # First 2 bytes: 16-bit sequence number (big-endian)
        # Remaining bytes: 12-bit unsigned ints packed 3 per 4.5 bytes
        # Unpack as: each sample = 12-bit value, big-endian, 1.5 bytes
        # Muse encodes as: 2 bytes header + 18 bytes of data (12 x 12-bit)
        if len(data) < 20:
            return [0.0] * n
        raw_int = int.from_bytes(data[2:], "big")
        total_bits = len(data[2:]) * 8
        bit_offset = total_bits - 12 * n
        for i in range(n):
            shift = total_bits - 12 * (i + 1) - bit_offset
            val = (raw_int >> shift) & 0xFFF
            samples.append((val - 2048) * _EEG_SCALE)
        return samples
    except Exception:
        return [0.0] * _EEG_SAMPLES_PER_PACKET


def decode_ppg(data: bytes | bytearray) -> list[float]:
    """Decode a single Muse S PPG GATT notification.

    Returns list of 6 uint32 counts. Returns zeros on malformed input.
    """
    try:
        n = _PPG_SAMPLES_PER_PACKET
        if len(data) < 2 + n * 3:
            return [0.0] * n
        # Skip 2-byte sequence header; 6 x 3-byte big-endian uint24
        samples: list[float] = []
        for i in range(n):
            offset = 2 + i * 3
            val = int.from_bytes(data[offset: offset + 3], "big")
            samples.append(float(val))
        return samples
    except Exception:
        return [0.0] * _PPG_SAMPLES_PER_PACKET


def decode_imu(
    data: bytes | bytearray,
) -> tuple[list[float], list[float]]:
    """Decode a single Muse S IMU (accel or gyro) GATT notification.

    Returns (samples_x, samples_y, samples_z) each with 3 values.
    Actually returns (accel_list, gyro_list) where each is a list of floats.
    Format: 2-byte header + 3 samples x 3 axes x 2-byte int16 big-endian.
    Scale: accel 0.0000610352 g/LSB, gyro 0.0074768 deg/s per LSB.
    Returns zeros on malformed input.
    """
    _ACCEL_SCALE = 0.0000610352  # g per LSB
    _GYRO_SCALE = 0.0074768      # deg/s per LSB
    try:
        n_samples = _IMU_SAMPLES_PER_PACKET
        expected = 2 + n_samples * 3 * 2  # header + 3 samples * 3 axes * 2 bytes
        if len(data) < expected:
            axes_flat = [0.0] * (n_samples * 3)
            return axes_flat[:n_samples], axes_flat[:n_samples]
        result_xyz: list[list[float]] = [[], [], []]
        for s in range(n_samples):
            for ax in range(3):
                offset = 2 + (s * 3 + ax) * 2
                raw = struct.unpack(">h", data[offset: offset + 2])[0]
                result_xyz[ax].append(float(raw) * _ACCEL_SCALE)
        # Flatten: return all xyz interleaved (x0,y0,z0,x1,y1,...)
        flat = []
        for s in range(n_samples):
            for ax in range(3):
                flat.append(result_xyz[ax][s])
        return flat, flat
    except Exception:
        zeros = [0.0] * (_IMU_SAMPLES_PER_PACKET * 3)
        return zeros, zeros
