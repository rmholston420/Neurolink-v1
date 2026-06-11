"""Muse BLE packet decoders for EEG, PPG, and IMU.

Ported from Rigpa-v2 ble_bridge.py decode routines.

All BLE packet formats are FIXED by Muse firmware.
DO NOT modify packet sizes, bit shifts, or scale factors.
"""
from __future__ import annotations

import struct

# EEG: 20-byte packet → 12 float samples
# First 2 bytes = sequence/status, then 3 bytes per sample * 12
_EEG_MIN_PACKET_LEN: int = 14  # 2 header + 12 sample bytes
_EEG_SCALE: float = 0.48828125  # uV per LSB (from Muse SDK)
_EEG_OFFSET: float = 2048.0     # 12-bit unsigned centre

# PPG: 20-byte packet → 6 samples (3-byte each, big-endian)
# Minimum meaningful packet: 2-byte header + 1 full 3-byte sample = 5 bytes
_PPG_SAMPLE_SIZE: int = 3
_PPG_MIN_PACKET_LEN: int = 2 + _PPG_SAMPLE_SIZE * 2  # at least 2 full samples = 8 bytes
_PPG_SAMPLES_PER_PACKET: int = 6

# IMU: 20-byte packet → 9 accel + 9 gyro int16 values (big-endian)
# Minimum meaningful packet: 2-byte header + at least 2 int16 values = 6 bytes
_IMU_MIN_PACKET_LEN: int = 6
_IMU_N_VALUES: int = 9  # 3 axes * 3 samples
_ACCEL_SCALE: float = 0.0000610352   # g per LSB (Muse SDK)
_GYRO_SCALE: float = 0.0074768       # deg/s per LSB (Muse SDK)


def decode_eeg(data: bytes) -> list[float]:
    """Decode a Muse EEG BLE characteristic notification.

    Returns:
        List of up to 12 float samples in uV, or empty list for short/invalid packets.
    """
    if len(data) < _EEG_MIN_PACKET_LEN:
        return []

    # Bytes 2..14: 12 samples, packed as 12-bit big-endian (unusual Muse encoding)
    # Each sample is packed in 1.5 bytes; use 3-byte groups for 2 samples.
    payload = data[2:]  # Skip header
    samples = []
    i = 0
    while i + 2 < len(payload) and len(samples) < 12:
        b0, b1, b2 = payload[i], payload[i + 1], payload[i + 2]
        s1 = ((b0 << 4) | (b1 >> 4)) & 0xFFF
        s2 = ((b1 & 0xF) << 8) | b2
        samples.append((s1 - _EEG_OFFSET) * _EEG_SCALE)
        if len(samples) < 12:
            samples.append((s2 - _EEG_OFFSET) * _EEG_SCALE)
        i += 3

    return samples[:12]


def decode_ppg(data: bytes) -> list[float]:
    """Decode a Muse PPG BLE characteristic notification.

    Returns:
        List of up to 6 float samples, or empty list for short/invalid packets.
    """
    if len(data) < _PPG_MIN_PACKET_LEN:
        return []

    payload = data[2:]  # Skip 2-byte header
    samples = []
    for i in range(0, min(len(payload), _PPG_SAMPLES_PER_PACKET * _PPG_SAMPLE_SIZE), _PPG_SAMPLE_SIZE):
        if i + 2 < len(payload):
            val = (payload[i] << 16) | (payload[i + 1] << 8) | payload[i + 2]
            samples.append(float(val))

    return samples[:_PPG_SAMPLES_PER_PACKET]


def decode_imu(data: bytes) -> tuple[list[float], list[float]]:
    """Decode a Muse IMU (accel or gyro) BLE characteristic notification.

    Returns:
        (accel_flat, gyro_flat): Two lists of 9 float values each.
        Returns empty lists for short/invalid packets.
    """
    if len(data) < _IMU_MIN_PACKET_LEN:
        return [], []

    payload = data[2:]  # Skip header
    n_int16 = min(_IMU_N_VALUES, len(payload) // 2)
    values = []
    for i in range(n_int16):
        raw = struct.unpack_from(">h", payload, i * 2)[0]
        values.append(float(raw))

    # Pad to 9 if shorter
    while len(values) < _IMU_N_VALUES:
        values.append(0.0)

    accel = [v * _ACCEL_SCALE for v in values[:_IMU_N_VALUES]]
    gyro = [v * _GYRO_SCALE for v in values[:_IMU_N_VALUES]]

    return accel, gyro
