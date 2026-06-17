"""Muse BLE packet decoders for EEG, PPG, and IMU.

All BLE packet formats are FIXED by Muse firmware.
DO NOT modify packet sizes, bit shifts, or scale factors.
"""

from __future__ import annotations

import struct

# Minimum EEG packet length: 2-byte header + at least one full 3-byte triple = 6.
# A 5-byte packet (header + 3 bytes = only one partial triple offset) cannot
# yield any samples and must return [].
_EEG_MIN_PACKET_LEN: int = 6
_EEG_SCALE: float = 0.48828125
_EEG_OFFSET: float = 2048.0

_PPG_SAMPLE_SIZE: int = 3
_PPG_MIN_PACKET_LEN: int = 12
_PPG_SAMPLES_PER_PACKET: int = 6

_IMU_PACKET_LEN: int = 20
_IMU_MIN_PACKET_LEN: int = _IMU_PACKET_LEN
_IMU_N_VALUES: int = 9
_ACCEL_SCALE: float = 0.0000610352
_GYRO_SCALE: float = 0.0074768


def decode_eeg(data: bytes) -> list[float]:
    """Decode a Muse EEG BLE characteristic notification.

    Returns:
        List of up to 12 float samples in uV, or empty list for short/invalid packets.
    """
    if len(data) < _EEG_MIN_PACKET_LEN:
        return []

    payload = data[2:]
    samples: list[float] = []
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
    """Decode a Muse PPG BLE characteristic notification."""
    if len(data) < _PPG_MIN_PACKET_LEN:
        return []

    payload = data[2:]
    samples: list[float] = []
    for i in range(
        0,
        min(len(payload), _PPG_SAMPLES_PER_PACKET * _PPG_SAMPLE_SIZE),
        _PPG_SAMPLE_SIZE,
    ):
        if i + 2 < len(payload):
            val = (payload[i] << 16) | (payload[i + 1] << 8) | payload[i + 2]
            samples.append(float(val))

    return samples[:_PPG_SAMPLES_PER_PACKET]


def decode_imu(data: bytes) -> tuple[list[float], list[float]]:
    """Decode a Muse IMU BLE characteristic notification."""
    if len(data) < _IMU_MIN_PACKET_LEN:
        return [], []

    payload = data[2:]
    n_int16 = min(_IMU_N_VALUES, len(payload) // 2)
    values: list[float] = []
    for i in range(n_int16):
        raw = struct.unpack_from(">h", payload, i * 2)[0]
        values.append(float(raw))

    while len(values) < _IMU_N_VALUES:
        values.append(0.0)

    accel = [v * _ACCEL_SCALE for v in values[:_IMU_N_VALUES]]
    gyro = [v * _GYRO_SCALE for v in values[:_IMU_N_VALUES]]

    return accel, gyro
