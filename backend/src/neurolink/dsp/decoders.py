"""Raw GATT frame decoders for Muse S EEG, PPG, and IMU.

Ported verbatim from Rigpa-v2 muse_decoders.py.
DO NOT MODIFY the bit-manipulation constants below.
All protocol constants are firmware-level.
"""
from __future__ import annotations

import struct

# ── EEG decoder ──────────────────────────────────────────────────────────────────────
#
# The Muse EEG characteristic sends 20-byte GATT payloads.
# Each payload contains a 16-bit timestamp (bytes 0-1) followed by 12 packed
# 12-bit samples (bytes 2-19). The unpacking is a custom bit-packing scheme.
#
# DO NOT MODIFY - these constants encode Muse S firmware packet format.

_EEG_SCALE: float = 0.48828125  # uV per LSB = 4096 / 8388607 * 1000 * ... (FIXED)
_EEG_SAMPLES_PER_PACKET: int = 12
_EEG_PACKET_LEN: int = 20


def decode_eeg(data: bytes) -> list[float]:
    """Decode a 20-byte Muse EEG GATT packet to 12 uV samples.

    Args:
        data: 20-byte GATT characteristic notification payload

    Returns:
        List of 12 float values in microvolts (uV).
        Returns empty list for malformed packets.
    """
    if len(data) < _EEG_PACKET_LEN:
        return []

    samples: list[float] = []
    # Bytes 2-19 carry 12 packed 12-bit samples
    raw = data[2:]
    bit_pos = 0
    buf = int.from_bytes(raw, "big")
    total_bits = len(raw) * 8

    for _ in range(_EEG_SAMPLES_PER_PACKET):
        shift = total_bits - bit_pos - 12
        if shift < 0:
            break
        val = (buf >> shift) & 0xFFF
        # Sign-extend 12-bit two's complement
        if val >= 2048:
            val -= 4096
        samples.append(float(val) * _EEG_SCALE)
        bit_pos += 12

    return samples


# ── PPG decoder ──────────────────────────────────────────────────────────────────────
#
# PPG characteristic: 20-byte payload with timestamp (bytes 0-1) and
# 6 packed 24-bit samples (bytes 2-19).

_PPG_SAMPLES_PER_PACKET: int = 6
_PPG_PACKET_LEN: int = 20


def decode_ppg(data: bytes) -> list[float]:
    """Decode a 20-byte Muse PPG GATT packet to 6 raw ADC samples.

    Args:
        data: 20-byte GATT characteristic notification payload

    Returns:
        List of 6 float ADC values.
        Returns empty list for malformed packets.
    """
    if len(data) < _PPG_PACKET_LEN:
        return []

    samples: list[float] = []
    # Bytes 2-19: six 24-bit big-endian unsigned integers
    for i in range(_PPG_SAMPLES_PER_PACKET):
        offset = 2 + i * 3
        if offset + 3 > len(data):
            break
        val = struct.unpack(">I", b"\x00" + data[offset:offset + 3])[0]
        samples.append(float(val))
    return samples


# ── IMU decoder ──────────────────────────────────────────────────────────────────────
#
# Accel and Gyro characteristics: 20-byte payload with timestamp (bytes 0-1)
# and 9 packed 16-bit signed integers (3 samples x 3 axes).

_IMU_SCALE_ACCEL: float = 0.0000610  # g per LSB (FIXED, FS=±2g range)
_IMU_SCALE_GYRO: float = 0.00875     # deg/s per LSB (FIXED, FS=250 dps range)
_IMU_SAMPLES_PER_PACKET: int = 3
_IMU_AXES: int = 3
_IMU_PACKET_LEN: int = 20


def decode_imu(data: bytes) -> tuple[list[float], list[float]]:
    """Decode a 20-byte Muse IMU GATT packet.

    Each packet has 3 samples, each with 3 axes (x, y, z).
    Scales to physical units.

    Args:
        data: 20-byte GATT payload

    Returns:
        (accel_flat, gyro_flat) where each is a list of 9 floats
        [x0, y0, z0, x1, y1, z1, x2, y2, z2].
        Accel in g; gyro in deg/s.
        Returns ([], []) for malformed packets.

    Note:
        The same decoder is used for both accel and gyro characteristics.
        The caller should route data to the correct buffer.
        The returned (flat_accel, flat_gyro) are both computed from the
        same packet but scaled differently. The caller uses whichever
        channel is appropriate.
    """
    if len(data) < _IMU_PACKET_LEN:
        return [], []

    accel_flat: list[float] = []
    gyro_flat: list[float] = []
    offset = 2  # skip 2-byte timestamp

    for _ in range(_IMU_SAMPLES_PER_PACKET):
        for _axis in range(_IMU_AXES):
            if offset + 2 > len(data):
                break
            raw = struct.unpack(">h", data[offset:offset + 2])[0]  # signed int16
            accel_flat.append(float(raw) * _IMU_SCALE_ACCEL)
            gyro_flat.append(float(raw) * _IMU_SCALE_GYRO)
            offset += 2

    return accel_flat, gyro_flat
