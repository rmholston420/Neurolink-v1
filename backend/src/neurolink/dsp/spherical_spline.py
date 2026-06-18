"""Stage 2 — Spherical spline interpolation for bad EEG channels.

Implements the Perrin et al. (1989) spherical spline method used by
MNE-Python's ``raw.interpolate_bads()``.  This is a pure-NumPy
implementation with no MNE runtime dependency.

Reference
---------
Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989).
Spherical splines for scalp potential and current density mapping.
Electroencephalography and Clinical Neurophysiology, 72(2), 184-187.

Muse electrode layout (unit-sphere coordinates)
-----------------------------------------------
Positions are approximate standard-10-20 directions projected onto
the unit sphere.  The Muse has four EEG electrodes; AUX is a
non-cortical reference and is excluded from the spline.

Channel index → name → (x, y, z) unit vector
    0  TP9   left  posterior temporal
    1  AF7   left  prefrontal
    2  AF8   right prefrontal
    3  TP10  right posterior temporal
    4  AUX   excluded

Legendre series
---------------
Truncated at degree N=7 (Perrin recommends 4–10 for sparse arrays;
7 balances smoothness and accuracy for 4 sites).  The regularisation
parameter lambda=1e-5 matches MNE defaults.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Electrode positions — approximate 10-20 unit-sphere vectors for Muse S
# Computed as: pos / ||pos|| from standard 10-20 Cartesian coordinates.
# TP9  ≈ T9  (left posterior temporal)
# AF7  ≈ AF7 (left anterior prefrontal)
# AF8  ≈ AF8 (right anterior prefrontal)
# TP10 ≈ T10 (right posterior temporal)
# ──────────────────────────────────────────────────────────────────────────────
_MUSE_POSITIONS: dict[str, np.ndarray] = {
    "TP9": np.array([-0.5878, -0.3090, 0.7431], dtype=np.float64),
    "AF7": np.array([-0.5440, 0.6550, 0.5240], dtype=np.float64),
    "AF8": np.array([0.5440, 0.6550, 0.5240], dtype=np.float64),
    "TP10": np.array([0.5878, -0.3090, 0.7431], dtype=np.float64),
}

# Spline parameters
_N_LEGENDRE: int = 7  # Legendre series truncation degree
_LAMBDA: float = 1e-5  # Tikhonov regularisation (MNE default)
_EEG_CHANNELS: list[str] = ["TP9", "AF7", "AF8", "TP10"]
_CHANNEL_IDX: dict[str, int] = {"TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3, "AUX": 4}


# ──────────────────────────────────────────────────────────────────────────────
# Legendre polynomial helpers
# ──────────────────────────────────────────────────────────────────────────────


def _g_function(cos_dist: np.ndarray, n_legendre: int = _N_LEGENDRE) -> np.ndarray:
    """Compute the G (smoothing) matrix entries.

    G(x) = (1/4pi) * sum_{n=1}^{N} ((2n+1) / (n(n+1))^m) * P_n(x)
    with m=4 (4th-order Laplacian) following Perrin 1989.

    Args:
        cos_dist: Array of cosine distances (dot products of unit vectors).
        n_legendre: Legendre series truncation degree.

    Returns:
        G values, same shape as cos_dist.
    """
    result = np.zeros_like(cos_dist, dtype=np.float64)
    # Recurrence relation for Legendre polynomials
    p_prev = np.ones_like(cos_dist, dtype=np.float64)  # P_0
    p_curr = cos_dist.copy().astype(np.float64)  # P_1
    m_order = 4  # Perrin 1989 uses m=4
    for n in range(1, n_legendre + 1):
        coef = (2 * n + 1) / (n**m_order * (n + 1) ** m_order)
        result += coef * (p_curr if n > 0 else p_prev)
        # Advance recurrence: P_{n+1}(x) = ((2n+1)x P_n - n P_{n-1}) / (n+1)
        p_next = ((2 * n + 1) * cos_dist * p_curr - n * p_prev) / (n + 1)
        p_prev = p_curr
        p_curr = p_next
    return result / (4.0 * np.pi)


def _cosine_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute N×N cosine-distance matrix for an array of unit vectors.

    Args:
        positions: (N, 3) array of unit-sphere coordinates.

    Returns:
        (N, N) symmetric matrix of dot products.
    """
    # Clamp to [-1, 1] to guard against floating-point overshoot in acos
    return np.clip(positions @ positions.T, -1.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Public interpolation function
# ──────────────────────────────────────────────────────────────────────────────


def interpolate_bad_channels(
    eeg: np.ndarray,
    bad_channels: list[str],
) -> np.ndarray:
    """Interpolate bad EEG channels using spherical splines.

    Only Muse EEG channels (TP9, AF7, AF8, TP10) participate in the
    spline.  AUX is never interpolated.  If no bad channels are in the
    EEG set, the array is returned unchanged.

    Fallback rules:
    * If only 1 good channel remains (3 bad), fills with channel mean.
    * If the spline matrix is singular (numerically), fills with mean.
    * If bad_channels is empty, returns eeg unchanged (no copy).

    Args:
        eeg:          (n_channels, n_samples) float32 array.  n_channels
                      should be 5 (TP9, AF7, AF8, TP10, AUX) but handles
                      fewer gracefully.
        bad_channels: List of channel name strings to interpolate.

    Returns:
        Array of the same shape with bad channels replaced.
    """
    # Filter to EEG-only bad channels that are actually present
    bad_eeg = [
        ch for ch in bad_channels if ch in _EEG_CHANNELS and _CHANNEL_IDX.get(ch, 99) < eeg.shape[0]
    ]
    if not bad_eeg:
        return eeg

    good_eeg = [
        ch for ch in _EEG_CHANNELS if ch not in bad_eeg and _CHANNEL_IDX.get(ch, 99) < eeg.shape[0]
    ]

    out = eeg.copy()

    # ── Fallback: not enough good channels for a spline ─────────────────────
    if len(good_eeg) < 2:
        log.warning(
            "stage2_spline_fallback_mean",
            bad=bad_eeg,
            good=good_eeg,
        )
        # Fill with mean of all available good EEG channels
        if good_eeg:
            good_idx = [_CHANNEL_IDX[ch] for ch in good_eeg]
            mean_sig = np.mean(out[good_idx], axis=0)
        else:
            mean_sig = np.zeros(eeg.shape[1], dtype=np.float32)
        for ch in bad_eeg:
            out[_CHANNEL_IDX[ch]] = mean_sig
        return out

    # ── Build spline system ──────────────────────────────────────────────────
    # Use only good-channel positions to solve for spline coefficients,
    # then evaluate at bad-channel positions.
    n_good = len(good_eeg)
    good_pos = np.array([_MUSE_POSITIONS[ch] for ch in good_eeg], dtype=np.float64)  # (n_good, 3)

    # G matrix among good channels
    cos_good = _cosine_matrix(good_pos)  # (n_good, n_good)
    G_good = _g_function(cos_good)  # (n_good, n_good)

    # Perrin 1989 augmented system:
    # [ G+λI   1 ] [ c ] = [ v ]
    # [   1^T   0 ] [ d ]   [ 0 ]
    size = n_good + 1
    A = np.zeros((size, size), dtype=np.float64)
    A[:n_good, :n_good] = G_good + _LAMBDA * np.eye(n_good)
    A[:n_good, n_good] = 1.0
    A[n_good, :n_good] = 1.0
    # A[n_good, n_good] stays 0

    # Right-hand side: good-channel signal (n_good, n_samples)
    good_idx = [_CHANNEL_IDX[ch] for ch in good_eeg]
    V_good = out[good_idx].astype(np.float64)  # (n_good, n_samples)
    rhs = np.zeros((size, V_good.shape[1]), dtype=np.float64)
    rhs[:n_good] = V_good

    # Solve: A @ coefs = rhs
    try:
        coefs = np.linalg.solve(A, rhs)  # (n_good+1, n_samples)
    except np.linalg.LinAlgError:
        log.warning("stage2_spline_singular", bad=bad_eeg)
        mean_sig = np.mean(V_good, axis=0).astype(np.float32)
        for ch in bad_eeg:
            out[_CHANNEL_IDX[ch]] = mean_sig
        return out

    c = coefs[:n_good]  # spline weights  (n_good, n_samples)
    d = coefs[n_good]  # constant term   (n_samples,)

    # ── Evaluate spline at bad-channel positions ──────────────────────────
    for bad_ch in bad_eeg:
        bad_pos = _MUSE_POSITIONS[bad_ch]  # (3,)
        # Cosine distances from bad electrode to all good electrodes
        cos_bad_good = np.clip(good_pos @ bad_pos, -1.0, 1.0)  # (n_good,)
        g_vec = _g_function(cos_bad_good[np.newaxis, :]).squeeze(0)  # (n_good,)
        # interpolated = G(bad, good) @ c + d
        interpolated = (g_vec @ c + d).astype(np.float32)  # (n_samples,)
        out[_CHANNEL_IDX[bad_ch]] = interpolated
        log.debug(
            "stage2_channel_interpolated",
            channel=bad_ch,
            good_channels=good_eeg,
        )

    return out
