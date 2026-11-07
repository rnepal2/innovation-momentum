"""Invariant graph-mode power and finite-window density measurements."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window, periodogram
from traffic_core import graph_basis

FLOOR = 1e-12
BANDS = {"long": (1, 2), "intermediate": (3, 6), "short": (7, 12)}


def cycle_modes(n=24):
    if n != 24:
        raise ValueError("This protocol fixes 24 vehicles")
    adjacency = np.roll(np.eye(n), 1, axis=0) + np.roll(np.eye(n), -1, axis=0)
    eigenvalues, basis = graph_basis(adjacency)
    groups = [
        np.flatnonzero(np.isclose(eigenvalues, 2 - 2 * np.cos(2 * np.pi * q / n), atol=1e-9))
        for q in range(n // 2 + 1)
    ]
    if sum(map(len, groups)) != n or any(len(group) == 0 for group in groups):
        raise ValueError("Incomplete eigenspace grouping")
    return basis, groups


def spatial_power(response, basis=None, groups=None):
    x = np.asarray(response, dtype=float)
    if x.ndim != 2 or x.shape[1] != 24 or not np.isfinite(x).all():
        raise ValueError("Expected finite time-by-24 response")
    if basis is None:
        basis, groups = cycle_modes(x.shape[1])
    mean = x.mean(axis=1)
    residual = x - mean[:, None]
    z = residual @ basis
    energy = np.stack([np.sum(z[:, group] ** 2, axis=1) / x.shape[1] for group in groups], axis=1)
    total = energy[:, 1:].sum(axis=1)
    shares = np.full_like(energy, np.nan)
    valid = total > FLOOR
    shares[valid] = energy[valid] / total[valid, None]
    return {
        "mean": mean,
        "residual": residual,
        "energy": energy,
        "shares": shares,
        "total": total,
        "raw_power": np.mean(x * x, axis=1),
    }


def fft_power(response):
    x = np.asarray(response)
    residual = x - x.mean(axis=1, keepdims=True)
    energy = np.abs(np.fft.rfft(residual, axis=1, norm="ortho")) ** 2 / x.shape[1]
    energy[:, 1:-1] *= 2
    return energy


def joint_density(response, start, length=32, dt=1.0, detrend=False):
    x = np.array(response[start : start + length], dtype=float, copy=True)
    if len(x) != length:
        raise ValueError("Incomplete spectral window")
    spatial = spatial_power(x)
    basis, groups = cycle_modes(x.shape[1])
    frequencies, psd = periodogram(
        spatial["residual"] @ basis,
        fs=1 / dt,
        window="hann",
        detrend=detrend,
        scaling="density",
        axis=0,
    )
    density = np.stack([psd[:, group].sum(axis=1) / x.shape[1] for group in groups], axis=1)
    energy = density.sum(axis=0) * (frequencies[1] - frequencies[0])
    return frequencies, density, energy


def weighted_spatial_power(response, start, length=32, detrend=False):
    x = np.asarray(response[start : start + length], dtype=float)
    if len(x) != length:
        raise ValueError("Incomplete power window")
    x = x - x.mean(axis=1, keepdims=True)
    if detrend == "constant":
        x = x - x.mean(axis=0, keepdims=True)
    elif detrend is not False:
        raise ValueError("Only constant or no temporal detrending is registered")
    w = get_window("hann", length) ** 2
    return np.sum(fft_power(x) * w[:, None], axis=0) / w.sum()


def window_summary(response, length=32, detrend=False):
    early_start, late_start = 20 - length // 2, 68 - length // 2
    _, _, early = joint_density(response, early_start, length, detrend=detrend)
    _, _, late = joint_density(response, late_start, length, detrend=detrend)
    valid = (early > FLOOR) & (late > FLOOR)
    growth = np.full_like(early, np.nan)
    growth[valid] = np.log(late[valid] / early[valid]) / (2 * 48)
    return early, late, growth


def band_power(energy):
    return {name: np.sum(energy[..., lo : hi + 1], axis=-1) for name, (lo, hi) in BANDS.items()}


def signed_density(response, start=4, length=64, dt=1.0):
    """Two-sided temporal density by signed ordinary spatial Fourier frequency."""
    x = np.array(response[start : start + length], dtype=float, copy=True)
    if len(x) != length:
        raise ValueError("Incomplete signed-density window")
    x -= x.mean(axis=1, keepdims=True)
    z = np.fft.fft(x, axis=1, norm="ortho")
    f, p = periodogram(
        z, fs=1 / dt, window="hann", detrend=False, return_onesided=False, scaling="density", axis=0
    )
    return (
        np.fft.fftshift(np.fft.fftfreq(x.shape[1])),
        np.fft.fftshift(f),
        np.fft.fftshift(p, axes=(0, 1)) / x.shape[1],
    )
