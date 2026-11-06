"""Numerical controls and past-only features for the traffic research pilot."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.fft import dct
from scipy.signal import get_window, periodogram
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Design:
    nodes: int = 24
    window: int = 64
    horizon: int = 16
    dt: float = 2.0
    measurement_noise: float = 0.02
    train: int = 320
    validation: int = 96
    test: int = 192


MECHANISMS = ("aligned", "hidden_basis", "independent", "shared_forcing", "directed")
GEOMETRIES = ("path", "branch")
EPS = 1e-6


def road_graph(kind: str, n: int = 24) -> np.ndarray:
    if n < 12 or kind not in GEOMETRIES:
        raise ValueError("Expected path/branch graph with at least twelve nodes")
    w = np.zeros((n, n))
    main = n if kind == "path" else n - 6
    for i in range(main - 1):
        w[i, i + 1] = 1 if kind == "path" else 0.7 + 0.1 * (i % 7)
    if kind == "branch":
        w[main // 2, main] = 0.8
        for i in range(main, n - 1):
            w[i, i + 1] = 0.65 + 0.12 * (i % 3)
    return w + w.T


def graph_basis(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.allclose(w, w.T) or np.any(w < 0):
        raise ValueError("Weights must be symmetric and nonnegative")
    return np.linalg.eigh(np.diag(w.sum(axis=1)) - w)


def coordinate_bands(n: int, eigenvalues: np.ndarray | None = None) -> list[np.ndarray]:
    cuts = [0, n // 3, 2 * n // 3, n]
    if eigenvalues is not None:
        for j in (1, 2):
            while cuts[j] < n and np.isclose(
                eigenvalues[cuts[j]], eigenvalues[cuts[j] - 1], atol=1e-9, rtol=1e-7
            ):
                cuts[j] += 1
        cuts = sorted(set(cuts))
    return [np.arange(a, b) for a, b in pairwise(cuts) if b > a]


def dct_basis(n: int) -> np.ndarray:
    return dct(np.eye(n), type=2, norm="ortho", axis=0).T


def random_basis(n: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q * np.where(np.diag(r) < 0, -1, 1)


def evolution(kind: str, u: np.ndarray, bands: list[np.ndarray]) -> np.ndarray:
    n = len(u)
    rates = np.empty(n)
    for band, rate in zip(bands, (0.985, 1.008, 0.94)):
        rates[band] = rate
    if len(bands) != 3:
        raise ValueError("This control requires three complete eigenspace bands")
    if kind in ("independent", "shared_forcing"):
        return 0.97 * np.eye(n)
    if kind == "directed":
        return 0.92 * np.eye(n) + 0.15 * np.eye(n, k=1)
    basis = random_basis(n, 819) if kind == "hidden_basis" else u
    if kind not in ("aligned", "hidden_basis"):
        raise ValueError(kind)
    return (basis * rates) @ basis.T


def simulate(a: np.ndarray, seed: int, design: Design, shared: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(a)
    x = rng.normal(size=n) * np.exp(rng.uniform(-1, 1))
    noise = np.exp(rng.uniform(np.log(0.015), np.log(0.15)))
    observations = []
    for _ in range(design.window + design.horizon):
        observations.append(x + rng.normal(scale=design.measurement_noise, size=n))
        forcing = rng.normal(scale=noise, size=n)
        if shared:
            forcing += rng.normal(scale=3 * noise) * np.ones(n)
        x = a @ x + forcing
    return np.asarray(observations)


def amplitude(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def amplification(x: np.ndarray, origin: int, horizon: int) -> float:
    if origin < 3 or horizon < 4 or origin + horizon >= len(x):
        raise ValueError("Need four observations at origin and future endpoint")
    now = amplitude(x[origin - 3 : origin + 1])
    future = amplitude(x[origin + horizon - 3 : origin + horizon + 1])
    return float(np.log((future + EPS) / (now + EPS)))


def past_window(x: np.ndarray, origin: int, length: int) -> np.ndarray:
    if origin < length - 1 or origin >= len(x):
        raise ValueError("Incomplete history")
    window = np.asarray(x[origin - length + 1 : origin + 1], dtype=float)
    if not np.isfinite(window).all():
        raise ValueError("Nonfinite observations")
    return window


def modal_psd(x: np.ndarray, u: np.ndarray, dt: float = 2.0):
    return periodogram(x @ u, fs=1 / dt, window="hann", detrend="constant", axis=0)


def state_features(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=1)) + EPS
    scale = float(np.mean(rms[-4:]))
    row, col = np.nonzero(np.triu(w, 1))
    local = x[:, row] - x[:, col]
    coord = np.diff(x, axis=1)
    return np.array(
        [
            np.log(scale),
            np.log(np.mean(rms)),
            np.log(np.mean(rms[:8])),
            np.log(np.mean(rms[-8:])),
            np.log(rms[-1]),
            np.log(np.mean(rms[-8:]) / np.mean(rms[:8])),
            np.mean(np.diff(np.log(rms))[-8:]),
            np.std(np.diff(np.log(rms))),
            np.mean(x[-4:]) / scale,
            np.std(x[-4:]) / scale,
            np.mean(np.diff(x, axis=0) ** 2) / scale**2,
            np.mean(coord[-8:] ** 2) / scale**2,
            np.mean(local[-8:] ** 2) / scale**2,
            np.mean(local[:8] ** 2) / scale**2,
            np.mean(x[-8:] ** 4) / scale**4,
        ]
    )


def spectral_features(x: np.ndarray, u: np.ndarray, bands: list[np.ndarray], dt=2.0):
    z = x @ u
    f, psd = modal_psd(x, u, dt)
    df = f[1] - f[0]
    masks = (f <= f[-1] / 8, f > f[-1] / 8)
    scale = np.mean(x[-4:] ** 2) * x.shape[1] + EPS
    power, dynamics = [], []
    for band in bands:
        power.append(np.log1p(np.mean(np.sum(z[-8:, band] ** 2, axis=1)) / scale))
        for mask in masks:
            power.append(np.log1p(psd[np.ix_(mask, band)].sum() * df / scale))
        early = np.mean(z[:8, band] ** 2)
        late = np.mean(z[-8:, band] ** 2)
        rho = np.sum(z[1:, band] * z[:-1, band]) / (np.sum(z[:-1, band] ** 2) + EPS)
        dynamics.extend([0.5 * np.log((late + EPS) / (early + EPS)), rho])
    return np.asarray(power), np.r_[power, dynamics]


def cross_spectral_features(x: np.ndarray, dt=2.0) -> np.ndarray:
    """Welch CPSD leading eigenvalue; a deliberately limited envelope comparator."""
    segments = np.stack([x[t : t + 16] for t in range(0, len(x) - 15, 8)])
    segments -= segments.mean(axis=1, keepdims=True)
    taper = get_window("hann", 16)
    z = np.fft.rfft(segments * taper[None, :, None], axis=1)
    cpsd = np.einsum("sfi,sfj->fij", z, z.conj()) / (len(segments) * (1 / dt) * np.sum(taper**2))
    cpsd[1:-1] *= 2
    eig = np.linalg.eigvalsh(cpsd).real
    total = np.maximum(eig.sum(axis=1), EPS)
    power = eig[:, -1] * (1 / dt / 16)
    scale = np.mean(x[-4:] ** 2) * x.shape[1] + EPS
    return np.array(
        [
            np.log1p(power[:2].sum() / scale),
            np.log1p(power[2:].sum() / scale),
            np.average(eig[:, -1] / total, weights=total),
            np.max(eig[:, -1] / total),
        ]
    )


def fit_pca(windows: np.ndarray) -> np.ndarray:
    data = windows.reshape(-1, windows.shape[-1])
    # A spatial covariance basis; the synthetic zero reference is known by design.
    _, u = np.linalg.eigh(data.T @ data / len(data))
    return u[:, ::-1]


def ridge_predict(train_x, train_y, val_x, val_y, test_x):
    choices = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_x, train_y)
        choices.append((float(np.mean((model.predict(val_x) - val_y) ** 2)), alpha, model))
    loss, alpha, model = min(choices, key=lambda item: item[:2])
    return model.predict(test_x), {"alpha": alpha, "validation_mse": loss}, model


def fit_transition(windows: np.ndarray, rank: int, delay: int = 1):
    if delay not in (1, 2):
        raise ValueError("Only delay one/two is specified")
    n = windows.shape[-1]
    if delay == 1:
        x, y = windows[:, :-1].reshape(-1, n), windows[:, 1:].reshape(-1, n)
    else:
        x = np.concatenate([windows[:, :-2], windows[:, 1:-1]], axis=2).reshape(-1, 2 * n)
        y = windows[:, 2:].reshape(-1, n)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    r = min(rank, np.count_nonzero(s > s[0] * 1e-10))
    coef = (vt[:r].T / s[:r]) @ u[:, :r].T @ y
    residual = y - x @ coef
    q = residual.T @ residual / len(residual)
    if delay == 1:
        return coef.T, q
    a = np.block([[np.zeros((n, n)), np.eye(n)], [coef[:n].T, coef[n:].T]])
    return a, np.block([[np.zeros((n, n)), np.zeros((n, n))], [np.zeros((n, n)), q]])


def fit_band_transition(windows: np.ndarray, u: np.ndarray, bands: list[np.ndarray]):
    z = windows @ u
    rates = np.zeros(len(u))
    for band in bands:
        rates[band] = np.sum(z[:, 1:, band] * z[:, :-1, band]) / np.sum(z[:, :-1, band] ** 2)
    a = (u * rates) @ u.T
    residual = (windows[:, 1:] - windows[:, :-1] @ a.T).reshape(-1, len(u))
    return a, residual.T @ residual / len(residual)


def forecast_amplitude(
    windows: np.ndarray, a: np.ndarray, q: np.ndarray, horizon: int, adaptive: bool = False
):
    n = windows.shape[-1]
    delay = len(a) // n
    state = windows[:, -delay:].reshape(len(windows), -1).copy()
    covariance = np.zeros_like(a)
    if adaptive:
        if delay == 1:
            before, after = windows[:, :-1], windows[:, 1:]
        else:
            before = np.concatenate([windows[:, :-2], windows[:, 1:-1]], axis=2)
            after = np.concatenate([windows[:, 1:-1], windows[:, 2:]], axis=2)
        residual = after - before @ a.T
        episode_q = np.einsum("nti,ntj->nij", residual, residual) / residual.shape[1]
        selector = np.diag(np.r_[np.zeros(len(a) - n), np.ones(n)])
        impulse = np.eye(len(a))
        noise_gain = np.zeros_like(a)
    energies = []
    for _ in range(horizon):
        state = state @ a.T
        if adaptive:
            noise_gain += impulse.T @ selector @ impulse
            impulse = a @ impulse
            variance = np.einsum("ij,nji->n", noise_gain, episode_q)
        else:
            covariance = a @ covariance @ a.T + q
            variance = np.trace(covariance[-n:, -n:])
        energies.append((np.sum(state[:, -n:] ** 2, axis=1) + variance) / n)
    future = np.sqrt(np.mean(energies[-4:], axis=0))
    now = np.sqrt(np.mean(windows[:, -4:] ** 2, axis=(1, 2)))
    return np.log((future + EPS) / (now + EPS))


def coupling_fraction(a: np.ndarray, u: np.ndarray) -> float:
    b = u.T @ a @ u
    return float(np.sum((b - np.diag(np.diag(b))) ** 2) / np.sum(b**2))


def phase_forecast(window: np.ndarray, horizon: int, ridge=1e-4):
    """Independent complex AR(2) on spatial FFT modes of a real periodic signal."""
    z = np.fft.rfft(window, axis=1, norm="ortho")
    history = [z[-2].copy(), z[-1].copy()]
    coefficients = []
    for k in range(z.shape[1]):
        predictors = np.column_stack([z[1:-1, k], z[:-2, k]])
        target = z[2:, k]
        gram = predictors.conj().T @ predictors
        penalty = max(float(np.trace(gram).real) / 2, 1e-12) * ridge
        coef = np.linalg.solve(gram + penalty * np.eye(2), predictors.conj().T @ target)
        coefficients.append(coef)
    coefficients = np.asarray(coefficients)
    future = []
    for _ in range(horizon):
        new = coefficients[:, 0] * history[-1] + coefficients[:, 1] * history[-2]
        history.append(new)
        future.append(np.fft.irfft(new, n=window.shape[1], norm="ortho"))
    now = amplitude(window[-4:])
    return float(np.log((amplitude(np.asarray(future)[-4:]) + EPS) / (now + EPS)))


def raw_var2_forecast(window: np.ndarray, horizon: int, ridge=1e-4):
    n = window.shape[1]
    x = np.c_[window[1:-1], window[:-2]]
    y = window[2:]
    gram = x.T @ x
    # The predictor dimension is the appropriate scale for isotropic raw-coordinate ridge.
    penalty = max(float(np.trace(gram)) / len(gram), 1e-12) * ridge
    coef = np.linalg.solve(gram + penalty * np.eye(2 * n), x.T @ y)
    history = [window[-2].copy(), window[-1].copy()]
    future = []
    for _ in range(horizon):
        new = np.r_[history[-1], history[-2]] @ coef
        history.append(new)
        future.append(new)
    return float(
        np.log((amplitude(np.asarray(future)[-4:]) + EPS) / (amplitude(window[-4:]) + EPS))
    )


def transport_fixture(n=128, steps=48):
    locations = np.arange(n)
    packet = np.exp(-0.5 * ((locations - 12) / 4) ** 2)
    x = np.array([np.roll(packet, t) for t in range(steps)])
    return x, np.sqrt(np.mean(x[:, 48:64] ** 2, axis=1))
