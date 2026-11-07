"""IDM gap-speed linearization and independently testable ring mode operators."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def acceleration(gap, speed, leader_speed, accel=0.8, tau=1.0, decel=2.0, desired=30.0):
    gap, speed, leader_speed = np.broadcast_arrays(gap, speed, leader_speed)
    target = 2 + np.maximum(
        0, tau * speed + speed * (speed - leader_speed) / (2 * np.sqrt(accel * decel))
    )
    return accel * (1 - (speed / desired) ** 4 - (target / gap) ** 2)


def equilibrium(circumference, taus=1.0, n=24):
    taus = np.broadcast_to(taus, (n,)).copy()

    def gaps(v):
        return (2 + taus * v) / np.sqrt(1 - (v / 30) ** 4)

    v = brentq(lambda v: gaps(v).sum() + 5 * n - circumference, 0, 29.999)
    return v, gaps(v)


def derivatives(gap, speed, accel=0.8, tau=1.0, decel=2.0):
    target = 2 + tau * speed
    cross = speed / (2 * np.sqrt(accel * decel))
    fs = 2 * accel * target**2 / gap**3
    fv = -accel * (4 * speed**3 / 30**4 + 2 * target / gap**2 * (tau + cross))
    fl = 2 * accel * target / gap**2 * cross
    return fs, fv, fl


def full_operator(gaps, speed, accel=0.8, taus=1.0, dt=None):
    n = len(gaps)
    fs, fv, fl = derivatives(np.asarray(gaps), speed, accel, np.broadcast_to(taus, n))
    fs, fv, fl = [np.broadcast_to(v, n) for v in [fs, fv, fl]]
    lead = np.roll(np.eye(n), 1, axis=1)
    difference = lead - np.eye(n)
    speed_gap = np.diag(fs)
    speed_speed = np.diag(fv) + np.diag(fl) @ lead
    if dt is None:
        return np.block([[np.zeros((n, n)), difference], [speed_gap, speed_speed]])
    next_speed = np.eye(n) + dt * speed_speed
    return np.block(
        [
            [np.eye(n) + dt**2 * difference @ speed_gap, dt * difference @ next_speed],
            [dt * speed_gap, next_speed],
        ]
    )


def mode_operator(q, gap, speed, accel=0.8, tau=1.0, dt=None, n=24):
    fs, fv, fl = derivatives(gap, speed, accel, tau)
    phase = np.exp(2j * np.pi * q / n)
    eta = phase - 1
    speed_speed = fv + fl * phase
    if dt is None:
        return np.array([[0, eta], [fs, speed_speed]], dtype=complex)
    return np.array(
        [
            [1 + dt**2 * eta * fs, dt * eta * (1 + dt * speed_speed)],
            [dt * fs, 1 + dt * speed_speed],
        ],
        dtype=complex,
    )


def mode_response(gaps, speeds, q):
    return (
        np.column_stack([np.fft.fft(gaps, axis=1)[:, q], np.fft.fft(speeds, axis=1)[:, q]])
        / gaps.shape[1]
    )


def predict_mode(initial, operator, steps, stride):
    sampled = np.linalg.matrix_power(operator, stride)
    states = [np.asarray(initial, dtype=complex)]
    for _ in range(steps - 1):
        states.append(sampled @ states[-1])
    return np.asarray(states)


def initialize(gaps, speed, q=0, amplitude=0.0, kind="speed", phase_offset=17.0, phase=0.0):
    n = len(gaps)
    wave = np.cos(2 * np.pi * q * np.arange(n) / n + phase)
    perturbed_gaps = np.array(gaps, copy=True)
    speeds = np.full(n, speed)
    if kind == "gap":
        perturbed_gaps += amplitude * wave
    elif kind == "speed":
        speeds += amplitude * wave
    else:
        raise ValueError(kind)
    positions = np.r_[0, np.cumsum(perturbed_gaps[:-1] + 5)] + phase_offset
    return positions, speeds


def relative_error(actual, predicted):
    denominator = np.linalg.norm(actual)
    return (
        float(np.linalg.norm(actual - predicted) / denominator) if denominator > 1e-14 else np.nan
    )
