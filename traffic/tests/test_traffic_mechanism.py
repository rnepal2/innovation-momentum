"""Mechanistic state, Jacobian, and discrete sampling checks."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Research dependencies required")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechanism_core import (
    acceleration,
    derivatives,
    equilibrium,
    full_operator,
    initialize,
    mode_operator,
    predict_mode,
)


class MechanismTests(unittest.TestCase):
    def test_equilibrium_and_heterogeneous_conservation(self):
        for taus in [1.0, np.linspace(0.6, 1.4, 24)]:
            v, s = equilibrium(479.92, taus)
            self.assertAlmostEqual(s.sum() + 120, 479.92)
            np.testing.assert_allclose(acceleration(s, v, v, tau=taus), 0, atol=1e-12)

    def test_derivatives_against_centered_differences(self):
        v, s = equilibrium(479.92)
        state = np.array([s[0], v, v])
        numerical = []
        for j in range(3):
            delta = np.zeros(3)
            delta[j] = 1e-5
            numerical.append(
                (acceleration(*(state + delta)) - acceleration(*(state - delta))) / 2e-5
            )
        np.testing.assert_allclose(derivatives(s[0], v), numerical, rtol=1e-7, atol=1e-10)

    def test_full_operator_matches_every_fourier_block(self):
        v, s = equilibrium(479.92)
        for dt in [None, 0.025, 0.05]:
            a = full_operator(s, v, dt=dt)
            for q in range(13):
                wave = np.exp(2j * np.pi * q * np.arange(24) / 24)
                for state in [np.array([1, 0.3j]), np.array([0.4j, 1])]:
                    full = np.r_[state[0] * wave, state[1] * wave]
                    block = mode_operator(q, s[0], v, dt=dt) @ state
                    np.testing.assert_allclose(
                        a @ full, np.r_[block[0] * wave, block[1] * wave], atol=1e-12
                    )

    def test_initial_gap_mode_is_physical_and_conserved(self):
        v, s = equilibrium(479.92)
        for q in range(1, 13):
            p, speeds = initialize(s, v, q, 0.1, "gap")
            measured = (np.roll(p, -1) - p) % 479.92 - 5
            np.testing.assert_allclose(
                measured, s + 0.1 * np.cos(2 * np.pi * q * np.arange(24) / 24), atol=1e-12
            )
            self.assertAlmostEqual(measured.sum(), 479.92 - 120)
            np.testing.assert_allclose(speeds, v)

    def test_semiimplicit_map_against_nonlinear_difference(self):
        v, s = equilibrium(479.92)
        dt = 0.025
        state = np.r_[s, np.full(24, v)]

        def update(z):
            gaps, speeds = z[:24], z[24:]
            next_speed = speeds + dt * acceleration(gaps, speeds, np.roll(speeds, -1))
            return np.r_[gaps + dt * (np.roll(next_speed, -1) - next_speed), next_speed]

        numerical = np.column_stack(
            [
                (update(state + np.eye(48)[j] * 1e-5) - update(state - np.eye(48)[j] * 1e-5)) / 2e-5
                for j in range(48)
            ]
        )
        np.testing.assert_allclose(full_operator(s, v, dt=dt), numerical, atol=3e-10)

    def test_sampling_and_continuous_limit(self):
        v, s = equilibrium(479.92)
        a = mode_operator(2, s[0], v)
        dt = 1e-6
        np.testing.assert_allclose(
            (mode_operator(2, s[0], v, dt=dt) - np.eye(2)) / dt, a, atol=1e-6
        )
        step = mode_operator(2, s[0], v, dt=0.025)
        predicted = predict_mode([1.0, 0.2], step, 11, 4)
        np.testing.assert_allclose(
            predicted[-1], np.linalg.matrix_power(step, 40) @ np.array([1, 0.2])
        )
