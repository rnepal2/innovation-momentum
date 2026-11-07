"""Aliasing, target identification, and exact known-noise residual controls."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Research dependencies required")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechanism_core import equilibrium, initialize
from mode_consistency import (
    covariance,
    dictionary,
    history_design,
    noise_draws,
    observed_state,
    real_operator,
    recovery,
    residual,
    score,
    threshold,
    window_score,
    window_threshold,
)
from run_mechanism import recording_stride


class ConsistencyTests(unittest.TestCase):
    def test_simulator_millisecond_clock_contract(self):
        self.assertEqual(recording_stride(0.025), 4)
        self.assertEqual(recording_stride(0.01), 10)
        for dt in [0.0125, 0.013, 0, -0.1, np.nan]:
            with self.assertRaises(ValueError):
                recording_stride(dt)

    def test_target_identifiable_despite_nuisance_alias(self):
        _, full, rank, _ = recovery(np.arange(24), True)
        self.assertTrue(full)
        self.assertEqual(rank, 9)
        _, identifiable, rank, _ = recovery(np.arange(0, 24, 2), True)
        self.assertTrue(identifiable)
        self.assertEqual(rank, 7)
        indices = np.arange(0, 24, 2)
        c = recovery(indices, True)[0]
        expected = np.zeros((2, 9))
        expected[0, 5], expected[1, 6] = 0.5, -0.5
        np.testing.assert_allclose(c @ dictionary([1, 2, 3, 11])[indices], expected, atol=1e-12)
        self.assertFalse(recovery(np.arange(0, 24, 3), True)[1])

    def test_alias_fixture_and_naive_recovery(self):
        n = np.arange(24)
        wave = np.cos(2 * np.pi * 11 * n / 24)
        indices = np.arange(0, 24, 3)
        np.testing.assert_allclose(wave[indices], np.cos(2 * np.pi * 3 * indices / 24), atol=1e-12)
        c, _, _, _ = recovery(indices, False)
        np.testing.assert_allclose(c @ wave[indices], [0.5, 0], atol=1e-12)

    def test_dictionary_recovery_and_physical_phase(self):
        u = dictionary([1, 2, 3, 11])
        indices = np.sort(np.random.default_rng(26016).choice(24, 12, replace=False))
        c, valid, rank, _ = recovery(indices, True)
        self.assertTrue(valid)
        self.assertEqual(rank, 9)
        coefficients = np.arange(9) / 10
        np.testing.assert_allclose(
            c @ (u @ coefficients)[indices], [coefficients[5] / 2, -coefficients[6] / 2], atol=1e-12
        )
        v, gaps = equilibrium(479.92)
        for phase in [0, 0.3, 1.0]:
            p, s = initialize(gaps, v, 3, 0.02, "gap", phase=phase)
            actual = (np.roll(p, -1) - p) % 479.92 - 5 - gaps
            np.testing.assert_allclose(
                actual, 0.02 * np.cos(2 * np.pi * 3 * np.arange(24) / 24 + phase), atol=1e-12
            )
            np.testing.assert_allclose(s, v)

    def test_covariance_propagates_sensor_errors(self):
        c, _, _, _ = recovery(np.arange(24), True)
        expected = covariance(c, 0.05, 0.005)
        rng = np.random.default_rng(1)
        response = np.stack(
            [rng.normal(scale=0.05, size=(100000, 24)), rng.normal(scale=0.005, size=(100000, 24))],
            axis=1,
        )
        states = observed_state(response, np.arange(24), c)
        np.testing.assert_allclose(
            np.diag(np.cov(states, rowvar=False)), np.diag(expected), rtol=0.02
        )

    def test_real_operator_and_exact_linear_residual(self):
        a = np.array([[0.9 + 0.02j, 0.1], [0.03, 0.95 - 0.01j]])
        z = np.array([0.2 + 0.3j, 0.1 - 0.1j])
        state = np.r_[z.real, z.imag]
        np.testing.assert_allclose(real_operator(a) @ state, np.r_[(a @ z).real, (a @ z).imag])
        history = [state]
        for _ in range(40):
            history.append(real_operator(a) @ history[-1])
        np.testing.assert_allclose(residual(np.asarray(history), real_operator(a)), 0, atol=1e-15)

    def test_calibrated_noise_null_uses_independent_draws(self):
        a = 0.95 * np.eye(4)
        cov = np.diag([0.01, 0.02, 0.01, 0.02])
        cutoff, inverse = threshold(a, cov, draws=4000, seed=80000)
        values = score(noise_draws(cov, 5000, 41, 70000), a, inverse)
        self.assertAlmostEqual(values.mean(), 4.0, delta=0.05)
        self.assertTrue(0.03 < float(np.mean(values > cutoff)) < 0.07)

    def test_whole_history_unknown_initial_state_and_null(self):
        from scipy.stats import chi2

        a = np.array([[0.97, 0.03, 0, 0], [0, 0.95, 0, 0], [0, 0, 0.97, -0.03], [0, 0, 0, 0.95]])
        cov = np.diag([0.01, 0.02, 0.01, 0.02])
        cutoff, whitening, basis = window_threshold(a, cov, draws=4000, seed=180000)
        design = history_design(a)
        self.assertEqual(np.linalg.matrix_rank(design), 4)
        state = (design @ np.array([0.1, 0.2, 0.3, 0.4])).reshape(41, 4)
        self.assertLess(window_score(state, whitening, basis), 1e-28)
        values = window_score(state + noise_draws(cov, 5000, 41, 170000), whitening, basis)
        self.assertAlmostEqual(values.mean(), 160 / 41, delta=0.04)
        self.assertTrue(0.03 < float(np.mean(values > cutoff)) < 0.07)
        self.assertAlmostEqual(cutoff, chi2.ppf(0.95, 160) / 41, delta=0.08)
