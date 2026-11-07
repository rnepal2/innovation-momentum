"""Normalization, invariance, and interpretation checks for traffic spectra."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if any(importlib.util.find_spec(name) is None for name in ["scipy", "sklearn", "matplotlib"]):
    raise unittest.SkipTest("Research tests require uv sync --extra research")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_forcing_spectra import scaling_table, summarize_response
from spectral_diagnostics import (
    BANDS,
    band_power,
    cycle_modes,
    fft_power,
    joint_density,
    signed_density,
    spatial_power,
    weighted_spatial_power,
    window_summary,
)


class SpectralDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(37).normal(size=(91, 24)) + 3

    def test_mean_decomposition_and_fourier_equivalence(self):
        p = spatial_power(self.x)
        np.testing.assert_allclose(p["mean"] ** 2 + p["total"], p["raw_power"], atol=1e-12)
        np.testing.assert_allclose(p["energy"], fft_power(self.x), atol=1e-12)
        np.testing.assert_allclose(p["shares"][:, 1:].sum(axis=1), 1)

    def test_repeated_eigenspace_rotation(self):
        basis, groups = cycle_modes()
        rotated = basis.copy()
        rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
        for group in groups:
            if len(group) == 2:
                rotated[:, group] = basis[:, group] @ rotation
        np.testing.assert_allclose(
            spatial_power(self.x, rotated, groups)["energy"],
            spatial_power(self.x)["energy"],
            atol=1e-12,
        )

    def test_density_integration_independent_time_domain(self):
        for length in [24, 32, 40]:
            for detrend in [False, "constant"]:
                f, density, integrated = joint_density(self.x, 0, length, dt=0.5, detrend=detrend)
                expected = weighted_spatial_power(self.x, 0, length, detrend)
                np.testing.assert_allclose(integrated, expected, atol=1e-12)
                np.testing.assert_allclose(
                    density.sum(axis=0) * (f[1] - f[0]), expected, atol=1e-12
                )

    def test_pulse_dimension_reference_and_amplitude_scaling(self):
        x = np.zeros((1, 24))
        x[0, 0] = 1
        p = spatial_power(x)
        fractions = band_power(p["shares"])
        for name, expected in zip(BANDS, [4 / 23, 8 / 23, 11 / 23]):
            self.assertAlmostEqual(fractions[name][0], expected)
        np.testing.assert_allclose(
            spatial_power(2 * self.x)["energy"], 4 * spatial_power(self.x)["energy"]
        )
        np.testing.assert_allclose(
            spatial_power(2 * self.x)["shares"], spatial_power(self.x)["shares"]
        )

    def test_uniform_and_zero_have_no_reportable_spatial_share(self):
        for x in [np.zeros((91, 24)), np.full((91, 24), 2.0)]:
            p = spatial_power(x)
            self.assertTrue(np.isnan(p["shares"]).all())
            np.testing.assert_allclose(p["total"], 0, atol=1e-24)
            np.testing.assert_allclose(p["raw_power"], p["mean"] ** 2)

    def test_known_growth_and_window_isolation(self):
        t, n = np.arange(91)[:, None], np.arange(24)[None, :]
        x = np.exp(0.015 * t) * np.cos(2 * np.pi * 3 * n / 24 - 0.2 * t)
        _, _, growth = window_summary(x)
        self.assertAlmostEqual(growth[3], 0.015)
        before = joint_density(x, 4)[1]
        x[36:] = 1e8
        np.testing.assert_array_equal(before, joint_density(x, 4)[1])

    def test_signed_density_direction_and_no_input_mutation(self):
        t, n = np.arange(64)[:, None], np.arange(24)[None, :]
        x = np.cos(2 * np.pi * 3 * n / 24 - 2 * np.pi * 0.125 * t) + 2
        before = x.copy()
        q, f, density = signed_density(x, start=0, length=64)
        positive_q = np.flatnonzero(np.isclose(q, 3 / 24))[0]
        self.assertAlmostEqual(f[np.argmax(density[:, positive_q])], -0.125)
        self.assertAlmostEqual(density.sum() * (f[1] - f[0]), 0.5)
        np.testing.assert_array_equal(x, before)

    def test_invalid_and_incomplete_inputs(self):
        with self.assertRaises(ValueError):
            spatial_power(np.ones((4, 23)))
        with self.assertRaises(ValueError):
            joint_density(self.x, 80)
        with self.assertRaises(ValueError):
            cycle_modes(23)

    def test_forcing_summary_recovers_quadratic_fixture(self):
        x = np.random.default_rng(53).normal(size=(241, 24))
        rows = []
        for amplitude in [0.05, 0.1, 0.2]:
            summary, trajectory = summarize_response(amplitude * x, amplitude)
            for row in summary:
                rows.append(
                    {"accel": 0.8, "seed": 31, "dt": 0.025, "realized_drop": amplitude, **row}
                )
            np.testing.assert_allclose(trajectory.normalized_power, spatial_power(x)["total"])
        table = scaling_table(pd.DataFrame(rows))
        np.testing.assert_allclose(table.power_exponent, 2, atol=1e-12)
        np.testing.assert_allclose(table.normalized_power_max_min, 1, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
