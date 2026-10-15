"""Mathematical and availability checks for the isolated traffic benchmark."""

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

if any(importlib.util.find_spec(name) is None for name in ["scipy", "sklearn", "matplotlib"]):
    raise unittest.SkipTest("Research tests require uv sync --extra research")

TRAFFIC = Path(__file__).resolve().parents[1] / "research/2026-10/traffic"
sys.path.insert(0, str(TRAFFIC))
import audit_data
from provenance import archive_sources
from run_experiments import seed_groups
from scipy.signal import get_window
from sklearn.linear_model import Ridge
from sumo_pilot import real_fourier_basis
from traffic_core import (
    Design,
    amplification,
    coordinate_bands,
    coupling_fraction,
    dct_basis,
    evolution,
    fit_band_transition,
    fit_pca,
    fit_transition,
    forecast_amplitude,
    graph_basis,
    modal_psd,
    past_window,
    phase_forecast,
    random_basis,
    ridge_predict,
    road_graph,
    simulate,
    spectral_features,
    transport_fixture,
)


class TrafficTests(unittest.TestCase):
    def setUp(self):
        self.e, self.u = graph_basis(road_graph("path"))
        self.bands = coordinate_bands(24, self.e)
        self.x = np.random.default_rng(1).normal(size=(96, 24))

    def test_parseval_and_density_integration(self):
        self.assertTrue(np.allclose(self.u.T @ self.u, np.eye(24)))
        self.assertAlmostEqual(np.sum(self.x**2), np.sum((self.x @ self.u) ** 2))
        f, psd = modal_psd(self.x, self.u, 2)
        z = self.x @ self.u
        taper = get_window("hann", len(z))
        expected = np.sum(((z - z.mean(axis=0)) * taper[:, None]) ** 2, axis=0) / np.sum(taper**2)
        self.assertTrue(np.allclose(psd.sum(axis=0) * (f[1] - f[0]), expected))

    def test_path_dct_and_sign_invariance(self):
        a = spectral_features(self.x, self.u, self.bands)[1]
        b = spectral_features(self.x, dct_basis(24), self.bands)[1]
        self.assertTrue(np.allclose(a, b, atol=1e-12))
        self.assertTrue(np.allclose(a, spectral_features(self.x, -self.u, self.bands)[1]))

    def test_cycle_matches_ordinary_fourier_subspaces(self):
        w = np.roll(np.eye(24), 1, axis=0) + np.roll(np.eye(24), -1, axis=0)
        e, u = graph_basis(w)
        f = real_fourier_basis(24)
        self.assertTrue(np.allclose(f.T @ f, np.eye(24), atol=1e-12))
        for band in coordinate_bands(24, e):
            self.assertTrue(np.allclose(u[:, band] @ u[:, band].T, f[:, band] @ f[:, band].T))

    def test_phase_model_recovers_traveling_wave_growth(self):
        t = np.arange(94)[:, None]
        position = np.arange(24)[None, :]
        x = 1.005**t * np.cos(2 * np.pi * position / 24 - 0.3 * t)
        actual = amplification(x, 63, 30)
        self.assertAlmostEqual(actual, 30 * np.log(1.005), places=6)
        self.assertLess(abs(phase_forecast(x[:64], 30) - actual), 0.01)
        before = phase_forecast(x[:64], 30)
        x[64:] = 1e9
        self.assertEqual(phase_forecast(x[:64], 30), before)
        self.assertAlmostEqual(phase_forecast(np.zeros((32, 24)), 30), 0)

    def test_repeated_eigenspace_is_not_split(self):
        e = np.repeat(np.arange(4), 3)
        bands = coordinate_bands(12, e)
        for value in np.unique(e):
            self.assertEqual(sum(np.any(e[b] == value) for b in bands), 1)
        self.assertEqual(sum(map(len, bands)), len(e))

    def test_subspace_rotation_invariance(self):
        rotated = self.u.copy()
        for j, band in enumerate(self.bands):
            rotated[:, band] = self.u[:, band] @ random_basis(len(band), j)
        self.assertTrue(
            np.allclose(
                spectral_features(self.x, self.u, self.bands)[1],
                spectral_features(self.x, rotated, self.bands)[1],
                atol=1e-12,
            )
        )

    def test_future_mutation_does_not_change_features(self):
        before = spectral_features(past_window(self.x, 63, 64), self.u, self.bands)[1]
        self.x[64:] = 1e8
        after = spectral_features(past_window(self.x, 63, 64), self.u, self.bands)[1]
        self.assertTrue(np.array_equal(before, after))
        with self.assertRaises(ValueError):
            past_window(self.x, 20, 64)

    def test_scalar_evolution_and_covariance_forecast(self):
        rate = 1.01
        x = rate ** np.arange(80)[:, None] * np.ones((1, 24))
        self.assertAlmostEqual(amplification(x, 63, 16), 16 * np.log(rate), places=6)
        pred = forecast_amplitude(x[None, :64], np.eye(24) * rate, np.zeros((24, 24)), 16)
        self.assertAlmostEqual(pred[0], amplification(x, 63, 16), places=12)
        expected = np.log((np.sqrt(1 + np.mean(np.arange(13, 17))) + 1e-6) / (1 + 1e-6))
        pred = forecast_amplitude(np.ones((1, 64, 24)), np.eye(24), np.eye(24), 16)
        self.assertAlmostEqual(pred[0], expected, places=6)

    def test_transition_recovery(self):
        rng = np.random.default_rng(2)
        a = evolution("aligned", self.u, self.bands)
        windows = []
        for _ in range(20):
            x = rng.normal(size=24)
            trajectory = [x]
            for _ in range(63):
                x = a @ x
                trajectory.append(x)
            windows.append(trajectory)
        windows = np.asarray(windows)
        fitted, q = fit_transition(windows, 24)
        self.assertTrue(np.allclose(a, fitted, atol=1e-10))
        self.assertLess(np.linalg.norm(q), 1e-20)
        fitted, _ = fit_band_transition(windows, self.u, self.bands)
        self.assertTrue(np.allclose(a, fitted, atol=1e-10))

    def test_known_alignment_and_nonnormality(self):
        a = evolution("aligned", self.u, self.bands)
        self.assertLess(coupling_fraction(a, self.u), 1e-20)
        a = evolution("directed", self.u, self.bands)
        self.assertLess(np.max(np.abs(np.linalg.eigvals(a))), 1)
        self.assertGreater(np.linalg.norm(a, 2), 1)
        self.assertGreater(coupling_fraction(a, self.u), 0)

    def test_adaptive_covariance_matches_explicit_recursion(self):
        rng = np.random.default_rng(17)
        windows = rng.normal(size=(2, 64, 24))
        a = 0.9 * np.eye(24) + 0.04 * np.eye(24, k=1)
        predicted = forecast_amplitude(windows, a, np.eye(24), 16, adaptive=True)
        for i, x in enumerate(windows):
            residual = x[1:] - x[:-1] @ a.T
            q = residual.T @ residual / len(residual)
            actual = forecast_amplitude(x[None], a, q, 16)[0]
            self.assertAlmostEqual(predicted[i], actual, places=12)
        hankel, q = fit_transition(windows, 48, delay=2)
        residual = (
            np.concatenate([windows[:, 1:-1], windows[:, 2:]], axis=2)
            - np.concatenate([windows[:, :-2], windows[:, 1:-1]], axis=2) @ hankel.T
        )
        for i, x in enumerate(windows):
            q = residual[i].T @ residual[i] / len(residual[i])
            self.assertAlmostEqual(
                forecast_amplitude(x[None], hankel, q, 16)[0],
                forecast_amplitude(x[None], hankel, np.eye(48), 16, adaptive=True)[0],
                places=12,
            )

    def test_full_coordinate_ridge_rotation_equivalence(self):
        rng = np.random.default_rng(1)
        x, y, test = rng.normal(size=(80, 24)), rng.normal(size=80), rng.normal(size=(10, 24))
        first = Ridge(alpha=4).fit(x, y).predict(test)
        second = Ridge(alpha=4).fit(x @ self.u, y).predict(test @ self.u)
        self.assertTrue(np.allclose(first, second, atol=1e-12))

    def test_training_selection_is_independent_of_test_inputs(self):
        rng = np.random.default_rng(14)
        train, validation = rng.normal(size=(40, 5)), rng.normal(size=(20, 5))
        y, v = rng.normal(size=40), rng.normal(size=20)
        _, first, model = ridge_predict(train, y, validation, v, np.ones((4, 5)))
        _, second, _ = ridge_predict(train, y, validation, v, np.ones((4, 5)) * 1e6)
        self.assertEqual(first, second)
        self.assertTrue(np.allclose(model[0].mean_, train.mean(axis=0)))
        windows = rng.normal(size=(20, 64, 24))
        basis = fit_pca(windows)
        covariance = windows.reshape(-1, 24).T @ windows.reshape(-1, 24)
        self.assertTrue(
            np.allclose(
                basis.T @ covariance @ basis, np.diag(np.diag(basis.T @ covariance @ basis))
            )
        )

    def test_seed_groups_and_simulation_reproducibility(self):
        groups = seed_groups(Design())
        self.assertFalse(set(groups["test"]) & set(seed_groups(Design(), "adaptive")["test"]))
        values = list(groups.values())
        for i, first in enumerate(values):
            for second in values[i + 1 :]:
                self.assertFalse(set(first) & set(second))
        a = np.eye(24) * 0.97
        self.assertTrue(np.array_equal(simulate(a, 7, Design()), simulate(a, 7, Design())))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            hashes = archive_sources(out, TRAFFIC)
            self.assertIn("traffic_core.py", hashes)
            self.assertEqual(
                (out / "frozen_sources/protocol.md.txt").read_bytes(),
                (TRAFFIC / "protocol.md").read_bytes(),
            )

    def test_transport_label_ignores_entry_into_region(self):
        x, region = transport_fixture()
        rms = np.sqrt(np.mean(x * x, axis=1))
        self.assertLess(np.ptp(rms), 1e-14)
        self.assertAlmostEqual(amplification(x, 15, 16), 0, places=12)
        self.assertGreater(region.max(), 1000 * region[0])

    def test_failed_download_preserves_accepted_input(self):
        class Response(io.BytesIO):
            def __init__(self, value):
                super().__init__(value)
                self.headers = {}

        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "accepted.csv"
            dest.write_bytes(b"accepted")
            for kwargs in [
                {"max_bytes": 3},
                {"max_bytes": 100, "expected_size": 9},
                {"max_bytes": 100, "expected_sha": "bad"},
            ]:
                with (
                    patch.object(audit_data, "urlopen", return_value=Response(b"bad data")),
                    self.assertRaises(ValueError),
                ):
                    audit_data.fetch_verified("https://example.com/sample", dest, **kwargs)
                self.assertEqual(dest.read_bytes(), b"accepted")
                self.assertEqual(list(Path(directory).iterdir()), [dest])

    def test_data_units_and_duplicate_audit(self):
        frame = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "space": [62.0, 61.99, 61.98],
                "speed": [36.0, 36.0, 36.0],
                "v_id": [1, 1, 1],
            }
        )
        result = audit_data.profile_trajectories(frame)
        self.assertLess(result["kinematic_abs_error_mph"]["1"], 1e-9)
        self.assertEqual(result["duplicate_vehicle_time"], 0)
        result = audit_data.profile_trajectories(pd.concat([frame, frame.iloc[:1]]))
        self.assertEqual(result["duplicate_vehicle_time"], 1)

    def test_missing_endpoints_are_counted(self):
        frame = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "space": [62.0, 61.99, np.nan],
                "speed": [36.0, 36.0, np.nan],
                "v_id": [1, 1, 1],
            }
        )
        result = audit_data.profile_trajectories(frame)
        self.assertEqual(result["nonfinite_physical_rows"], 1)
        self.assertEqual(result["nonfinite_at_last_timestamp"], 1)
        self.assertAlmostEqual(result["finite_fraction"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
