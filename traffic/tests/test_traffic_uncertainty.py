"""Joint covariance, GLS, and noise-aware transition-estimation controls."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Research dependencies required")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mode_consistency import real_operator
from uncertainty_core import (
    JointGLS,
    fit_transition,
    full_history_design,
    joint_noise,
    temporal_covariance,
)


class UncertaintyTests(unittest.TestCase):
    def test_restricted_fit_recovers_exact_model(self):
        from analyze_physics_fit import fit_physics
        from mechanism_core import equilibrium, mode_operator

        speed, gaps = equilibrium(479.92, 1.8)
        scales = np.diag([1 / gaps[0], 1 / speed, 1 / gaps[0], 1 / speed])
        op = (
            scales
            @ real_operator(
                np.linalg.matrix_power(mode_operator(3, gaps[0], speed, 0.3, 1.8, dt=0.01), 50)
            )
            @ np.linalg.inv(scales)
        )
        initial = np.random.default_rng(8).normal(size=(8, 4)) * 0.001
        histories = np.stack([initial @ np.linalg.matrix_power(op, t).T for t in range(21)], axis=1)
        _, info = fit_physics(histories, 1e-7 * np.eye(4), scales, speed, gaps[0], 0.01)
        self.assertEqual(info["status"], "eligible")
        self.assertAlmostEqual(info["inferred_headway"], 1.8, places=10)
        self.assertAlmostEqual(info["estimated_acceleration"], 0.3, places=5)

    def test_acceleration_only_ambiguity_has_opposite_growth_signs(self):
        from mechanism_core import equilibrium, mode_operator

        speed, gaps = equilibrium(479.92, 1.8)
        a = mode_operator(3, gaps[0], speed, 0.3, 1.8)
        b = mode_operator(3, gaps[0], speed, 1.5, 1.8)
        self.assertGreater(np.linalg.eigvals(a).real.max(), 0)
        self.assertLess(np.linalg.eigvals(b).real.max(), 0)
        np.testing.assert_array_equal((a - b)[0], 0)

    def test_relative_study_path_receipt(self):
        from analyze_uncertainty import ROOT, load_study

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory)
            (path / "scenarios.csv").write_text("id\nfixture\n")
            (path / "manifest.json").write_text("{}")
            _, arrays, receipt = load_study("unused", path.relative_to(ROOT))
            self.assertEqual(arrays, {})
            self.assertEqual(receipt["result_directory"], str(path.relative_to(ROOT)))

    def test_joint_covariance_whitens_and_matches_explicit_gls(self):
        op = real_operator(np.array([[0.95, 0.01j], [0.01, 0.98 - 0.02j]]))
        cov = np.diag([0.01, 0.002, 0.01, 0.002])
        tc = temporal_covariance(0.8, 5, True)
        gls = JointGLS(op, cov, tc)
        full = np.kron(tc, cov)
        w = np.kron(gls.time_white, gls.state_white)
        np.testing.assert_allclose(w @ full @ w.T, np.eye(164), atol=1e-13)
        samples = joint_noise(cov, tc, 5000, 23)
        scores = gls.score(samples)
        self.assertAlmostEqual(scores.mean(), 160 / 41, delta=0.03)
        self.assertTrue(0.03 < np.mean(scores > gls.cutoff) < 0.07)
        d = full_history_design(op)
        y = samples[0].ravel()
        fit = np.linalg.solve(d.T @ np.linalg.solve(full, d), d.T @ np.linalg.solve(full, y))
        error = y - d @ fit
        self.assertAlmostEqual(scores[0], error @ np.linalg.solve(full, error) / 41, places=10)

    def test_nuisance_rank_and_exact_history(self):
        op = real_operator(np.array([[0.95 + 0.03j, 0.01], [0.02, 0.97 - 0.01j]]))
        for nuisance, rank in [(0, 4), (1, 8), (2, 12)]:
            gls = JointGLS(op, np.eye(4), nuisance=nuisance)
            self.assertEqual(gls.rank, rank)
            d = full_history_design(op, nuisance=nuisance)
            self.assertLess(gls.score((d @ np.ones(rank)).reshape(41, 4)), 1e-11)

    def test_exact_ols_tls_and_corrected_moments(self):
        rng = np.random.default_rng(11)
        a = np.array([[0.9 + 0.05j, 0.1], [-0.05j, 0.97]])
        z = rng.normal(size=(10000, 2)) + 1j * rng.normal(size=(10000, 2))
        pairs = np.stack([z, z @ a.T], axis=1)
        histories = np.concatenate([pairs.real, pairs.imag], axis=2)
        for method in ["ols", "tls", "noise_corrected"]:
            fit, _, status = fit_transition(histories, 1e-12 * np.eye(4), method)
            self.assertEqual(status, "eligible")
            np.testing.assert_allclose(fit, real_operator(a), atol=1e-9)
        noisy = histories + rng.normal(scale=0.3, size=histories.shape)
        ols = fit_transition(noisy, 0.09 * np.eye(4), "ols")[0]
        for method in ["tls", "noise_corrected"]:
            estimate = fit_transition(noisy, 0.09 * np.eye(4), method)[0]
            self.assertLess(
                np.linalg.norm(estimate - real_operator(a)), np.linalg.norm(ols - real_operator(a))
            )

    def test_nonpositive_moment_abstention(self):
        fit, _, status = fit_transition(np.zeros((2, 3, 4)), np.eye(4), "noise_corrected")
        self.assertIsNone(fit)
        self.assertEqual(status, "nonpositive_corrected_gram")

    def test_tls_matches_augmented_snapshot_projection(self):
        rng = np.random.default_rng(18)
        h = rng.normal(size=(8, 21, 4))
        estimate = fit_transition(h, 0.5 * np.eye(4), "tls")[0]
        z = h[..., :2] + 1j * h[..., 2:]
        x, y = z[:, :-1].reshape(-1, 2).T, z[:, 1:].reshape(-1, 2).T
        _, _, vh = np.linalg.svd(np.vstack([x, y]), full_matrices=False)
        v = vh[:2].conj().T
        projected_x, projected_y = x @ v @ v.conj().T, y @ v @ v.conj().T
        expected = projected_y @ np.linalg.pinv(projected_x)
        np.testing.assert_allclose(estimate, real_operator(expected), atol=1e-12)
