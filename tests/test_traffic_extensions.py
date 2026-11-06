"""Graph-band approximation and empirical coverage guardrails."""

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if any(importlib.util.find_spec(name) is None for name in ["scipy", "matplotlib"]):
    raise unittest.SkipTest("Research dependencies required")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/2026-10/traffic"))
from collect_ngsim import acquire, complete_strip, profile
from mechanism_core import derivatives, equilibrium, full_operator
from run_heterogeneity import basis_for, coupling, patterns, projected


class ExtensionTests(unittest.TestCase):
    def test_accepted_acquisition_cannot_change_query_or_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text("[]")
            receipt = {
                "snapshot.json": {
                    "url": "https://example.com/a",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
            self.assertEqual(
                acquire(
                    "https://example.com/a", "snapshot.json", receipt, data_root=Path(directory)
                ),
                path,
            )
            with self.assertRaises(ValueError):
                acquire(
                    "https://example.com/b", "snapshot.json", receipt, data_root=Path(directory)
                )
            with self.assertRaises(FileExistsError):
                acquire("https://example.com/a", "snapshot.json", {}, data_root=Path(directory))
            self.assertEqual(path.read_text(), "[]")

    def test_patterns_fix_mean_and_bounds(self):
        for taus in patterns().values():
            self.assertAlmostEqual(taus.mean(), 1.0)
            self.assertGreaterEqual(taus.min(), 0.7 - 1e-12)
            self.assertLessEqual(taus.max(), 1.3 + 1e-12)
            v, _ = equilibrium(479.92, taus)
            self.assertAlmostEqual(v, equilibrium(479.92)[0])

    def test_basis_controls_and_homogeneous_exact_bands(self):
        v, s = equilibrium(479.92)
        a = full_operator(s, v)
        for seed in [None, 2000, 2001]:
            u = basis_for(seed=seed)
            np.testing.assert_allclose(u.T @ u, np.eye(24), atol=1e-12)
            np.testing.assert_allclose(abs(u[:, 0]), 1 / np.sqrt(24), atol=1e-12)
        self.assertLess(coupling(a, basis_for(), s, v), 1e-20)
        self.assertGreater(coupling(a, basis_for(seed=2000), s, v), 0.01)
        step = full_operator(s, v, dt=0.025)
        t, masked = projected(step, basis_for())
        np.testing.assert_allclose(masked, (t.T @ step @ t)[1:, 1:], atol=1e-12)

    def test_alternating_weight_control_is_uniform(self):
        v, s = equilibrium(479.92, patterns()["alternating"])
        fs, _, _ = derivatives(s, v, 0.8, patterns()["alternating"])
        weights = np.sqrt(fs * np.roll(fs, -1))
        np.testing.assert_allclose(weights, weights[0])

    def test_count_based_strip_and_missingness(self):
        records = [
            {
                "global_time": 1000 + t * 5000,
                "local_y": (b * 50 + 1) / 0.3048,
                "v_vel": float(t + b),
            }
            for t in range(36)
            for b in range(12)
            if b != 3
        ]
        f = pd.DataFrame(records)
        selected, info = complete_strip(f, 1000, 50)
        self.assertTrue(info["eligible"])
        self.assertEqual(info["spatial_range_m"], [200, 600])
        f["v_vel"] = np.random.default_rng(1).normal(size=len(f))
        _, changed = complete_strip(f, 1000, 50)
        self.assertEqual(info, changed)
        self.assertFalse(selected.isna().any().any())

    def test_incomplete_time_range_is_ineligible(self):
        f = pd.DataFrame(
            {"global_time": [1000] * 12, "local_y": np.arange(12) * 50 / 0.3048, "v_vel": 1.0}
        )
        _, info = complete_strip(f, 1000, 50)
        self.assertFalse(info["eligible"])

    def test_profile_detects_duplicates_and_units(self):
        f = pd.DataFrame(
            {
                "vehicle_id": [1, 1, 1],
                "global_time": [1000, 1100, 1100],
                "local_y": [0, 1, 1],
                "v_vel": [10.0, 10.0, 10.0],
                "lane_id": [1, 1, 1],
                "location": ["test"] * 3,
            }
        )
        p = profile(f)
        self.assertEqual(p["duplicate_keys"], 1)
        self.assertAlmostEqual(p["speed_position_discrepancy_mps"]["0.5"], 0)
        self.assertEqual(p["positive_dt_seconds"]["0.5"], 0.1)
