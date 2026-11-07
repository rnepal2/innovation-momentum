"""Fixed supports, no imputation, and explicit trace-quality exclusion."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Research dependencies required")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from empirical_resolution import fixed_field, flagged_traces, spectrum


class ResolutionTests(unittest.TestCase):
    def fixture(self):
        times, locations = np.meshgrid(np.arange(180) * 1000, np.arange(13) * 25 + 10)
        return pd.DataFrame(
            {
                "global_time": times.ravel(),
                "local_y": locations.ravel() / 0.3048,
                "v_vel": (20 + 0.01 * times.ravel() / 1000 + locations.ravel() / 100) / 0.3048,
            }
        )

    def test_fixed_support_and_no_fill(self):
        frame = self.fixture()
        field, info = fixed_field(frame, 0, 325, 25, 1)
        self.assertTrue(info["eligible"])
        self.assertEqual(field.shape, (180, 13))
        _, info = fixed_field(frame.iloc[1:], 0, 325, 25, 1)
        self.assertFalse(info["eligible"])
        self.assertEqual(info["missing_cells"], 1)
        _, info = fixed_field(frame, 0, 325, 50, 5)
        self.assertEqual(info["reason"], "support_not_divisible")
        _, info = fixed_field(frame, 0, 150, 25, 5)
        self.assertEqual(info["reason"], "fewer_than_eight_spatial_bins")

    def test_temporal_binning_and_boundary_exclusion(self):
        raw = self.fixture()
        outside = pd.DataFrame(
            {"global_time": [-1, 180000], "local_y": [10, 10], "v_vel": [1e9, 1e9]}
        )
        expected, _ = fixed_field(raw, 0, 325, 25, 5)
        actual, _ = fixed_field(pd.concat([raw, outside]), 0, 325, 25, 5)
        np.testing.assert_array_equal(actual, expected)
        self.assertAlmostEqual(actual.iloc[0, 0], 20 + 0.02 + 0.1)

    def test_entire_trace_flag_is_explicit(self):
        frame = pd.DataFrame(
            {
                "vehicle_id": [1, 1, 1, 2, 2],
                "global_time": [0, 100, 200, 0, 100],
                "local_y": np.array([0, 1, 100, 0, 1]) / 0.3048,
                "v_vel": np.full(5, 10 / 0.3048),
            }
        )
        np.testing.assert_array_equal(flagged_traces(frame), [1])

    def test_centering_removes_stationary_profile(self):
        field = np.tile(np.arange(13), (36, 1))
        power, error = spectrum(field, 5, False)
        centered, _ = spectrum(field, 5, True)
        self.assertAlmostEqual(power[1:].sum(), np.var(np.arange(13)))
        self.assertLess(error, 1e-12)
        self.assertLess(centered.sum(), 1e-24)
