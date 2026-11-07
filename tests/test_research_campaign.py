"""Small correctness tests for the isolated research campaign."""

# ruff: noqa: E402
import gzip
import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

if any(importlib.util.find_spec(name) is None for name in ["matplotlib", "scipy", "sklearn"]):
    raise unittest.SkipTest("Research tests require uv sync --extra research")

CAMPAIGN = Path(__file__).resolve().parents[1] / "research/campaign"
sys.path.insert(0, str(CAMPAIGN))
import common
import fetch_corpus
from common import graph, permutation_within_activity
from h1 import baseline_features, local_predict, poisson_deviance
from h2 import heat_features, independent_three
from scipy import sparse
from sklearn.metrics import average_precision_score
from summarize import weighted_ap


class ResearchTests(unittest.TestCase):
    def test_failed_fetch_preserves_accepted_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / "data"
            dest.mkdir()
            accepted = dest / "robotics_works.jsonl.gz"
            accepted.write_bytes(b"accepted corpus")
            with (
                patch.object(fetch_corpus, "DEST", dest),
                patch.object(fetch_corpus, "ROOT", root),
                patch.object(fetch_corpus, "get_page", side_effect=RuntimeError("unavailable")),
            ):
                with self.assertRaises(RuntimeError):
                    fetch_corpus.main()
            self.assertEqual(accepted.read_bytes(), b"accepted corpus")

    def test_count_mismatch_preserves_accepted_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / "data"
            dest.mkdir()
            accepted = dest / "robotics_works.jsonl.gz"
            accepted.write_bytes(b"accepted corpus")
            with (
                patch.object(fetch_corpus, "DEST", dest),
                patch.object(fetch_corpus, "ROOT", root),
                patch.object(
                    fetch_corpus,
                    "get_page",
                    return_value={"meta": {"count": 10, "next_cursor": None}, "results": []},
                ),
                patch.object(fetch_corpus.time, "sleep"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Incomplete corpus"):
                    fetch_corpus.main()
            self.assertEqual(accepted.read_bytes(), b"accepted corpus")

    def test_deduplication_retains_earliest_date(self):
        title = "A sufficiently long and identical title describing robot manipulation research"
        template = dict(title=title, authors=["a"], doi=None, references=[], abstract="word " * 30)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with gzip.open(root / "robotics_works.jsonl.gz", "wt") as f:
                for identifier, date in [("later", "2015-01-01"), ("earlier", "2010-02-01")]:
                    f.write(json.dumps(dict(template, id=identifier, date=date)) + "\n")
            with patch.object(common, "DATA", root), patch.object(common, "OUT", root):
                kept = common.load_corpus("openalex")
            self.assertEqual([p["id"] for p in kept], ["earlier"])

    def test_weighted_average_precision_matches_sklearn_with_ties(self):
        y = np.array([1, 0, 0, 1, 1, 0])
        p = np.array([0.8, 0.8, 0.5, 0.5, 0.2, 0.1])
        w = np.array([2, 0, 1, 3, 1, 2])
        self.assertAlmostEqual(weighted_ap(y, p, w), average_precision_score(y, p, sample_weight=w))

    def test_disjoint_author_test_matches_brute_force(self):
        rng = np.random.default_rng(4)
        for _ in range(100):
            groups = [set(rng.choice(8, rng.integers(0, 5), replace=False)) for _ in range(10)]
            expected = any(
                a and b and c and a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
                for a, b, c in itertools.combinations(groups, 3)
            )
            self.assertEqual(independent_three(groups), expected)

    def test_empty_authors_cannot_establish_independence(self):
        self.assertFalse(independent_three([[], [], []]))
        self.assertFalse(independent_three([["a"], ["b"], ["a", "b"]]))
        self.assertTrue(independent_three([["a"], ["b"], ["c"]]))

    def test_local_prediction_ignores_future_targets(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(60, 4, 3))
        y = rng.normal(size=(60, 4))
        first = local_predict(x, y, np.arange(40), [40], 10)
        y[40:] = 10000
        self.assertTrue(np.allclose(first, local_predict(x, y, np.arange(40), [40], 10)))

    def test_local_features_are_past_only(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(60, 4))
        volume = rng.normal(size=60)
        first = baseline_features(x, volume, 3)[40].copy()
        x[41:] += 1000
        volume[41:] += 1000
        self.assertTrue(np.allclose(first, baseline_features(x, volume, 3)[40]))

    def test_graph_ignores_future_papers(self):
        x = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32))
        months = np.array([1, 2, 50])
        a = graph(x, months, 2, 3, k=1)
        x = sparse.vstack([x, sparse.csr_matrix([[0, 1, 1]])])
        b = graph(x, np.r_[months, 51], 2, 3, k=1)
        self.assertTrue(np.allclose(a, b))

    def test_activity_permutation_preserves_quartiles(self):
        counts = np.arange(40)
        perm = permutation_within_activity(counts, 9)
        self.assertEqual(set(perm), set(range(40)))
        self.assertTrue(np.array_equal(counts // 10, counts[perm] // 10))

    def test_heat_features_and_deviance(self):
        s = np.array([[0.0, 1.0], [1.0, 0.0]])
        features = heat_features(s, np.array([0]), np.array([1]))
        self.assertEqual(features.shape, (1, 6))
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(
            np.allclose(
                poisson_deviance(np.array([0.0, 1.0, 4.0]), np.array([1.0, 1.0, 4.0])), [2, 0, 0]
            )
        )


if __name__ == "__main__":
    unittest.main()
