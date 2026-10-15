"""Correctness checks for the cross-community adoption pilot."""

# ruff: noqa: E402
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ["scipy", "sklearn", "networkx"]):
    raise unittest.SkipTest("Adoption tests require uv sync --extra research")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/2026-10/adoption"))
import adoption_features
from adoption_data import METHODS, ONTOLOGY, asserted_concepts, extract
from adoption_features import author_network, breadth, outcome, propagation, row_transition
from experiment import features
from summarize_adoption import weighted_ap
from version_audit import first_abstract


class AdoptionTests(unittest.TestCase):
    def paper(self, identifier, date, authors):
        return dict(
            id=identifier,
            date=date,
            day=pd.Timestamp(date),
            authors=authors,
            methods=[0],
            applications=[0],
        )

    def test_alias_boundaries_and_hyphens(self):
        m, a = extract("We use semi-supervised learning for image segmentation.", "medical")
        self.assertIn(METHODS.index("semi_supervised_learning"), m)
        self.assertIn(list(ONTOLOGY["medical"]).index("segmentation"), a)
        self.assertEqual(extract("CNN SVM unavailable", "medical"), ([], []))
        self.assertEqual(extract("reconstructionist", "medical"), ([], []))

    def test_background_comparison_and_negation(self):
        m, a, _ = asserted_concepts(
            "",
            "We propose convolutional neural networks for image segmentation. The results outperform support vector machines.",
            "medical",
        )
        self.assertIn(METHODS.index("convolutional_network"), m)
        self.assertNotIn(METHODS.index("support_vector_machine"), m)
        self.assertTrue(a)
        self.assertEqual(
            asserted_concepts(
                "",
                "Learning from demonstration is a common technique for reinforcement learning.",
                "robotics",
            )[:2],
            ([], []),
        )
        self.assertEqual(
            asserted_concepts("A survey of reinforcement learning", "", "robotics")[:2], ([], [])
        )
        self.assertEqual(
            asserted_concepts("", "We propose a replacement of the dexterous hand.", "robotics")[1],
            [],
        )

    def test_new_authors_and_time_bins(self):
        p = [
            self.paper("1", "2021-01-01", ["b"]),
            self.paper("2", "2021-04-01", ["c"]),
            self.paper("3", "2021-08-01", ["d"]),
        ]
        result = outcome(p, 0, 0, pd.Timestamp("2020-12-31"), {"s"})
        self.assertEqual(result["event"], 1)
        self.assertEqual(result["event_day"], "2021-08-01")
        self.assertEqual(outcome(p, 0, 0, pd.Timestamp("2020-12-31"), {"b"})["event"], 0)
        self.assertEqual(
            outcome(list(reversed(p)), 0, 0, pd.Timestamp("2020-12-31"), {"s"})["event_day"],
            result["event_day"],
        )

    def test_empty_shared_authors_and_censoring(self):
        p = [
            self.paper("1", "2021-01-01", ["b"]),
            self.paper("2", "2021-08-01", ["b"]),
            self.paper("3", "2023-01-01", ["c"]),
            self.paper("4", "2022-01-01", []),
        ]
        result = outcome(p, 0, 0, pd.Timestamp("2020-12-31"), set())
        self.assertEqual(result["event"], 0)
        self.assertEqual(result["papers"], 2)

    def test_large_consortium_does_not_prove_independence(self):
        p = [
            self.paper("1", "2021-01-01", [str(i) for i in range(51)]),
            self.paper("2", "2021-08-01", ["a"]),
            self.paper("3", "2021-09-01", ["b"]),
        ]
        self.assertEqual(outcome(p, 0, 0, pd.Timestamp("2020-12-31"), set())["event"], 0)
        _, _, paper_labels, audit = author_network(p[:1])
        self.assertEqual(paper_labels.tolist(), [-1])
        self.assertEqual(audit["excluded_large_authorship_papers"], 1)

    def test_citation_direction(self):
        flow = sparse.csr_matrix([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        q, r = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        self.assertEqual(propagation(q, row_transition(flow), r), [0.0, 1.0, 0.0])
        self.assertEqual(propagation(q, row_transition(flow.T), r), [0.0, 0.0, 0.0])

    def test_breadth_and_determinism(self):
        values = breadth({"a", "b"}, {"a": 0, "b": 1})
        self.assertTrue(np.allclose(values, [np.log(3), 1, 0.5]))
        p = [self.paper("1", "2020-01-01", ["c", "a"]), self.paper("2", "2020-02-01", ["a", "b"])]
        first = author_network(p)
        second = author_network(p)
        self.assertEqual(first[1], second[1])
        self.assertTrue(np.array_equal(first[2], second[2]))

    def test_features_ignore_outcome_labels(self):
        row = dict(
            base=[0] * 15,
            static=[1] * 4,
            exposure=[2] * 4,
            breadth=[3] * 3,
            undirected=[4] * 3,
            directed=[5] * 3,
            reverse=[6] * 3,
            null=[[7] * 3] * 10,
            label=object(),
        )
        self.assertEqual(features([row], "directed").shape, (1, 32))
        row["label"] = {"event": 100000}
        self.assertEqual(
            features([row], "null_0").tolist()[0][:23], features([row], "exposure").tolist()[0]
        )

    def test_identity_weighted_ap_handles_ties_and_empty_events(self):
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([0.8, 0.8, 0.5, 0.2, 0.2])
        w = np.array([1, 2, 0, 3, 1])
        self.assertAlmostEqual(weighted_ap(y, p, w), average_precision_score(y, p, sample_weight=w))
        self.assertTrue(np.isnan(weighted_ap(np.zeros(5), p, w)))

    def test_pdf_abstract_transport_is_normalized(self):
        body = "We propose a convolutional neural net-\nwork for image segmentation. "
        body += "The method uses a carefully evaluated collection of medical images. " * 3
        abstract = first_abstract("Abstract\n" + body + "\n1 Introduction\nOther text.")
        self.assertIn("neural network", abstract)
        self.assertNotIn("\n", abstract)
        self.assertNotIn("Other text", abstract)
        self.assertIsNone(first_abstract("Abstract\nA tiny fragment\n1 Introduction"))

    def test_snapshot_features_invariant_to_future_records(self):
        past = []
        for i in range(41):
            paper = self.paper(str(i), "2020-01-01", [f"a{i}"])
            paper.update(
                abstract="medical learning segmentation " + ("brain" if i % 2 else "image"),
                methods=[0] if i <= 20 else [],
                applications=[0] if i >= 20 else [],
                references=["future_anchor"] if i == 25 else [],
            )
            past.append(paper)
        future = [
            self.paper("f1", "2021-02-01", ["new1"]),
            self.paper("f2", "2021-08-01", ["new2"]),
            self.paper("f3", "2022-01-01", ["new3"]),
        ]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "medical_prepared_meta.json").write_text("{}")
            anchors = {method: [] for method in METHODS}
            anchors[METHODS[0]] = [
                dict(id="https://openalex.org/future_anchor", publication_date="2021-01-01")
            ]
            (folder / "anchors.json").write_text(json.dumps(dict(methods=anchors)))
            with (
                patch.object(adoption_features, "DATA", folder),
                patch.object(adoption_features, "OUT", folder),
            ):
                first = adoption_features.snapshot(past, "medical", "2020-12-31")
                (folder / "medical_2020-12-31_snapshot.json").unlink()
                second = adoption_features.snapshot(past + future, "medical", "2020-12-31")
        self.assertEqual(first["audit"], second["audit"])
        self.assertEqual(second["audit"]["future_dated_references_removed"], 1)
        for a, b in zip(first["rows"], second["rows"]):
            for stage in ["temporal", "static", "exposure", "breadth", "directed", "null_0"]:
                np.testing.assert_array_equal(features([a], stage), features([b], stage))
        self.assertEqual(first["rows"][0]["label"]["event"], 0)
        self.assertEqual(second["rows"][0]["label"]["event"], 1)


if __name__ == "__main__":
    unittest.main()
