"""Historical receipts remain readable after moving the traffic study."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA, HERE, ROOT, recorded_path


class ProjectPathsTests(unittest.TestCase):
    def test_historical_and_current_paths_resolve_to_same_file(self):
        for old, current in [
            ("data/research_2026/traffic/sumo/run.npz", DATA / "sumo/run.npz"),
            ("data/research_2026/traffic_novelty/paper.pdf", DATA / "literature/paper.pdf"),
            ("research/2026-10/traffic/results/harmonics", HERE / "results/harmonics"),
            ("research/2026-10/campaign/results", ROOT / "research/campaign/results"),
            (
                "artifacts/traffic_noise_resolution_repeat/summary.csv",
                HERE / "artifacts/noise_resolution_repeat/summary.csv",
            ),
            (
                "artifacts/traffic_manuscript_layout/validation.json",
                HERE / "artifacts/manuscript_layout/validation.json",
            ),
        ]:
            with self.subTest(old=old):
                self.assertEqual(recorded_path(old), current)
                self.assertEqual(recorded_path(ROOT / old), current)
                self.assertEqual(recorded_path(current.relative_to(ROOT)), current)

    def test_unrelated_paths_are_preserved(self):
        self.assertEqual(
            recorded_path("data/processed/panel.csv"), ROOT / "data/processed/panel.csv"
        )
        self.assertEqual(
            recorded_path("data/research_2026/traffic_other"),
            ROOT / "data/research_2026/traffic_other",
        )
        self.assertEqual(recorded_path("/tmp/independent-run"), Path("/tmp/independent-run"))

    def test_archived_sources_still_match_their_freezes(self):
        checked = 0
        for path in (HERE / "results").rglob("freeze.json"):
            freeze = json.loads(path.read_text())
            sources = freeze.get("archived_sources_sha256", freeze.get("sources", {}))
            for name, expected in sources.items():
                archived = path.parent / "frozen_sources" / (name + ".txt")
                if not (path.parent / "frozen_sources").exists():
                    continue
                with self.subTest(archive=archived.relative_to(HERE)):
                    actual = hashlib.sha256(archived.read_bytes()).hexdigest()
                    self.assertEqual(actual, expected)
                    checked += 1
        self.assertGreater(checked, 800)


if __name__ == "__main__":
    unittest.main()
