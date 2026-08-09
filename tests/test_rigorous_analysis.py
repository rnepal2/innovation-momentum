import hashlib
import json

import numpy as np
import pandas as pd

from analyze_agentic_and_visuals import summarize_agentic_terms
from paths import project_root
from rigorous_topic_dynamics import centered_log_ratio, persistent_shift, validate_inputs

ROOT = project_root()


def test_centered_log_ratio_is_centered_by_year():
    counts = pd.DataFrame(
        [[0, 10, 20], [5, 5, 90]],
        index=[2021, 2022],
        columns=["a", "b", "c"],
    )
    transformed = centered_log_ratio(counts)
    assert np.allclose(transformed.mean(axis=1), 0.0, atol=1e-12)
    assert np.isfinite(transformed.to_numpy()).all()


def test_persistent_shift_uses_three_year_means():
    signal = pd.DataFrame({"topic": np.arange(7, dtype=float)}, index=range(2019, 2026))
    result = persistent_shift(signal, 2022)
    assert result["topic"] == 3.0


def test_committed_topic_panel_closes_and_has_expected_shape():
    topics = pd.read_csv(ROOT / "data/processed/openalex_ai_topics.csv")
    counts = pd.read_csv(ROOT / "data/processed/openalex_ai_topic_year_counts.csv")
    validate_inputs(topics, counts)
    assert topics["topic_id"].nunique() == 77
    assert len(counts) == 77 * 37


def test_manifest_matches_committed_inputs():
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete_evaluation_end_year"] == 2025
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_agentic_family_index_removes_exact_duplicate_series():
    counts = pd.read_csv(ROOT / "data/processed/openalex_agentic_precursor_counts.csv")
    term_summary, _, _ = summarize_agentic_terms(counts)
    audit = term_summary.set_index("term_id")
    assert audit.loc["ai_agents", "duplicate_of"] == "ai_agent"
    assert not bool(audit.loc["ai_agents", "included_in_family_index"])
    assert bool(audit.loc["ai_agent", "included_in_family_index"])
