"""Reproduce exploratory checks without changing the published report or its evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rigorous_topic_dynamics import make_panel, score_cutoff

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "evidence"
FEATURES = [
    "spectral_emergence_score",
    "spectral_midband_momentum",
    "raw_momentum_3y",
    "baseline_growth_3y",
]


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def filter_diagnostics(adjacency: np.ndarray) -> dict[str, object]:
    degrees = adjacency.sum(axis=1)
    inverse = np.zeros_like(degrees)
    inverse[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
    laplacian = np.eye(len(degrees)) - inverse[:, None] * adjacency * inverse[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
    pivot = float(np.median(eigenvalues[eigenvalues > 1e-8]))
    weights = {
        "local": eigenvalues / (eigenvalues + pivot + 1e-9),
        "midband": np.exp(-((eigenvalues - pivot) ** 2) / (2 * (pivot + 1e-9) ** 2)),
    }
    results: dict[str, object] = {}
    for name, weight in weights.items():
        operator = (eigenvectors * weight) @ eigenvectors.T
        scalar = float(np.trace(operator) / len(operator))
        identity = scalar * np.eye(len(operator))
        total_energy = float(np.sum(operator**2))
        identity_fraction = float(np.sum(identity**2) / total_energy)
        assert np.isclose(
            total_energy,
            np.sum(identity**2) + np.sum((operator - identity) ** 2),
        )
        results[name] = {
            "best_scalar_identity_coefficient": scalar,
            "scalar_identity_fraction_squared_frobenius_norm": identity_fraction,
            "weight_quantiles_0_05_50_95_100": np.quantile(weight, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        }
    return results


def main() -> None:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        source = ROOT / item["path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]

    topics = pd.read_csv(ROOT / "data/processed/openalex_ai_topics.csv")
    topics = topics.sort_values("topic_id").reset_index(drop=True)
    counts = pd.read_csv(ROOT / "data/processed/openalex_ai_topic_year_counts.csv")
    panel = make_panel(counts, topics["topic_id"].tolist(), "primary")
    clr = panel["clr"]
    rows = []
    holdout_details: dict[str, object] = {}
    filter_details: dict[str, object] = {}

    for cutoff in range(2005, 2023):
        scores, _, adjacency, _ = score_cutoff(panel, topics, cutoff)
        eligible = scores[scores["eligible"]].copy()
        future_component = clr.loc[cutoff + 1 : cutoff + 3].mean() - clr.loc[cutoff]
        known_component = clr.loc[cutoff] - clr.loc[cutoff - 2 : cutoff].mean()
        persistent = clr.loc[cutoff + 1 : cutoff + 3].mean() - clr.loc[cutoff - 2 : cutoff].mean()
        decomposition_error = float(np.max(np.abs(persistent - future_component - known_component)))
        assert decomposition_error < 1e-12
        mapped_future = eligible["topic_id"].map(future_component).to_numpy()
        mapped_known = eligible["topic_id"].map(known_component).to_numpy()
        mapped_persistent = eligible["topic_id"].map(persistent).to_numpy()
        assert np.allclose(mapped_persistent, eligible["persistent_shift"])
        for feature in FEATURES:
            values = eligible[feature].to_numpy()
            rows.append(
                {
                    "cutoff_year": cutoff,
                    "feature": feature,
                    "n_topics": len(eligible),
                    "persistent_shift_rho": rank_correlation(values, mapped_persistent),
                    "future_only_rho": rank_correlation(values, mapped_future),
                }
            )
        if cutoff == 2022:
            filter_details = filter_diagnostics(adjacency)
            holdout_details = {
                "n_topics": len(eligible),
                "known_component_vs_persistent_shift_rho": rank_correlation(mapped_known, mapped_persistent),
                "known_component_vs_future_only_rho": rank_correlation(mapped_known, mapped_future),
                "maximum_target_decomposition_error": decomposition_error,
            }

    comparisons = pd.DataFrame(rows)
    holdout = comparisons[comparisons["cutoff_year"] == 2022]
    summary = comparisons.groupby("feature", as_index=False)[
        ["persistent_shift_rho", "future_only_rho"]
    ].mean()
    published = pd.read_csv(ROOT / "reports/evidence/holdout_metrics_primary.csv").set_index("feature")
    for row in holdout.itertuples():
        assert np.isclose(row.persistent_shift_rho, published.loc[row.feature, "spearman"])

    result = {
        "review_date": "2026-10-14",
        "data_fetched_at": manifest["fetched_at"],
        "status": "exploratory diagnostic; no new untouched holdout",
        "filter_measure": "squared Frobenius projection onto span(identity), divided by operator norm squared",
        "filter_diagnostics": filter_details,
        "target_decomposition": holdout_details,
        "holdout_comparison": holdout.to_dict(orient="records"),
        "rolling_mean_comparison": summary.to_dict(orient="records"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(OUTPUT / "target_comparison.csv", index=False)
    (OUTPUT / "current_design.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(holdout.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Evidence written to {OUTPUT}")


if __name__ == "__main__":
    main()
