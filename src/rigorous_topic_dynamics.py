#!/usr/bin/env python3
"""Build the composition-aware OpenAlex topic analysis and published article."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paths import project_root
from spectral import graph_fourier_basis, spearman_corr, spectral_node_scores, zscore

ROOT = project_root()
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
EVIDENCE_DIR = REPORT_DIR / "evidence"
ARTIFACT_DIR = ROOT / "artifacts" / "report"
FIGURE_DIR = ARTIFACT_DIR / "figures"

YEAR_START = 1990
CUTOFF_YEAR = 2022
COMPLETE_END_YEAR = 2025
PANEL_END_YEAR = 2026
ROLLING_START_YEAR = 2005
MIN_CUTOFF_COUNT = 250
PSEUDOCOUNT = 0.5
RANDOM_SEED = 20260619
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 20_000
N_GRAPH_NULL = 1_000

INK = "#1f2933"
MUTED = "#667085"
GRID = "#d9dee8"
BLUE = "#3b6ea8"
TEAL = "#2a9d8f"
RUST = "#b25d31"
VIOLET = "#7a5195"
GOLD = "#d18f1f"

FEATURES = [
    "spectral_emergence_score",
    "spectral_midband_momentum",
    "spectral_local_momentum",
    "raw_momentum_3y",
    "baseline_growth_3y",
    "raw_acceleration_1y",
    "cutoff_log_count",
]

FEATURE_LABELS = {
    "spectral_emergence_score": "Fixed spectral composite",
    "spectral_midband_momentum": "Midband spectral momentum",
    "spectral_local_momentum": "Local spectral momentum",
    "raw_momentum_3y": "Standardized 3-year momentum",
    "baseline_growth_3y": "CLR growth baseline",
    "raw_acceleration_1y": "1-year acceleration",
    "cutoff_log_count": "Topic size at cutoff",
}

SCOPE_CONFIG = {
    "primary": {
        "label": "Primary topic",
        "count_col": "primary_topic_count",
        "share_col": "primary_share_per_million_ai",
    },
    "any_topic": {
        "label": "Any topic assignment",
        "count_col": "ai_subfield_any_topic_count",
        "share_col": "ai_any_share_per_million_ai",
    },
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def tooltip_attr(value: object) -> str:
    return esc(value).replace("\n", "&#10;")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def scale(value: float, domain: tuple[float, float], target: tuple[float, float]) -> float:
    lo, hi = domain
    a, b = target
    if hi == lo:
        return (a + b) / 2
    return a + (value - lo) / (hi - lo) * (b - a)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def centered_log_ratio(counts: pd.DataFrame, pseudocount: float = PSEUDOCOUNT) -> pd.DataFrame:
    """Apply an additive-smoothed centered log-ratio transform by year."""
    logged = np.log(counts.astype(float) + pseudocount)
    return logged.sub(logged.mean(axis=1), axis=0)


def zscore_by_training_window(signal: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    train = signal.loc[signal.index <= cutoff]
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0, np.nan).fillna(1.0)
    return signal.sub(mean, axis=1).div(std, axis=1)


def correlation_graph(train_signal: pd.DataFrame, top_k: int = 7) -> np.ndarray:
    values = train_signal.to_numpy(dtype=float).T
    corr = np.nan_to_num(np.corrcoef(values), nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)
    adjacency = np.zeros_like(corr)
    for i in range(len(corr)):
        neighbors = np.argsort(corr[i])[-top_k:]
        adjacency[i, neighbors] = corr[i, neighbors]
    return np.maximum(adjacency, adjacency.T)


def _tokens(value: object) -> set[str]:
    stop = {
        "and", "artificial", "based", "data", "intelligence", "learning", "machine",
        "method", "methods", "model", "models", "research", "system", "systems", "using",
    }
    words = re.findall(r"[a-z0-9][a-z0-9-]{2,}", str(value).lower())
    return {word for word in words if word not in stop}


def semantic_similarity(topics: pd.DataFrame, topic_ids: list[str]) -> np.ndarray:
    meta = topics.set_index("topic_id").loc[topic_ids]
    token_sets = [
        _tokens(f"{row.display_name} {row.description} {row.keywords}")
        for row in meta.itertuples()
    ]
    matrix = np.zeros((len(token_sets), len(token_sets)), dtype=float)
    for i, left in enumerate(token_sets):
        for j in range(i + 1, len(token_sets)):
            right = token_sets[j]
            union = left | right
            value = len(left & right) / len(union) if union else 0.0
            matrix[i, j] = matrix[j, i] = value
    return matrix


def mixed_graph(
    train_signal: pd.DataFrame,
    semantic: np.ndarray,
    top_k: int,
    correlation_weight: float,
) -> np.ndarray:
    values = train_signal.to_numpy(dtype=float).T
    corr = np.nan_to_num(np.corrcoef(values), nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)
    base = correlation_weight * corr + (1.0 - correlation_weight) * semantic
    np.fill_diagonal(base, 0.0)
    adjacency = np.zeros_like(base)
    for i in range(len(base)):
        neighbors = np.argsort(base[i])[-top_k:]
        adjacency[i, neighbors] = base[i, neighbors]
    return np.maximum(adjacency, adjacency.T)


def make_panel(counts: pd.DataFrame, topic_ids: list[str], scope_id: str) -> dict[str, pd.DataFrame]:
    cfg = SCOPE_CONFIG[scope_id]
    years = list(range(YEAR_START, PANEL_END_YEAR + 1))
    wide_count = (
        counts.pivot(index="year", columns="topic_id", values=cfg["count_col"])
        .reindex(index=years, columns=topic_ids)
        .fillna(0.0)
    )
    wide_share = (
        counts.pivot(index="year", columns="topic_id", values=cfg["share_col"])
        .reindex(index=years, columns=topic_ids)
        .fillna(0.0)
    )
    return {
        "counts": wide_count,
        "shares_per_million": wide_share,
        "clr": centered_log_ratio(wide_count),
    }


def persistent_shift(signal: pd.DataFrame, cutoff: int) -> pd.Series:
    future = signal.loc[cutoff + 1 : cutoff + 3].mean(axis=0)
    recent = signal.loc[cutoff - 2 : cutoff].mean(axis=0)
    return future - recent


def feature_metrics(scores: pd.DataFrame, target_col: str = "persistent_shift") -> pd.DataFrame:
    eligible = scores[scores["eligible"]].copy()
    target = eligible[target_col].to_numpy(dtype=float)
    top_n = min(10, len(eligible))
    actual_top = set(eligible.nlargest(top_n, target_col)["topic_id"])
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        ranked = eligible.sort_values(feature, ascending=False)["topic_id"].tolist()
        hits = sum(topic in actual_top for topic in ranked[:top_n])
        seen = 0
        precision_sum = 0.0
        for rank, topic in enumerate(ranked, start=1):
            if topic in actual_top:
                seen += 1
                precision_sum += seen / rank
        rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "spearman": spearman_corr(eligible[feature].to_numpy(), target),
                "top10_hits": hits,
                "average_precision": precision_sum / top_n,
                "n_topics": len(eligible),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def score_cutoff(
    panel: dict[str, pd.DataFrame],
    topics: pd.DataFrame,
    cutoff: int,
    top_k: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    clr = panel["clr"]
    wide_count = panel["counts"]
    normalized = zscore_by_training_window(clr, cutoff)
    adjacency = correlation_graph(normalized.loc[:cutoff], top_k=top_k)
    scores, _ = spectral_node_scores(normalized, adjacency, cutoff)
    target = persistent_shift(clr, cutoff)
    baseline = (clr.loc[cutoff] - clr.loc[cutoff - 3]) / 3.0
    labels = topics.set_index("topic_id")["display_name"]
    scores["label"] = scores["topic_id"].map(labels)
    scores["persistent_shift"] = scores["topic_id"].map(target)
    scores["baseline_growth_3y"] = scores["topic_id"].map(baseline)
    scores["cutoff_count"] = scores["topic_id"].map(wide_count.loc[cutoff]).astype(int)
    scores["cutoff_log_count"] = np.log1p(scores["cutoff_count"])
    scores["eligible"] = scores["cutoff_count"] >= MIN_CUTOFF_COUNT
    scores["cutoff_year"] = cutoff
    scores["eval_end_year"] = cutoff + 3
    metrics = feature_metrics(scores)
    metrics["cutoff_year"] = cutoff
    metrics["eval_end_year"] = cutoff + 3
    return scores, metrics, adjacency, normalized


def run_scope(
    counts: pd.DataFrame,
    topics: pd.DataFrame,
    scope_id: str,
) -> dict[str, object]:
    topic_ids = topics["topic_id"].tolist()
    panel = make_panel(counts, topic_ids, scope_id)
    scores, metrics, adjacency, normalized = score_cutoff(panel, topics, CUTOFF_YEAR)
    metrics["scope"] = scope_id
    metrics["scope_label"] = SCOPE_CONFIG[scope_id]["label"]
    rolling_metrics: list[pd.DataFrame] = []
    rolling_scores: list[pd.DataFrame] = []
    for cutoff in range(ROLLING_START_YEAR, CUTOFF_YEAR + 1):
        cutoff_scores, cutoff_metrics, _, _ = score_cutoff(panel, topics, cutoff)
        cutoff_metrics["scope"] = scope_id
        cutoff_metrics["scope_label"] = SCOPE_CONFIG[scope_id]["label"]
        cutoff_scores["scope"] = scope_id
        rolling_metrics.append(cutoff_metrics)
        rolling_scores.append(cutoff_scores)
    return {
        "panel": panel,
        "scores": scores,
        "metrics": metrics,
        "adjacency": adjacency,
        "normalized": normalized,
        "rolling": pd.concat(rolling_metrics, ignore_index=True),
        "rolling_scores": pd.concat(rolling_scores, ignore_index=True),
    }


def _rank_residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    ranked = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    ranked_controls = np.column_stack(
        [pd.Series(controls[:, i]).rank(method="average").to_numpy(dtype=float) for i in range(controls.shape[1])]
    )
    design = np.column_stack([np.ones(len(ranked)), ranked_controls])
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    left = _rank_residual(np.asarray(x, dtype=float), np.asarray(controls, dtype=float))
    right = _rank_residual(np.asarray(y, dtype=float), np.asarray(controls, dtype=float))
    return float(np.corrcoef(left, right)[0, 1])


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * float(pvalues[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def infer_holdout(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = scores[scores["eligible"]].reset_index(drop=True)
    target = data["persistent_shift"].to_numpy(dtype=float)
    observed = np.array([spearman_corr(data[f].to_numpy(), target) for f in FEATURES])
    rng = np.random.default_rng(RANDOM_SEED)
    boot = np.empty((N_BOOTSTRAP, len(FEATURES)), dtype=float)
    n = len(data)
    arrays = [data[f].to_numpy(dtype=float) for f in FEATURES]
    for b in range(N_BOOTSTRAP):
        sample = rng.integers(0, n, size=n)
        sampled_target = target[sample]
        boot[b] = [spearman_corr(values[sample], sampled_target) for values in arrays]

    target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=float)
    target_rank = (target_rank - target_rank.mean()) / target_rank.std()
    feature_ranks = np.column_stack(
        [pd.Series(values).rank(method="average").to_numpy(dtype=float) for values in arrays]
    )
    feature_ranks = (feature_ranks - feature_ranks.mean(axis=0)) / feature_ranks.std(axis=0)
    extreme = np.zeros(len(FEATURES), dtype=int)
    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(target_rank)
        permuted_rho = permuted @ feature_ranks / n
        extreme += np.abs(permuted_rho) >= np.abs(observed)
    pvalues = (extreme + 1) / (N_PERMUTATIONS + 1)
    adjusted = holm_adjust(pvalues)
    inference = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_label": [FEATURE_LABELS[f] for f in FEATURES],
            "spearman": observed,
            "bootstrap_low": np.quantile(boot, 0.025, axis=0),
            "bootstrap_high": np.quantile(boot, 0.975, axis=0),
            "permutation_p": pvalues,
            "holm_p": adjusted,
            "n_topics": n,
        }
    ).sort_values("spearman", ascending=False)

    pairs = [
        ("spectral_emergence_score", "baseline_growth_3y"),
        ("spectral_emergence_score", "raw_momentum_3y"),
        ("spectral_midband_momentum", "raw_momentum_3y"),
    ]
    pair_rows = []
    feature_index = {feature: index for index, feature in enumerate(FEATURES)}
    for left, right in pairs:
        delta = boot[:, feature_index[left]] - boot[:, feature_index[right]]
        pair_rows.append(
            {
                "left_feature": left,
                "right_feature": right,
                "left_label": FEATURE_LABELS[left],
                "right_label": FEATURE_LABELS[right],
                "observed_delta": observed[feature_index[left]] - observed[feature_index[right]],
                "bootstrap_low": float(np.quantile(delta, 0.025)),
                "bootstrap_high": float(np.quantile(delta, 0.975)),
            }
        )
    return inference, pd.DataFrame(pair_rows)


def endpoint_sensitivity(scores: pd.DataFrame, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eligible = scores[scores["eligible"]].copy()
    clr = panel["clr"]
    rows = []
    endpoint_features = [
        "spectral_emergence_score",
        "spectral_midband_momentum",
        "raw_momentum_3y",
        "baseline_growth_3y",
        "raw_acceleration_1y",
    ]
    targets: list[tuple[str, pd.Series, bool]] = []
    for year in range(2023, PANEL_END_YEAR + 1):
        targets.append(("Jun 2026" if year == 2026 else str(year), clr.loc[year] - clr.loc[CUTOFF_YEAR], year < 2026))
    targets.append(("2023-2025 mean", persistent_shift(clr, CUTOFF_YEAR), True))
    for label, target, complete in targets:
        mapped = eligible["topic_id"].map(target).to_numpy(dtype=float)
        for feature in endpoint_features:
            rows.append(
                {
                    "endpoint": label,
                    "complete_data": complete,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "spearman": spearman_corr(eligible[feature].to_numpy(), mapped),
                    "n_topics": len(eligible),
                }
            )
    return pd.DataFrame(rows)


def volume_sensitivity(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0, 100, 250, 500, 1000]:
        data = scores[scores["cutoff_count"] >= threshold]
        for feature in ["spectral_emergence_score", "raw_momentum_3y", "baseline_growth_3y"]:
            rows.append(
                {
                    "minimum_2022_count": threshold,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "spearman": spearman_corr(data[feature].to_numpy(), data["persistent_shift"].to_numpy()),
                    "n_topics": len(data),
                }
            )
    return pd.DataFrame(rows)


def graph_ablation(
    panel: dict[str, pd.DataFrame],
    topics: pd.DataFrame,
    base_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clr = panel["clr"]
    normalized = zscore_by_training_window(clr, CUTOFF_YEAR)
    train = normalized.loc[:CUTOFF_YEAR]
    target = persistent_shift(clr, CUTOFF_YEAR)
    eligible_ids = set(base_scores.loc[base_scores["eligible"], "topic_id"])
    semantic = semantic_similarity(topics, train.columns.tolist())
    variants = [
        ("Correlation k=3", 3, 1.0),
        ("Correlation k=5", 5, 1.0),
        ("Correlation k=7", 7, 1.0),
        ("Correlation k=10", 10, 1.0),
        ("Correlation k=15", 15, 1.0),
        ("Hybrid k=5", 5, 0.76),
        ("Hybrid k=7", 7, 0.76),
        ("Hybrid k=10", 10, 0.76),
        ("Semantic k=7", 7, 0.0),
    ]
    rows = []
    for label, top_k, corr_weight in variants:
        adjacency = mixed_graph(train, semantic, top_k, corr_weight)
        spectral_scores, _ = spectral_node_scores(normalized, adjacency, CUTOFF_YEAR)
        data = spectral_scores[spectral_scores["topic_id"].isin(eligible_ids)].copy()
        mapped_target = data["topic_id"].map(target).to_numpy(dtype=float)
        rows.append(
            {
                "variant": label,
                "top_k": top_k,
                "correlation_weight": corr_weight,
                "uses_current_metadata": corr_weight < 1.0,
                "spearman": spearman_corr(data["spectral_emergence_score"].to_numpy(), mapped_target),
                "midband_spearman": spearman_corr(data["spectral_midband_momentum"].to_numpy(), mapped_target),
                "edges": int(np.count_nonzero(adjacency) // 2),
            }
        )

    data = base_scores[base_scores["eligible"]].copy()
    target_values = data["persistent_shift"].to_numpy(dtype=float)
    full_ids = base_scores["topic_id"].tolist()
    eligible_positions = [full_ids.index(topic_id) for topic_id in data["topic_id"]]
    adjacency = correlation_graph(train, top_k=7)
    rng = np.random.default_rng(RANDOM_SEED + 7)
    null_rhos = []
    for _ in range(N_GRAPH_NULL):
        order = rng.permutation(len(full_ids))
        randomized = adjacency[np.ix_(order, order)]
        randomized_scores, _ = spectral_node_scores(normalized, randomized, CUTOFF_YEAR)
        values = randomized_scores.iloc[eligible_positions]["spectral_emergence_score"].to_numpy(dtype=float)
        null_rhos.append(spearman_corr(values, target_values))
    observed = float(
        pd.DataFrame(rows).loc[lambda frame: frame["variant"] == "Correlation k=7", "spearman"].iloc[0]
    )
    null = pd.DataFrame({"null_spearman": null_rhos})
    null["observed_spearman"] = observed
    null["upper_tail_p"] = (sum(value >= observed for value in null_rhos) + 1) / (len(null_rhos) + 1)
    return pd.DataFrame(rows), null


def nested_model_selection(
    panel: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Select graph density and score weights using only outcomes available at each cutoff."""
    weight_sets = {
        "fixed_50_30_20": (0.50, 0.30, 0.20),
        "spectral_equal": (0.50, 0.50, 0.00),
        "equal_with_accel": (1 / 3, 1 / 3, 1 / 3),
        "local_only": (1.00, 0.00, 0.00),
        "midband_only": (0.00, 1.00, 0.00),
    }
    history_rows = []
    clr = panel["clr"]
    counts = panel["counts"]
    for cutoff in range(ROLLING_START_YEAR, CUTOFF_YEAR + 1):
        normalized = zscore_by_training_window(clr, cutoff)
        target = persistent_shift(clr, cutoff)
        eligible = counts.loc[cutoff] >= MIN_CUTOFF_COUNT
        for top_k in [3, 5, 7, 10, 15]:
            adjacency = correlation_graph(normalized.loc[:cutoff], top_k=top_k)
            component_scores, _ = spectral_node_scores(normalized, adjacency, cutoff)
            topic_ids = component_scores["topic_id"]
            mask = topic_ids.map(eligible).to_numpy(dtype=bool)
            outcome = topic_ids.map(target).to_numpy(dtype=float)[mask]
            local = zscore(component_scores["spectral_local_momentum"].to_numpy(dtype=float))
            midband = zscore(component_scores["spectral_midband_momentum"].to_numpy(dtype=float))
            accel = zscore(component_scores["raw_acceleration_1y"].to_numpy(dtype=float))
            for weight_name, (local_w, mid_w, accel_w) in weight_sets.items():
                candidate = local_w * local + mid_w * midband + accel_w * accel
                history_rows.append(
                    {
                        "cutoff_year": cutoff,
                        "eval_end_year": cutoff + 3,
                        "top_k": top_k,
                        "weight_set": weight_name,
                        "candidate": f"k{top_k}_{weight_name}",
                        "spearman": spearman_corr(candidate[mask], outcome),
                        "n_topics": int(mask.sum()),
                    }
                )
    history = pd.DataFrame(history_rows)
    baseline_by_cutoff = {}
    for cutoff in range(ROLLING_START_YEAR, CUTOFF_YEAR + 1):
        target = persistent_shift(clr, cutoff)
        baseline = (clr.loc[cutoff] - clr.loc[cutoff - 3]) / 3.0
        eligible = counts.loc[cutoff] >= MIN_CUTOFF_COUNT
        baseline_by_cutoff[cutoff] = spearman_corr(
            baseline[eligible].to_numpy(dtype=float), target[eligible].to_numpy(dtype=float)
        )

    selected_rows = []
    for cutoff in range(ROLLING_START_YEAR, CUTOFF_YEAR + 1):
        prior = history[history["eval_end_year"] <= cutoff]
        prior_cutoffs = prior["cutoff_year"].nunique()
        if prior_cutoffs < 3:
            continue
        candidate_means = prior.groupby("candidate")["spearman"].mean().sort_values(ascending=False)
        selected = str(candidate_means.index[0])
        test_row = history[(history["cutoff_year"] == cutoff) & (history["candidate"] == selected)].iloc[0]
        selected_rows.append(
            {
                "cutoff_year": cutoff,
                "eval_end_year": cutoff + 3,
                "selected_candidate": selected,
                "training_cutoffs": prior_cutoffs,
                "training_mean_spearman": float(candidate_means.iloc[0]),
                "selected_spearman": float(test_row["spearman"]),
                "baseline_spearman": baseline_by_cutoff[cutoff],
                "delta_vs_baseline": float(test_row["spearman"] - baseline_by_cutoff[cutoff]),
                "n_topics": int(test_row["n_topics"]),
            }
        )
    selected = pd.DataFrame(selected_rows)
    low, high = moving_block_interval(selected["delta_vs_baseline"].to_numpy())
    diagnostics = {
        "mean_selected": float(selected["selected_spearman"].mean()),
        "mean_baseline": float(selected["baseline_spearman"].mean()),
        "mean_delta": float(selected["delta_vs_baseline"].mean()),
        "block_low": low,
        "block_high": high,
        "wins": int((selected["delta_vs_baseline"] > 0).sum()),
        "n_cutoffs": len(selected),
    }
    return history, selected, diagnostics


def moving_block_interval(values: np.ndarray, block_length: int = 3, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED + 19)
    values = np.asarray(values, dtype=float)
    n = len(values)
    starts = np.arange(n - block_length + 1)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.choice(starts))
            sampled.extend(values[start : start + block_length])
        means[i] = np.mean(sampled[:n])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def rolling_summary(rolling: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    summary = (
        rolling.groupby(["feature", "feature_label"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            mean_top10_hits=("top10_hits", "mean"),
            mean_n_topics=("n_topics", "mean"),
        )
        .sort_values("mean_spearman", ascending=False)
    )
    pivot = rolling.pivot(index="cutoff_year", columns="feature", values="spearman")
    delta = pivot["spectral_emergence_score"] - pivot["baseline_growth_3y"]
    low, high = moving_block_interval(delta.to_numpy())
    diagnostics = {
        "mean_delta": float(delta.mean()),
        "block_low": low,
        "block_high": high,
        "spectral_wins": int((delta > 0).sum()),
        "n_cutoffs": len(delta),
    }
    return summary, diagnostics


def spectral_energy(normalized: pd.DataFrame, adjacency: np.ndarray) -> pd.DataFrame:
    eigvals, eigvecs = graph_fourier_basis(adjacency)
    low_cut = np.quantile(eigvals, 0.33)
    high_cut = np.quantile(eigvals, 0.66)
    rows = []
    for year in normalized.index:
        coeff = eigvecs.T @ normalized.loc[year].to_numpy(dtype=float)
        energy = coeff**2
        total = float(energy.sum())
        rows.append(
            {
                "year": year,
                "low_share": float(energy[eigvals <= low_cut].sum() / total),
                "mid_share": float(energy[(eigvals > low_cut) & (eigvals <= high_cut)].sum() / total),
                "high_share": float(energy[eigvals > high_cut].sum() / total),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(counts: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        counts.groupby("year", as_index=False)
        .agg(
            primary_assignments=("primary_topic_count", "sum"),
            any_topic_assignments=("ai_subfield_any_topic_count", "sum"),
            ai_works=("total_ai_subfield_count", "max"),
        )
        .sort_values("year")
    )
    coverage["primary_per_work"] = coverage["primary_assignments"] / coverage["ai_works"]
    coverage["any_topic_assignments_per_work"] = coverage["any_topic_assignments"] / coverage["ai_works"]
    return coverage


def top_shift_table(scores: pd.DataFrame, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    data = scores[scores["eligible"]].nlargest(12, "persistent_shift").copy()
    shares = panel["shares_per_million"]
    counts = panel["counts"]
    data["count_2025"] = data["topic_id"].map(counts.loc[COMPLETE_END_YEAR]).astype(int)
    data["share_per_million_2022"] = data["topic_id"].map(shares.loc[CUTOFF_YEAR])
    data["share_per_million_2025"] = data["topic_id"].map(shares.loc[COMPLETE_END_YEAR])
    data["share_multiple_2025_vs_2022"] = (
        (data["share_per_million_2025"] + 0.1) / (data["share_per_million_2022"] + 0.1)
    )
    data["spectral_rank"] = data["spectral_emergence_score"].rank(ascending=False, method="min")
    return data[
        [
            "topic_id", "label", "cutoff_count", "count_2025", "persistent_shift",
            "share_per_million_2022", "share_per_million_2025", "share_multiple_2025_vs_2022",
            "spectral_emergence_score", "spectral_rank",
        ]
    ]


def random_top10_probability(n_topics: int, hits: int, top_n: int = 10) -> float:
    denominator = math.comb(n_topics, top_n)
    return sum(
        math.comb(top_n, k) * math.comb(n_topics - top_n, top_n - k)
        for k in range(hits, top_n + 1)
        if top_n - k <= n_topics - top_n
    ) / denominator


def write_manifest() -> dict[str, object]:
    relative_paths = [
        "data/raw/openalex_ai_topics_raw.json",
        "data/raw/openalex_agentic_precursor_counts_raw.json",
        "data/processed/openalex_ai_topics.csv",
        "data/processed/openalex_ai_topic_year_counts.csv",
        "data/processed/openalex_agentic_precursor_counts.csv",
    ]
    files = []
    for relative in relative_paths:
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    counts = pd.read_csv(ROOT / "data/processed/openalex_ai_topic_year_counts.csv", nrows=1)
    manifest = {
        "snapshot": "OpenAlex aggregate export used by the published analysis",
        "fetched_at": str(counts.loc[0, "fetched_at"]),
        "analysis_cutoff_year": CUTOFF_YEAR,
        "complete_evaluation_end_year": COMPLETE_END_YEAR,
        "partial_panel_end": "June 2026",
        "files": files,
    }
    write_text(DATA_DIR / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        "<style>.p{cursor:crosshair}.p:hover{stroke:#111827;stroke-width:2.5}.s:hover{stroke-width:4}</style>",
    ]


def draw_uncertainty(inference: pd.DataFrame, path: Path) -> None:
    data = inference.sort_values("spearman", ascending=True).reset_index(drop=True)
    width, height = 820, 470
    left, right, top, bottom = 250, 45, 30, 58
    plot_w = width - left - right
    row_h = (height - top - bottom) / len(data)
    xmin = min(-0.35, float(data["bootstrap_low"].min()) - 0.05)
    xmax = max(0.75, float(data["bootstrap_high"].max()) + 0.05)
    sx = lambda value: scale(value, (xmin, xmax), (left, left + plot_w))
    lines = _svg_start(width, height)
    for tick in np.arange(-0.2, 0.81, 0.2):
        if xmin <= tick <= xmax:
            x = sx(float(tick))
            color = "#98a2b3" if abs(tick) < 1e-9 else GRID
            lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height-bottom}" stroke="{color}"/>')
            lines.append(f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="11" fill="{MUTED}">{tick:.1f}</text>')
    for i, row in enumerate(data.itertuples()):
        y = top + row_h * (i + 0.5)
        color = RUST if row.feature == "spectral_emergence_score" else TEAL if row.feature == "baseline_growth_3y" else BLUE
        tooltip = f"{row.feature_label}\nSpearman: {row.spearman:.3f}\nBootstrap 95% interval: {row.bootstrap_low:.3f} to {row.bootstrap_high:.3f}\nHolm-adjusted permutation p: {row.holm_p:.4f}"
        lines.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="{INK}">{esc(row.feature_label)}</text>')
        lines.append(f'<line x1="{sx(row.bootstrap_low):.1f}" x2="{sx(row.bootstrap_high):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<circle class="p" data-tooltip="{tooltip_attr(tooltip)}" cx="{sx(row.spearman):.1f}" cy="{y:.1f}" r="6" fill="{color}"><title>{esc(tooltip)}</title></circle>')
    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-5}" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Spearman correlation with persistent 2023-2025 CLR shift</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_prediction(scores: pd.DataFrame, path: Path) -> None:
    data = scores[scores["eligible"]].copy()
    width, height = 820, 520
    left, right, top, bottom = 78, 35, 30, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = float(data["spectral_emergence_score"].min()), float(data["spectral_emergence_score"].max())
    ymin, ymax = float(data["persistent_shift"].min()), float(data["persistent_shift"].max())
    pad_x, pad_y = (xmax - xmin) * 0.07, (ymax - ymin) * 0.08
    xmin, xmax, ymin, ymax = xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y
    sx = lambda value: scale(value, (xmin, xmax), (left, left + plot_w))
    sy = lambda value: scale(value, (ymin, ymax), (top + plot_h, top))
    lines = _svg_start(width, height)
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10.5" fill="{MUTED}">{tick:.2f}</text>')
    top_labels = set(data.nlargest(5, "persistent_shift")["topic_id"]) | set(data.nlargest(3, "spectral_emergence_score")["topic_id"])
    for row in data.itertuples():
        x, y = sx(row.spectral_emergence_score), sy(row.persistent_shift)
        tooltip = f"{row.label}\n2022 count: {row.cutoff_count:,}\nSpectral score: {row.spectral_emergence_score:.2f}\nPersistent CLR shift: {row.persistent_shift:.3f}"
        lines.append(f'<circle class="p" data-tooltip="{tooltip_attr(tooltip)}" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{RUST}" fill-opacity="0.68"><title>{esc(tooltip)}</title></circle>')
        if row.topic_id in top_labels:
            label = row.label if len(row.label) <= 29 else row.label[:26].rstrip() + "..."
            anchor = "end" if x > left + plot_w * 0.72 else "start"
            dx = -7 if anchor == "end" else 7
            lines.append(f'<text x="{x+dx:.1f}" y="{y-7:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="10.5" font-weight="700" fill="{INK}">{esc(label)}</text>')
    lines.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" stroke="#98a2b3"/>')
    lines.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#98a2b3"/>')
    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Fixed 2022 spectral composite</text>')
    lines.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Persistent CLR shift</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_endpoint_sensitivity(data: pd.DataFrame, path: Path) -> None:
    order = ["2023", "2024", "2025", "Jun 2026", "2023-2025 mean"]
    features = [
        ("spectral_emergence_score", RUST),
        ("spectral_midband_momentum", BLUE),
        ("raw_momentum_3y", VIOLET),
        ("baseline_growth_3y", TEAL),
        ("raw_acceleration_1y", GOLD),
    ]
    width, height = 820, 510
    left, right, top, bottom = 72, 205, 30, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    sx = lambda index: scale(index, (0, len(order) - 1), (left, left + plot_w))
    sy = lambda value: scale(value, (-0.35, 0.65), (top + plot_h, top))
    lines = _svg_start(width, height)
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{"#98a2b3" if tick == 0 else GRID}"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{MUTED}">{tick:.1f}</text>')
    for index, label in enumerate(order):
        x = sx(index)
        if label == "Jun 2026":
            lines.append(f'<rect x="{x-43:.1f}" y="{top}" width="86" height="{plot_h}" fill="#f6f7fb"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial" font-size="10.5" fill="{MUTED}">{esc(label)}</text>')
    for feature, color in features:
        rows = data[data["feature"] == feature].set_index("endpoint").loc[order]
        points = [(sx(i), sy(float(row.spearman))) for i, row in enumerate(rows.itertuples())]
        lines.append(f'<polyline class="s" points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        for label, row, (x, y) in zip(order, rows.itertuples(), points):
            tooltip = f"{FEATURE_LABELS[feature]}\nOutcome: {label}\nSpearman: {row.spearman:.3f}"
            lines.append(f'<circle class="p" data-tooltip="{tooltip_attr(tooltip)}" cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"><title>{esc(tooltip)}</title></circle>')
    legend_x = left + plot_w + 28
    for i, (feature, color) in enumerate(features):
        y = top + 24 + i * 45
        lines.append(f'<line x1="{legend_x}" x2="{legend_x+22}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        label = FEATURE_LABELS[feature]
        lines.append(f'<text x="{legend_x+30}" y="{y+4}" font-family="Arial" font-size="10.8" fill="{INK}">{esc(label)}</text>')
    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Outcome window</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_rolling_delta(rolling: pd.DataFrame, path: Path) -> None:
    pivot = rolling.pivot(index="cutoff_year", columns="feature", values="spearman")
    series = [
        ("spectral_emergence_score", "Composite minus baseline", RUST),
        ("spectral_midband_momentum", "Midband minus baseline", BLUE),
    ]
    width, height = 820, 460
    left, right, top, bottom = 68, 185, 30, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    years = pivot.index.to_list()
    deltas = np.concatenate([(pivot[f] - pivot["baseline_growth_3y"]).to_numpy() for f, _, _ in series])
    ymin, ymax = min(-0.45, float(deltas.min()) - 0.05), max(0.2, float(deltas.max()) + 0.05)
    sx = lambda year: scale(year, (min(years), max(years)), (left, left + plot_w))
    sy = lambda value: scale(value, (ymin, ymax), (top + plot_h, top))
    lines = _svg_start(width, height)
    for tick in np.arange(-0.4, 0.21, 0.1):
        if ymin <= tick <= ymax:
            y = sy(float(tick))
            lines.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{"#111827" if abs(tick) < 1e-8 else GRID}" opacity="{0.55 if abs(tick) < 1e-8 else 1}"/>')
    for year in range(min(years), max(years) + 1, 2):
        x = sx(year)
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="10.5" fill="{MUTED}">{year}</text>')
    legend_x = left + plot_w + 28
    for index, (feature, label, color) in enumerate(series):
        values = pivot[feature] - pivot["baseline_growth_3y"]
        points = [(sx(int(year)), sy(float(value))) for year, value in values.items()]
        lines.append(f'<polyline class="s" points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2.8"/>')
        for year, value, (x, y) in zip(values.index, values, points):
            tooltip = f"{label}\nCutoff: {year}\nSpearman difference: {value:.3f}"
            lines.append(f'<circle class="p" data-tooltip="{tooltip_attr(tooltip)}" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{esc(tooltip)}</title></circle>')
        y = top + 30 + index * 46
        lines.append(f'<line x1="{legend_x}" x2="{legend_x+22}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+30}" y="{y+4}" font-family="Arial" font-size="10.8" fill="{INK}">{esc(label)}</text>')
        lines.append(f'<text x="{legend_x+30}" y="{y+21}" font-family="Arial" font-size="10.2" fill="{MUTED}">mean {values.mean():+.2f}</text>')
    lines.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Spearman difference</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_ablation(ablation: pd.DataFrame, null: pd.DataFrame, path: Path) -> None:
    data = ablation.sort_values("spearman").reset_index(drop=True)
    width, height = 820, 500
    left, right, top, bottom = 210, 50, 32, 55
    plot_w = width - left - right
    row_h = (height - top - bottom) / len(data)
    null_low, null_high = null["null_spearman"].quantile([0.025, 0.975])
    sx = lambda value: scale(value, (-0.35, 0.75), (left, left + plot_w))
    lines = _svg_start(width, height)
    lines.append(f'<rect x="{sx(null_low):.1f}" y="{top}" width="{sx(null_high)-sx(null_low):.1f}" height="{height-top-bottom}" fill="#eef1f6"/>')
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6]:
        x = sx(tick)
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height-bottom}" stroke="{"#98a2b3" if tick == 0 else GRID}"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="11" fill="{MUTED}">{tick:.1f}</text>')
    for index, row in enumerate(data.itertuples()):
        y = top + row_h * (index + 0.5)
        color = RUST if row.variant == "Correlation k=7" else VIOLET if row.uses_current_metadata else BLUE
        tooltip = f"{row.variant}\nSpearman: {row.spearman:.3f}\nEdges: {row.edges}\nUses current metadata: {'yes' if row.uses_current_metadata else 'no'}"
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11.5" fill="{INK}">{esc(row.variant)}</text>')
        lines.append(f'<circle class="p" data-tooltip="{tooltip_attr(tooltip)}" cx="{sx(row.spearman):.1f}" cy="{y:.1f}" r="6" fill="{color}"><title>{esc(tooltip)}</title></circle>')
    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-4}" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Holdout Spearman; gray band is the randomized-graph 95% interval</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_uncertainty_mobile(inference: pd.DataFrame, path: Path) -> None:
    data = inference.sort_values("spearman", ascending=False).reset_index(drop=True)
    width, height = 360, 585
    left, right, top, bottom = 42, 18, 18, 42
    plot_w = width - left - right
    row_h = (height - top - bottom) / len(data)
    sx = lambda value: scale(value, (-0.35, 0.75), (left, left + plot_w))
    lines = _svg_start(width, height)
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6]:
        x = sx(tick)
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height-bottom}" stroke="{"#98a2b3" if tick == 0 else GRID}"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="10" fill="{MUTED}">{tick:.1f}</text>')
    for index, row in enumerate(data.itertuples()):
        label_y = top + row_h * index + 17
        mark_y = label_y + 25
        color = RUST if row.feature == "spectral_emergence_score" else TEAL if row.feature == "baseline_growth_3y" else BLUE
        tooltip = f"{row.feature_label}\nSpearman: {row.spearman:.3f}\nBootstrap 95% interval: {row.bootstrap_low:.3f} to {row.bootstrap_high:.3f}"
        lines.append(f'<text x="{left}" y="{label_y:.1f}" font-family="Arial" font-size="11" font-weight="700" fill="{INK}">{esc(row.feature_label)}</text>')
        lines.append(f'<line x1="{sx(row.bootstrap_low):.1f}" x2="{sx(row.bootstrap_high):.1f}" y1="{mark_y:.1f}" y2="{mark_y:.1f}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<circle class="p" data-tooltip="{tooltip_attr(tooltip)}" cx="{sx(row.spearman):.1f}" cy="{mark_y:.1f}" r="5.5" fill="{color}"><title>{esc(tooltip)}</title></circle>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_endpoint_mobile(data: pd.DataFrame, path: Path) -> None:
    order = ["2023", "2024", "2025", "Jun 2026", "2023-2025 mean"]
    short = ["2023", "2024", "2025", "2026*", "3-year"]
    features = [
        ("spectral_emergence_score", "Composite", RUST),
        ("spectral_midband_momentum", "Midband", BLUE),
        ("raw_momentum_3y", "Raw momentum", VIOLET),
        ("baseline_growth_3y", "CLR baseline", TEAL),
        ("raw_acceleration_1y", "Acceleration", GOLD),
    ]
    width, height = 360, 410
    left, right, top, bottom = 42, 14, 20, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    sx = lambda index: scale(index, (0, len(order) - 1), (left, left + plot_w))
    sy = lambda value: scale(value, (-0.35, 0.65), (top + plot_h, top))
    lines = _svg_start(width, height)
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{"#98a2b3" if tick == 0 else GRID}"/>')
        lines.append(f'<text x="{left-6}" y="{y+3:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="{MUTED}">{tick:.1f}</text>')
    for index, label in enumerate(short):
        lines.append(f'<text x="{sx(index):.1f}" y="{top+plot_h+19}" text-anchor="middle" font-family="Arial" font-size="9" fill="{MUTED}">{label}</text>')
    for feature, label, color in features:
        rows = data[data["feature"] == feature].set_index("endpoint").loc[order]
        points = [(sx(i), sy(float(row.spearman))) for i, row in enumerate(rows.itertuples())]
        lines.append(f'<polyline class="s" points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for point in points:
            lines.append(f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="3.2" fill="{color}"/>')
    for index, (_, label, color) in enumerate(features):
        col, row = index % 2, index // 2
        x, y = 24 + col * 170, height - 55 + row * 18
        lines.append(f'<line x1="{x}" x2="{x+18}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{x+24}" y="{y+3}" font-family="Arial" font-size="9.5" fill="{INK}">{label}</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_rolling_mobile(rolling: pd.DataFrame, path: Path) -> None:
    pivot = rolling.pivot(index="cutoff_year", columns="feature", values="spearman")
    series = [
        ("spectral_emergence_score", "Composite", RUST),
        ("spectral_midband_momentum", "Midband", BLUE),
    ]
    width, height = 360, 350
    left, right, top, bottom = 42, 14, 30, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    years = pivot.index.to_list()
    deltas = np.concatenate([(pivot[f] - pivot["baseline_growth_3y"]).to_numpy() for f, _, _ in series])
    ymin, ymax = min(-0.45, float(deltas.min()) - 0.05), max(0.2, float(deltas.max()) + 0.05)
    sx = lambda year: scale(year, (min(years), max(years)), (left, left + plot_w))
    sy = lambda value: scale(value, (ymin, ymax), (top + plot_h, top))
    lines = _svg_start(width, height)
    for tick in [-0.4, -0.2, 0.0, 0.2]:
        if ymin <= tick <= ymax:
            y = sy(tick)
            lines.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{"#111827" if tick == 0 else GRID}"/>')
            lines.append(f'<text x="{left-6}" y="{y+3:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="{MUTED}">{tick:.1f}</text>')
    for year in range(min(years), max(years) + 1, 3):
        lines.append(f'<text x="{sx(year):.1f}" y="{top+plot_h+19}" text-anchor="middle" font-family="Arial" font-size="9" fill="{MUTED}">{year}</text>')
    for index, (feature, label, color) in enumerate(series):
        values = pivot[feature] - pivot["baseline_growth_3y"]
        points = [(sx(int(year)), sy(float(value))) for year, value in values.items()]
        lines.append(f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for point in points:
            lines.append(f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="3" fill="{color}"/>')
        x = 78 + index * 150
        lines.append(f'<line x1="{x}" x2="{x+18}" y1="16" y2="16" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{x+24}" y="19" font-family="Arial" font-size="9.5" fill="{INK}">{label}</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_ablation_mobile(ablation: pd.DataFrame, null: pd.DataFrame, path: Path) -> None:
    data = ablation.sort_values("spearman", ascending=False).reset_index(drop=True)
    width, height = 360, 595
    left, right, top, bottom = 38, 18, 18, 38
    plot_w = width - left - right
    row_h = (height - top - bottom) / len(data)
    null_low, null_high = null["null_spearman"].quantile([0.025, 0.975])
    sx = lambda value: scale(value, (-0.35, 0.75), (left, left + plot_w))
    lines = _svg_start(width, height)
    lines.append(f'<rect x="{sx(null_low):.1f}" y="{top}" width="{sx(null_high)-sx(null_low):.1f}" height="{height-top-bottom}" fill="#eef1f6"/>')
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6]:
        x = sx(tick)
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height-bottom}" stroke="{"#98a2b3" if tick == 0 else GRID}"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="9" fill="{MUTED}">{tick:.1f}</text>')
    for index, row in enumerate(data.itertuples()):
        label_y = top + index * row_h + 16
        mark_y = label_y + 19
        color = RUST if row.variant == "Correlation k=7" else VIOLET if row.uses_current_metadata else BLUE
        lines.append(f'<text x="{left}" y="{label_y:.1f}" font-family="Arial" font-size="10.5" font-weight="700" fill="{INK}">{esc(row.variant)}</text>')
        lines.append(f'<circle cx="{sx(row.spearman):.1f}" cy="{mark_y:.1f}" r="5.2" fill="{color}"/>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def draw_energy(energy: pd.DataFrame, path: Path) -> None:
    width, height = 820, 440
    left, right, top, bottom = 65, 140, 30, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    years = energy["year"].astype(int).to_list()
    sx = lambda year: scale(year, (min(years), max(years)), (left, left + plot_w))
    sy = lambda value: scale(value, (0, 1), (top + plot_h, top))
    bands = [("low_share", "Low", BLUE), ("mid_share", "Mid", GOLD), ("high_share", "High", RUST)]
    lines = _svg_start(width, height)
    cumulative = np.zeros(len(energy))
    x_values = [sx(year) for year in years]
    for column, label, color in bands:
        values = energy[column].to_numpy(dtype=float)
        upper = cumulative + values
        points = list(zip(x_values, [sy(value) for value in upper])) + list(zip(reversed(x_values), [sy(value) for value in reversed(cumulative)]))
        lines.append(f'<polygon points="{polyline(points)}" fill="{color}" fill-opacity="0.78"/>')
        cumulative = upper
    for year in range(1990, PANEL_END_YEAR + 1, 5):
        lines.append(f'<text x="{sx(year):.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="10.5" fill="{MUTED}">{year}</text>')
    for index, (_, label, color) in enumerate(bands):
        y = top + 28 + index * 37
        lines.append(f'<rect x="{left+plot_w+25}" y="{y-12}" width="18" height="12" fill="{color}" fill-opacity="0.78"/>')
        lines.append(f'<text x="{left+plot_w+51}" y="{y-2}" font-family="Arial" font-size="11" fill="{INK}">{label} frequency</text>')
    lines.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Energy share</text>')
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def _svg_embed(name: str, label: str, mobile_name: str | None = None) -> str:
    path = FIGURE_DIR / name
    svg = path.read_text(encoding="utf-8")
    if mobile_name is None:
        return f'<div class="inline-svg" role="img" aria-label="{esc(label)}">{svg}</div>'
    mobile_svg = (FIGURE_DIR / mobile_name).read_text(encoding="utf-8")
    return (
        f'<div class="inline-svg desktop-svg" role="img" aria-label="{esc(label)}">{svg}</div>'
        f'<div class="mobile-svg" role="img" aria-label="{esc(label)}">{mobile_svg}</div>'
    )


def _html_rows(rows: list[list[object]]) -> str:
    return "\n".join("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)


def write_article(
    primary: dict[str, object],
    any_topic: dict[str, object],
    inference: pd.DataFrame,
    pair_inference: pd.DataFrame,
    endpoints: pd.DataFrame,
    ablation: pd.DataFrame,
    graph_null: pd.DataFrame,
    volume: pd.DataFrame,
    rolling_primary: pd.DataFrame,
    rolling_diagnostics: dict[str, float],
    nested_selection: pd.DataFrame,
    nested_diagnostics: dict[str, float],
    energy: pd.DataFrame,
    coverage: pd.DataFrame,
    top_shifts: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    scores = primary["scores"]
    metrics = primary["metrics"]
    any_metrics = any_topic["metrics"]
    eligible = scores[scores["eligible"]]
    spectral = inference.set_index("feature").loc["spectral_emergence_score"]
    baseline = inference.set_index("feature").loc["baseline_growth_3y"]
    pair = pair_inference.set_index(["left_feature", "right_feature"])
    spectral_delta = pair.loc[("spectral_emergence_score", "baseline_growth_3y")]
    controls = np.column_stack(
        [eligible["raw_momentum_3y"], eligible["raw_acceleration_1y"], eligible["cutoff_log_count"]]
    )
    partial = partial_spearman(
        eligible["spectral_emergence_score"].to_numpy(),
        eligible["persistent_shift"].to_numpy(),
        controls,
    )
    spectral_raw_corr = spearman_corr(
        eligible["spectral_emergence_score"].to_numpy(), eligible["raw_momentum_3y"].to_numpy()
    )
    mid_raw_corr = spearman_corr(
        eligible["spectral_midband_momentum"].to_numpy(), eligible["raw_momentum_3y"].to_numpy()
    )
    null_p = float(graph_null["upper_tail_p"].iloc[0])
    null_low, null_high = graph_null["null_spearman"].quantile([0.025, 0.975])
    primary_hits = int(metrics.set_index("feature").loc["spectral_emergence_score", "top10_hits"])
    random_hit_p = random_top10_probability(len(eligible), primary_hits)
    coverage_lookup = coverage.set_index("year")
    ratio_1990 = coverage_lookup.loc[1990, "any_topic_assignments_per_work"]
    ratio_2022 = coverage_lookup.loc[2022, "any_topic_assignments_per_work"]
    ratio_2026 = coverage_lookup.loc[2026, "any_topic_assignments_per_work"]
    agentic_audit_path = ARTIFACT_DIR / "tables" / "agentic_duplicate_audit.csv"
    duplicate_count = 0
    if agentic_audit_path.exists():
        duplicate_count = int((pd.read_csv(agentic_audit_path)["duplicate_of"].fillna("") != "").sum())
    duplicate_sentence = (
        "One exact duplicate time series is removed before family aggregation."
        if duplicate_count == 1
        else f"{duplicate_count} exact duplicate time series are removed before family aggregation."
    )
    fetched_date = str(manifest["fetched_at"]).split("T")[0]

    def pvalue(value: float) -> str:
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    metric_rows = _html_rows(
        [
            [
                row.feature_label,
                f"{row.spearman:.2f}",
                f"{row.bootstrap_low:.2f} to {row.bootstrap_high:.2f}",
                pvalue(row.holm_p),
                f"{int(metrics.set_index('feature').loc[row.feature, 'top10_hits'])}/10",
            ]
            for row in inference.itertuples()
        ]
    )
    scope_rows = _html_rows(
        [
            [
                FEATURE_LABELS[feature],
                f"{metrics.set_index('feature').loc[feature, 'spearman']:.2f}",
                f"{any_metrics.set_index('feature').loc[feature, 'spearman']:.2f}",
            ]
            for feature in ["spectral_emergence_score", "spectral_midband_momentum", "raw_momentum_3y", "baseline_growth_3y"]
        ]
    )
    rolling_rows = _html_rows(
        [
            [row.feature_label, f"{row.mean_spearman:.2f}", f"{row.median_spearman:.2f}", f"{row.mean_top10_hits:.1f}"]
            for row in rolling_primary.itertuples()
            if row.feature in ["baseline_growth_3y", "raw_momentum_3y", "spectral_midband_momentum", "spectral_emergence_score"]
        ]
    )
    top_rows = _html_rows(
        [
            [row.label, f"{row.cutoff_count:,}", f"{row.count_2025:,}", f"{row.share_multiple_2025_vs_2022:.2f}x", f"{row.persistent_shift:.2f}"]
            for row in top_shifts.head(10).itertuples()
        ]
    )
    volume_rows = _html_rows(
        [
            [
                int(threshold),
                int(group["n_topics"].iloc[0]),
                f"{group.set_index('feature').loc['spectral_emergence_score', 'spearman']:.2f}",
                f"{group.set_index('feature').loc['baseline_growth_3y', 'spearman']:.2f}",
            ]
            for threshold, group in volume.groupby("minimum_2022_count")
        ]
    )

    tooltip_script = """
<div id="chart-tooltip" role="status" aria-live="polite"></div>
<script>
(() => {
  const tooltip = document.getElementById('chart-tooltip');
  if (!tooltip) return;
  const place = (event) => {
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
    const box = tooltip.getBoundingClientRect();
    if (box.right > window.innerWidth - 10) tooltip.style.left = `${event.clientX - box.width - 14}px`;
    if (box.bottom > window.innerHeight - 10) tooltip.style.top = `${event.clientY - box.height - 14}px`;
  };
  document.querySelectorAll('[data-tooltip]').forEach((node) => {
    node.addEventListener('pointerenter', (event) => {
      tooltip.textContent = node.getAttribute('data-tooltip');
      tooltip.classList.add('visible');
      place(event);
    });
    node.addEventListener('pointermove', place);
    node.addEventListener('pointerleave', () => tooltip.classList.remove('visible'));
  });
})();
</script>
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Innovation Momentum: Spectral Dynamics in AI Research</title>
  <style>
    :root {{ --ink:{INK}; --muted:{MUTED}; --line:{GRID}; --paper:#fff; --band:#f6f7fb; --blue:{BLUE}; --teal:{TEAL}; --rust:{RUST}; --gold:{GOLD}; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height:1.68; }}
    a {{ color:#235f96; text-decoration-thickness:1px; text-underline-offset:3px; }}
    .page {{ width:min(1120px, calc(100% - 40px)); margin:0 auto; }}
    header {{ padding:64px 0 30px; border-bottom:1px solid var(--line); }}
    .kicker {{ margin:0 0 10px; color:var(--rust); font-size:12px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
    h1 {{ max-width:900px; margin:0; font-family:Georgia, "Times New Roman", serif; font-size:clamp(38px, 6vw, 72px); font-weight:500; line-height:1.03; letter-spacing:0; }}
    .dek {{ max-width:820px; margin:22px 0 0; color:#475467; font-family:Georgia, "Times New Roman", serif; font-size:20px; line-height:1.48; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:24px; color:var(--muted); font-size:12.5px; }}
    nav {{ position:sticky; top:0; z-index:10; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); }}
    nav .page {{ display:flex; gap:18px; overflow-x:auto; padding:11px 0; scrollbar-width:thin; }}
    nav a {{ flex:0 0 auto; color:#475467; font-size:12.5px; font-weight:700; text-decoration:none; }}
    main section {{ padding:44px 0; border-bottom:1px solid var(--line); scroll-margin-top:48px; }}
    h2 {{ max-width:850px; margin:0 0 18px; font-family:Georgia, "Times New Roman", serif; font-size:34px; font-weight:500; line-height:1.16; letter-spacing:0; }}
    h3 {{ margin:0 0 10px; font-size:15px; line-height:1.3; letter-spacing:0; }}
    p {{ max-width:790px; margin:0 0 16px; }}
    .abstract {{ max-width:900px; margin-top:26px; color:#344054; font-size:17px; }}
    .result-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; margin:24px 0; border:1px solid var(--line); background:var(--line); }}
    .result {{ min-width:0; padding:18px; background:#fff; }}
    .result strong {{ display:block; color:var(--rust); font-size:29px; line-height:1.05; }}
    .result span {{ display:block; margin-top:8px; color:var(--muted); font-size:12.5px; line-height:1.4; }}
    .method-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:22px; }}
    .method {{ min-width:0; padding:16px; border:1px solid var(--line); }}
    .method p {{ margin:0; color:#475467; font-size:13.5px; }}
    .note {{ max-width:820px; margin:20px 0; padding:13px 16px; border-left:4px solid var(--gold); background:#fbfaf6; color:#344054; }}
    figure {{ margin:24px 0 0; padding:14px; border:1px solid var(--line); background:#fff; }}
    .chart-shell {{ max-width:100%; overflow-x:auto; overscroll-behavior-inline:contain; scrollbar-color:#98a2b3 #eef1f6; }}
    .inline-svg {{ min-width:640px; }}
    .inline-svg svg {{ display:block; width:100%; height:auto; }}
    .mobile-svg {{ display:none; }}
    figcaption {{ max-width:850px; margin-top:12px; padding-top:10px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px; line-height:1.5; }}
    .table-wrap {{ max-width:100%; margin-top:20px; overflow-x:auto; border:1px solid var(--line); }}
    table {{ width:100%; min-width:680px; border-collapse:collapse; background:#fff; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:12.5px; }}
    th {{ background:var(--band); color:var(--ink); font-weight:750; }}
    td {{ color:#344054; }}
    code {{ padding:1px 4px; background:#f1f3f6; font-size:.92em; }}
    footer {{ padding:34px 0 54px; color:var(--muted); font-size:12.5px; }}
    #chart-tooltip {{ position:fixed; z-index:30; max-width:min(320px,calc(100vw - 28px)); padding:9px 11px; border:1px solid rgba(31,41,51,.18); background:rgba(255,255,255,.97); box-shadow:0 10px 28px rgba(31,41,51,.15); color:var(--ink); font-size:12px; line-height:1.4; white-space:pre-line; pointer-events:none; opacity:0; }}
    #chart-tooltip.visible {{ opacity:1; }}
    @media (max-width:720px) {{
      .page {{ width:min(100% - 24px,1120px); }}
      header {{ padding-top:38px; }}
      h1 {{ font-size:42px; }}
      .dek {{ font-size:17px; }}
      main section {{ padding:34px 0; }}
      h2 {{ font-size:28px; }}
      .result-grid,.method-grid {{ grid-template-columns:1fr; }}
      figure {{ margin-left:-4px; margin-right:-4px; padding:10px; }}
      .inline-svg {{ min-width:600px; }}
      .desktop-svg {{ display:none; }}
      .mobile-svg {{ display:block; min-width:0; }}
      .mobile-svg svg {{ display:block; width:100%; height:auto; }}
      table {{ min-width:620px; }}
    }}
    @media (prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}
  </style>
</head>
<body>
<header>
  <div class="page">
    <p class="kicker">OpenAlex field dynamics</p>
    <h1>Innovation Momentum: Spectral Dynamics in AI Research</h1>
    <p class="dek">A retrospective test of whether graph Fourier features add useful information to ordinary publication growth.</p>
    <p class="abstract">This study treats topic shares as a composition, uses complete 2023-2025 data for the primary outcome, and fixes the graph at the 2022 cutoff. The spectral composite reaches a Spearman correlation of {spectral.spearman:.2f}; the CLR growth baseline reaches {baseline.spearman:.2f}. Their paired bootstrap difference is {spectral_delta.observed_delta:+.2f}, with a 95% interval of {spectral_delta.bootstrap_low:+.2f} to {spectral_delta.bootstrap_high:+.2f}. Rolling cutoffs favor the growth baseline.</p>
    <div class="meta"><span>OpenAlex snapshot: {esc(fetched_date)}</span><span>Snapshot taxonomy: 77 AI topics</span><span>Main holdout: {len(eligible)} topics</span><span>Partial 2026 data used as sensitivity only</span></div>
  </div>
</header>
<nav><div class="page"><a href="#result">Result</a><a href="#design">Design</a><a href="#holdout">Holdout</a><a href="#stability">Stability</a><a href="#topics">Topics</a><a href="#agentic">Agentic terms</a><a href="#limits">Limits</a></div></nav>
<main class="page">
  <section id="result">
    <h2>Main Result</h2>
    <p>The 2022 holdout shows a positive association. Uncertainty around the incremental gain is wide, and the rolling record favors the growth baseline. The evidence supports graph spectra as a descriptive view of field motion. Predictive value beyond recent growth remains unresolved.</p>
    <div class="result-grid">
      <div class="result"><strong>{spectral.spearman:.2f}</strong><span>spectral composite correlation with the persistent 2023-2025 shift</span></div>
      <div class="result"><strong>{baseline.spearman:.2f}</strong><span>unstandardized CLR growth baseline on the same topics</span></div>
      <div class="result"><strong>{int(rolling_diagnostics['spectral_wins'])}/{int(rolling_diagnostics['n_cutoffs'])}</strong><span>rolling cutoffs where the spectral composite beats the baseline</span></div>
    </div>
    <p>The composite captures {primary_hits} of the ten largest persistent shifts. Under random ranking, the probability of at least that many hits is {random_hit_p:.2f}. The top-ten result provides little separation on its own.</p>
  </section>

  <section id="design">
    <h2>Research Design</h2>
    <p><a href="https://help.openalex.org/hc/en-us/articles/24736129405719-Topics">OpenAlex assigns</a> one primary topic to each work and may assign several secondary topics. The primary-topic counts sum to the AI-subfield denominator in every year of this snapshot. That closure makes ordinary share changes dependent across topics. A pseudocount of 0.5 is added before applying a centered log-ratio transform within each year.</p>
    <div class="method-grid">
      <div class="method"><h3>Outcome</h3><p>The primary outcome is mean topic CLR in 2023-2025 minus mean CLR in 2020-2022. It measures a sustained relative shift and excludes the incomplete 2026 endpoint.</p></div>
      <div class="method"><h3>Eligibility</h3><p>Topics require at least {MIN_CUTOFF_COUNT} primary-topic works at the cutoff. The 2022 holdout retains {len(eligible)} of 77 topics. Threshold sensitivity is reported below.</p></div>
      <div class="method"><h3>Graph</h3><p>The main specification uses a positive-correlation k-nearest-neighbor graph with k=7, estimated through the cutoff. Topic descriptions are excluded from this graph.</p></div>
      <div class="method"><h3>Pre-specified score</h3><p>The spectral composite uses weights set before evaluation against the persistent-shift outcome: 50% local spectral momentum, 30% midband momentum, and 20% acceleration.</p></div>
    </div>
    <p class="note">This is a current-taxonomy retrospective holdout. The June 2026 OpenAlex topic universe and historical assignments were downloaded together. Rolling cutoffs restrict count histories, while taxonomy construction and later OpenAlex revisions remain outside the time split.</p>
    <p>The any-topic panel also uses a compositional denominator. Its CLR transform is equivalent to normalizing each topic count by total topic assignments for that year. Assignments per AI work rise from {ratio_1990:.2f} in 1990 to {ratio_2022:.2f} in 2022 and {ratio_2026:.2f} in June 2026.</p>
    <div class="table-wrap"><table><thead><tr><th>Feature</th><th>Primary-topic rho</th><th>Any-topic rho</th></tr></thead><tbody>{scope_rows}</tbody></table></div>
  </section>

  <section id="holdout">
    <h2>The Holdout Has Signal And Wide Comparative Uncertainty</h2>
    <p>The fixed spectral composite has a bootstrap interval of {spectral.bootstrap_low:.2f} to {spectral.bootstrap_high:.2f}. Its Holm-adjusted permutation p-value is {esc(pvalue(spectral.holm_p))}. The paired interval for its advantage over the CLR baseline includes zero, so the holdout does not establish a stable improvement.</p>
    <figure><h3>Feature correlations with uncertainty</h3><div class="chart-shell">{_svg_embed('holdout_uncertainty.svg', 'Feature correlations and bootstrap intervals', 'holdout_uncertainty_mobile.svg')}</div><figcaption>Figure 1. Spearman correlations for the eligible 2022 holdout. Intervals use 10,000 topic-row bootstrap samples. Topic dependence created by the composition makes these intervals approximate.</figcaption></figure>
    <div class="table-wrap"><table><thead><tr><th>Feature</th><th>Spearman</th><th>Bootstrap 95% interval</th><th>Holm p</th><th>Top-10 hits</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
    <figure><h3>Where the fixed composite lands</h3><div class="chart-shell">{_svg_embed('prediction_scatter.svg', 'Spectral composite against persistent topic shift')}</div><figcaption>Figure 2. Each point is an eligible OpenAlex topic. Vertical position is the composition-aware outcome, measured against the full topic set.</figcaption></figure>
    <p>The composite is strongly related to ordinary momentum. Its Spearman correlation with standardized three-year momentum is {spectral_raw_corr:.2f}; the midband component correlates with that feature at {mid_raw_corr:.2f}. After rank-residualizing the composite and outcome on raw momentum, acceleration, and topic size, the partial correlation is {partial:.2f}.</p>
  </section>

  <section id="stability">
    <h2>Endpoint And Graph Choices Matter</h2>
    <p>Single-year outcomes change the ordering. The 2024 endpoint favors acceleration, while the partial June 2026 endpoint favors the growth baseline. Averaging complete post-cutoff years produces the strongest reading for the fixed composite.</p>
    <figure><h3>Endpoint sensitivity</h3><div class="chart-shell">{_svg_embed('endpoint_sensitivity.svg', 'Endpoint sensitivity of feature correlations', 'endpoint_sensitivity_mobile.svg')}</div><figcaption>Figure 3. The shaded June 2026 column is incomplete. The final column is the primary persistent-shift outcome.</figcaption></figure>
    <p>The correlation-only graph avoids present-day topic descriptions in the main score. Across k values of 3, 5, 7, 10, and 15, the composite correlation ranges from {ablation[ablation['correlation_weight'] == 1.0]['spearman'].min():.2f} to {ablation[ablation['correlation_weight'] == 1.0]['spearman'].max():.2f}. Hybrid and semantic variants use current metadata and belong to sensitivity analysis.</p>
    <figure><h3>Graph ablations and randomized labels</h3><div class="chart-shell">{_svg_embed('graph_ablation.svg', 'Graph ablation and randomized graph null', 'graph_ablation_mobile.svg')}</div><figcaption>Figure 4. The gray band spans the central 95% of 1,000 graph-label randomizations. The observed k=7 result has a one-sided graph-null probability of {null_p:.3f}; the null interval is {null_low:.2f} to {null_high:.2f}.</figcaption></figure>
    <p>The observed k=7 score falls below the central randomized-graph interval. Most graph-label randomizations produce an equal or higher composite correlation. This test finds no evidence that the learned topic connections create the holdout association.</p>
    <p>Rolling tests begin in 2005, giving every graph at least 16 annual observations. Each outcome uses the following three complete years, and each cutoff applies the same count rule. The baseline mean correlation is {rolling_primary.set_index('feature').loc['baseline_growth_3y','mean_spearman']:.2f}; the spectral mean is {rolling_primary.set_index('feature').loc['spectral_emergence_score','mean_spearman']:.2f}. Their mean difference is {rolling_diagnostics['mean_delta']:+.2f}. A three-cutoff moving-block bootstrap gives {rolling_diagnostics['block_low']:+.2f} to {rolling_diagnostics['block_high']:+.2f}.</p>
    <figure><h3>Rolling advantage over the CLR baseline</h3><div class="chart-shell">{_svg_embed('rolling_delta.svg', 'Rolling spectral correlation minus baseline correlation', 'rolling_delta_mobile.svg')}</div><figcaption>Figure 5. Adjacent outcomes overlap, so the cutoff results are serially dependent. Values below zero favor the baseline.</figcaption></figure>
    <div class="table-wrap"><table><thead><tr><th>Feature</th><th>Mean rho</th><th>Median rho</th><th>Mean top-10 hits</th></tr></thead><tbody>{rolling_rows}</tbody></table></div>
    <p>A nested check selects graph density and one of five fixed score-weight forms using outcomes fully observed by each outer cutoff. It covers {int(nested_diagnostics['n_cutoffs'])} outer cutoffs. The selected models average {nested_diagnostics['mean_selected']:.2f}, compared with {nested_diagnostics['mean_baseline']:.2f} for the baseline. They exceed the baseline in {int(nested_diagnostics['wins'])} of {int(nested_diagnostics['n_cutoffs'])} tests. The mean difference is {nested_diagnostics['mean_delta']:+.2f}, with a moving-block interval of {nested_diagnostics['block_low']:+.2f} to {nested_diagnostics['block_high']:+.2f}.</p>
  </section>

  <section id="topics">
    <h2>Persistent Relative Movers</h2>
    <p>The table ranks sustained shifts in the topic composition. A high value means a topic gained share against other OpenAlex AI topics. It does not measure scientific quality, citation impact, or absolute societal importance.</p>
    <div class="table-wrap"><table><thead><tr><th>Topic</th><th>2022 count</th><th>2025 count</th><th>Share multiple</th><th>Persistent CLR shift</th></tr></thead><tbody>{top_rows}</tbody></table></div>
    <p>The main threshold removes three very small 2022 series. The broad result changes gradually under stricter thresholds.</p>
    <div class="table-wrap"><table><thead><tr><th>Minimum 2022 count</th><th>Topics</th><th>Spectral rho</th><th>Baseline rho</th></tr></thead><tbody>{volume_rows}</tbody></table></div>
  </section>

  <section id="agentic">
    <h2>Agentic Vocabulary Has Older Precursors</h2>
    <p>The phrase panel provides a companion description. It sums OpenAlex title-and-abstract query hits within capability families. {duplicate_sentence} Other queries can still match the same work, so family values are query-hit indices rather than paper counts.</p>
    <figure><h3>Agentic-AI phrase-family histories</h3><div class="chart-shell">{_svg_embed('agentic_precursor_river.svg', 'Agentic AI query-hit family histories')}</div><figcaption>Figure 6. Query-hit intensity per million AI works. Recent LLM-agent terms rise after 2022; planning, retrieval, dialogue, code use, and classic agent terms have longer histories.</figcaption></figure>
    <p>The phrase evidence supports a narrow interpretation: recent naming connects to older capability areas. Aggregate query counts cannot establish a paper-level lineage. Work-level deduplication, citations, and text embeddings are required for that claim.</p>
  </section>

  <section id="limits">
    <h2>What The Analysis Can Support</h2>
    <p>Graph Fourier features summarize whether topic motion is broad or localized. Low-frequency energy falls from {energy.set_index('year').loc[2022,'low_share']:.0%} in 2022 to {energy.set_index('year').loc[2025,'low_share']:.0%} in 2025 in the fixed 2022 graph. This is a coordinate-dependent description of the field.</p>
    <figure><h3>Spectral energy in the fixed 2022 graph</h3><div class="chart-shell">{_svg_embed('spectral_energy.svg', 'Low mid and high spectral energy shares')}</div><figcaption>Figure 7. Annual energy shares use the 2022 correlation graph for all years. Later values describe movement in that fixed basis.</figcaption></figure>
    <p>Further evaluation should freeze historical OpenAlex vintages, construct paper-level citation and embedding graphs, define outcomes before scoring, and evaluate models on non-overlapping forward blocks. Independent data would provide a cleaner test after model selection.</p>
    <p>The evidence supports the spectral representation as a descriptive tool for field dynamics. One 2022 holdout is favorable. The rolling results and paired uncertainty leave incremental predictive value unresolved.</p>
  </section>
</main>
<footer><div class="page"><p>Rebuild with <code>uv run innovation-build-report</code>. The frozen aggregate inputs, SHA-256 manifest, and evidence tables are committed with the article. Source: <a href="https://api.openalex.org/works">OpenAlex Works API</a> and <a href="https://api.openalex.org/topics">OpenAlex Topics API</a>.</p></div></footer>
{tooltip_script}
</body>
</html>
"""
    write_text(REPORT_DIR / "index.html", html_text)


def write_evidence_tables(tables: dict[str, pd.DataFrame]) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(EVIDENCE_DIR / f"{name}.csv", index=False)
    write_text(
        EVIDENCE_DIR / "README.md",
        "# Evidence tables\n\nThese CSV files are generated by `uv run innovation-build-report` from the committed OpenAlex snapshot.\n",
    )


def validate_inputs(topics: pd.DataFrame, counts: pd.DataFrame) -> None:
    expected_rows = len(topics) * (PANEL_END_YEAR - YEAR_START + 1)
    panel = counts[counts["year"].between(YEAR_START, PANEL_END_YEAR)]
    if len(panel) != expected_rows:
        raise ValueError(f"Expected {expected_rows} topic-year rows; found {len(panel)}")
    if panel[["year", "topic_id"]].duplicated().any():
        raise ValueError("Duplicate topic-year rows found")
    if panel[["primary_topic_count", "ai_subfield_any_topic_count", "total_ai_subfield_count"]].isna().any().any():
        raise ValueError("Missing count values found")
    primary_check = panel.groupby("year").agg(
        assigned=("primary_topic_count", "sum"), denominator=("total_ai_subfield_count", "max")
    )
    if not np.array_equal(primary_check["assigned"].to_numpy(), primary_check["denominator"].to_numpy()):
        raise ValueError("Primary topic counts do not close to the annual denominator")


def main() -> None:
    topics_path = DATA_DIR / "processed" / "openalex_ai_topics.csv"
    counts_path = DATA_DIR / "processed" / "openalex_ai_topic_year_counts.csv"
    agentic_path = DATA_DIR / "processed" / "openalex_agentic_precursor_counts.csv"
    for path in [topics_path, counts_path, agentic_path]:
        if not path.exists():
            raise SystemExit(f"Missing {path}. Refresh the OpenAlex inputs before rebuilding.")

    topics = pd.read_csv(topics_path).sort_values("topic_id").reset_index(drop=True)
    counts = pd.read_csv(counts_path)
    counts = counts[counts["year"].between(YEAR_START, PANEL_END_YEAR)].copy()
    validate_inputs(topics, counts)
    manifest = write_manifest()

    from analyze_agentic_and_visuals import main as build_agentic_companion

    build_agentic_companion()

    primary = run_scope(counts, topics, "primary")
    any_topic = run_scope(counts, topics, "any_topic")
    inference, pair_inference = infer_holdout(primary["scores"])
    endpoints = endpoint_sensitivity(primary["scores"], primary["panel"])
    volume = volume_sensitivity(primary["scores"])
    ablation, graph_null = graph_ablation(primary["panel"], topics, primary["scores"])
    rolling_primary, rolling_diagnostics = rolling_summary(primary["rolling"])
    rolling_any, _ = rolling_summary(any_topic["rolling"])
    nested_history, nested_selection, nested_diagnostics = nested_model_selection(primary["panel"])
    energy = spectral_energy(primary["normalized"], primary["adjacency"])
    coverage = coverage_table(counts)
    top_shifts = top_shift_table(primary["scores"], primary["panel"])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    draw_uncertainty(inference, FIGURE_DIR / "holdout_uncertainty.svg")
    draw_prediction(primary["scores"], FIGURE_DIR / "prediction_scatter.svg")
    draw_endpoint_sensitivity(endpoints, FIGURE_DIR / "endpoint_sensitivity.svg")
    draw_rolling_delta(primary["rolling"], FIGURE_DIR / "rolling_delta.svg")
    draw_ablation(ablation, graph_null, FIGURE_DIR / "graph_ablation.svg")
    draw_uncertainty_mobile(inference, FIGURE_DIR / "holdout_uncertainty_mobile.svg")
    draw_endpoint_mobile(endpoints, FIGURE_DIR / "endpoint_sensitivity_mobile.svg")
    draw_rolling_mobile(primary["rolling"], FIGURE_DIR / "rolling_delta_mobile.svg")
    draw_ablation_mobile(ablation, graph_null, FIGURE_DIR / "graph_ablation_mobile.svg")
    draw_energy(energy, FIGURE_DIR / "spectral_energy.svg")

    write_evidence_tables(
        {
            "holdout_metrics_primary": primary["metrics"],
            "holdout_metrics_any_topic": any_topic["metrics"],
            "holdout_inference": inference,
            "paired_feature_differences": pair_inference,
            "endpoint_sensitivity": endpoints,
            "volume_sensitivity": volume,
            "graph_ablation": ablation,
            "graph_randomization_null": graph_null,
            "rolling_metrics_primary": primary["rolling"],
            "rolling_metrics_any_topic": any_topic["rolling"],
            "rolling_summary_primary": rolling_primary,
            "rolling_summary_any_topic": rolling_any,
            "nested_candidate_history": nested_history,
            "nested_model_selection": nested_selection,
            "coverage_by_year": coverage,
            "top_persistent_shifts": top_shifts,
            "spectral_energy": energy,
        }
    )
    write_article(
        primary,
        any_topic,
        inference,
        pair_inference,
        endpoints,
        ablation,
        graph_null,
        volume,
        rolling_primary,
        rolling_diagnostics,
        nested_selection,
        nested_diagnostics,
        energy,
        coverage,
        top_shifts,
        manifest,
    )
    print(inference.to_string(index=False))
    print(f"Wrote {REPORT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
