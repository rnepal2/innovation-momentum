"""Graph spectral helpers for innovation momentum experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / (np.nanstd(values) + eps)


def rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with rank 1 assigned to the smallest value."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rank_average(np.asarray(x, dtype=float))
    ry = rank_average(np.asarray(y, dtype=float))
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def build_topic_graph(
    train_matrix: pd.DataFrame,
    topic_meta: pd.DataFrame,
    top_k: int = 5,
    semantic_floor: float = 0.25,
) -> np.ndarray:
    """Build a pre-cutoff graph from historical co-movement plus semantic groups.

    Rows in train_matrix are years, columns are topic ids.
    """
    values = train_matrix.to_numpy(dtype=float).T
    corr = np.corrcoef(values)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)

    n = corr.shape[0]
    adjacency = np.zeros_like(corr)
    for i in range(n):
        neighbors = np.argsort(corr[i])[-top_k:]
        adjacency[i, neighbors] = corr[i, neighbors]
    adjacency = np.maximum(adjacency, adjacency.T)

    clusters = topic_meta.set_index("topic_id").loc[train_matrix.columns, "cluster"].to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if clusters[i] == clusters[j]:
                adjacency[i, j] = max(adjacency[i, j], semantic_floor)
                adjacency[j, i] = adjacency[i, j]

    for i in range(n):
        if adjacency[i].sum() == 0:
            j = int(np.argmax(corr[i]))
            if i != j:
                adjacency[i, j] = max(corr[i, j], 0.05)
                adjacency[j, i] = adjacency[i, j]
    return adjacency


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    return np.eye(adjacency.shape[0]) - (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :])


def graph_fourier_basis(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    laplacian = normalized_laplacian(adjacency)
    eigvals, eigvecs = np.linalg.eigh(laplian_symmetrized(laplacian))
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def laplian_symmetrized(laplacian: np.ndarray) -> np.ndarray:
    return 0.5 * (laplacian + laplacian.T)


def spectral_node_scores(
    train_signal_by_year: pd.DataFrame,
    adjacency: np.ndarray,
    cutoff_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute node-level spectral momentum scores at the cutoff year."""
    years = train_signal_by_year.index.to_list()
    for required in [cutoff_year, cutoff_year - 1, cutoff_year - 2, cutoff_year - 3]:
        if required not in years:
            raise ValueError(f"Missing required year for spectral score: {required}")

    topics = train_signal_by_year.columns.to_list()
    X = train_signal_by_year.loc[:, topics]
    eigvals, eigvecs = graph_fourier_basis(adjacency)

    dx_1 = X.loc[cutoff_year].to_numpy() - X.loc[cutoff_year - 1].to_numpy()
    dx_3 = (X.loc[cutoff_year].to_numpy() - X.loc[cutoff_year - 3].to_numpy()) / 3.0
    accel = dx_1 - (X.loc[cutoff_year - 1].to_numpy() - X.loc[cutoff_year - 2].to_numpy())

    alpha_dx = eigvecs.T @ dx_3
    nonzero = eigvals[eigvals > 1e-8]
    pivot = float(np.median(nonzero)) if len(nonzero) else 1.0
    low_weight = 1.0 / (1.0 + eigvals)
    high_weight = eigvals / (eigvals + pivot + 1e-9)
    mid_weight = np.exp(-((eigvals - pivot) ** 2) / (2.0 * (pivot + 1e-9) ** 2))

    smooth_momentum = eigvecs @ (alpha_dx * low_weight)
    local_momentum = eigvecs @ (alpha_dx * high_weight)
    midband_momentum = eigvecs @ (alpha_dx * mid_weight)

    score = (
        0.50 * zscore(local_momentum)
        + 0.30 * zscore(midband_momentum)
        + 0.20 * zscore(accel)
    )
    scores = pd.DataFrame(
        {
            "topic_id": topics,
            "spectral_emergence_score": score,
            "spectral_local_momentum": local_momentum,
            "spectral_midband_momentum": midband_momentum,
            "spectral_smooth_momentum": smooth_momentum,
            "raw_momentum_3y": dx_3,
            "raw_acceleration_1y": accel,
        }
    )

    coeffs = []
    for year in X.index:
        alpha = eigvecs.T @ X.loc[year].to_numpy()
        total_energy = float(np.sum(alpha**2))
        coeffs.append(
            {
                "year": int(year),
                "spectral_total_energy": total_energy,
                "spectral_low_energy": float(np.sum(alpha[eigvals <= np.quantile(eigvals, 0.33)] ** 2)),
                "spectral_mid_energy": float(
                    np.sum(
                        alpha[
                            (eigvals > np.quantile(eigvals, 0.33))
                            & (eigvals <= np.quantile(eigvals, 0.66))
                        ]
                        ** 2
                    )
                ),
                "spectral_high_energy": float(np.sum(alpha[eigvals > np.quantile(eigvals, 0.66)] ** 2)),
            }
        )
    return scores, pd.DataFrame(coeffs)
