#!/usr/bin/env python3
"""Run the AI-topic spectral dynamics analysis and article build."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paths import project_root
from spectral import (
    graph_fourier_basis,
    spearman_corr,
    spectral_node_scores,
)

ROOT = project_root()
REPORTS_DIR = ROOT / "reports"
REPORT_BUILD_DIR = ROOT / "artifacts" / "report"

YEAR_START = 1990
YEAR_CUTOFF = 2022
YEAR_END = 2026
EVAL_END_LABEL = "June 2026"
EVAL_WINDOW_LABEL = f"2023 to {EVAL_END_LABEL}"
TARGET_WINDOW_LABEL = f"{YEAR_CUTOFF} to {EVAL_END_LABEL}"
FUTURE_GAIN_COL = f"future_gain_{YEAR_CUTOFF}_to_{YEAR_END}"
FUTURE_CONTEXT_COL = f"future_gain_2023_{YEAR_END}_vs_2020_{YEAR_CUTOFF}"
COUNT_END_COL = f"count_{YEAR_END}"
SHARE_END_COL = f"share_per_million_{YEAR_END}"
FUTURE_MULTIPLE_COL = f"future_multiple_share_{YEAR_END}_vs_{YEAR_CUTOFF}"
PANEL_YEARS_LABEL = f"{YEAR_START} to {EVAL_END_LABEL}"

INK = "#1f2933"
MUTED = "#667085"
GRID = "#e5e7ef"
PAPER = "#ffffff"

FAMILY_COLORS = {
    "language_knowledge": "#3b6ea8",
    "learning_methods": "#2a9d8f",
    "agents_reasoning": "#b25d31",
    "security_privacy": "#7a5195",
    "robotics_control_sensing": "#cc6677",
    "computing_infrastructure": "#5c677d",
    "applied_domain_ai": "#8c8c52",
    "general_ai": "#8a8f98",
}

FAMILY_LABELS = {
    "language_knowledge": "Language & knowledge",
    "learning_methods": "Learning methods",
    "agents_reasoning": "Agents & reasoning",
    "security_privacy": "Security & privacy",
    "robotics_control_sensing": "Robotics, control & sensing",
    "computing_infrastructure": "Computing infrastructure",
    "applied_domain_ai": "Applied-domain AI",
    "general_ai": "General AI",
}

FEATURE_LABELS = {
    "spectral_emergence_score": "Spectral emergence",
    "spectral_local_momentum": "High-frequency local momentum",
    "spectral_midband_momentum": "Midband spectral momentum",
    "raw_momentum_3y": "Raw 3-year momentum",
    "raw_acceleration_1y": "Raw 1-year acceleration",
    "baseline_growth_3y": "Publication-growth baseline",
    "baseline_accel": "Acceleration baseline",
    "share_per_million_cutoff": "Topic size at cutoff",
}

COUNT_SCOPES = {
    "primary": {
        "label": "Primary-topic",
        "count_col": "primary_topic_count",
        "share_col": "primary_share_per_million_ai",
        "description": "works whose primary topic is the given AI topic",
    },
    "any_topic": {
        "label": "Any-topic within AI",
        "count_col": "ai_subfield_any_topic_count",
        "share_col": "ai_any_share_per_million_ai",
        "description": "AI-subfield works where the topic appears anywhere in the topic list",
    },
}

STOPWORDS = {
    "advanced",
    "algorithm",
    "algorithms",
    "analysis",
    "application",
    "applications",
    "approach",
    "artificial",
    "based",
    "cluster",
    "covers",
    "data",
    "deep",
    "development",
    "field",
    "focuses",
    "intelligence",
    "learning",
    "machine",
    "method",
    "methods",
    "model",
    "modeling",
    "models",
    "paper",
    "papers",
    "research",
    "system",
    "systems",
    "technique",
    "techniques",
    "technologies",
    "technology",
    "topic",
    "topics",
    "wide",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def tooltip_attr(value: object) -> str:
    return esc(value).replace("\n", "&#10;")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    lo, hi = domain
    a, b = range_
    if hi == lo:
        return (a + b) / 2.0
    return a + (value - lo) / (hi - lo) * (b - a)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def polygon(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{v:02x}" for v in rgb)


def blend(c1: str, c2: str, t: float) -> str:
    t = clamp(t)
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return rgb_to_hex(
        (
            round(r1 + (r2 - r1) * t),
            round(g1 + (g2 - g1) * t),
            round(b1 + (b2 - b1) * t),
        )
    )


def heat_color(value: float) -> str:
    value = clamp(value)
    if value < 0.5:
        return blend("#f6f7fb", "#8fb7d9", value / 0.5)
    return blend("#8fb7d9", "#b94e48", (value - 0.5) / 0.5)


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def wrap_label(text: str, width: int = 28, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = word if not current else f"{current} {word}"
        if len(proposal) <= width:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1][: max(3, width - 3)].rstrip() + "..."
    return lines or [text[:width]]


def markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def topic_text(row: pd.Series) -> str:
    return " ".join(
        [
            safe_text(row.get("display_name")),
            safe_text(row.get("keywords")),
            safe_text(row.get("description")),
        ]
    ).lower()


def has_term(text: str, term: str) -> bool:
    term = term.strip().lower()
    if not term:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def has_any(text: str, terms: list[str]) -> bool:
    return any(has_term(text, term) for term in terms)


def classify_topic(row: pd.Series) -> str:
    text = topic_text(row)
    rules = [
        (
            "applied_domain_ai",
            [
                "geochemistry",
                "geologic",
                "seismology",
                "earthquake",
                "solar",
                "photovoltaic",
                "healthcare",
                "cancer",
                "histopathology",
                "education",
                "pedagogy",
                "tutoring",
                "service",
                "hospitality",
                "law",
                "intellectual property",
                "organizational",
                "employee",
                "psychiatry",
                "mental health",
                "neuroscience",
                "multimedia in education",
            ],
        ),
        (
            "agents_reasoning",
            [
                "agent",
                "agents",
                "multi-agent",
                "planning",
                "reasoning",
                "game",
                "games",
                "negotiation",
                "answer set",
                "knowledge representation",
                "case-based reasoning",
                "monte carlo tree search",
            ],
        ),
        (
            "language_knowledge",
            [
                "language",
                "linguistic",
                "speech",
                "dialogue",
                "text",
                "document",
                "sentiment",
                "opinion",
                "authorship",
                "readability",
                "translation",
                "topic modeling",
                "semantic web",
                "ontology",
                "knowledge graph",
                "question answering",
                "hate speech",
            ],
        ),
        (
            "robotics_control_sensing",
            [
                "robot",
                "robotics",
                "control",
                "tracking",
                "sensor",
                "sensors",
                "wireless",
                "modulation",
                "fuzzy",
                "kalman",
                "particle filter",
                "autonomous",
            ],
        ),
        (
            "security_privacy",
            [
                "cryptography",
                "cryptographic",
                "cryptanalysis",
                "privacy",
                "privacy-preserving",
                "security",
                "secure",
                "e-voting",
                "cipher",
                "ciphers",
                "side-channel",
                "coding theory",
                "homomorphic encryption",
            ],
        ),
        (
            "learning_methods",
            [
                "neural",
                "graph neural",
                "classification",
                "clustering",
                "anomaly",
                "few-shot",
                "domain adaptation",
                "bayesian",
                "gaussian",
                "gradient",
                "optimization",
                "reinforcement learning",
                "adversarial",
                "reservoir",
                "extreme learning",
                "active learning",
                "stream mining",
                "explainable",
                "xai",
                "imbalanced",
            ],
        ),
        (
            "computing_infrastructure",
            [
                "python",
                " r language",
                "statistical computing",
                "software",
                "programming",
                "type systems",
                "compression",
                "computer science",
                "computational physics",
                "data analysis",
            ],
        ),
    ]
    for family, terms in rules:
        if has_any(text, terms):
            return family
    return "general_ai"


def tokenize(row: pd.Series) -> set[str]:
    text = topic_text(row)
    words = re.findall(r"[a-z0-9][a-z0-9-]{2,}", text)
    return {w for w in words if w not in STOPWORDS and not w.isdigit()}


def prepare_topic_meta(topics: pd.DataFrame) -> pd.DataFrame:
    meta = topics.copy()
    meta["label"] = meta["display_name"]
    meta["family"] = meta.apply(classify_topic, axis=1)
    meta["cluster"] = meta["family"]
    meta["token_set"] = meta.apply(tokenize, axis=1)
    meta["is_core_ai"] = ~meta["family"].isin(["applied_domain_ai"])
    return meta.sort_values("topic_id").reset_index(drop=True)


def semantic_similarity(meta: pd.DataFrame) -> np.ndarray:
    token_sets = meta["token_set"].to_list()
    n = len(token_sets)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                value = 0.0
            else:
                inter = len(a & b)
                union = len(a | b)
                overlap = inter / max(1, min(len(a), len(b)))
                jaccard = inter / max(1, union)
                value = 0.65 * jaccard + 0.35 * overlap
            sim[i, j] = sim[j, i] = value
    return sim


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for neighbor in np.where(adjacency[node] > 0)[0]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        comps.append(comp)
    return comps


def build_openalex_topic_graph(
    train_signal: pd.DataFrame,
    meta: pd.DataFrame,
    top_k: int = 7,
) -> np.ndarray:
    values = train_signal.to_numpy(dtype=float).T
    corr = np.corrcoef(values)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)

    sem = semantic_similarity(meta.set_index("topic_id").loc[train_signal.columns].reset_index())
    families = meta.set_index("topic_id").loc[train_signal.columns, "family"].to_numpy()
    base = 0.76 * corr + 0.24 * sem
    np.fill_diagonal(base, 0.0)

    n = base.shape[0]
    adjacency = np.zeros_like(base)
    for i in range(n):
        neighbors = np.argsort(base[i])[-top_k:]
        for j in neighbors:
            if base[i, j] > 0:
                adjacency[i, j] = base[i, j]
    adjacency = np.maximum(adjacency, adjacency.T)

    for i in range(n):
        for j in range(i + 1, n):
            if families[i] == families[j]:
                adjacency[i, j] = max(adjacency[i, j], 0.10 + 0.14 * sem[i, j])
                adjacency[j, i] = adjacency[i, j]

    # Make the graph connected by linking isolated components through the best
    # available pre-cutoff correlation/semantic edge. A connected graph makes the
    # low-frequency modes easier to interpret.
    while True:
        comps = connected_components(adjacency)
        if len(comps) <= 1:
            break
        main = comps[0]
        other = comps[1]
        best = (0.0, main[0], other[0])
        for i in main:
            for j in other:
                weight = max(base[i, j], 0.05)
                if weight > best[0]:
                    best = (weight, i, j)
        _, i, j = best
        adjacency[i, j] = adjacency[j, i] = max(best[0], 0.05)
    return adjacency


def zscore_by_train(log_share: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    train = log_share.loc[log_share.index <= cutoff]
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0, np.nan).fillna(1.0)
    return (log_share - mean) / std


def compute_energy(signal: pd.DataFrame, adjacency: np.ndarray) -> pd.DataFrame:
    eigvals, eigvecs = graph_fourier_basis(adjacency)
    low_cut = np.quantile(eigvals, 0.33)
    high_cut = np.quantile(eigvals, 0.66)
    rows = []
    for year in signal.index:
        alpha = eigvecs.T @ signal.loc[year].to_numpy(dtype=float)
        energy = alpha**2
        total = float(np.sum(energy))
        probs = energy / total if total > 0 else np.zeros_like(energy)
        entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])) / math.log(len(probs)))
        centroid = float(np.sum(eigvals * probs)) if total > 0 else 0.0
        low = float(np.sum(energy[eigvals <= low_cut]))
        mid = float(np.sum(energy[(eigvals > low_cut) & (eigvals <= high_cut)]))
        high = float(np.sum(energy[eigvals > high_cut]))
        rows.append(
            {
                "year": int(year),
                "spectral_total_energy": total,
                "spectral_low_energy": low,
                "spectral_mid_energy": mid,
                "spectral_high_energy": high,
                "spectral_low_energy_share": low / total if total else 0.0,
                "spectral_mid_energy_share": mid / total if total else 0.0,
                "spectral_high_energy_share": high / total if total else 0.0,
                "spectral_centroid": centroid,
                "spectral_entropy": entropy,
            }
        )
    return pd.DataFrame(rows)


def feature_metrics(score_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    target = score_df[target_col].to_numpy(dtype=float)
    actual_top10 = set(score_df.sort_values(target_col, ascending=False).head(10)["topic_id"])
    rows = []
    for col in FEATURE_LABELS:
        if col not in score_df:
            continue
        ranked = score_df.sort_values(col, ascending=False)["topic_id"].to_list()
        hits = sum(1 for topic_id in ranked[:10] if topic_id in actual_top10)
        precision_sum = 0.0
        seen_hits = 0
        for idx, topic_id in enumerate(ranked, start=1):
            if topic_id in actual_top10:
                seen_hits += 1
                precision_sum += seen_hits / idx
        rows.append(
            {
                "feature": col,
                "feature_label": FEATURE_LABELS.get(col, col),
                "spearman_vs_future_gain": spearman_corr(score_df[col].to_numpy(dtype=float), target),
                "top10_hits": hits,
                "average_precision_actual_top10": precision_sum / max(1, len(actual_top10)),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_vs_future_gain", ascending=False)


def run_cutoff_backtest(
    log_share: pd.DataFrame,
    wide_share: pd.DataFrame,
    meta: pd.DataFrame,
    cutoff: int,
    eval_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    signal = zscore_by_train(log_share, cutoff)
    train_signal = signal.loc[signal.index <= cutoff]
    adjacency = build_openalex_topic_graph(train_signal, meta)
    spectral_scores, _ = spectral_node_scores(signal, adjacency, cutoff)

    future_gain = log_share.loc[eval_year] - log_share.loc[cutoff]
    baseline_growth = (log_share.loc[cutoff] - log_share.loc[cutoff - 3]) / 3.0
    baseline_accel = (log_share.loc[cutoff] - log_share.loc[cutoff - 1]) - (
        log_share.loc[cutoff - 1] - log_share.loc[cutoff - 2]
    )
    score_df = spectral_scores.merge(meta.drop(columns=["token_set"]), on="topic_id", how="left")
    score_df["future_gain"] = score_df["topic_id"].map(future_gain.to_dict())
    score_df["baseline_growth_3y"] = score_df["topic_id"].map(baseline_growth.to_dict())
    score_df["baseline_accel"] = score_df["topic_id"].map(baseline_accel.to_dict())
    score_df["share_per_million_cutoff"] = score_df["topic_id"].map(wide_share.loc[cutoff].to_dict())
    metrics = feature_metrics(score_df, "future_gain")
    metrics["cutoff_year"] = cutoff
    metrics["eval_year"] = eval_year
    metrics["horizon_years"] = eval_year - cutoff
    return score_df, metrics, adjacency


def build_wide_tables(
    counts: pd.DataFrame,
    topic_ids: list[str],
    count_col: str,
    share_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = list(range(YEAR_START, YEAR_END + 1))
    wide_share = (
        counts.pivot(index="year", columns="topic_id", values=share_col)
        .reindex(index=years, columns=topic_ids)
        .fillna(0.0)
    )
    wide_count = (
        counts.pivot(index="year", columns="topic_id", values=count_col)
        .reindex(index=years, columns=topic_ids)
        .fillna(0.0)
    )
    return wide_share, wide_count, np.log1p(wide_share)


def enrich_score_frame(
    score_df: pd.DataFrame,
    log_share: pd.DataFrame,
    wide_share: pd.DataFrame,
    wide_count: pd.DataFrame,
    scope_id: str,
    scope_label: str,
) -> pd.DataFrame:
    score_df = score_df.rename(columns={"future_gain": FUTURE_GAIN_COL}).copy()
    score_df[FUTURE_CONTEXT_COL] = score_df["topic_id"].map(
        (log_share.loc[2023:YEAR_END].mean(axis=0) - log_share.loc[2020:YEAR_CUTOFF].mean(axis=0)).to_dict()
    )
    score_df["count_2022"] = score_df["topic_id"].map(wide_count.loc[YEAR_CUTOFF].to_dict()).astype(int)
    score_df[COUNT_END_COL] = score_df["topic_id"].map(wide_count.loc[YEAR_END].to_dict()).astype(int)
    score_df["share_per_million_2022"] = score_df["topic_id"].map(wide_share.loc[YEAR_CUTOFF].to_dict())
    score_df[SHARE_END_COL] = score_df["topic_id"].map(wide_share.loc[YEAR_END].to_dict())
    score_df[FUTURE_MULTIPLE_COL] = (
        (score_df[SHARE_END_COL] + 0.1) / (score_df["share_per_million_2022"] + 0.1)
    )
    score_df["family_label"] = score_df["family"].map(FAMILY_LABELS)
    score_df["spectral_rank"] = score_df["spectral_emergence_score"].rank(ascending=False, method="min")
    score_df["future_gain_rank"] = score_df[FUTURE_GAIN_COL].rank(ascending=False, method="min")
    score_df["baseline_growth_rank"] = score_df["baseline_growth_3y"].rank(ascending=False, method="min")
    score_df["size_rank"] = score_df["share_per_million_2022"].rank(ascending=False, method="min")
    score_df["count_scope"] = scope_id
    score_df["count_scope_label"] = scope_label
    return score_df.sort_values("spectral_emergence_score", ascending=False)


def add_scope_columns(metrics: pd.DataFrame, scope_id: str, scope_label: str) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["count_scope"] = scope_id
    metrics["count_scope_label"] = scope_label
    return metrics


def run_scope_panel(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    scope_id: str,
) -> dict[str, object]:
    scope = COUNT_SCOPES[scope_id]
    topic_ids = meta["topic_id"].to_list()
    wide_share, wide_count, log_share = build_wide_tables(
        counts,
        topic_ids,
        scope["count_col"],
        scope["share_col"],
    )
    score_df, _, adjacency = run_cutoff_backtest(log_share, wide_share, meta, YEAR_CUTOFF, YEAR_END)
    score_df = enrich_score_frame(score_df, log_share, wide_share, wide_count, scope_id, scope["label"])
    metrics = add_scope_columns(feature_metrics(score_df, FUTURE_GAIN_COL), scope_id, scope["label"])

    rolling_rows = []
    rolling_score_snapshots = []
    for cutoff in range(2000, YEAR_CUTOFF + 1):
        eval_year = cutoff + 3
        cutoff_scores, cutoff_metrics, _ = run_cutoff_backtest(log_share, wide_share, meta, cutoff, eval_year)
        cutoff_scores = cutoff_scores.copy()
        cutoff_scores["cutoff_year"] = cutoff
        cutoff_scores["eval_year"] = eval_year
        cutoff_scores["count_scope"] = scope_id
        cutoff_scores["count_scope_label"] = scope["label"]
        rolling_score_snapshots.append(cutoff_scores)
        rolling_rows.append(add_scope_columns(cutoff_metrics, scope_id, scope["label"]))

    return {
        "wide_share": wide_share,
        "wide_count": wide_count,
        "log_share": log_share,
        "score_df": score_df,
        "metrics": metrics,
        "adjacency": adjacency,
        "rolling": pd.concat(rolling_rows, ignore_index=True),
        "rolling_scores": pd.concat(rolling_score_snapshots, ignore_index=True),
    }


def scope_coverage_tables(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_by_year = (
        counts.groupby("year", as_index=False)
        .agg(
            primary_topic_assignments=("primary_topic_count", "sum"),
            any_topic_assignments=("ai_subfield_any_topic_count", "sum"),
            total_ai_subfield_works=("total_ai_subfield_count", "max"),
        )
        .sort_values("year")
    )
    coverage_by_year["any_to_primary_ratio"] = (
        coverage_by_year["any_topic_assignments"] / coverage_by_year["primary_topic_assignments"].replace(0, np.nan)
    )
    span = coverage_by_year[coverage_by_year["year"].between(YEAR_START, YEAR_END)]
    scope_summary = pd.DataFrame(
        [
            {
                "metric": "topics",
                "value": int(counts["topic_id"].nunique()),
                "note": "AI-subfield topic series",
            },
            {
                "metric": "complete_years",
                "value": int(YEAR_END - YEAR_START + 1),
                "note": f"{YEAR_START}-{YEAR_END}",
            },
            {
                "metric": f"primary_assignments_{YEAR_START}_{YEAR_END}",
                "value": int(span["primary_topic_assignments"].sum()),
                "note": "summed across topic-year cells",
            },
            {
                "metric": f"any_topic_assignments_{YEAR_START}_{YEAR_END}",
                "value": int(span["any_topic_assignments"].sum()),
                "note": "summed across topic-year cells",
            },
            {
                "metric": "any_to_primary_ratio_total",
                "value": float(span["any_topic_assignments"].sum() / span["primary_topic_assignments"].sum()),
                "note": "broader assignment coverage over the full panel",
            },
            {
                "metric": "any_to_primary_ratio_2022",
                "value": float(
                    coverage_by_year.loc[coverage_by_year["year"] == YEAR_CUTOFF, "any_to_primary_ratio"].iloc[0]
                ),
                "note": "broader assignment coverage at the prediction cutoff",
            },
            {
                "metric": f"any_to_primary_ratio_{YEAR_END}",
                "value": float(
                    coverage_by_year.loc[coverage_by_year["year"] == YEAR_END, "any_to_primary_ratio"].iloc[0]
                ),
                "note": "broader assignment coverage at the holdout endpoint",
            },
        ]
    )
    return coverage_by_year, scope_summary


def compare_topic_rank_shifts(primary_scores: pd.DataFrame, any_scores: pd.DataFrame) -> pd.DataFrame:
    left_cols = [
        "topic_id",
        "label",
        "family_label",
        "spectral_rank",
        "future_gain_rank",
        FUTURE_GAIN_COL,
        "count_2022",
        COUNT_END_COL,
        FUTURE_MULTIPLE_COL,
    ]
    merged = primary_scores[left_cols].merge(
        any_scores[left_cols],
        on="topic_id",
        suffixes=("_primary", "_any_topic"),
    )
    merged["label"] = merged["label_primary"]
    merged["family_label"] = merged["family_label_primary"]
    merged["spectral_rank_delta_any_minus_primary"] = (
        merged["spectral_rank_any_topic"] - merged["spectral_rank_primary"]
    )
    merged["future_rank_delta_any_minus_primary"] = (
        merged["future_gain_rank_any_topic"] - merged["future_gain_rank_primary"]
    )
    merged["abs_spectral_rank_delta"] = merged["spectral_rank_delta_any_minus_primary"].abs()
    merged["abs_future_rank_delta"] = merged["future_rank_delta_any_minus_primary"].abs()
    return merged[
        [
            "topic_id",
            "label",
            "family_label",
            "spectral_rank_primary",
            "spectral_rank_any_topic",
            "spectral_rank_delta_any_minus_primary",
            "future_gain_rank_primary",
            "future_gain_rank_any_topic",
            "future_rank_delta_any_minus_primary",
            f"{FUTURE_GAIN_COL}_primary",
            f"{FUTURE_GAIN_COL}_any_topic",
            "count_2022_primary",
            "count_2022_any_topic",
            f"{COUNT_END_COL}_primary",
            f"{COUNT_END_COL}_any_topic",
            f"{FUTURE_MULTIPLE_COL}_primary",
            f"{FUTURE_MULTIPLE_COL}_any_topic",
            "abs_spectral_rank_delta",
            "abs_future_rank_delta",
        ]
    ].sort_values(["abs_spectral_rank_delta", "abs_future_rank_delta"], ascending=False)


def label_fit(text: str, max_len: int = 34) -> str:
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."


def svg_interaction_style() -> str:
    return (
        "<style>"
        ".hover-point,.hover-cell,.hover-series{cursor:crosshair;}"
        ".hover-point:hover{stroke:#111827;stroke-width:2.2;opacity:1;}"
        ".hover-cell:hover{stroke:#111827;stroke-width:1.2;}"
        ".hover-series:hover{stroke-width:4.2;}"
        "</style>"
    )


def draw_prediction_scatter(score_df: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    width, height = 1180, 760
    left, right, top, bottom = 90, 300, 92, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    x_col = "spectral_emergence_score"
    y_col = FUTURE_GAIN_COL
    x = score_df[x_col].to_numpy(dtype=float)
    y = score_df[y_col].to_numpy(dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xpad = (xmax - xmin) * 0.08 or 1.0
    ypad = (ymax - ymin) * 0.12 or 1.0
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    def sx(v: float) -> float:
        return scale(v, (xmin, xmax), (left, left + plot_w))

    def sy(v: float) -> float:
        return scale(v, (ymin, ymax), (top + plot_h, top))

    spearman = metrics.loc[metrics["feature"] == x_col, "spearman_vs_future_gain"].iloc[0]
    highlight_ids = set(score_df.sort_values(y_col, ascending=False).head(10)["topic_id"]) | set(
        score_df.sort_values(x_col, ascending=False).head(8)["topic_id"]
    )
    label_ids = set(score_df.sort_values(y_col, ascending=False).head(5)["topic_id"]) | set(
        score_df.sort_values(x_col, ascending=False).head(4)["topic_id"]
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>',
        svg_interaction_style(),
        f'<text x="{left}" y="38" font-family="Arial" font-size="21" font-weight="700" fill="{INK}">Can Pre-2023 Spectral Momentum Anticipate {EVAL_WINDOW_LABEL} AI Topic Growth?</text>',
        f'<text x="{left}" y="62" font-family="Arial" font-size="12.5" fill="{MUTED}">Each point is one topic time series. Spearman r={spearman:.2f}; hover for names, ranks, counts, and growth.</text>',
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        xv = xmin + frac * (xmax - xmin)
        yv = ymin + frac * (ymax - ymin)
        lines.append(f'<line x1="{sx(xv):.1f}" x2="{sx(xv):.1f}" y1="{top}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'<text x="{sx(xv):.1f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Arial" font-size="11" fill="{MUTED}">{xv:.1f}</text>')
        lines.append(f'<text x="{left - 10}" y="{sy(yv) + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{MUTED}">{yv:.1f}</text>')
    if xmin < 0 < xmax:
        lines.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2" opacity="0.45"/>')
    if ymin < 0 < ymax:
        lines.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#111827" stroke-width="1.2" opacity="0.45"/>')

    for row in score_df.sort_values("count_2022").itertuples():
        px, py = sx(float(getattr(row, x_col))), sy(float(getattr(row, y_col)))
        color = FAMILY_COLORS.get(row.family, FAMILY_COLORS["general_ai"])
        r = 4.5 + min(9.5, math.sqrt(max(float(row.count_2022), 0.0)) * 0.035)
        opacity = 0.90 if row.topic_id in highlight_ids else 0.52
        tooltip = (
            f"{row.label}\n"
            f"Family: {row.family_label}\n"
            f"2022 count: {int(row.count_2022):,}; {EVAL_END_LABEL} count: {int(getattr(row, COUNT_END_COL)):,}\n"
            f"Spectral score: {float(getattr(row, x_col)):.2f}\n"
            f"Future growth: {float(getattr(row, y_col)):.2f}\n"
            f"Spectral rank: {float(row.spectral_rank):.0f}; future rank: {float(row.future_gain_rank):.0f}"
        )
        lines.append(f'<circle class="hover-point" data-tooltip="{tooltip_attr(tooltip)}" cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="{color}" opacity="{opacity:.2f}" stroke="#ffffff" stroke-width="0.8"><title>{esc(tooltip)}</title></circle>')

    placed: list[tuple[float, float, float, float]] = []
    for row in score_df[score_df["topic_id"].isin(label_ids)].sort_values(y_col, ascending=False).itertuples():
        px, py = sx(float(getattr(row, x_col))), sy(float(getattr(row, y_col)))
        text = label_fit(row.label, 30)
        w = max(70, len(text) * 6.0)
        h = 18
        candidates = [(9, -10, "start"), (9, 18, "start"), (-9, -10, "end"), (-9, 18, "end")]
        best = None
        for dx, dy, anchor in candidates:
            tx = px + dx
            ty = py + dy
            x0 = tx if anchor == "start" else tx - w
            y0 = ty - h + 3
            x1, y1 = x0 + w, y0 + h
            penalty = 0
            if x0 < left or x1 > left + plot_w:
                penalty += 10
            if y0 < top or y1 > top + plot_h:
                penalty += 10
            for bx0, by0, bx1, by1 in placed:
                if not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1):
                    penalty += 1
            if best is None or penalty < best[0]:
                best = (penalty, tx, ty, anchor, x0, y0, x1, y1)
        _, tx, ty, anchor, x0, y0, x1, y1 = best
        placed.append((x0, y0, x1, y1))
        lines.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="11" font-weight="700" fill="{INK}">{esc(text)}</text>')

    legend_x = left + plot_w + 32
    legend_y = top - 2
    lines.append(f'<rect x="{legend_x - 14}" y="{legend_y - 16}" width="246" height="270" rx="0" fill="#ffffff" stroke="#d9dee8" stroke-width="1"/>')
    lines.append(f'<text x="{legend_x}" y="{top + 8}" font-family="Arial" font-size="12.5" font-weight="700" fill="{INK}">Topic families</text>')
    for idx, (family, label) in enumerate(FAMILY_LABELS.items()):
        y0 = top + 34 + idx * 28
        lines.append(f'<circle cx="{legend_x + 6}" cy="{y0 - 4}" r="5.5" fill="{FAMILY_COLORS[family]}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{y0}" font-family="Arial" font-size="11.2" fill="{MUTED}">{esc(label)}</text>')
    lines.extend(
        [
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 30}" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">2022 graph Fourier emergence score</text>',
            f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Log growth in primary-topic share, 2022 to {EVAL_END_LABEL}</text>',
            "</svg>",
        ]
    )
    write_text(out, "\n".join(lines))


def draw_emergence_heatmap(
    wide_share: pd.DataFrame,
    score_df: pd.DataFrame,
    out: Path,
    n_topics: int = 30,
) -> None:
    years = list(range(YEAR_START, YEAR_END + 1))
    selected = score_df.sort_values(FUTURE_GAIN_COL, ascending=False).head(n_topics)
    row_h, col_w = 21, 25
    left, right, top, bottom = 310, 230, 96, 76
    plot_w, plot_h = col_w * len(years), row_h * len(selected)
    width, height = left + plot_w + right, top + plot_h + bottom
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>',
        svg_interaction_style(),
        f'<text x="{left}" y="38" font-family="Arial" font-size="21" font-weight="700" fill="{INK}">Emergence Heatmap: Topics That Grew Most After The 2022 Cutoff</text>',
        f'<text x="{left}" y="62" font-family="Arial" font-size="12.5" fill="{MUTED}">Rows are topic histories; color is row-normalized log share, so each row shows when that topic became intense relative to its own past.</text>',
    ]
    for yi, year in enumerate(years):
        x = left + yi * col_w
        if year in [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2022, 2025, 2026]:
            lines.append(f'<text x="{x + col_w / 2:.1f}" y="{top - 14}" text-anchor="middle" font-family="Arial" font-size="10.5" fill="{MUTED}">{year}</text>')
    cutoff_x = left + (YEAR_CUTOFF - YEAR_START) * col_w + col_w / 2
    lines.append(f'<line x1="{cutoff_x:.1f}" x2="{cutoff_x:.1f}" y1="{top - 8}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.3" stroke-dasharray="4 4" opacity="0.70"/>')
    lines.append(f'<text x="{cutoff_x + 6:.1f}" y="{top + plot_h + 26}" font-family="Arial" font-size="11" fill="#111827">2022 cutoff</text>')
    metric_x = left + plot_w + 12
    lines.append(f'<rect x="{metric_x - 10}" y="{top - 30}" width="204" height="{plot_h + 42}" rx="0" fill="#ffffff" stroke="#d9dee8" stroke-width="1" opacity="0.92"/>')
    lines.append(f'<text x="{left + plot_w + 16}" y="{top - 14}" font-family="Arial" font-size="10.5" font-weight="700" fill="{INK}">2022->{EVAL_END_LABEL} share/million</text>')

    for ri, row in enumerate(selected.itertuples()):
        y = top + ri * row_h
        series = np.log1p(wide_share.loc[years, row.topic_id].to_numpy(dtype=float))
        lo, hi = float(np.min(series)), float(np.max(series))
        denom = hi - lo if hi > lo else 1.0
        family_color = FAMILY_COLORS.get(row.family, FAMILY_COLORS["general_ai"])
        lines.append(f'<rect x="{left - 14}" y="{y + 3}" width="5" height="{row_h - 5}" fill="{family_color}"/>')
        label_lines = wrap_label(row.label, width=36, max_lines=1)
        lines.append(f'<text x="{left - 22}" y="{y + 15}" text-anchor="end" font-family="Arial" font-size="10.8" fill="{INK}">{esc(label_lines[0])}</text>')
        raw_share = wide_share.loc[years, row.topic_id].to_numpy(dtype=float)
        for yi, value in enumerate(series):
            x = left + yi * col_w
            year = years[yi]
            color = heat_color((float(value) - lo) / denom)
            intensity = (float(value) - lo) / denom
            tooltip = (
                f"{row.label}\n"
                f"Year: {year}\n"
                f"Attention share: {float(raw_share[yi]):.1f} per million AI works\n"
                f"Within-row intensity: {intensity:.2f}\n"
                f"2022->{EVAL_END_LABEL} multiple: {getattr(row, FUTURE_MULTIPLE_COL):.1f}x"
            )
            lines.append(f'<rect class="hover-cell" data-tooltip="{tooltip_attr(tooltip)}" x="{x}" y="{y}" width="{col_w - 1}" height="{row_h - 1}" fill="{color}"><title>{esc(tooltip)}</title></rect>')
        metric = f"{row.share_per_million_2022:.0f}->{getattr(row, SHARE_END_COL):.0f}; {getattr(row, FUTURE_MULTIPLE_COL):.1f}x"
        lines.append(f'<text x="{left + plot_w + 16}" y="{y + 15}" font-family="Arial" font-size="10.8" fill="{MUTED}">{esc(metric)}</text>')

    # Color scale.
    scale_x, scale_y = left + plot_w - 212, top + plot_h + 42
    for i in range(80):
        lines.append(f'<rect x="{scale_x + i * 2}" y="{scale_y}" width="2" height="10" fill="{heat_color(i / 79)}"/>')
    lines.append(f'<text x="{scale_x}" y="{scale_y + 28}" font-family="Arial" font-size="10.5" fill="{MUTED}">low within-row intensity</text>')
    lines.append(f'<text x="{scale_x + 160}" y="{scale_y + 28}" text-anchor="end" font-family="Arial" font-size="10.5" fill="{MUTED}">high</text>')
    lines.append("</svg>")
    write_text(out, "\n".join(lines))


def draw_topic_atlas(score_df: pd.DataFrame, adjacency: pd.DataFrame, out: Path) -> None:
    topic_order = adjacency.index.to_list()
    eigvals, eigvecs = graph_fourier_basis(adjacency.to_numpy(dtype=float))
    coords = pd.DataFrame(
        {
            "topic_id": topic_order,
            "x": eigvecs[:, 1],
            "y": eigvecs[:, 2] if eigvecs.shape[1] > 2 else eigvecs[:, 1],
        }
    )
    df = score_df.merge(coords, on="topic_id", how="left")
    width, height = 1360, 900
    left, right, top, bottom = 82, 280, 94, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    xpad = max(0.01, (df["x"].max() - df["x"].min()) * 0.13)
    ypad = max(0.01, (df["y"].max() - df["y"].min()) * 0.13)
    xdom = (float(df["x"].min() - xpad), float(df["x"].max() + xpad))
    ydom = (float(df["y"].min() - ypad), float(df["y"].max() + ypad))

    def sx(v: float) -> float:
        return scale(v, xdom, (left, left + plot_w))

    def sy(v: float) -> float:
        return scale(v, ydom, (top + plot_h, top))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>',
        svg_interaction_style(),
        f'<text x="{left}" y="38" font-family="Arial" font-size="21" font-weight="700" fill="{INK}">A Fourier Atlas Of The AI Topic Graph</text>',
        f'<text x="{left}" y="62" font-family="Arial" font-size="12.5" fill="{MUTED}">Position uses the first two non-constant graph Laplacian eigenvectors from pre-2023 topic motion. Larger nodes grew more from 2022 to {EVAL_END_LABEL}; dark outlines mark top-decile 2022 spectral emergence.</text>',
    ]
    weights = adjacency.to_numpy(dtype=float)
    nonzero = weights[weights > 0]
    threshold = float(np.quantile(nonzero, 0.80)) if len(nonzero) else 0.0
    coord_map = df.set_index("topic_id")[["x", "y"]].to_dict("index")
    for i, src in enumerate(topic_order):
        for j in range(i + 1, len(topic_order)):
            w = weights[i, j]
            if w < threshold:
                continue
            dst = topic_order[j]
            x1, y1 = coord_map[src]["x"], coord_map[src]["y"]
            x2, y2 = coord_map[dst]["x"], coord_map[dst]["y"]
            lines.append(f'<line x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" x2="{sx(x2):.1f}" y2="{sy(y2):.1f}" stroke="#d5dae4" stroke-width="{0.6 + 2.0 * w:.2f}" opacity="0.48"/>')

    max_gain = max(0.01, float(df[FUTURE_GAIN_COL].max()))
    outline_cut = float(df["spectral_emergence_score"].quantile(0.90))
    for row in df.sort_values(FUTURE_GAIN_COL, ascending=True).itertuples():
        x, y = sx(float(row.x)), sy(float(row.y))
        gain = max(0.0, float(getattr(row, FUTURE_GAIN_COL)))
        r = 5.0 + 18.0 * math.sqrt(gain / max_gain)
        color = FAMILY_COLORS.get(row.family, FAMILY_COLORS["general_ai"])
        stroke = "#111827" if row.spectral_emergence_score >= outline_cut else "#ffffff"
        stroke_w = 2.5 if row.spectral_emergence_score >= outline_cut else 0.9
        tooltip = (
            f"{row.label}\n"
            f"Family: {row.family_label}\n"
            f"Future growth rank: {float(row.future_gain_rank):.0f}\n"
            f"Spectral emergence rank: {float(row.spectral_rank):.0f}\n"
            f"2022->{EVAL_END_LABEL} share multiple: {getattr(row, FUTURE_MULTIPLE_COL):.1f}x\n"
            f"Graph coordinates: ({float(row.x):.3f}, {float(row.y):.3f})"
        )
        lines.append(f'<circle class="hover-point" data-tooltip="{tooltip_attr(tooltip)}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.76" stroke="{stroke}" stroke-width="{stroke_w}"><title>{esc(tooltip)}</title></circle>')

    label_topics = set(df.sort_values(FUTURE_GAIN_COL, ascending=False).head(8)["topic_id"])
    label_topics |= {"T10456", "T10906", "T10181", "T12031"}
    placed: list[tuple[float, float, float, float]] = []
    center_x, center_y = left + plot_w / 2, top + plot_h / 2
    for row in df[df["topic_id"].isin(label_topics)].sort_values(FUTURE_GAIN_COL, ascending=False).itertuples():
        px, py = sx(float(row.x)), sy(float(row.y))
        text = label_fit(row.label, 32)
        w = max(85, len(text) * 6.0)
        h = 18
        outward_x = 1 if px >= center_x else -1
        outward_y = 1 if py >= center_y else -1
        candidates = [
            (outward_x * 12, outward_y * 20, "start" if outward_x > 0 else "end"),
            (outward_x * 12, -outward_y * 20, "start" if outward_x > 0 else "end"),
            (-outward_x * 12, outward_y * 20, "start" if outward_x < 0 else "end"),
            (16, -18, "start"),
            (-16, -18, "end"),
            (16, 24, "start"),
            (-16, 24, "end"),
        ]
        special_candidates = {
            "T12026": [(18, 40, "start")],  # XAI sits inside a dense application-methods cluster.
            "T12128": [(18, -30, "start")],
            "T13702": [(-18, -32, "end")],
        }
        candidates = special_candidates.get(row.topic_id, []) + candidates
        best = None
        for dx, dy, anchor in candidates:
            tx, ty = px + dx, py + dy
            x0 = tx if anchor == "start" else tx - w
            y0 = ty - h + 3
            x1, y1 = x0 + w, y0 + h
            penalty = 0
            if x0 < left or x1 > left + plot_w or y0 < top or y1 > top + plot_h:
                penalty += 10
            for bx0, by0, bx1, by1 in placed:
                if not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1):
                    penalty += 1
            if best is None or penalty < best[0]:
                best = (penalty, tx, ty, anchor, x0, y0, x1, y1)
        _, tx, ty, anchor, x0, y0, x1, y1 = best
        placed.append((x0, y0, x1, y1))
        lines.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{tx:.1f}" y2="{ty - 5:.1f}" stroke="#98a2b3" stroke-width="0.8" opacity="0.65"/>')
        lines.append(f'<rect x="{x0 - 2:.1f}" y="{y0 - 2:.1f}" width="{w + 4:.1f}" height="{h + 3:.1f}" fill="#ffffff" opacity="0.84"/>')
        lines.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="11" font-weight="700" fill="{INK}">{esc(text)}</text>')

    legend_x = left + plot_w + 34
    lines.append(f'<rect x="{legend_x - 14}" y="{top - 14}" width="240" height="426" rx="0" fill="#ffffff" stroke="#d9dee8" stroke-width="1"/>')
    lines.append(f'<text x="{legend_x}" y="{top + 10}" font-family="Arial" font-size="12.5" font-weight="700" fill="{INK}">Families</text>')
    for idx, (family, label) in enumerate(FAMILY_LABELS.items()):
        y0 = top + 36 + idx * 27
        lines.append(f'<circle cx="{legend_x + 7}" cy="{y0 - 4}" r="5.5" fill="{FAMILY_COLORS[family]}"/>')
        lines.append(f'<text x="{legend_x + 20}" y="{y0}" font-family="Arial" font-size="11.2" fill="{MUTED}">{esc(label)}</text>')
    lines.append(f'<text x="{legend_x}" y="{top + 300}" font-family="Arial" font-size="12.5" font-weight="700" fill="{INK}">Reading guide</text>')
    guide = [
        "Near nodes moved similarly before 2023.",
        "Large nodes accelerated after 2022.",
        "Outlined nodes were spectral early signals.",
        "Edges show strongest pre-2023 ties.",
    ]
    for idx, text in enumerate(guide):
        lines.append(f'<text x="{legend_x}" y="{top + 326 + idx * 22}" font-family="Arial" font-size="11.2" fill="{MUTED}">{esc(text)}</text>')
    lines.append(f'<text x="{left + plot_w / 2}" y="{height - 34}" text-anchor="middle" font-family="Arial" font-size="11.5" fill="{MUTED}">Axes are graph Fourier coordinates, not ordinary semantic dimensions.</text>')
    lines.append("</svg>")
    write_text(out, "\n".join(lines))


def draw_rolling_backtest(rolling: pd.DataFrame, out: Path) -> None:
    width, height = 1180, 680
    left, right, top, bottom = 82, 300, 88, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    features = [
        ("spectral_emergence_score", "#b25d31"),
        ("spectral_midband_momentum", "#3b6ea8"),
        ("baseline_growth_3y", "#2a9d8f"),
        ("raw_acceleration_1y", "#7a5195"),
        ("share_per_million_cutoff", "#5c677d"),
    ]
    years = sorted(rolling["cutoff_year"].unique())
    yvals = rolling["spearman_vs_future_gain"].to_numpy(dtype=float)
    ymin, ymax = min(-0.4, float(np.nanmin(yvals)) - 0.08), max(0.8, float(np.nanmax(yvals)) + 0.08)

    def sx(year: int) -> float:
        return scale(year, (min(years), max(years)), (left, left + plot_w))

    def sy(value: float) -> float:
        return scale(value, (ymin, ymax), (top + plot_h, top))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>',
        svg_interaction_style(),
        f'<text x="{left}" y="38" font-family="Arial" font-size="21" font-weight="700" fill="{INK}">Rolling Three-Year Backtest: Is The Signal Stable Before 2022?</text>',
        f'<text x="{left}" y="62" font-family="Arial" font-size="12.5" fill="{MUTED}">For each cutoff, the graph is rebuilt using only earlier years, then features are ranked against the next three years of topic-share growth.</text>',
    ]
    for tick in [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]:
        if tick < ymin or tick > ymax:
            continue
        y = sy(tick)
        stroke = "#111827" if tick == 0 else GRID
        opacity = 0.50 if tick == 0 else 1.0
        lines.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1" opacity="{opacity}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{MUTED}">{tick:.1f}</text>')
    for year in range(min(years), max(years) + 1, 2):
        x = sx(year)
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="#f1f3f8" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Arial" font-size="10.5" fill="{MUTED}">{year}</text>')

    summary = []
    for feature, color in features:
        rows = rolling[rolling["feature"] == feature].sort_values("cutoff_year")
        points = [(sx(int(r.cutoff_year)), sy(float(r.spearman_vs_future_gain))) for r in rows.itertuples()]
        lines.append(f'<polyline class="hover-series" points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>')
        for r, (x, y) in zip(rows.itertuples(), points):
            tooltip = (
                f"{FEATURE_LABELS[feature]}\n"
                f"Cutoff: {int(r.cutoff_year)}; evaluated through {int(r.eval_year)}\n"
                f"Spearman vs future growth: {float(r.spearman_vs_future_gain):.2f}\n"
                f"Top-10 future growers captured: {int(r.top10_hits)}/10\n"
                f"Average precision: {float(r.average_precision_actual_top10):.2f}"
            )
            lines.append(f'<circle class="hover-point" data-tooltip="{tooltip_attr(tooltip)}" cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}" opacity="0.88"><title>{esc(tooltip)}</title></circle>')
        summary.append((feature, color, float(rows["spearman_vs_future_gain"].mean()), float(rows["top10_hits"].mean())))

    legend_x = left + plot_w + 32
    lines.append(f'<rect x="{legend_x - 14}" y="{top - 16}" width="268" height="270" rx="0" fill="#ffffff" stroke="#d9dee8" stroke-width="1"/>')
    lines.append(f'<text x="{legend_x}" y="{top + 8}" font-family="Arial" font-size="12.5" font-weight="700" fill="{INK}">Feature comparison</text>')
    for idx, (feature, color, avg_corr, avg_hits) in enumerate(summary):
        y0 = top + 36 + idx * 45
        lines.append(f'<line x1="{legend_x}" x2="{legend_x + 24}" y1="{y0 - 4}" y2="{y0 - 4}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        lines.append(f'<text x="{legend_x + 34}" y="{y0}" font-family="Arial" font-size="11.2" font-weight="700" fill="{INK}">{esc(FEATURE_LABELS[feature])}</text>')
        lines.append(f'<text x="{legend_x + 34}" y="{y0 + 17}" font-family="Arial" font-size="10.5" fill="{MUTED}">mean r={avg_corr:.2f}; top-10 hits={avg_hits:.1f}</text>')
    lines.extend(
        [
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Cutoff year</text>',
            f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Spearman rank correlation vs next-three-year growth</text>',
            "</svg>",
        ]
    )
    write_text(out, "\n".join(lines))


def draw_spectral_energy_stack(energy: pd.DataFrame, out: Path) -> None:
    width, height = 1180, 700
    left, right, top, bottom = 82, 300, 88, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    years = energy["year"].astype(int).to_list()
    bands = [
        ("spectral_low_energy_share", "Low frequency: field-wide motion", "#3b6ea8"),
        ("spectral_mid_energy_share", "Mid frequency: family-level transfer", "#d18f1f"),
        ("spectral_high_energy_share", "High frequency: localized novelty", "#cc6677"),
    ]

    def sx(year: int) -> float:
        return scale(year, (min(years), max(years)), (left, left + plot_w))

    def sy(value: float) -> float:
        return scale(value, (0.0, 1.0), (top + plot_h, top))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>',
        svg_interaction_style(),
        f'<text x="{left}" y="38" font-family="Arial" font-size="21" font-weight="700" fill="{INK}">Spectral Energy Transfer Across The AI Topic Graph</text>',
        f'<text x="{left}" y="62" font-family="Arial" font-size="12.5" fill="{MUTED}">A fixed pre-2023 graph Fourier basis decomposes each year into broad, mid-scale, and localized topic motion; the black line tracks spectral entropy.</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{MUTED}">{tick:.2f}</text>')
    for year in range(min(years), max(years) + 1, 5):
        x = sx(year)
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Arial" font-size="10.5" fill="{MUTED}">{year}</text>')

    cumulative = np.zeros(len(energy), dtype=float)
    xs = [sx(y) for y in years]
    for col, _, color in bands:
        values = energy[col].to_numpy(dtype=float)
        lower = cumulative.copy()
        upper = cumulative + values
        upper_points = [(xs[i], sy(float(upper[i]))) for i in range(len(xs))]
        lower_points = [(xs[i], sy(float(lower[i]))) for i in range(len(xs) - 1, -1, -1)]
        lines.append(f'<polygon points="{polygon(upper_points + lower_points)}" fill="{color}" opacity="0.82"/>')
        cumulative = upper

    entropy = energy["spectral_entropy"].to_numpy(dtype=float)
    entropy_points = [(xs[i], sy(float(entropy[i]))) for i in range(len(xs))]
    lines.append(f'<polyline points="{polyline(entropy_points)}" fill="none" stroke="#111827" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
    cutoff_x = sx(YEAR_CUTOFF)
    lines.append(f'<line x1="{cutoff_x:.1f}" x2="{cutoff_x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-dasharray="4 4" opacity="0.55"/>')
    lines.append(f'<text x="{cutoff_x + 8:.1f}" y="{top + 16}" font-family="Arial" font-size="11" fill="#111827">2022 cutoff</text>')
    step = xs[1] - xs[0] if len(xs) > 1 else 12
    for i, row in enumerate(energy.itertuples()):
        tooltip = (
            f"{int(row.year)}\n"
            f"Low-frequency energy: {float(row.spectral_low_energy_share):.0%}\n"
            f"Mid-frequency energy: {float(row.spectral_mid_energy_share):.0%}\n"
            f"High-frequency energy: {float(row.spectral_high_energy_share):.0%}\n"
            f"Spectral entropy: {float(row.spectral_entropy):.2f}"
        )
        x0 = max(left, xs[i] - step / 2)
        x1 = min(left + plot_w, xs[i] + step / 2)
        lines.append(f'<rect class="hover-cell" data-tooltip="{tooltip_attr(tooltip)}" x="{x0:.1f}" y="{top}" width="{x1 - x0:.1f}" height="{plot_h}" fill="#ffffff" opacity="0" pointer-events="all"><title>{esc(tooltip)}</title></rect>')

    legend_x = left + plot_w + 32
    lines.append(f'<rect x="{legend_x - 14}" y="{top - 16}" width="268" height="196" rx="0" fill="#ffffff" stroke="#d9dee8" stroke-width="1"/>')
    lines.append(f'<text x="{legend_x}" y="{top + 8}" font-family="Arial" font-size="12.5" font-weight="700" fill="{INK}">Bands</text>')
    for idx, (_, label, color) in enumerate(bands):
        y0 = top + 38 + idx * 36
        lines.append(f'<rect x="{legend_x}" y="{y0 - 14}" width="23" height="14" fill="{color}" opacity="0.82"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y0 - 3}" font-family="Arial" font-size="11.2" fill="{MUTED}">{esc(label)}</text>')
    lines.append(f'<line x1="{legend_x}" x2="{legend_x + 23}" y1="{top + 152}" y2="{top + 152}" stroke="#111827" stroke-width="2.4"/>')
    lines.append(f'<text x="{legend_x + 32}" y="{top + 156}" font-family="Arial" font-size="11.2" fill="{MUTED}">Spectral entropy</text>')
    lines.extend(
        [
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Year</text>',
            f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Share of spectral energy / entropy</text>',
            "</svg>",
        ]
    )
    write_text(out, "\n".join(lines))


def draw_scope_robustness(
    primary_metrics: pd.DataFrame,
    any_metrics: pd.DataFrame,
    primary_rolling: pd.DataFrame,
    any_rolling: pd.DataFrame,
    coverage_by_year: pd.DataFrame,
    out: Path,
) -> None:
    features = [
        "spectral_emergence_score",
        "raw_acceleration_1y",
        "baseline_growth_3y",
        "spectral_midband_momentum",
        "share_per_million_cutoff",
    ]
    scope_colors = {
        "primary": "#3b6ea8",
        "any_topic": "#b25d31",
    }
    metric_frames = [
        primary_metrics.assign(panel_scope="primary"),
        any_metrics.assign(panel_scope="any_topic"),
    ]
    holdout = pd.concat(metric_frames, ignore_index=True)
    rolling = (
        pd.concat(
            [
                primary_rolling.assign(panel_scope="primary"),
                any_rolling.assign(panel_scope="any_topic"),
            ],
            ignore_index=True,
        )
        .groupby(["panel_scope", "feature", "feature_label"], as_index=False)
        .agg(
            spearman_vs_future_gain=("spearman_vs_future_gain", "mean"),
            top10_hits=("top10_hits", "mean"),
            average_precision_actual_top10=("average_precision_actual_top10", "mean"),
        )
    )
    all_values = pd.concat(
        [
            holdout[holdout["feature"].isin(features)]["spearman_vs_future_gain"],
            rolling[rolling["feature"].isin(features)]["spearman_vs_future_gain"],
        ],
        ignore_index=True,
    )
    ymin = min(-0.35, float(all_values.min()) - 0.08)
    ymax = max(0.75, float(all_values.max()) + 0.08)
    width, height = 1180, 760
    left, right, top, bottom = 86, 300, 92, 84
    plot_w, panel_h, panel_gap = width - left - right, 218, 126

    def sy(value: float, panel_top: float) -> float:
        return scale(value, (ymin, ymax), (panel_top + panel_h, panel_top))

    def yearly(year: int, col: str) -> float:
        return float(coverage_by_year.loc[coverage_by_year["year"] == year, col].iloc[0])

    year_2022_primary = yearly(YEAR_CUTOFF, "primary_topic_assignments")
    year_2022_any = yearly(YEAR_CUTOFF, "any_topic_assignments")
    year_end_primary = yearly(YEAR_END, "primary_topic_assignments")
    year_end_any = yearly(YEAR_END, "any_topic_assignments")
    ratio_2022 = year_2022_any / year_2022_primary
    ratio_end = year_end_any / year_end_primary

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>',
        svg_interaction_style(),
        f'<text x="{left}" y="38" font-family="Arial" font-size="21" font-weight="700" fill="{INK}">Coverage Robustness: Clean Topic Counts Versus Wider Topic Assignments</text>',
        f'<text x="{left}" y="62" font-family="Arial" font-size="12.5" fill="{MUTED}">Primary-topic counts are the main signal; any-topic counts widen recall inside the same AI corpus. The question is whether the {TARGET_WINDOW_LABEL} result survives that wider measurement.</text>',
    ]

    def draw_panel(panel_top: float, title: str, df: pd.DataFrame, metric_label: str) -> None:
        zero_y = sy(0.0, panel_top)
        lines.append(f'<text x="{left}" y="{panel_top - 18}" font-family="Arial" font-size="14" font-weight="700" fill="{INK}">{esc(title)}</text>')
        for tick in [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]:
            if tick < ymin or tick > ymax:
                continue
            y = sy(tick, panel_top)
            stroke = "#111827" if tick == 0 else GRID
            opacity = 0.52 if tick == 0 else 1.0
            lines.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1" opacity="{opacity}"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10.5" fill="{MUTED}">{tick:.1f}</text>')
        group_w = plot_w / len(features)
        for idx, feature in enumerate(features):
            cx = left + group_w * (idx + 0.5)
            for scope_id, dx in [("primary", -12), ("any_topic", 12)]:
                row = df[(df["panel_scope"] == scope_id) & (df["feature"] == feature)].iloc[0]
                value = float(row["spearman_vs_future_gain"])
                y = sy(value, panel_top)
                bar_y = min(y, zero_y)
                bar_h = max(2.0, abs(zero_y - y))
                color = scope_colors[scope_id]
                tooltip = (
                    f"{COUNT_SCOPES[scope_id]['label']}\n"
                    f"{FEATURE_LABELS[feature]}\n"
                    f"{metric_label}: {value:.2f}\n"
                    f"Top-10 hits: {float(row['top10_hits']):.1f}/10\n"
                    f"Average precision: {float(row['average_precision_actual_top10']):.2f}"
                )
                lines.append(f'<rect class="hover-cell" data-tooltip="{tooltip_attr(tooltip)}" x="{cx + dx - 7:.1f}" y="{bar_y:.1f}" width="14" height="{bar_h:.1f}" fill="{color}" opacity="0.88"><title>{esc(tooltip)}</title></rect>')
                lines.append(f'<circle class="hover-point" data-tooltip="{tooltip_attr(tooltip)}" cx="{cx + dx:.1f}" cy="{y:.1f}" r="4.1" fill="#ffffff" stroke="{color}" stroke-width="2"><title>{esc(tooltip)}</title></circle>')
            wrapped = wrap_label(FEATURE_LABELS[feature], width=18, max_lines=2)
            for li, label in enumerate(wrapped):
                lines.append(f'<text x="{cx:.1f}" y="{panel_top + panel_h + 24 + li * 13}" text-anchor="middle" font-family="Arial" font-size="10.4" fill="{MUTED}">{esc(label)}</text>')
        lines.append(f'<line x1="{left}" x2="{left}" y1="{panel_top}" y2="{panel_top + panel_h}" stroke="#98a2b3"/>')
        lines.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{panel_top + panel_h}" y2="{panel_top + panel_h}" stroke="#98a2b3"/>')

    draw_panel(top, f"A. 2022 holdout: Spearman versus {TARGET_WINDOW_LABEL} growth", holdout, "Holdout Spearman")
    draw_panel(top + panel_h + panel_gap, "B. Rolling 2000-2022: mean Spearman over three-year horizons", rolling, "Mean rolling Spearman")

    legend_x = left + plot_w + 36
    lines.append(f'<rect x="{legend_x - 14}" y="{top - 16}" width="248" height="386" rx="0" fill="#ffffff" stroke="#d9dee8" stroke-width="1"/>')
    lines.append(f'<text x="{legend_x}" y="{top + 10}" font-family="Arial" font-size="12.5" font-weight="700" fill="{INK}">Scopes</text>')
    for idx, scope_id in enumerate(["primary", "any_topic"]):
        y0 = top + 42 + idx * 48
        lines.append(f'<rect x="{legend_x}" y="{y0 - 15}" width="24" height="14" fill="{scope_colors[scope_id]}" opacity="0.88"/>')
        lines.append(f'<text x="{legend_x + 34}" y="{y0 - 4}" font-family="Arial" font-size="11.5" font-weight="700" fill="{INK}">{esc(COUNT_SCOPES[scope_id]["label"])}</text>')
        lines.append(f'<text x="{legend_x + 34}" y="{y0 + 13}" font-family="Arial" font-size="10.6" fill="{MUTED}">{esc(COUNT_SCOPES[scope_id]["description"])}</text>')
    lines.append(f'<text x="{legend_x}" y="{top + 158}" font-family="Arial" font-size="12.5" font-weight="700" fill="{INK}">Coverage change</text>')
    coverage_lines = [
        f"2022 assignments: {int(year_2022_primary):,} -> {int(year_2022_any):,}",
        f"2022 wider-scope ratio: {ratio_2022:.2f}x",
        f"{EVAL_END_LABEL} assignments: {int(year_end_primary):,} -> {int(year_end_any):,}",
        f"{EVAL_END_LABEL} wider-scope ratio: {ratio_end:.2f}x",
        "Same 77 topics and same AI-year denominator.",
    ]
    for idx, text in enumerate(coverage_lines):
        y0 = top + 184 + idx * 28
        lines.append(f'<text x="{legend_x}" y="{y0}" font-family="Arial" font-size="11.1" fill="{MUTED}">{esc(text)}</text>')
    lines.append(f'<text x="24" y="{top + 274}" transform="rotate(-90 24 {top + 274})" text-anchor="middle" font-family="Arial" font-size="12" fill="{INK}">Spearman rank correlation</text>')
    lines.append("</svg>")
    write_text(out, "\n".join(lines))


def write_research_questions(
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    any_metrics: pd.DataFrame | None = None,
    any_rolling: pd.DataFrame | None = None,
    scope_summary: pd.DataFrame | None = None,
) -> None:
    best = metrics.iloc[0]
    spectral = metrics.loc[metrics["feature"] == "spectral_emergence_score"].iloc[0]
    baseline = metrics.loc[metrics["feature"] == "baseline_growth_3y"].iloc[0]
    avg = rolling.groupby("feature")["spearman_vs_future_gain"].mean().sort_values(ascending=False)
    if any_metrics is not None and any_rolling is not None:
        any_spectral = any_metrics.loc[any_metrics["feature"] == "spectral_emergence_score"].iloc[0]
        any_baseline = any_metrics.loc[any_metrics["feature"] == "baseline_growth_3y"].iloc[0]
        any_avg = any_rolling.groupby("feature")["spearman_vs_future_gain"].mean().sort_values(ascending=False)
        ratio_2022 = (
            float(scope_summary.loc[scope_summary["metric"] == "any_to_primary_ratio_2022", "value"].iloc[0])
            if scope_summary is not None
            else float("nan")
        )
        robustness_text = f"""
## Coverage Robustness

The expanded analysis now tests both count scopes. The primary-topic scope remains
the main measurement because each work contributes one topic assignment. The wider
any-topic scope counts secondary topic assignments inside the same AI-subfield
corpus; at the 2022 cutoff it contains {ratio_2022:.2f} times as many topic
assignments.

- Primary-topic spectral emergence: Spearman {spectral['spearman_vs_future_gain']:.3f};
  growth baseline {baseline['spearman_vs_future_gain']:.3f}.
- Any-topic spectral emergence: Spearman {any_spectral['spearman_vs_future_gain']:.3f};
  growth baseline {any_baseline['spearman_vs_future_gain']:.3f}.
- Best mean rolling primary feature: `{avg.index[0]}` at {avg.iloc[0]:.3f}.
- Best mean rolling any-topic feature: `{any_avg.index[0]}` at {any_avg.iloc[0]:.3f}.

The wider scope is therefore used as a sensitivity test, not as a replacement for
the cleaner primary-topic signal.
"""
    else:
        robustness_text = ""
    text = f"""# Research Question Map

The study is framed as a sequence of falsifiable questions rather than a single
broad claim.

## Primary Question

Can graph-spectral dynamics measured before 2023 reveal which AI research topics
would accelerate from {EVAL_WINDOW_LABEL} better than simple bibliometric baselines?

Current answer: the best 2022 feature is `{best['feature']}` with Spearman
correlation {best['spearman_vs_future_gain']:.3f} against {TARGET_WINDOW_LABEL} topic-share
growth. The composite spectral emergence score scores {spectral['spearman_vs_future_gain']:.3f},
while the raw 3-year growth baseline scores {baseline['spearman_vs_future_gain']:.3f}.

## Question Families

1. Measurement validity: What should a node represent: phrase, OpenAlex topic,
   paper cluster, patent class, repo ecosystem, or learned embedding region?
   The current analysis uses 77 AI-subfield topics with primary-topic annual counts.

2. Temporal validity: Does the method use only information available before the
   cutoff? The graph is rebuilt for every rolling cutoff using only prior years.

3. Prediction: Do spectral features rank future accelerators better than size,
   raw growth, and acceleration? The analysis reports Spearman, top-10 hits, and average
   precision against future top-10 growers.

4. Lead time: How early do signals appear? The rolling test uses cutoffs from 2000 through
   2022 against the next three years.

5. Naming shock: Are new labels predictable from old capability bundles? The
   agentic-AI phrase study says exact labels are late, but precursor families
   existed before 2023.

6. Mode interpretation: Do low-, mid-, and high-frequency bands correspond to
   field-wide motion, family-level transfer, and localized novelty? The analysis measures
   annual spectral energy shares and entropy in a fixed pre-2023 basis.

7. Field merger: Which topics bridge older regions into newer regions? The first atlas uses
   graph edges and spectral layout; a citation/embedding graph
   is the next stronger test.

8. Robustness: Does the answer survive measurement choices? The report now compares primary
   topic counts against AI-subfield any-topic counts, which widen recall without
   leaving the same public source and denominator.

9. Negative controls: Where should the method fail? It should struggle with pure
   naming events, policy-driven shocks, and topics that OpenAlex taxonomy assigns
   imperfectly. Those failures are useful because they separate vocabulary drift
   from capability drift.

10. Theory-building: Are there slowly varying spectral quantities? The first pass starts with
    spectral energy shares, centroid, and entropy. A stronger future claim would
    require showing stability across datasets and longer horizons.

## Rolling Backtest Summary

Mean Spearman correlation across cutoffs:

{markdown_table(avg.reset_index().rename(columns={'feature': 'feature', 'spearman_vs_future_gain': 'mean_spearman'}))}

{robustness_text}

## Current Boundary

The current evidence supports a modest claim: graph Fourier features provide a
different and sometimes predictive view of AI topic motion. It does not yet prove
a general dynamical law of innovation. The next evidence layer should add paper
citations or paper embeddings so the graph is built from research relationships,
not only topic-level co-movement and metadata.
"""
    write_text(REPORT_BUILD_DIR / "research_questions.md", text)


def write_topic_findings(
    score_df: pd.DataFrame,
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    energy: pd.DataFrame,
    any_metrics: pd.DataFrame | None = None,
    any_rolling: pd.DataFrame | None = None,
    scope_summary: pd.DataFrame | None = None,
) -> None:
    best = metrics.iloc[0]
    spectral = metrics.loc[metrics["feature"] == "spectral_emergence_score"].iloc[0]
    baseline = metrics.loc[metrics["feature"] == "baseline_growth_3y"].iloc[0]
    size = metrics.loc[metrics["feature"] == "share_per_million_cutoff"].iloc[0]
    top_future = score_df.sort_values(FUTURE_GAIN_COL, ascending=False).head(12)
    top_spectral = score_df.sort_values("spectral_emergence_score", ascending=False).head(12)
    avg_rolling = rolling.groupby(["feature", "feature_label"], as_index=False).agg(
        mean_spearman=("spearman_vs_future_gain", "mean"),
        mean_top10_hits=("top10_hits", "mean"),
        mean_average_precision=("average_precision_actual_top10", "mean"),
    ).sort_values("mean_spearman", ascending=False)
    energy_note = energy.loc[energy["year"].isin([2000, 2010, 2022, YEAR_END])][
        [
            "year",
            "spectral_low_energy_share",
            "spectral_mid_energy_share",
            "spectral_high_energy_share",
            "spectral_entropy",
        ]
    ]
    ratio_2022 = (
        float(scope_summary.loc[scope_summary["metric"] == "any_to_primary_ratio_2022", "value"].iloc[0])
        if scope_summary is not None
        else float("nan")
    )
    if any_metrics is not None and any_rolling is not None:
        any_spectral = any_metrics.loc[any_metrics["feature"] == "spectral_emergence_score"].iloc[0]
        any_baseline = any_metrics.loc[any_metrics["feature"] == "baseline_growth_3y"].iloc[0]
        scope_metric_rows = pd.concat([metrics, any_metrics], ignore_index=True)
        scope_metric_rows = scope_metric_rows[
            scope_metric_rows["feature"].isin(
                [
                    "spectral_emergence_score",
                    "raw_acceleration_1y",
                    "baseline_growth_3y",
                    "spectral_midband_momentum",
                    "share_per_million_cutoff",
                ]
            )
        ][
            [
                "count_scope_label",
                "feature_label",
                "spearman_vs_future_gain",
                "top10_hits",
                "average_precision_actual_top10",
            ]
        ]
        any_avg_rolling = (
            any_rolling.groupby(["feature", "feature_label"], as_index=False)
            .agg(
                mean_spearman=("spearman_vs_future_gain", "mean"),
                mean_top10_hits=("top10_hits", "mean"),
                mean_average_precision=("average_precision_actual_top10", "mean"),
            )
            .sort_values("mean_spearman", ascending=False)
        )
        robustness_block = f"""## Coverage Robustness

The expanded run adds a wider count scope without changing the source, field
restriction, topic set, or yearly denominator. At the 2022 cutoff, any-topic
assignments are {ratio_2022:.2f} times the primary-topic assignments. This captures
papers where a topic appears as a secondary AI topic rather than the primary one.

- Primary-topic spectral emergence: Spearman {spectral['spearman_vs_future_gain']:.3f};
  top-10 hits {int(spectral['top10_hits'])}/10.
- Any-topic spectral emergence: Spearman {any_spectral['spearman_vs_future_gain']:.3f};
  top-10 hits {int(any_spectral['top10_hits'])}/10.
- Primary-topic growth baseline: Spearman {baseline['spearman_vs_future_gain']:.3f}.
- Any-topic growth baseline: Spearman {any_baseline['spearman_vs_future_gain']:.3f}.

{markdown_table(scope_metric_rows)}

Mean rolling any-topic results:

{markdown_table(any_avg_rolling[['feature_label', 'mean_spearman', 'mean_top10_hits', 'mean_average_precision']])}
"""
    else:
        robustness_block = ""
    findings = f"""# Findings: AI Topic Spectral Dynamics

## Dataset

The field-level panel expands beyond the 32 curated phrase queries. It uses all
77 topics assigned to the Artificial Intelligence subfield and annual public
works counts from {YEAR_START} through {EVAL_END_LABEL}. The main signal
uses `primary_topic.id` counts normalized by total AI-subfield works. The fetcher
also stores AI-subfield `topics.id` counts and the analysis now reports that
wider scope as a robustness check.

This matters because the phrase panel was useful but narrow: it could see
modern labels such as RAG and agentic AI, but it was not a full field-level map.
The topic panel is broader and makes the graph Fourier test less hand-curated.

## Main Backtest Result

Using only pre-2023 data, the strongest 2022 feature is `{best['feature_label']}`,
with Spearman correlation {best['spearman_vs_future_gain']:.3f} against {TARGET_WINDOW_LABEL}
topic-share growth.

- Spectral emergence score: Spearman {spectral['spearman_vs_future_gain']:.3f};
  top-10 hits {int(spectral['top10_hits'])}/10.
- Raw 3-year growth baseline: Spearman {baseline['spearman_vs_future_gain']:.3f};
  top-10 hits {int(baseline['top10_hits'])}/10.
- Topic size at the 2022 cutoff: Spearman {size['spearman_vs_future_gain']:.3f};
  top-10 hits {int(size['top10_hits'])}/10.

The result should be read as an early empirical test, not a proof. With partial
2026 included, the raw 3-year momentum feature is slightly stronger than spectral
emergence in the primary holdout. The useful spectral claim is narrower: graph
relative motion remains close to the best raw feature and survives the wider
any-topic robustness scope.

## Topics With Largest Actual Post-2022 Growth

{markdown_table(top_future[['label', 'family_label', 'count_2022', COUNT_END_COL, 'share_per_million_2022', SHARE_END_COL, FUTURE_MULTIPLE_COL, 'spectral_rank', 'baseline_growth_rank']])}

## Strongest 2022 Spectral Emergence Signals

{markdown_table(top_spectral[['label', 'family_label', 'count_2022', COUNT_END_COL, FUTURE_GAIN_COL, 'spectral_emergence_score', 'future_gain_rank', 'baseline_growth_rank']])}

## Feature Correlations At The 2022 Cutoff

{markdown_table(metrics[['feature_label', 'spearman_vs_future_gain', 'top10_hits', 'average_precision_actual_top10']])}

{robustness_block}

## Rolling Cutoff Summary

{markdown_table(avg_rolling[['feature_label', 'mean_spearman', 'mean_top10_hits', 'mean_average_precision']])}

## Spectral Energy Checkpoints

{markdown_table(energy_note)}

## Reading

1. The topic-level dataset turns the analysis from a phrase demo into a field map:
   77 topics, 36 complete years plus partial 2026, and a primary-topic normalization.
2. The 2022 test asks the user's core question directly: could pre-2023 motion
   anticipate the {EVAL_WINDOW_LABEL} period?
3. The rolling backtest is the guardrail against cherry-picking 2022. It checks
   whether the spectral signal is a recurring pattern or just a one-off fit.
4. Agentic AI still needs the phrase layer because topic labels are not
   fine-grained enough to represent the phrase "agentic AI" directly.
5. The next scientific upgrade is a citation or embedding graph. Topic counts
   are good for a first field-level experiment, but real knowledge-space geometry
   should use relationships between papers, authors, citations, and semantic
   neighborhoods.

## Files

- `data/processed/openalex_ai_topics.csv`
- `data/processed/openalex_ai_topic_year_counts.csv`
- `data/processed/topic_scores.csv`
- `data/processed/topic_scores_any_topic.csv`
- `data/processed/topic_graph_adjacency.csv`
- `data/processed/topic_graph_adjacency_any_topic.csv`
- `data/processed/spectral_energy_by_year.csv`
- `artifacts/report/tables/feature_correlations.csv`
- `artifacts/report/tables/feature_correlations_by_scope.csv`
- `artifacts/report/tables/rolling_cutoff_metrics.csv`
- `artifacts/report/tables/rolling_cutoff_metrics_by_scope.csv`
- `artifacts/report/tables/coverage_scope_summary.csv`
- `artifacts/report/tables/scope_rank_shifts.csv`
- `artifacts/report/figures/prediction_scatter.svg`
- `artifacts/report/figures/scope_robustness.svg`
- `artifacts/report/figures/emergence_heatmap.svg`
- `artifacts/report/figures/topic_atlas.svg`
- `artifacts/report/figures/rolling_backtest.svg`
- `artifacts/report/figures/spectral_energy_stack.svg`
"""
    write_text(REPORT_BUILD_DIR / "topic_findings.md", findings)


def write_visual_interpretations(
    score_df: pd.DataFrame,
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    any_metrics: pd.DataFrame | None = None,
    scope_summary: pd.DataFrame | None = None,
) -> None:
    spectral = metrics.loc[metrics["feature"] == "spectral_emergence_score"].iloc[0]
    baseline = metrics.loc[metrics["feature"] == "baseline_growth_3y"].iloc[0]
    best_rolling = rolling.groupby("feature_label")["spearman_vs_future_gain"].mean().sort_values(ascending=False).head(1)
    if any_metrics is not None and scope_summary is not None:
        any_spectral = any_metrics.loc[any_metrics["feature"] == "spectral_emergence_score"].iloc[0]
        ratio_2022 = float(scope_summary.loc[scope_summary["metric"] == "any_to_primary_ratio_2022", "value"].iloc[0])
        scope_note = f"""
## Coverage Robustness

- The any-topic scope widens 2022 assignment coverage by {ratio_2022:.2f}x while
  keeping the same 77 topics and AI-year denominator.
- Primary-topic spectral emergence has Spearman {spectral['spearman_vs_future_gain']:.2f};
  the wider any-topic scope has {any_spectral['spearman_vs_future_gain']:.2f}.
- Use this visual to decide whether a result depends on clean but narrow primary
  labels or survives broader topic assignment.
"""
    else:
        scope_note = ""
    text = f"""# Visual Interpretation Notes

## Agentic Precursor River

- The modern LLM-agent vocabulary grows sharply after 2022, but older families
  such as retrieval, dialogue, planning, and classic agents were already present.
- This supports the naming-shock interpretation: the label changed faster than
  the underlying capability stack.
- Use this visual when explaining why exact phrase prediction is not enough.

## Agentic Naming Bridge

- The bridge reframes "agentic AI" as a recombination of planning, memory,
  assistant interfaces, and tool use.
- The useful research question is not only "did the phrase appear?" but "did the
  prerequisite capability bundle already have momentum?"
- This is the intuitive counterpart to the graph-spectral argument.

## Prediction Scatter

- The 2022 spectral emergence score has Spearman {spectral['spearman_vs_future_gain']:.2f}
  against {TARGET_WINDOW_LABEL} topic-share growth.
- Compare it against the raw growth baseline at {baseline['spearman_vs_future_gain']:.2f};
  the gap tells us whether graph-relative motion added information.
- Points high and right are the clearest "momentum space" candidates: they were
  unusual before 2023 and grew afterward.

{scope_note}

## Emergence Heatmap

- The heatmap shows whether top post-2022 growers were abrupt breaks or had
  visible warm-up periods before 2022.
- Rows that brighten before the cutoff are predictable accelerations; rows that
  ignite only after the cutoff are naming, policy, or exogenous shocks.
- Because each row is normalized to itself, this chart emphasizes timing rather
  than raw scale.

## Fourier Topic Atlas

- Nearby topics had similar pre-2023 dynamics in the graph Fourier construction.
- Large outlined nodes are the most interesting: they combine later growth with
  high pre-cutoff spectral emergence.
- Isolated large nodes are warning signs that the current graph lacks a richer
  relationship layer, such as citations or paper embeddings.

## Rolling Backtest

- The rolling chart checks whether 2022 is special or whether the method works
  across earlier cutoffs.
- The best average rolling feature is `{best_rolling.index[0]}` with mean Spearman
  {best_rolling.iloc[0]:.2f}.
- If spectral lines only win near 2022, the method may be detecting the LLM-era
  transition rather than a general law.

## Spectral Energy Stack

- Low-frequency energy means broad AI-wide movement; high-frequency energy means
  localized novelty in specific topics.
- A rising entropy line means motion is spread across more spectral modes rather
  than concentrated in a few smooth modes.
- This chart is the closest visual analogue to Fourier energy in physics.

## Question Map

- The question map prevents the analysis from becoming a single-chart prediction
  demo.
- It separates measurement, dynamics, prediction, robustness, and theory claims.
- The current analysis answers the first layer; the next layer needs citations,
  paper embeddings, patents, or GitHub repositories.
"""
    write_text(REPORT_BUILD_DIR / "visual_interpretations.md", text)


def write_article(
    score_df: pd.DataFrame,
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    energy: pd.DataFrame,
    any_metrics: pd.DataFrame,
    any_rolling: pd.DataFrame,
    scope_summary: pd.DataFrame,
    coverage_by_year: pd.DataFrame,
) -> None:
    spectral = metrics.loc[metrics["feature"] == "spectral_emergence_score"].iloc[0]
    baseline = metrics.loc[metrics["feature"] == "baseline_growth_3y"].iloc[0]
    acceleration = metrics.loc[metrics["feature"] == "raw_acceleration_1y"].iloc[0]
    best = metrics.iloc[0]
    any_spectral = any_metrics.loc[any_metrics["feature"] == "spectral_emergence_score"].iloc[0]
    any_baseline = any_metrics.loc[any_metrics["feature"] == "baseline_growth_3y"].iloc[0]
    rolling_summary = (
        rolling.groupby(["feature", "feature_label"], as_index=False)
        .agg(
            mean_spearman=("spearman_vs_future_gain", "mean"),
            mean_top10_hits=("top10_hits", "mean"),
            mean_ap=("average_precision_actual_top10", "mean"),
        )
        .sort_values("mean_spearman", ascending=False)
    )
    any_rolling_summary = (
        any_rolling.groupby(["feature", "feature_label"], as_index=False)
        .agg(
            mean_spearman=("spearman_vs_future_gain", "mean"),
            mean_top10_hits=("top10_hits", "mean"),
            mean_ap=("average_precision_actual_top10", "mean"),
        )
        .sort_values("mean_spearman", ascending=False)
    )
    rolling_best = rolling_summary.head(1).iloc[0]
    any_rolling_best = any_rolling_summary.head(1).iloc[0]
    top_future = score_df.sort_values(FUTURE_GAIN_COL, ascending=False).head(8)
    top_spectral = score_df.sort_values("spectral_emergence_score", ascending=False).head(6)
    energy_lookup = energy.set_index("year")
    energy_2022 = energy_lookup.loc[YEAR_CUTOFF]
    energy_end = energy_lookup.loc[YEAR_END]
    scope_metrics = pd.concat([metrics, any_metrics], ignore_index=True)
    scope_metrics = scope_metrics[
        scope_metrics["feature"].isin(
            [
                "spectral_emergence_score",
                "raw_acceleration_1y",
                "baseline_growth_3y",
                "spectral_midband_momentum",
                "share_per_million_cutoff",
            ]
        )
    ]

    def summary_value(metric: str) -> float:
        return float(scope_summary.loc[scope_summary["metric"] == metric, "value"].iloc[0])

    ratio_total = summary_value("any_to_primary_ratio_total")
    ratio_2022 = summary_value("any_to_primary_ratio_2022")
    coverage_2022 = coverage_by_year.loc[coverage_by_year["year"] == YEAR_CUTOFF].iloc[0]

    def num(value: float, digits: int = 2) -> str:
        return f"{float(value):,.{digits}f}"

    def intish(value: float) -> str:
        return f"{int(round(float(value))):,}"

    def rows_to_html(rows: list[list[object]]) -> str:
        return "\n".join(
            "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        )

    top_future_rows = rows_to_html(
        [
            [
                row.label,
                row.family_label,
                intish(row.count_2022),
                intish(getattr(row, COUNT_END_COL)),
                f"{getattr(row, FUTURE_MULTIPLE_COL):.1f}x",
                intish(row.spectral_rank),
            ]
            for row in top_future.itertuples()
        ]
    )
    metrics_rows = rows_to_html(
        [
            [
                row.feature_label,
                f"{row.spearman_vs_future_gain:.2f}",
                f"{int(row.top10_hits)}/10",
                f"{row.average_precision_actual_top10:.2f}",
            ]
            for row in metrics.itertuples()
        ]
    )
    scope_rows = rows_to_html(
        [
            [
                row.count_scope_label,
                row.feature_label,
                f"{row.spearman_vs_future_gain:.2f}",
                f"{int(row.top10_hits)}/10",
                f"{row.average_precision_actual_top10:.2f}",
            ]
            for row in scope_metrics.itertuples()
        ]
    )
    rolling_rows = rows_to_html(
        [
            [
                row.feature_label,
                f"{row.mean_spearman:.2f}",
                f"{row.mean_top10_hits:.1f}",
                f"{row.mean_ap:.2f}",
            ]
            for row in rolling_summary.itertuples()
        ]
    )
    top_spectral_labels = ", ".join(top_spectral["label"].head(4).to_list())

    def svg_embed(src: str, label: str, width: int, height: int) -> str:
        try:
            svg_text = (REPORT_BUILD_DIR / src).read_text(encoding="utf-8")
        except FileNotFoundError:
            return f'<img src="{esc(src)}" alt="{esc(label)}">'
        return f'<div class="inline-svg" role="img" aria-label="{esc(label)}">{svg_text}</div>'

    tooltip_script = """
<div id="chart-tooltip" role="status" aria-live="polite"></div>
<script>
(() => {
  const tooltip = document.getElementById('chart-tooltip');
  if (!tooltip) return;

  function moveTooltip(clientX, clientY) {
    const pad = 14;
    tooltip.style.left = `${clientX + pad}px`;
    tooltip.style.top = `${clientY + pad}px`;
    const rect = tooltip.getBoundingClientRect();
    if (rect.right > window.innerWidth - 10) {
      tooltip.style.left = `${clientX - rect.width - pad}px`;
    }
    if (rect.bottom > window.innerHeight - 10) {
      tooltip.style.top = `${clientY - rect.height - pad}px`;
    }
  }

  function showTooltip(text, clientX, clientY) {
    tooltip.textContent = text;
    tooltip.classList.add('is-visible');
    moveTooltip(clientX, clientY);
  }

  function hideTooltip() {
    tooltip.classList.remove('is-visible');
  }

  document.querySelectorAll('[data-tooltip]').forEach((target) => {
    const enter = (event) => {
      showTooltip(target.getAttribute('data-tooltip'), event.clientX, event.clientY);
    };
    const move = (event) => {
      moveTooltip(event.clientX, event.clientY);
    };
    target.addEventListener('pointerenter', enter);
    target.addEventListener('pointermove', move);
    target.addEventListener('pointerleave', hideTooltip);
    target.addEventListener('mouseenter', enter);
    target.addEventListener('mouseover', enter);
    target.addEventListener('mousemove', move);
    target.addEventListener('mouseleave', hideTooltip);
    target.addEventListener('mouseout', hideTooltip);
  });
})();
</script>
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Innovation Momentum: Spectral Dynamics in AI Research</title>
  <script>
    window.MathJax = {{
      tex: {{
        displayMath: [['\\\\[', '\\\\]']],
        inlineMath: [['\\\\(', '\\\\)']]
      }},
      chtml: {{ scale: 0.92 }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #667085;
      --line: #d9dee8;
      --paper: #ffffff;
      --band: #f6f7fb;
      --gold: #d18f1f;
      --blue: #3b6ea8;
      --teal: #2a9d8f;
      --rust: #b25d31;
      --violet: #7a5195;
      --olive: #8c8c52;
    }}
    * {{
      box-sizing: border-box;
    }}
    html {{
      scroll-behavior: smooth;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--paper);
      overflow-x: hidden;
    }}
    a {{
      color: var(--blue);
      text-decoration-color: rgba(59, 110, 168, 0.35);
      text-underline-offset: 2px;
    }}
    .page {{
      width: 100%;
      max-width: 1160px;
      margin: 0 auto;
      padding: 0 clamp(18px, 3.4vw, 46px);
    }}
    h1 {{
      margin: 0;
      max-width: 980px;
      font-size: clamp(28px, 3vw, 42px);
      line-height: 1.1;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: clamp(20px, 2vw, 28px);
      line-height: 1.18;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 17px;
      letter-spacing: 0;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.56;
      font-size: 15.2px;
    }}
    p {{
      max-width: 990px;
    }}
    header {{
      padding: 42px 0 28px;
      border-bottom: 1px solid var(--line);
    }}
    section {{
      padding: 36px 0;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      margin: 0 0 14px;
      color: var(--rust);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .dek {{
      max-width: 940px;
      margin: 16px 0 0;
      color: #475467;
      font-size: clamp(16px, 1.4vw, 18px);
      line-height: 1.48;
    }}
    .abstract {{
      max-width: 980px;
      margin: 22px 0 0;
      padding: 14px 16px;
      border-left: 4px solid var(--blue);
      background: #f7f9fc;
      color: #344054;
      font-size: 14.5px;
      line-height: 1.56;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 24px;
      color: var(--muted);
      font-size: 13px;
    }}
    .toc {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 28px;
      max-width: 1050px;
    }}
    .toc a {{
      display: inline-block;
      padding: 8px 10px;
      border: 1px solid var(--line);
      color: var(--ink);
      font-size: 13px;
      text-decoration: none;
      background: #fff;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(142px, 1fr));
      gap: 1px;
      background: var(--line);
      margin: 24px 0 0;
      max-width: 1080px;
    }}
    .kpi {{
      background: var(--paper);
      padding: 14px 16px;
      min-height: 104px;
    }}
    .kpi strong {{
      display: block;
      font-size: 24px;
      color: var(--ink);
    }}
    .kpi span {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(250px, 0.75fr);
      gap: 24px;
      align-items: start;
    }}
    .summary-copy {{
      max-width: 980px;
    }}
    .summary-grid > *,
    .method-grid > *,
    .two-col > * {{
      min-width: 0;
    }}
    .callout {{
      border-left: 4px solid var(--rust);
      background: #fff8f4;
      padding: 16px 18px;
    }}
    .callout p {{
      margin: 0;
      color: #344054;
    }}
    .method-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 20px;
    }}
    .method-card {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 16px;
      min-width: 0;
    }}
    .formula {{
      margin: 12px 0 0;
      padding: 10px 12px;
      background: #f6f7fb;
      color: #111827;
      font-size: 13.5px;
      line-height: 1.45;
      overflow-x: auto;
      white-space: normal;
    }}
    .formula mjx-container[jax="CHTML"][display="true"] {{
      margin: 0;
    }}
    .figure-block {{
      max-width: 100%;
      margin: 20px 0 0;
      padding: 14px;
      border: 1px solid var(--line);
      background: #fff;
      overflow: hidden;
    }}
    .visual {{
      width: 100%;
      margin: 0;
      border-top: 0;
      padding-top: 0;
      overflow-x: auto;
      contain: inline-size;
      -webkit-overflow-scrolling: touch;
    }}
    .visual img {{
      width: 100%;
      min-width: 680px;
      height: auto;
      display: block;
    }}
    .inline-svg {{
      width: 100%;
      min-width: 680px;
    }}
    .inline-svg svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    #chart-tooltip {{
      position: fixed;
      z-index: 30;
      max-width: min(320px, calc(100vw - 32px));
      padding: 10px 12px;
      border: 1px solid rgba(31, 41, 51, 0.16);
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 12px 30px rgba(31, 41, 51, 0.16);
      color: var(--ink);
      font-size: 12.5px;
      line-height: 1.42;
      white-space: pre-line;
      pointer-events: none;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 120ms ease, transform 120ms ease;
    }}
    #chart-tooltip.is-visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    figcaption {{
      max-width: 980px;
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.5;
    }}
    .note {{
      border-left: 4px solid var(--gold);
      padding-left: 16px;
      color: var(--ink);
    }}
    .interpretation {{
      max-width: 980px;
      margin-top: 14px;
      padding-left: 20px;
    }}
    .table-wrap {{
      max-width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      margin-top: 18px;
      border: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      background: #fff;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      color: var(--ink);
      background: #f6f7fb;
      font-weight: 700;
    }}
    td {{
      color: #344054;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 28px;
      align-items: start;
    }}
    .pill {{
      display: inline-block;
      margin: 0 6px 8px 0;
      padding: 5px 8px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      background: #fff;
    }}
    .tag-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 12px 0 0;
    }}
    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      background: #fff;
      color: #344054;
      font-size: 12px;
      line-height: 1;
    }}
    .tag::before {{
      content: "";
      width: 7px;
      height: 7px;
      background: var(--blue);
    }}
    .tag[data-tone="rust"]::before {{
      background: var(--rust);
    }}
    .tag[data-tone="teal"]::before {{
      background: var(--teal);
    }}
    .result-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      background: var(--line);
      margin-top: 18px;
      max-width: none;
    }}
    .result-strip div {{
      background: #fff;
      padding: 12px 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .result-strip strong {{
      display: block;
      color: var(--ink);
      font-size: 18px;
      margin-bottom: 4px;
    }}
    .small {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    footer {{
      padding: 36px 0 56px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    @media (max-width: 820px) {{
      .summary-grid, .method-grid, .two-col {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 28px;
      }}
      .page {{
        padding: 0 16px;
      }}
      .figure-block {{
        padding: 10px;
      }}
      .visual img {{
        min-width: 700px;
      }}
      .inline-svg {{
        min-width: 700px;
      }}
      .result-strip {{
        grid-template-columns: 1fr;
      }}
      .toc a {{
        font-size: 12.5px;
      }}
    }}
  </style>
</head>
<body>
<div class="page">
  <header>
    <p class="eyebrow">Graph-spectral backtest</p>
    <h1>Innovation Momentum: Spectral Dynamics in AI Research</h1>
    <p class="dek">Fourier transforms often replace an awkward coordinate description with a basis in which motion is easier to see. Let us test an analogous move for scientific research fields evolution: transform AI topic histories into graph-spectral modes, measure momentum before 2023, and compare those signals with the growth that followed.</p>
    <p class="abstract">A public AI-topic panel from 1990 through {EVAL_END_LABEL} is used to ask whether pre-2023 graph Fourier features anticipate {EVAL_WINDOW_LABEL} topic growth. The answer is positive but limited: spectral emergence improves on the raw three-year growth baseline in the 2022 holdout, while simple growth remains the most stable signal in rolling backtests. Expanding the count scope from primary-topic labels to broader any-topic assignments increases 2022 assignment coverage by {ratio_2022:.2f}x and tests whether the conclusion is a measurement artifact.</p>
    <div class="meta">
      <span>Public research corpus</span>
      <span>AI subfield, 77 topic time series</span>
      <span>Primary and any-topic count scopes</span>
      <span>Cutoff: 2022</span>
      <span>Forward window: {EVAL_WINDOW_LABEL}</span>
    </div>
    <div class="kpis">
      <div class="kpi"><strong>77</strong><span>AI-topic time series in the field-level panel.</span></div>
      <div class="kpi"><strong>{PANEL_YEARS_LABEL}</strong><span>Annual panel; 2026 is partial through the current public snapshot.</span></div>
      <div class="kpi"><strong>{best['spearman_vs_future_gain']:.2f}</strong><span>Best 2022 holdout Spearman: {html.escape(best['feature_label'])}.</span></div>
      <div class="kpi"><strong>{baseline['spearman_vs_future_gain']:.2f}</strong><span>Raw 3-year growth baseline on the same task.</span></div>
      <div class="kpi"><strong>{ratio_2022:.2f}x</strong><span>Wider any-topic assignment coverage at the 2022 cutoff.</span></div>
      <div class="kpi"><strong>{rolling_best['mean_spearman']:.2f}</strong><span>Best mean rolling Spearman: {html.escape(rolling_best['feature_label'])}.</span></div>
    </div>
    <nav class="toc" aria-label="Report sections">
      <a href="#summary">Main Result</a>
      <a href="#data">Data</a>
      <a href="#robustness">Robustness</a>
      <a href="#method">Method</a>
      <a href="#evidence">Evidence</a>
      <a href="#agentic">Agentic AI</a>
      <a href="#final-summary">Summary</a>
    </nav>
  </header>

  <section id="summary">
    <h2>Main Result</h2>
    <div class="summary-copy">
      <p>Using only information available through 2022, the spectral emergence score reaches Spearman {spectral['spearman_vs_future_gain']:.2f} against {TARGET_WINDOW_LABEL} topic-share growth. The raw three-year growth baseline reaches {baseline['spearman_vs_future_gain']:.2f}. In this holdout, graph-relative motion contains information that a one-topic trend does not.</p>
      <p>The effect is not yet a general forecasting law. Raw 3-year momentum is slightly stronger by Spearman ({best['spearman_vs_future_gain']:.2f}), while raw one-year acceleration captures more top future growers ({int(acceleration['top10_hits'])}/10 versus {int(spectral['top10_hits'])}/10) despite a negative overall rank correlation. In rolling tests from 2000 through 2022, the simple publication-growth baseline remains the most stable feature.</p>
      <p>The useful claim is therefore diagnostic: spectral coordinates separate broad field motion, local novelty, and naming shocks in a way that ordinary counts flatten. A stronger claim requires paper-level relations such as citations, co-citation, authorship, and embeddings.</p>
    </div>
    <div class="result-strip">
      <div><strong>{best['spearman_vs_future_gain']:.2f}</strong>Primary-topic best holdout: {html.escape(best['feature_label'])}.</div>
      <div><strong>{spectral['spearman_vs_future_gain']:.2f}</strong>Primary-topic spectral emergence.</div>
      <div><strong>{any_spectral['spearman_vs_future_gain']:.2f}</strong>Any-topic spectral emergence under wider coverage.</div>
      <div><strong>{rolling_best['mean_spearman']:.2f}</strong>Best mean rolling primary-scope feature.</div>
    </div>
  </section>

  <section id="data">
    <h2>Data And Count Scope</h2>
    <p>The unit of analysis is a research topic inside the AI subfield. For each topic and year, works are normalized by total AI output in the same year. The target is attention share, not raw publication volume.</p>
    <div class="method-grid">
      <div class="method-card">
        <h3>Main panel</h3>
        <p>77 AI-subfield topics, annual years 1990 through {EVAL_END_LABEL}. The 2026 endpoint is partial, so it is treated as a forward-window observation rather than a complete calendar year.</p>
        <div class="tag-row"><span class="tag">public corpus</span><span class="tag" data-tone="teal">topic-level counts</span><span class="tag" data-tone="rust">partial 2026</span></div>
      </div>
      <div class="method-card">
        <h3>Two count scopes</h3>
        <p>The primary-topic scope is cleaner because each work contributes one topic. The any-topic scope is wider because it also counts secondary topic assignments inside the same AI corpus. In 2022 this expands assignment coverage from {int(coverage_2022['primary_topic_assignments']):,} to {int(coverage_2022['any_topic_assignments']):,}.</p>
        <div class="tag-row"><span class="tag">primary-topic</span><span class="tag" data-tone="rust">any-topic</span><span class="tag" data-tone="teal">same denominator</span></div>
      </div>
    </div>
    <p class="small">A phrase-level companion panel is used only where topic labels are too coarse, especially for "agentic AI" and other post-2022 vocabulary. The main field-level result remains topic based; the phrase panel is used to interpret naming shocks.</p>
  </section>

  <section id="robustness">
    <h2>Coverage Expansion Preserves A Weak Spectral Signal</h2>
    <p>The wider any-topic scope contains {ratio_total:.2f} times as many topic assignments over {PANEL_YEARS_LABEL} as the primary-topic scope. It is not automatically better: secondary labels increase recall but may blur the meaning of a topic. The value of the expanded dataset is that it tests whether the holdout result depends on narrow primary labels.</p>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/scope_robustness.svg", "Robustness comparison between primary-topic and any-topic count scopes", 1180, 760)}</div>
      <figcaption>Figure 1. Count-scope robustness. The 2022 holdout and the rolling backtest are recomputed under primary-topic and any-topic measurements. Hover over bars for feature-level values.</figcaption>
    </figure>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Scope</th><th>Feature</th><th>Spearman vs future growth</th><th>Top-10 hits</th><th>Average precision</th></tr></thead>
        <tbody>
          {scope_rows}
        </tbody>
      </table>
    </div>
    <p>The expanded scope leaves the central picture intact but shifts the emphasis. Spectral emergence remains above the three-year growth baseline, while raw one-year acceleration is the strongest 2022 holdout feature. Across rolling cutoffs, publication growth remains the most stable signal. That is a useful constraint on the theory.</p>
  </section>

  <section id="method">
    <h2>Method: Turn Topic Histories Into Spectral Motion</h2>
    <p>The physics analogy only matters if it becomes measurable. A topic's position is its normalized attention over time. Its space is a graph linking topics that moved together before the cutoff and share vocabulary. The graph Fourier transform decomposes that motion into smooth, mid-scale, and localized modes.</p>
    <div class="method-grid">
      <div class="method-card">
        <h3>1. Normalize topic attention</h3>
        <p>For topic <em>i</em> in year <em>t</em>, convert counts into attention share per million AI works, then apply log scaling and train-window z-scores.</p>
        <div class="formula">\\[
          \\begin{{aligned}}
          s_i(t)&=10^6\\frac{{c_i(t)}}{{C_{{AI}}(t)}} \\\\
          x_i(t)&=z_{{\\mathrm{{train}}}}\\!\\left(\\log(1+s_i(t))\\right)
          \\end{{aligned}}
        \\]</div>
      </div>
      <div class="method-card">
        <h3>2. Build a pre-cutoff graph</h3>
        <p>The adjacency matrix combines positive pre-2023 co-movement with metadata similarity, plus conservative within-family edges.</p>
        <div class="formula">\\[
          A_{{ij}}=0.76\\,\\operatorname{{corr}}_+\\!(x_i,x_j)+0.24\\,\\operatorname{{sim}}_{{ij}}
        \\]</div>
      </div>
      <div class="method-card">
        <h3>3. Transform into graph Fourier modes</h3>
        <p>The normalized graph Laplacian supplies the Fourier basis. Low-frequency modes vary smoothly across neighbors; high-frequency modes isolate local deviations.</p>
        <div class="formula">\\[
          \\begin{{aligned}}
          L&=I-D^{{-1/2}}AD^{{-1/2}} \\\\
          LU&=U\\Lambda \\\\
          \\hat{{x}}(t)&=U^\\top x(t)
          \\end{{aligned}}
        \\]</div>
      </div>
      <div class="method-card">
        <h3>4. Score momentum and energy</h3>
        <p>Momentum is recent movement in normalized attention. Spectral energy asks how much yearly signal lives in each graph frequency band.</p>
        <div class="formula">\\[
          \\begin{{aligned}}
          \\Delta x_i&=\\frac{{x_i(2022)-x_i(2019)}}{{3}} \\\\
          E_k(t)&=\\hat{{x}}_k(t)^2 \\\\
          \\mathrm{{score}}&=0.5\\,\\mathrm{{local}}+0.3\\,\\mathrm{{midband}}+0.2\\,\\mathrm{{acceleration}}
          \\end{{aligned}}
        \\]</div>
      </div>
    </div>
  </section>

  <section id="evidence">
    <h2>The 2022 Backtest Finds A Weak But Nontrivial Signal</h2>
    <p>The strongest primary-scope 2022 Spearman feature is {html.escape(best['feature_label'])} ({best['spearman_vs_future_gain']:.2f}). Spectral emergence remains close behind ({spectral['spearman_vs_future_gain']:.2f}) and is still the clearest transformed-coordinate diagnostic. In the scatter below, x is the pre-2023 graph Fourier emergence score and y is the actual {TARGET_WINDOW_LABEL} growth in normalized topic share.</p>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/prediction_scatter.svg", "Prediction scatter showing spectral emergence score versus future topic growth", 1180, 760)}</div>
      <figcaption>Figure 2. Pre-2023 spectral emergence versus future normalized growth through {EVAL_END_LABEL}. High-right points are the cleanest momentum candidates; high-left points are future growers the current graph missed. Hover over points for full topic names and values.</figcaption>
    </figure>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Feature</th><th>Spearman vs future growth</th><th>Top-10 hits</th><th>Average precision</th></tr></thead>
        <tbody>
          {metrics_rows}
        </tbody>
      </table>
    </div>
    <p>The spectral score improves on the raw three-year growth baseline in rank correlation, but its hit rate remains low. The transform sees some field-relative motion, while the topic graph is still too coarse to model the full capability stack.</p>
  </section>

  <section>
    <h2>The Big Growers Are A Mix Of Warm-Ups And Shocks</h2>
    <p>Some fast growers show visible pre-2023 warming. Others brighten mainly after the cutoff, which any pre-cutoff detector will struggle to anticipate. The heatmap separates those cases by normalizing each row to its own history.</p>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/emergence_heatmap.svg", "Heatmap of top post-2022 emerging AI topics", 1440, 802)}</div>
      <figcaption>Figure 3. Row-normalized histories for the top post-2022 growers. Red cells are high relative to that topic's own past; blue cells are low. Hover over cells for year-level share and intensity.</figcaption>
    </figure>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Topic</th><th>Family</th><th>2022 count</th><th>{EVAL_END_LABEL} count</th><th>Share multiple</th><th>Spectral rank</th></tr></thead>
        <tbody>
          {top_future_rows}
        </tbody>
      </table>
    </div>
    <p>A useful momentum system has to separate gradual acceleration from naming shocks and taxonomy lag. Treating every post-2022 rise as the same phenomenon would overstate what this model can predict.</p>
  </section>

  <section>
    <h2>The Fourier Atlas Maps Motion, Not Meaning</h2>
    <p>The atlas places topics using the first two non-constant graph Laplacian eigenvectors. It is not a semantic map. Nearby points are topics that moved similarly before the cutoff.</p>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/topic_atlas.svg", "Fourier atlas of AI topics", 1360, 900)}</div>
      <figcaption>Figure 4. Fourier atlas of topic motion. Node size shows post-2022 growth; dark outlines mark top-decile spectral emergence at the 2022 cutoff. Hover over nodes for full labels, ranks, and growth multiples.</figcaption>
    </figure>
    <p>Large outlined nodes are the cleanest successes: they were early spectral signals and later grew. The current top spectral labels include {esc(top_spectral_labels)}. Several fast growers remain weakly predicted, which points toward a richer graph based on citations and embeddings.</p>
  </section>

  <section>
    <h2>Rolling Cutoffs Keep The Claim Honest</h2>
    <p>Repeating the test for every cutoff from 2000 through 2022 shows why the claim should stay modest. The publication-growth baseline has the strongest mean Spearman ({rolling_best['mean_spearman']:.2f}). Midband spectral momentum remains competitive ({rolling_summary.loc[rolling_summary['feature'] == 'spectral_midband_momentum', 'mean_spearman'].iloc[0]:.2f}), but the current spectral construction is a complement to simple baselines, not a replacement.</p>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/rolling_backtest.svg", "Rolling backtest of spectral and baseline features", 1180, 680)}</div>
      <figcaption>Figure 5. Rolling three-year backtest. Each cutoff rebuilds the graph using only prior data and evaluates the next three years of topic-share growth. Hover over points for cutoff-level metrics.</figcaption>
    </figure>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Feature</th><th>Mean Spearman</th><th>Mean top-10 hits</th><th>Mean average precision</th></tr></thead>
        <tbody>
          {rolling_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>After 2022, The Field Looks Less Smooth</h2>
    <p>In 2022, {energy_2022['spectral_low_energy_share']:.0%} of spectral energy sits in the low-frequency band. By {EVAL_END_LABEL}, that falls to {energy_end['spectral_low_energy_share']:.0%}, while mid- and high-frequency energy rise. In this graph basis, the field moves from broad shared motion toward more localized divergence after the LLM shock.</p>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/spectral_energy_stack.svg", "Stacked spectral energy bands over time", 1180, 700)}</div>
      <figcaption>Figure 6. Annual graph Fourier energy shares. Low-frequency energy corresponds to broad field-wide motion; higher-frequency energy indicates local deviations across topic neighborhoods. Hover over the chart for year-level band shares.</figcaption>
    </figure>
    <p>Spectral energy is descriptive, not causal. It shows that the signal became less smooth in this graph basis; it does not prove why.</p>
  </section>

  <section id="agentic">
    <p>Topic labels are too broad to represent "agentic AI" directly, so a phrase-level companion analysis is still useful. Modern LLM-agent vocabulary rises sharply after 2022, while older components such as multi-agent systems, planning, retrieval, dialogue, and tool/code use were already present.</p>
    <p>Exact labels are often late. A momentum-space model should track latent capability bundles, not just surface vocabulary.</p>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/agentic_precursor_river.svg", "Agentic AI precursor river chart", 1180, 700)}</div>
      <figcaption>Figure 7. Phrase-family histories for agentic-AI precursors. LLM-agent vocabulary surges after 2022, but several precursor families are visible earlier.</figcaption>
    </figure>
    <figure class="figure-block">
      <div class="visual">{svg_embed("figures/agentic_naming_bridge.svg", "Agentic AI naming bridge", 1180, 700)}</div>
      <figcaption>Figure 8. Conceptual bridge from older agent research to modern agentic systems. This is a qualitative synthesis, not an additional predictive model.</figcaption>
    </figure>
  </section>

  <section id="final-summary">
    <h2>Summary</h2>
    <p>Graph Fourier features give a distinct, measurable view of AI topic motion. In the 2022 holdout, spectral emergence is weakly predictive of {TARGET_WINDOW_LABEL} topic-share growth and remains close to the best raw momentum feature. Topic size at the cutoff is negatively associated with future growth, so scale alone is not momentum.</p>
    <p>The phrase analysis also supports a useful interpretation of agentic AI: the modern label arrives late, while older components such as planning, retrieval, dialogue, and multi-agent systems were already visible. The practical lesson is to track latent capability bundles, not only surface vocabulary.</p>
    <p>The stronger test is a paper-level graph built from references, co-citation, authorship, and title/abstract embeddings, with phrase-to-topic bridges for RAG, tool use, LLM agents, chain-of-thought, and planning/action vocabularies. A robustness matrix should compare primary-topic counts, any-topic counts, phrase panels, arXiv categories, and citation-weighted signals before any broad claim about innovation dynamics is made.</p>
    <p>Three questions remain open: whether spectral energy transfers precede field mergers or only describe them after the fact; whether graph wavelets localize emerging capability bundles better than global Laplacian modes; and which quantities are stable enough to treat as slow variables in innovation space.</p>
  </section>

  <footer>
    <p>Reproducibility: <code>analyze_topic_dynamics</code> rebuilds the analysis from processed public data. The source APIs are <a href="https://api.openalex.org/works">Works</a> and <a href="https://api.openalex.org/topics">Topics</a>; generated data and support artifacts are kept outside the committed report.</p>
  </footer>
</div>
{tooltip_script}
</body>
</html>
"""
    write_text(REPORTS_DIR / "index.html", html_text)


def main() -> None:
    topics_path = ROOT / "data" / "processed" / "openalex_ai_topics.csv"
    counts_path = ROOT / "data" / "processed" / "openalex_ai_topic_year_counts.csv"
    if not topics_path.exists() or not counts_path.exists():
        raise SystemExit(
            "Missing expanded topic data. Run `uv run innovation-fetch-ai-topics` first."
        )

    topics = pd.read_csv(topics_path)
    counts = pd.read_csv(counts_path)
    meta = prepare_topic_meta(topics)
    topic_ids = meta["topic_id"].to_list()
    counts = counts[counts["year"].between(YEAR_START, YEAR_END)].copy()

    primary_panel = run_scope_panel(counts, meta, "primary")
    any_panel = run_scope_panel(counts, meta, "any_topic")

    wide_share = primary_panel["wide_share"]
    log_share = primary_panel["log_share"]
    score_df = primary_panel["score_df"]
    metrics = primary_panel["metrics"]
    adjacency = primary_panel["adjacency"]
    rolling = primary_panel["rolling"]
    rolling_scores = primary_panel["rolling_scores"]

    any_score_df = any_panel["score_df"]
    any_metrics = any_panel["metrics"]
    any_adjacency = any_panel["adjacency"]
    any_rolling = any_panel["rolling"]
    any_rolling_scores = any_panel["rolling_scores"]

    coverage_by_year, scope_summary = scope_coverage_tables(counts)
    scope_feature_metrics = pd.concat([metrics, any_metrics], ignore_index=True)
    rolling_by_scope = pd.concat([rolling, any_rolling], ignore_index=True)
    scope_rank_shifts = compare_topic_rank_shifts(score_df, any_score_df)

    signal = zscore_by_train(log_share, YEAR_CUTOFF)
    energy = compute_energy(signal, adjacency)
    total_by_year = counts[["year", "total_ai_subfield_count"]].drop_duplicates().set_index("year")
    energy["total_ai_subfield_count"] = energy["year"].map(total_by_year["total_ai_subfield_count"].to_dict())

    processed_dir = ROOT / "data" / "processed"
    tables_dir = REPORT_BUILD_DIR / "tables"
    figures_dir = REPORT_BUILD_DIR / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    adjacency_df = pd.DataFrame(adjacency, index=topic_ids, columns=topic_ids)
    any_adjacency_df = pd.DataFrame(any_adjacency, index=topic_ids, columns=topic_ids)
    score_df.to_csv(processed_dir / "topic_scores.csv", index=False)
    any_score_df.to_csv(processed_dir / "topic_scores_any_topic.csv", index=False)
    adjacency_df.to_csv(processed_dir / "topic_graph_adjacency.csv")
    any_adjacency_df.to_csv(processed_dir / "topic_graph_adjacency_any_topic.csv")
    energy.to_csv(processed_dir / "spectral_energy_by_year.csv", index=False)
    meta.drop(columns=["token_set"]).to_csv(processed_dir / "topic_metadata_classified.csv", index=False)
    metrics.to_csv(tables_dir / "feature_correlations.csv", index=False)
    any_metrics.to_csv(tables_dir / "feature_correlations_any_topic.csv", index=False)
    scope_feature_metrics.to_csv(tables_dir / "feature_correlations_by_scope.csv", index=False)
    rolling.to_csv(tables_dir / "rolling_cutoff_metrics.csv", index=False)
    any_rolling.to_csv(tables_dir / "rolling_cutoff_metrics_any_topic.csv", index=False)
    rolling_by_scope.to_csv(tables_dir / "rolling_cutoff_metrics_by_scope.csv", index=False)
    rolling_scores.to_csv(processed_dir / "rolling_cutoff_topic_scores.csv", index=False)
    any_rolling_scores.to_csv(processed_dir / "rolling_cutoff_topic_scores_any_topic.csv", index=False)
    coverage_by_year.to_csv(tables_dir / "coverage_by_year.csv", index=False)
    scope_summary.to_csv(tables_dir / "coverage_scope_summary.csv", index=False)
    scope_rank_shifts.to_csv(tables_dir / "scope_rank_shifts.csv", index=False)
    score_df.sort_values(FUTURE_GAIN_COL, ascending=False).head(30).to_csv(
        tables_dir / "top_emerging_topics.csv", index=False
    )
    any_score_df.sort_values(FUTURE_GAIN_COL, ascending=False).head(30).to_csv(
        tables_dir / "top_emerging_topics_any_topic.csv", index=False
    )

    draw_prediction_scatter(score_df, metrics, figures_dir / "prediction_scatter.svg")
    draw_scope_robustness(metrics, any_metrics, rolling, any_rolling, coverage_by_year, figures_dir / "scope_robustness.svg")
    draw_emergence_heatmap(wide_share, score_df, figures_dir / "emergence_heatmap.svg")
    draw_topic_atlas(score_df, adjacency_df, figures_dir / "topic_atlas.svg")
    draw_rolling_backtest(rolling, figures_dir / "rolling_backtest.svg")
    draw_spectral_energy_stack(energy, figures_dir / "spectral_energy_stack.svg")
    write_research_questions(metrics, rolling, any_metrics, any_rolling, scope_summary)
    write_topic_findings(score_df, metrics, rolling, energy, any_metrics, any_rolling, scope_summary)
    write_visual_interpretations(score_df, metrics, rolling, any_metrics, scope_summary)
    write_article(score_df, metrics, rolling, energy, any_metrics, any_rolling, scope_summary, coverage_by_year)

    print(metrics.to_string(index=False))
    print(f"Wrote {processed_dir / 'topic_scores.csv'}")
    print(f"Wrote {REPORT_BUILD_DIR / 'topic_findings.md'}")
    print(f"Wrote {REPORTS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
