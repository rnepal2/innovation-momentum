#!/usr/bin/env python3
"""Analyze agentic-AI precursors and generate presentation SVGs."""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from paths import project_root

ROOT = project_root()
REPORT_BUILD_DIR = ROOT / "artifacts" / "report"

YEAR_START = 1990
YEAR_CUTOFF = 2022
YEAR_END = 2026
END_LABEL = "June 2026"

COLORS = {
    "classic_agents": "#2f6fbb",
    "dialogue_assistants": "#8a5fbf",
    "retrieval_reasoning": "#d18f1f",
    "planning_reasoning": "#4d8f57",
    "tool_code_use": "#cc6677",
    "llm_agentic_terms": "#202c59",
    "muted": "#98a2b3",
    "ink": "#1f2933",
    "subtle": "#667085",
    "grid": "#e6e8ef",
}

CLUSTER_COLORS = {
    "field_core": "#667085",
    "learning_paradigms": "#4d8f57",
    "application_fields": "#2f6fbb",
    "generative_ai": "#d18f1f",
    "transformer_stack": "#8a5fbf",
    "llm_stack": "#cc6677",
    "agentic_systems": "#202c59",
}

FAMILY_LABELS = {
    "classic_agents": "Classic agents",
    "dialogue_assistants": "Dialogue assistants",
    "retrieval_reasoning": "Retrieval & QA",
    "planning_reasoning": "Planning & reasoning",
    "tool_code_use": "Tool & code use",
    "llm_agentic_terms": "LLM-agent terms",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    lo, hi = domain
    a, b = range_
    if hi == lo:
        return (a + b) / 2
    return a + (value - lo) / (hi - lo) * (b - a)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


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


def summarize_agentic_terms(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide_share = df.pivot(index="year", columns="term_id", values="share_per_million_ai")
    wide_count = df.pivot(index="year", columns="term_id", values="count")
    meta = df[["term_id", "label", "family", "query"]].drop_duplicates().set_index("term_id")
    rows = []
    for term_id in wide_share.columns:
        s = wide_share[term_id]
        first_year = None
        first_year_100 = None
        for year, value in s.items():
            if first_year is None and value >= 10:
                first_year = int(year)
            if first_year_100 is None and value >= 100:
                first_year_100 = int(year)
        rows.append(
            {
                "term_id": term_id,
                "label": meta.loc[term_id, "label"],
                "family": meta.loc[term_id, "family"],
                "query": meta.loc[term_id, "query"],
                "count_2022": int(wide_count.loc[YEAR_CUTOFF, term_id]),
                "count_2026": int(wide_count.loc[YEAR_END, term_id]),
                "share_per_million_2022": float(s.loc[YEAR_CUTOFF]),
                "share_per_million_2026": float(s.loc[YEAR_END]),
                "future_multiple_2026_vs_2022": float((s.loc[YEAR_END] + 0.1) / (s.loc[YEAR_CUTOFF] + 0.1)),
                "pre2023_avg_2018_2022": float(s.loc[2018:YEAR_CUTOFF].mean()),
                "first_year_share_ge_10": first_year,
                "first_year_share_ge_100": first_year_100,
            }
        )
    term_summary = pd.DataFrame(rows).sort_values("future_multiple_2026_vs_2022", ascending=False)

    family_year = (
        df[df["year"].between(YEAR_START, YEAR_END)]
        .groupby(["year", "family"], as_index=False)["share_per_million_ai"]
        .sum()
    )
    family_wide = family_year.pivot(index="year", columns="family", values="share_per_million_ai").fillna(0)
    family_rows = []
    for family in family_wide.columns:
        s = family_wide[family]
        family_rows.append(
            {
                "family": family,
                "label": FAMILY_LABELS.get(family, family),
                "share_per_million_2022": float(s.loc[YEAR_CUTOFF]),
                "share_per_million_2026": float(s.loc[YEAR_END]),
                "future_multiple_2026_vs_2022": float((s.loc[YEAR_END] + 0.1) / (s.loc[YEAR_CUTOFF] + 0.1)),
                "pre2023_avg_2018_2022": float(s.loc[2018:YEAR_CUTOFF].mean()),
            }
        )
    family_summary = pd.DataFrame(family_rows).sort_values("future_multiple_2026_vs_2022", ascending=False)
    return term_summary, family_summary, family_year


def draw_precursor_river(family_year: pd.DataFrame, family_summary: pd.DataFrame, out: Path) -> None:
    width, height = 1180, 680
    left, right, top, bottom = 92, 260, 86, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    years = list(range(YEAR_START, YEAR_END + 1))
    family_wide = family_year.pivot(index="year", columns="family", values="share_per_million_ai").fillna(0)
    max_y = float(family_wide.loc[:YEAR_END].max().max())

    def sx(year: int) -> float:
        return scale(year, (YEAR_START, YEAR_END), (left, left + plot_w))

    def sy(value: float) -> float:
        return top + plot_h - math.log1p(value) / math.log1p(max_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial" font-size="25" font-weight="700" fill="{COLORS["ink"]}">Agentic AI Was A Renaming Wave Over An Older Capability Stack</text>',
        f'<text x="{left}" y="58" font-family="Arial" font-size="13" fill="{COLORS["subtle"]}">AI-subfield phrase counts per million works; y-axis uses log scale so old and new vocabularies can be read together.</text>',
        f'<line x1="{sx(YEAR_CUTOFF):.1f}" x2="{sx(YEAR_CUTOFF):.1f}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2" stroke-dasharray="5 5" opacity="0.5"/>',
        f'<text x="{sx(YEAR_CUTOFF) + 6:.1f}" y="{top + 16}" font-family="Arial" font-size="12" fill="#111827">2022 cutoff</text>',
    ]
    for tick in [0, 10, 100, 1000, 10000, 30000]:
        if tick > max_y:
            continue
        y = sy(tick)
        lines.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">{tick:g}</text>')
    for year in range(1990, YEAR_END + 1, 5):
        x = sx(year)
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">{year}</text>')

    order = [
        "classic_agents",
        "dialogue_assistants",
        "planning_reasoning",
        "retrieval_reasoning",
        "tool_code_use",
        "llm_agentic_terms",
    ]
    for idx, family in enumerate(order):
        if family not in family_wide.columns:
            continue
        color = COLORS[family]
        pts = [(sx(y), sy(float(family_wide.loc[y, family]))) for y in years]
        lines.append(f'<polyline points="{polyline(pts)}" fill="none" stroke="{color}" stroke-width="4.2" stroke-linejoin="round" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{sx(YEAR_CUTOFF):.1f}" cy="{sy(float(family_wide.loc[YEAR_CUTOFF, family])):.1f}" r="4.5" fill="#fff" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<circle cx="{sx(YEAR_END):.1f}" cy="{sy(float(family_wide.loc[YEAR_END, family])):.1f}" r="5.5" fill="{color}" opacity="0.9"/>')

        legend_y = top + 30 + idx * 58
        legend_x = left + plot_w + 42
        summary = family_summary.set_index("family").loc[family]
        lines.extend(
            [
                f'<line x1="{legend_x}" x2="{legend_x + 22}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>',
                f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="13" font-weight="700" fill="{COLORS["ink"]}">{esc(FAMILY_LABELS[family])}</text>',
                f'<text x="{legend_x + 30}" y="{legend_y + 22}" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">2022 {summary["share_per_million_2022"]:.0f} -> {END_LABEL} {summary["share_per_million_2026"]:.0f}</text>',
                f'<text x="{legend_x + 30}" y="{legend_y + 39}" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">{summary["future_multiple_2026_vs_2022"]:.1f}x after cutoff</text>',
            ]
        )

    lines.extend(
        [
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#98a2b3"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="{COLORS["subtle"]}">Year</text>',
            f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="12" fill="{COLORS["subtle"]}">Phrase intensity per million AI works</text>',
            "</svg>",
        ]
    )
    write_text(out, "\n".join(lines))


def draw_agentic_bridge(term_summary: pd.DataFrame, family_summary: pd.DataFrame, out: Path) -> None:
    width, height = 1180, 700
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="70" y="38" font-family="Arial" font-size="25" font-weight="700" fill="{COLORS["ink"]}">A Bridge From Older Agent Research To LLM Agentic Systems</text>',
        f'<text x="70" y="62" font-family="Arial" font-size="13" fill="{COLORS["subtle"]}">Node size reflects partial-2026 phrase intensity; curves summarize the pre-2023 capability lineage exposed by the phrase panel.</text>',
    ]

    left_nodes = [
        ("classic_agents", "Classic agent systems", 120, 160),
        ("planning_reasoning", "Planning & action", 120, 255),
        ("dialogue_assistants", "Dialogue assistants", 120, 350),
        ("retrieval_reasoning", "Retrieval & QA", 120, 445),
        ("tool_code_use", "Tool/code use", 120, 540),
    ]
    right_nodes = [
        ("ai_agent", "AI agents", 900, 200),
        ("llm_agent", "LLM agents", 900, 315),
        ("agentic_ai", "Agentic AI", 900, 430),
        ("retrieval_augmented_generation", "RAG agents", 900, 545),
    ]
    center = (525, 350)
    family_map = family_summary.set_index("family")
    term_map = term_summary.set_index("term_id")

    def radius_from_share(value: float) -> float:
        return 16 + min(42, math.sqrt(max(value, 0)) * 0.12)

    # Draw soft capability flows.
    bridge_edges = [
        ("classic_agents", "ai_agent", "agency vocabulary"),
        ("planning_reasoning", "llm_agent", "planning loops"),
        ("dialogue_assistants", "ai_agent", "assistant interface"),
        ("retrieval_reasoning", "retrieval_augmented_generation", "external memory"),
        ("tool_code_use", "llm_agent", "acting through tools"),
        ("tool_code_use", "agentic_ai", "function/tool calls"),
        ("classic_agents", "agentic_ai", "system autonomy"),
    ]
    left_pos = {node_id: (x, y) for node_id, _, x, y in left_nodes}
    right_pos = {node_id: (x, y) for node_id, _, x, y in right_nodes}
    for src, dst, label in bridge_edges:
        x1, y1 = left_pos[src]
        x2, y2 = right_pos[dst]
        color = COLORS[src]
        c1x, c2x = x1 + 270, x2 - 270
        lines.append(
            f'<path d="M {x1 + 36:.1f} {y1:.1f} C {c1x:.1f} {y1:.1f}, {c2x:.1f} {y2:.1f}, {x2 - 42:.1f} {y2:.1f}" fill="none" stroke="{color}" stroke-width="3.3" opacity="0.28"/>'
        )

    # Center synthesis node.
    lines.extend(
        [
            f'<circle cx="{center[0]}" cy="{center[1]}" r="74" fill="#f6f7fb" stroke="#111827" stroke-width="1.2"/>',
            f'<text x="{center[0]}" y="{center[1] - 10}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="{COLORS["ink"]}">Agentic</text>',
            f'<text x="{center[0]}" y="{center[1] + 11}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="{COLORS["ink"]}">capability</text>',
            f'<text x="{center[0]}" y="{center[1] + 33}" text-anchor="middle" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">planning + memory + tools</text>',
            f'<text x="{center[0]}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="12" fill="{COLORS["subtle"]}">Flows summarize older lineages: autonomy, planning loops, assistant interfaces, external memory, and tool/function calls.</text>',
        ]
    )

    for family, label, x, y in left_nodes:
        row = family_map.loc[family]
        r = radius_from_share(float(row["share_per_million_2026"]))
        color = COLORS[family]
        lines.extend(
            [
                f'<circle cx="{x}" cy="{y}" r="{r:.1f}" fill="{color}" opacity="0.18" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>',
                f'<text x="{x + 52}" y="{y - 7}" font-family="Arial" font-size="14" font-weight="700" fill="{COLORS["ink"]}">{esc(label)}</text>',
                f'<text x="{x + 52}" y="{y + 12}" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">2022 {row["share_per_million_2022"]:.0f} -> {END_LABEL} {row["share_per_million_2026"]:.0f}; {row["future_multiple_2026_vs_2022"]:.1f}x</text>',
            ]
        )

    for term_id, label, x, y in right_nodes:
        row = term_map.loc[term_id]
        r = radius_from_share(float(row["share_per_million_2026"]))
        color = COLORS[row["family"]]
        lines.extend(
            [
                f'<circle cx="{x}" cy="{y}" r="{r:.1f}" fill="{color}" opacity="0.20" stroke="{color}" stroke-width="2.2"/>',
                f'<text x="{x + 56}" y="{y - 9}" font-family="Arial" font-size="14" font-weight="700" fill="{COLORS["ink"]}">{esc(label)}</text>',
                f'<text x="{x + 56}" y="{y + 10}" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">{row["count_2022"]:.0f} papers in 2022 -> {row["count_2026"]:.0f} by {END_LABEL}</text>',
                f'<text x="{x + 56}" y="{y + 27}" font-family="Arial" font-size="11" fill="{COLORS["subtle"]}">{row["future_multiple_2026_vs_2022"]:.1f}x normalized growth</text>',
            ]
        )
    lines.append("</svg>")
    write_text(out, "\n".join(lines))


def write_visual_design_note() -> None:
    note = """# Visualization Design Notes

The visual language borrows four common graph signal processing views and
translates them into the knowledge-evolution setting.

1. **Graph signal on nodes.** GSP papers commonly show a signal by coloring or
   extruding values on graph vertices. Here, the topic atlas uses node size and
   outline strength to show post-2022 growth and 2022 spectral emergence.
2. **Fourier modes and eigenvalue order.** The graph Laplacian eigenvectors define
   the graph Fourier basis; lower eigenvalues are smoother modes, while higher
   eigenvalues vary more rapidly across neighboring nodes. Here, the atlas positions
   topics using the first two non-constant Fourier coordinates.
3. **Spectral energy plots.** Classical and graph Fourier analyses often show how
   energy is distributed across frequencies. Here, the stacked energy view partitions
   yearly innovation motion into low, mid, and high graph-frequency bands.
4. **Vertex-frequency / wavelet intuition.** Windowed GFT and graph wavelets are
   used when frequency content is localized on parts of a graph. Here, the agentic
   bridge acts as a conceptual vertex-frequency view: it shows where the post-2022
   agentic signal localizes in older capability families.

Useful references:

- Shuman et al., "The Emerging Field of Signal Processing on Graphs."
- PyGSP Fourier basis examples.
- Shuman, Ricaud, and Vandergheynst, "Vertex-Frequency Analysis on Graphs."
- Hammond, Vandergheynst, and Gribonval, "Wavelets on Graphs via Spectral Graph Theory."
"""
    write_text(REPORT_BUILD_DIR / "visualization_design.md", note)


def main() -> None:
    agentic = pd.read_csv(ROOT / "data" / "processed" / "openalex_agentic_precursor_counts.csv")
    term_summary, family_summary, family_year = summarize_agentic_terms(agentic)

    tables_dir = REPORT_BUILD_DIR / "tables"
    figures_dir = REPORT_BUILD_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    term_summary.to_csv(tables_dir / "agentic_term_summary.csv", index=False)
    family_summary.to_csv(tables_dir / "agentic_family_summary.csv", index=False)
    family_year.to_csv(tables_dir / "agentic_family_year.csv", index=False)

    draw_precursor_river(family_year, family_summary, figures_dir / "agentic_precursor_river.svg")
    draw_agentic_bridge(term_summary, family_summary, figures_dir / "agentic_naming_bridge.svg")
    write_visual_design_note()

    fast_growers = term_summary.head(12)
    family_table = family_summary[
        ["label", "share_per_million_2022", "share_per_million_2026", "future_multiple_2026_vs_2022"]
    ]
    findings = f"""# Findings: Agentic AI Precursor Check

## Question

The topic-level panel cannot represent the modern phrase "agentic AI" directly.
This companion check asks whether the relevant signals were absent before 2023,
or whether the field was already moving under older phrases such as multi-agent
systems, autonomous agents, dialogue systems, planning, retrieval, tool use, and
code generation.

## Main Finding

The exact modern labels are late, but the capability stack is not. Phrase signals
show substantial pre-2023 activity in classic agent systems, retrieval and question
answering, planning/reasoning, and tool/code-use terms. The LLM-centered agentic
family then grows sharply after the 2022 cutoff.

## Family-Level Movement

{markdown_table(family_table)}

## Fastest-Growing Terms After 2022

{markdown_table(fast_growers[["label", "family", "count_2022", "count_2026", "share_per_million_2022", "share_per_million_2026", "future_multiple_2026_vs_2022", "pre2023_avg_2018_2022"]])}

## Reading

The mismatch is productive. A phrase-level detector sees the naming event, but a
better innovation-momentum model should track latent capability bundles.
For agentic AI, that bundle plausibly includes:

- Classic agent and multi-agent systems vocabulary.
- Planning, action selection, and automated reasoning.
- Dialogue and assistant interfaces.
- Retrieval and question answering.
- Tool use, function calling, and code generation.
- LLM-specific agent terms after 2022.

This points toward the next technical upgrade: build nodes from semantic/citation
clusters rather than exact phrases, then measure whether energy transfers from older
capability clusters into newer naming clusters.

## Measurement Note

The `title_and_abstract.search` query is a near-match proxy. In spot checks,
`agentic_system` behaves partly like an older multi-agent-system proxy rather than a
pure exact match for the modern phrase. That is still useful for this question, but it
should be labeled as an agent-system vocabulary proxy, not a clean modern term count.

## New Visuals

- `artifacts/report/figures/agentic_precursor_river.svg`
- `artifacts/report/figures/agentic_naming_bridge.svg`
- `artifacts/report/tables/agentic_term_summary.csv`
- `artifacts/report/tables/agentic_family_summary.csv`
- `artifacts/report/tables/agentic_family_year.csv`
"""
    write_text(REPORT_BUILD_DIR / "agentic_findings.md", findings)
    print(family_summary.to_string(index=False))
    print(f"Wrote {REPORT_BUILD_DIR / 'agentic_findings.md'}")


if __name__ == "__main__":
    main()
