# Traffic spectral experiments

Work is on `feature/traffic-spectral-amplification`. The published article on `main` is unchanged.

- [Manuscript](manuscript.md): the current argument, three figures, and one comparison table. An optional [HTML preview](manuscript.html) is also available.
- [Methods supplement](supplement.md): secondary results, numerical controls, and eligibility denominators.
- [Research review](review-2026-11-06.md): what the results establish and why the presentation changed.
- [Evidence index](manuscript-notes.md): source tables and remaining research gaps.
- [Reproduction](reproduction.md): protocols, runners, saved results, and acquisition receipts.
- [Research plan and references](traffic-research-plan.md).

The strongest controlled result is growing harmonic power in a mode whose equilibrium dynamics decay. The observation experiments quantify limits under assumed dynamics and noise. The current road-data analysis has no instability labels. GFT-specific forecast improvement and publication novelty remain unestablished.

From the repository root:

```bash
uv run --extra research --with pytest python -m pytest traffic/tests -q
uv run --extra research python traffic/render_manuscript_figures.py
```

Figure rendering uses checked-in derived inputs and does not rerun experiments. Full simulation arrays and acquired trajectories remain under ignored `traffic/data/`. Historical results, source snapshots, and superseded timing/rank analyses remain preserved.

`results/` contains the evidence record; `figures/` contains the manuscript exports. Tests are in `tests/`. Local repeat runs and rendering checks belong in ignored `artifacts/`. See [repository layout and retention](../research/repository-layout.md) for the path migration and cleanup policy.
