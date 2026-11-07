# Traffic spectral experiments

Work is on `feature/traffic-spectral-amplification`. The published article on `main` is unchanged.

- [Manuscript](manuscript.md): the current argument, three figures, and one comparison table. An optional [HTML preview](manuscript.html) is also available.
- [Methods supplement](supplement.md): secondary results, numerical controls, and eligibility denominators.
- [Research review](review-2026-11-06.md): what the results establish and why the presentation changed.
- [Evidence index](manuscript-notes.md): source tables and remaining research gaps.
- [Reproduction](reproduction.md): protocols, runners, saved results, and acquisition receipts.

The strongest controlled result is growing harmonic power in a mode whose equilibrium dynamics decay. The observation experiments quantify limits under assumed dynamics and noise. The current road-data analysis has no instability labels. GFT-specific forecast improvement and publication novelty remain unestablished.

From the repository root:

```bash
uv run --extra research --with pytest python -m pytest -q
uv run --extra research python research/2026-10/traffic/render_manuscript_figures.py
```

Figure rendering uses checked-in derived inputs and does not rerun experiments. Full simulation arrays and acquired trajectories remain under ignored `data/research_2026/traffic/`. Historical results, source snapshots, and superseded timing/rank analyses remain preserved.
