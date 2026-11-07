# Repository layout and retention

The original OpenAlex article keeps its existing `src/`, `configs/`, `data/`, and `reports/` paths. The root `index.html` still opens that article. `research/` contains its research-field and adoption follow-ups.

The traffic study is self-contained under `traffic/`: manuscript, figures, experiment runners, protocols, literature reviews, result tables, and tests. Its local raw inputs are in ignored `traffic/data/`; repeat analysis and browser checks are in ignored `traffic/artifacts/`.

## Path migration

| Previous location | Current location |
| --- | --- |
| `research/2026-10/traffic/` | `traffic/` |
| `research/2026-10/traffic-*.md` | `traffic/traffic-*.md` |
| Other `research/2026-10/` files | `research/` |
| Traffic tests in root `tests/` | `traffic/tests/` |
| `data/research_2026/traffic/` | `traffic/data/` |
| `data/research_2026/traffic_novelty/` | `traffic/data/literature/` |
| `artifacts/traffic_*` | `traffic/artifacts/` |

Historical result tables, manifests, freeze files, and frozen source snapshots retain their original bytes, including recorded paths. `traffic/project_paths.py` resolves those paths for the current analysis tools. Current commands and new receipts use the new layout. Replaying a frozen historical script directly requires its original layout, available in Git history.

## Retained evidence

Negative forecasting results, failed eligibility checks, timing/rank corrections, and their source archives remain part of the research record. Repeat trajectories used by validation receipts also remain local. Copyrighted source papers and full acquired datasets stay ignored.

## Cleanup

The restructure removes eight superseded PNG exports: all five plots in the three `*_timing_audit/` result directories and three `analysis_revision/before_*.png` plots. Their numerical tables, correction records, and frozen plotting sources remain available; no current document links to those images. Current manuscript figures are unchanged.

Unreferenced local `mechanism_smoke` and `mechanism_smoke_raw` directories, old browser-layout screenshots, and disposable Python/test/lint caches were removed. The browser check lives at `traffic/tests/check_manuscript_layout.cjs`; running it with Playwright and Chrome available writes fresh checks to `traffic/artifacts/manuscript_layout/`.
