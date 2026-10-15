# Cross-community Adoption Pilot

[Findings](findings.md), [executed notebook](analysis.ipynb), [frozen protocol](protocol.md) and [experiment ledger](ledger.md).

From the repository root, using the existing local input snapshots:

```bash
uv sync --extra research
uv run --extra research python research/2026-10/adoption/experiment.py
uv run --extra research python research/2026-10/adoption/summarize_adoption.py
uv run --extra research --with nbformat --with nbclient --with ipykernel python research/2026-10/adoption/reproduce.py
```

`reproduce.py` checks input hashes, rebuilds and executes the companion notebook, and writes the execution receipt. The notebook itself reads only committed result tables and can be rerun without API access or raw abstracts.

The notebook executes top-to-bottom; its figures were visually reviewed. Full HTML presentation inspection was unavailable because browser policy blocks local-file navigation.

Large inputs remain in ignored `data/adoption_2026/` and `data/research_2026/`. Their hashes are recorded in the preparation and acquisition manifests. Repeating an API query later does not reproduce a frozen snapshot. Exact raw-data replication requires those local files. The accepted medical corpus, canonical references, development choices and prospective ledger are preserved on ordinary reruns; use a separate study directory for a new freeze.

For initial acquisition only, `fetch_medical.py` and `resolve_anchors.py` read `OPENALEX_API_KEY` from the environment. The older robotics acquisition lives under `../campaign/`. `experiment.py --development-only` creates development choices only when no frozen choices exist. `version_audit.py` reuses cached arXiv metadata and PDFs, with a three-second delay for uncached requests. None of these scripts schedules a later run or publishes results.

Run tests with `uv run --extra research --with pytest pytest -q`. Full abstracts and PDFs are local research inputs; the committed result bundle contains metadata, aggregate results, predictions and review annotations.
