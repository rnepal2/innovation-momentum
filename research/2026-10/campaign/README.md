# Robotics research experiments

[Findings and decisions](findings.md) | [Protocol](protocol.md) | [Implementation](implementation.md) | [Experiment ledger](ledger.md)

This directory tests residual activity propagation and sustained joint use of scientific phrases. All work is on `feature/research-directions-2026`; the public article is unchanged.

## Run

```bash
uv sync --extra research
uv run --extra research python research/2026-10/campaign/run_campaign.py
```

The full run expects the local corpus under `data/research_2026/`. Add `--download` to reconstruct it through the cached, resumable fetchers. OpenAlex reads `OPENALEX_API_KEY` from the environment. arXiv harvesting uses one connection with a minimum 3.1-second request interval. A fresh download can differ as providers revise metadata. Cached corpus files, their checksums and retrieval queries identify the data actually analyzed.

Use `--checks-only` to regenerate figures, numerical summaries, lexical and calibration sensitivities, and raw-text validation without refitting. Tests:

```bash
uv run --extra research --with pytest python -m pytest -q
```

## Evidence

`results/` contains forecast losses, event metrics, model selections, phrase lists, source audits, validation cases and four inspected figures. The acquisition manifest identifies both downloaded corpora. The execution manifest records Python, library versions and script checksums. Large raw records, cached API pages and full prediction arrays remain local and gitignored. Sharing exact replication requires sharing those frozen files or an archive; a current API refetch alone cannot promise the same snapshot.

`results/first_pass/` retains diagnostic outputs before the chronological deduplication correction and stronger controls. Final summaries use only corrected outputs in `results/`. Follow-ups chosen after inspecting results are exploratory.
