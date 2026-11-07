# Innovation Momentum

A quick experiment to test whether graph-spectral movement in OpenAlex research topic counts can surface early motion in AI/ML research fields.

[Read the article](https://rnepal2.github.io/innovation-momentum/)

## Quick Start

```bash
uv run --with pytest pytest
```

## Rebuild

```bash
uv run innovation-build-report
```

This rebuilds the article, agentic companion figure, and evidence tables from the committed data snapshot.

## Refresh Data

```bash
uv run innovation-fetch-ai-topics
uv run innovation-fetch-phrase-panel
uv run innovation-fetch-agentic-terms
```

The published article is tracked at `reports/index.html`.

## Research

- [Traffic spectra](traffic/README.md): manuscript, experiments, and reproduction commands.
- [Research-field follow-ups](research/README.md): OpenAlex/arXiv and method-adoption studies.

The original OpenAlex code, data snapshot, and published article remain in `src/`, `data/`, and `reports/`.
