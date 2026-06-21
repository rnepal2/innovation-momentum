# Innovation Momentum

A small research-code repo for testing whether graph-spectral movement in OpenAlex topic counts can surface early motion in AI/ML research fields.

[Read the article](https://rnepal2.github.io/innovation-momentum/)

## Quick Start

```bash
uv run --with pytest pytest
```

## Rebuild

```bash
uv run innovation-build-report
uv run innovation-build-agentic
```

## Refresh Data

```bash
uv run innovation-fetch-ai-topics
uv run innovation-fetch-phrase-panel
uv run innovation-fetch-agentic-terms
```

The generated data and intermediate artifacts live under `data/` and `artifacts/`. The published article is tracked at `reports/index.html`.
