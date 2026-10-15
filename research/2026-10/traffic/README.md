# Traffic spectral experiments

Branch: `feature/traffic-spectral-amplification`. Start with the [findings](findings.md), [parent plan](../traffic-research-plan.md), and [decision ledger](ledger.md). The public article is unchanged.

## Recorded studies

| Study | Design | Outputs |
| --- | --- | --- |
| Numerical controls | [Initial protocol](protocol.md): ten mechanism/geometry cases, disjoint simulation seeds, 18 models per case. | [Initial results](results/initial/), including per-episode predictions and independent score validation. |
| Forcing follow-up | [Protocol](forcing-followup.md): adaptive past-window innovation covariance; fresh evaluation seeds. | [Results](results/adaptive/): 22 models per case. |
| Public data | Version-pinned I-24 virtual-trajectory demo and static VT-tools inspection. | [Audit](results/data_audit/): manifest, coverage, missingness, temporal availability, and descriptive speed field. |
| Physical traffic | [SUMO/IDM protocol](sumo-protocol.md): 108 paired pulse responses, including numerical-step replication. | [Results](results/sumo_placement/); the failed insertion attempt's metadata remain in `results/sumo/`. |
| Phase-aware dynamics | [Protocol](phase-followup.md): 32 fresh parameter settings, two integration steps, fixed Fourier AR(2) and raw VAR(2). | [Results](results/phase/): forecasts, paired contrasts, validation, and figures. |

The synthetic and physical studies test specified generating systems. Real-road predictive validation remains pending. The I-24 demo has unresolved causal availability and independent-day coverage, so it supplies no forecast score. Raw third-party inputs and generated full trajectories are kept under ignored `data/research_2026/traffic/`.

## Reproduction

Run at the repository root. Existing experiment directories are protected against overwriting; choose a new `--out` when repeating a study.

```bash
uv run --extra research python research/2026-10/traffic/run_experiments.py --out data/research_2026/traffic/repeat_initial
uv run --extra research python research/2026-10/traffic/run_experiments.py --stage adaptive --out data/research_2026/traffic/repeat_adaptive
uv run --extra research python research/2026-10/traffic/validate_results.py --out data/research_2026/traffic/repeat_initial
uv run --extra research python research/2026-10/traffic/audit_data.py --fetch
```

SUMO is an optional pinned runtime; these commands obtain it through `uv` without changing the project's dependency list:

```bash
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python research/2026-10/traffic/sumo_pilot.py --out data/research_2026/traffic/repeat_sumo
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python research/2026-10/traffic/sumo_phase.py --out data/research_2026/traffic/repeat_phase
```

`summarize_physical.py` validates the recorded physical studies against locally generated response files and rebuilds their figures. Run the repository tests with `uv run --extra research --with pytest python -m pytest -q`. Frozen configurations record seeds, source hashes, and protocol hashes. Preserve these files alongside later results.
