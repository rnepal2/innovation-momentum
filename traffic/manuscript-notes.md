# Evidence index

The [manuscript](manuscript.md) presents the current argument. This index separates measured results, mathematical checks, and unfinished research. The [decision ledger](ledger.md) preserves the sequence of exploratory choices and repairs.

## Claims and scope

| Result | Evidence | Interpretation |
| --- | --- | --- |
| Long-band share rises during both growth and decay | [Case table](results/spectral/case_summary.csv), [sensitivities](results/spectral/sensitivity.csv) | Descriptive counterexample to using concentration alone as an amplification flag. |
| Individual small-amplitude modes follow coupled gap-speed dynamics | [Full-field errors](results/mechanism/full_response_validation.csv), [Jacobian checks](results/mechanism/independent_validation.json) | Simulator/model agreement under specified parameters. |
| Fundamental-only initialization generates growing q=3 power while its equilibrium mode decays | [Powers](results/harmonics/responses.csv), [exponents](results/harmonics/exponents.csv), [matched-state control](results/harmonics/autonomous_comparison.csv) | Strongest controlled mechanism result; coupling path and generality remain unspecified. |
| Whole-window rejection depends on amplitude, noise, model, and reference | [Observation table](results/observation_window/diagnostics.csv), [restricted-fit table](results/uncertainty/physics_fit/diagnostics.csv), [reference nulls](results/uncertainty/reference_noise/exact_linear_nulls.csv) | Conditional model check. Noise draws and translated phases are not independent environments. |
| Restricted IDM and TLS have different rejection behavior | [Fitted operators](results/uncertainty/physics_fit/operators.csv), [protocol](physics-fit-followup.md) | Prior information, parameter count, and objective all change. No isolated physics-constraint effect. |
| A constructed input matches opposite-growth modal trajectories | [Construction](forcing_controls.py), [replay](results/uncertainty/forcing/ambiguity.csv) | Algebraic unknown-input limit for one linear modal subsystem. No new theorem or full nonlinear equivalence. |
| Empirical baseline and exclusion choices alter spectra | [Six blocks](results/ngsim_expansion/combined_blocks.csv), [exclusions and coverage](results/ngsim_expansion/spectra.csv) | Measurement sensitivity. No empirical instability or prospective prediction claim. |
| No established best-baseline forecast gain | [Earlier findings](findings.md), [phase comparison](results/phase/) | Forecasting hypothesis remains unsupported in tested settings. |

## Current figures

| Figure | Analytical purpose | Source |
| --- | --- | --- |
| 1 | Connect the actual spatial response, absolute harmonic evolution, and amplitude scaling | [Derived field](figures/harmonic_field.csv), harmonic powers above |
| 2 | Show the interaction of noise, amplitude, and headway under one explicitly restricted fit | [11-level noise sweep](results/noise_resolution/summary.csv), [adaptive protocol](noise-resolution-followup.md) |
| 3 | Show what was measured and compare baseline choices on identical fields | [First field](results/ngsim/us-101_25m_field.csv), six-block comparisons above |

[Source hashes](figures/sources.json), [field provenance](figures/harmonic_field_source.json), and [rendering code](render_manuscript_figures.py) make each figure inspectable. PNG and SVG exports contain the same content. Legacy plots remain in historical result directories and are excluded from the main presentation.

## Outstanding tests

1. Test driver-model transfer and estimate equilibrium/noise from separate calibration data. Exact IDM assumptions currently help determine the fitted operator.
2. Measure or constrain ongoing inputs. Spectral growth alone cannot identify intrinsic dynamics under unrestricted forcing.
3. Acquire independently labeled disturbance episodes and recording days, with verified preprocessing availability, before empirical warning evaluation.
4. Complete matched prior-method replication and a focused priority search before claiming a publishable methodological contribution.

The prior-art notes record access limits: the full 2014 spectral-envelope methods were unavailable; selected Takada paper/code and Avila/Mezic main text were inspected, without complete replication. Targeted noise-aware DMD methods were checked, without reproducing their published numerical examples. Current presentation changes do not resolve publication novelty.
