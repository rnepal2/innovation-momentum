# Experiment register and reproduction

Branch: `feature/traffic-spectral-amplification`. This register retains the complete experiment and command inventory. Start with the [manuscript](manuscript.md) or [evidence index](manuscript-notes.md) for the current interpretation. The public article is unchanged.

## Current manuscript figures

Run `uv run --extra research python traffic/render_manuscript_figures.py` from the repository root. It uses checked-in inputs to rebuild the three current figures as PNG and SVG and the local [HTML manuscript](manuscript.html), without rerunning experiments. Figure input hashes are in `figures/sources.json`. The optional `--refresh-data` re-exports the one specified harmonic field from its verified local raw array. Other plots below belong to the historical experiment record and are not all included in the current manuscript.

Figure 2 is a standalone 6.6-by-4.5-inch plot. The Markdown manuscript uses a native two-column Markdown table with a 528-by-360 preview image on the left and a concise caption on the right. It contains no raw HTML. The full-resolution PNG/SVG remain available. The optional HTML renderer uses the full-resolution image in bounded 58%/42% columns, with a 528 px image limit and automatic height. Backgrounds stay transparent; mobile columns stack below 760 px. The legend remains inside the upper-right axes with parameter units and the same math typography as Figure 1. `render_manuscript.py` rebuilds the HTML alone after writing or layout edits; it uses local MathML for display equations and needs no network access. Production HTML is never written by either renderer.

Figure 2 uses an [11-level noise follow-up](noise-resolution-followup.md). With the verified local raw arrays available, run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --extra research python traffic/analyze_noise_resolution.py --out traffic/artifacts/noise_resolution_repeat` to repeat fitting and calibration in a fresh directory. `validate_noise_resolution.py --out traffic/artifacts/noise_resolution_repeat` independently replays one fit's calibration and evaluation at each headway/noise combination. Saved checks are in [results/noise_resolution](results/noise_resolution/). The three figures share blue/gold/green categorical colors. Signed speed deviations use a blue-neutral-gold scale centered at zero; absolute speed uses neutral-to-blue. Line styles and marker shapes distinguish overlapping series without relying on color alone.

## Recorded studies

| Study | Design | Outputs |
| --- | --- | --- |
| Numerical controls | [Initial protocol](protocol.md): ten mechanism/geometry cases, disjoint simulation seeds, 18 models per case. | [Initial results](results/initial/), including per-episode predictions and independent score validation. |
| Forcing follow-up | [Protocol](forcing-followup.md): adaptive past-window innovation covariance; fresh evaluation seeds. | [Results](results/adaptive/): 22 models per case. |
| Public data | Version-pinned I-24 virtual-trajectory demo and static VT-tools inspection. | [Audit](results/data_audit/): manifest, coverage, missingness, temporal availability, and descriptive speed field. |
| Physical traffic | [SUMO/IDM protocol](sumo-protocol.md): 108 paired pulse responses, including numerical-step replication. | [Results](results/sumo_placement/); the failed insertion attempt's metadata remain in `results/sumo/`. |
| Phase-aware dynamics | [Protocol](phase-followup.md): 32 fresh parameter settings, two integration steps, fixed Fourier AR(2) and raw VAR(2). | [Results](results/phase/): forecasts, paired contrasts, validation, and figures. |
| Descriptive spectra | [Protocol](spectral-diagnostics-protocol.md): mean-separated eigenspace powers, temporal density integration, window sensitivities. | [Results](results/spectral/): all 172 existing responses, five figures, and independent FFT validation. |
| Pulse strength | Same protocol: 0.5/1/2 m/s drops, fixed headway, three acceleration settings, two steps. | [Results](results/spectral_forcing/): 54 paired responses through 240 s, normalized powers and scaling exponents. |
| Small-pulse scaling | Adaptive [follow-up](small-pulse-followup.md): 0.05/0.1/0.2 m/s drops, finer numerical steps. | [Results](results/spectral_small_pulse/): 54 more responses and amplitude-scaling checks. |
| Mechanism | [Protocol](mechanism-protocol.md): coupled gap-speed modes and individual initial wavelengths. | [Results](results/mechanism/): 192 responses, theory/phase checks, and full-field validation. |
| Heterogeneous drivers | [Follow-up](heterogeneity-followup.md): four fixed headway patterns and matched graph-band approximations. | [Results](results/heterogeneity/): 48 responses; full dynamics substantially outperform independent bands. |
| Nonlinear harmonics | Same follow-up: q=1 initial disturbances and q=1/2/3 amplitude scaling. | [Results](results/harmonics/): 24 responses; growing harmonic power in linearly decaying modes. |
| NGSIM collection | Fixed [cohort selection](ngsim-collection-followup.md), followed by [baseline sensitivity](empirical-sensitivity-followup.md). | [Results](results/ngsim/): two official blocks, 439,065 rows, coverage failures and quality flags retained. |
| Observation limits | [Protocol](observation-protocol.md): direct, harmonic, and alias controls; fixed masks and synthetic noise. | [Results](results/observation/): 56 paired responses; limited noisy sensitivity of adjacent innovations. |
| Whole-window consistency | Adaptive [follow-up](window-consistency-followup.md): new phases, known-operator trajectory fitting. | [Results](results/observation_window/): 56 further responses; conditional low-noise detection, sparse/strong-noise limitations. |
| Empirical resolution | Fixed-support bins and trace exclusions specified in the observation protocol. | [Results](results/ngsim_resolution/): 20 of 32 configurations eligible; all failures and population changes retained. |
| Operator and reference uncertainty | [Protocol](uncertainty-protocol.md): 64 fresh paired responses, parameter error, OLS/TLS/noise-corrected transitions, and joint observation covariance. | [Results](results/uncertainty/): calibration failures, conditional corrections, and explicit abstentions. |
| Restricted operator fit | Adaptive [follow-up](physics-fit-followup.md): 32 further paired responses; exact equilibrium and correct IDM prior. | [Results](results/uncertainty/physics_fit/): reduced operator error and substantially better conditional calibration. |
| Continuous dynamics and forcing | U4 in the uncertainty protocol: independent DOP853 integration and an acceleration-only ambiguity construction. | [Results](results/uncertainty/forcing/): 32 refined ODE responses, SUMO agreement, and matched trajectories with opposite intrinsic growth signs. |
| NGSIM expansion | [Fixed selection](ngsim-expansion-followup.md): four additional nonoverlapping blocks on unchanged spatial support. | [Results](results/ngsim_expansion/): 906,559 additional rows; same recording contexts and retained coverage failure after exclusion. |

The synthetic and physical studies test specified generating systems. Real-road predictive validation remains pending. The I-24 demo has unresolved causal availability and independent-day coverage, so it supplies no forecast score. Raw third-party inputs and generated full trajectories are kept under ignored `traffic/data/`.

## Reproduction

Run at the repository root. Existing experiment directories are protected against overwriting; choose a new `--out` when repeating a study.

```bash
uv run --extra research python traffic/run_experiments.py --out traffic/data/repeat_initial
uv run --extra research python traffic/run_experiments.py --stage adaptive --out traffic/data/repeat_adaptive
uv run --extra research python traffic/validate_results.py --out traffic/data/repeat_initial
uv run --extra research python traffic/audit_data.py --fetch
```

SUMO is an optional pinned runtime; these commands obtain it through `uv` without changing the project's dependency list:

```bash
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/sumo_pilot.py --out traffic/data/repeat_sumo
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/sumo_phase.py --out traffic/data/repeat_phase
```

`summarize_physical.py` validates the recorded physical studies against locally generated response files and rebuilds their figures. Run the repository tests with `uv run --extra research --with pytest python -m pytest -q`. Frozen configurations record seeds, source hashes, and protocol hashes. Preserve these files alongside later results.

The descriptive analysis uses saved arrays produced by the first two SUMO studies. Recreate those first on a fresh checkout. Then run:

```bash
uv run --extra research python traffic/run_spectral_diagnostics.py --out traffic/data/repeat_spectra
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/run_forcing_spectra.py --out traffic/data/repeat_forcing
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/run_forcing_spectra.py --stage small --out traffic/data/repeat_small_pulse
```

`validate_spectra.py` independently recalculates the checked-in result tables against local raw arrays and verifies their frozen hashes. `summarize_spectra.py` exports the manuscript's numerical support. Use `--raw traffic/data/another_raw_directory` with the pulse runner to keep repeat trajectories separate. The full trajectories are generated locally and remain ignored by Git.

The new mechanism studies use the same pinned SUMO runtime. Their runners accept `--out` and `--raw` for separate repeat directories:

```bash
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/run_mechanism.py --out traffic/data/new_mechanism --raw traffic/data/new_mechanism_raw
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/run_heterogeneity.py --stage heterogeneity --out traffic/data/new_heterogeneity --raw traffic/data/new_heterogeneity_raw
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/run_heterogeneity.py --stage harmonics --out traffic/data/new_harmonics --raw traffic/data/new_harmonics_raw
```

`validate_mechanism.py` checks full physical response and finite-difference Jacobians. `validate_extensions.py` independently checks graph-band projections, harmonic powers, frozen hashes, and the recorded repeat directories. `analyze_harmonic_forcing.py` compares observed q=3 power with autonomous linear response; `summarize_extensions.py` exports the manuscript support and phase comparison.

`collect_ngsim.py` acquires bounded official trajectory blocks. For a new snapshot, choose fresh `--out` and `--data` directories; existing inputs cannot be overwritten without a matching receipt. `analyze_ngsim_followup.py` checks the original accepted snapshot and builds its fixed-support temporal-centering sensitivity. Raw trajectories remain under ignored `data/`; the result directory contains source attribution, queries, counts, checksums, quality profiles, and derived fields. These short processed recordings support retrospective measurement only.

The observation runner uses the same optional pinned SUMO runtime:

```bash
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/run_observation.py --out traffic/data/new_observation --raw traffic/data/new_observation_raw
uv run --extra research --with eclipse-sumo==1.28.0 --with libsumo==1.28.0 python traffic/run_observation.py --window --out traffic/data/new_window --raw traffic/data/new_window_raw
```

`empirical_resolution.py --out <fresh_directory>` uses the accepted NGSIM inputs. `validate_observation.py --repeats --probe-clock`, with the pinned simulator runtime available, independently checks the recorded studies and repeat directories. See the [clock correction](timing-correction.md), [sampling correction](sampling-correction.md), and source archives for superseded runs and corrected reanalysis. Retained fine-step harmonic and observation results use a verified 0.01 s simulator step. `refresh_observation_analysis.py` records the one-time historical repair; fresh runners already use the corrected reconstruction.

For uncertainty studies, use the pinned simulator command with `collect_uncertainty.py`; add `--physics-followup` for its adaptive cohort. Both output and raw directories must be fresh. Analysis entry points are `analyze_uncertainty.py` with stage `operator_error`, `learned_operator`, or `reference_noise`; `analyze_physics_fit.py`; and `forcing_controls.py`. Each accepts a fresh `--out`; the latter also needs a fresh `--raw`. The analyses accept `--physical` to select a repeated physical cohort. `report_uncertainty.py` rebuilds figures and claim support; `validate_uncertainty.py` checks hashes, independent calculations, and retained repeat directories. Its receipt states the diagnostic replay's sampled scope.

`collect_ngsim_expansion.py` obtains the four specified additional official blocks, with the same count/hash safeguards and fresh snapshot directories. The six blocks now contain 1,345,624 rows; they are not independent days. Raw trajectories stay ignored. Source queries, data-use metadata, eligibility failures, and secondary exclusion results remain in the result directories.
