# Graph spectra and traffic-wave amplification

Created: 2026-10-15. Status: working research plan; experimental details remain to be frozen before evaluation.

This file records the proposed study, its relationship to earlier work, and the tests we intend to run. Update the experiment register and decision log as work proceeds. Preserve earlier decisions and distinguish planned work, completed checks, and measured results.

Implementation branch: `feature/traffic-spectral-amplification`. The initial plan preceded traffic experiments. Subsequent results are recorded in the [traffic study directory](traffic/README.md) and [findings](traffic/findings.md). The existing public article remains unchanged.

## 1. Research question and context

The starting idea is that a spectral representation might expose dynamical structure that is difficult to identify directly in spatial observations. Our earlier publication-data experiments found no stable graph-specific forecasting advantage in the tested settings; their scope and results are recorded in the [robotics campaign](campaign/findings.md) and [adoption follow-up](adoption/findings.md). Those findings motivate testing a system with measurable physical propagation and finer temporal observations. They do not establish that publication data generally lack useful spectral structure.

The proposed traffic question is:

> Under which sensing and road-structure conditions do graph-mode spectral measurements improve prediction of disturbance amplification beyond current traffic conditions, ordinary spatial spectra, cross-spectral methods, and learned dynamical modes?

The initial methods will be non-neural. Useful outcomes include a reproducible predictive improvement, a physical explanation for a restricted benefit, or a measured limit on when graph spectra help.

### Position relative to existing work

The [congestion-precursor audit](traffic-novelty-review.md) found direct non-neural GFT warning prior art in Takada et al. [R1]. The [focused amplification review](traffic-amplification-review.md) also found frequency-domain growth prediction, spectral-envelope analysis, and Koopman/DMD traffic studies [R2-R6]. Joint graph-temporal spectral density is established [R7].

The accessed literature did not establish an exact duplicate of the complete conditional-amplification comparison proposed here. This is a bounded search finding. Scientific priority remains unresolved, especially where full methods were unavailable. A new dataset or a change from coefficients to squared coefficients supplies insufficient evidence of a new method. Replication and comparative extension are acceptable study outcomes.

## 2. Hypotheses

**Traffic H1: incremental predictive value.** Past-only graph-band spectral measurements improve held-out prediction of subsequent disturbance amplification after conditioning on current amplitude, speed, density where observed, recent trends, spatial gradients, and ordinary temporal variability.

Evidence requires an improvement over the frozen strongest baseline on independent episodes or days. A favorable spectrum plot or an in-sample association cannot establish H1. Gains that disappear after current-state controls would support a descriptive interpretation.

**Traffic H2: graph-specific representation value.** The specified road graph provides a useful estimation constraint compared with matched ordinary spatial transforms, training-fitted PCA, and randomized graph/basis controls. The benefit may depend on irregular road geometry, sensor density, or missing observations.

H2 requires separating the effect of graph geometry from dimension reduction, regularization, and feature count. Equal performance on randomized graphs would weaken the graph-specific claim. A benefit confined to a uniform corridor must be compared with the corresponding sine/cosine representation.

**Exploratory mechanism questions.** Does modal persistence add information beyond modal power? Does cross-mode coupling explain failures of independent-mode models? Can directional or localized representations recover information lost by an undirected global basis? These questions become separate experiments after the primary comparisons; each needs a recorded design before new held-out evaluation.

## 3. Measurements and interpretation

Let `x(t)` be the observed spatial speed-deviation vector, with the reference state and preprocessing defined using past observations. For a fixed symmetric graph Laplacian:

```text
L = U Lambda U^T
z(t) = U^T x(t)
E[B, Omega](t) = sum over k in B of integral over Omega S_k(f; t) df
```

`S_k(f; t)` is a temporal power spectral density estimated from mode `k` over a window ending at `t`. `B` indexes graph-frequency bands; `Omega` is a temporal-frequency interval. Spatial graph frequency and temporal frequency have separate meanings and units. Record sampling interval, window length, taper, detrending, PSD normalization, band boundaries, and frequency resolution.

Several constraints guide interpretation:

- An orthogonal transform preserves the information in the full observed vector. Predictive gains arise through an estimator's constraints, feature construction, or noise handling. Compare these choices directly.
- Laplacian eigenvalues describe the chosen spatial operator. For `x[t+1] = A x[t] + e[t]`, the transformed dynamics are `z[t+1] = (U^T A U) z[t] + U^T e[t]`. Growth and damping depend on the evolution operator; off-diagonal terms quantify mode coupling.
- Greater band power can reflect stronger forcing, a larger existing disturbance, changed noise, or altered damping. Vary forcing and damping independently in controls and estimate temporal response separately.
- Joint PSD interpretations require a stationarity assessment [R7]. Windowed modal spectra remain computable when modes are coupled, but they omit cross-mode dependence. Examine the cross-spectral matrix as well.
- Eigenvector signs are arbitrary. Individual coefficients within repeated-eigenvalue subspaces also depend on the basis choice. Prefer sign-invariant energies and complete eigenspace bands; test sensitivity near repeated eigenvalues.
- Specify spatial quadrature and graph normalization. Irregular sensor spacing can change the meaning of an unweighted energy sum. Keep the physical amplitude measurement distinct from the graph-coordinate convention.

### Outcome and observation unit

The intended traffic outcome is growth of a moving disturbance's amplitude over a fixed future horizon. A packet entering a fixed sensor region can increase local energy while preserving its amplitude. Include constant-amplitude transport as an explicit control.

The observational unit is an eligible disturbance episode at a forecast origin. Define eligibility through past information only. Measure packet amplitude in physical space using a frozen speed-deviation or equivalent physical measurement, independent of the graph bands being tested. Tracking and amplitude extraction must pass synthetic checks before empirical labels are accepted.

A candidate continuous endpoint is `log((A_future + epsilon) / (A_origin + epsilon))`, with a measurement-based noise floor. This is provisional. Freeze the amplitude definition, tracking rule, noise floor, horizon, and aggregation after a label-blind coverage audit. Include current amplitude as a control and inspect regression-to-the-mean effects. Predicting future amplitude conditional on current amplitude is a sensitivity analysis. Classifying growth versus decay is secondary and needs a predeclared indeterminate range near the noise floor.

Spatial amplification along a platoon, growth at a fixed detector, and moving-packet growth are distinct endpoints. Any comparison must state which endpoint it measures. Report packets that exit the observation region, failed tracking, merges, and missing labels. If reliable packet tracking is unavailable, record a narrower endpoint and revised question before evaluating it.

## 4. Comparisons

Every predictive comparison receives the same past observations and a comparable tuning budget. Graph geometry counts as additional prior information and must be documented. Select preprocessing, bands, ranks, feature counts, and penalties on development data only.

| Model or control | Purpose |
| --- | --- |
| Persistence, recent amplitude trend, and compact traffic-state regression | Establish the value of current conditions and ordinary temporal prediction. |
| Spatial variance, gradients, neighbor differences, and temporal spectra | Test simple spatial/temporal explanations for an apparent spectral gain. |
| Ordinary spatial DFT/DCT with matched temporal spectra and bands | Test whether graph geometry adds value on a corridor; document boundary assumptions. |
| Training-fitted PCA with matched retained dimension | Separate graph constraints from covariance-based compression. |
| Cross-spectral matrix / spectral-envelope comparator | Compare oscillation strength and spatial coherence with [R3]. Obtain full methods before claiming an exact extended spectral-envelope reproduction. A simplified implementation must carry its own name and definition. |
| DMD and Hankel-DMD/Koopman comparator | Compare a specified spatial basis with estimated temporal evolution [R5-R6]; choose rank and delay using development data. |
| GFT-band power; then power plus modal persistence/coupling | Isolate the proposed measurement and the contribution of temporal dynamics. |
| Random orthogonal bases, graph-label permutations, and audited rewiring | Assess representation specificity with matched dimensions and penalties. Record which graph properties each null preserves. |

Use a full-coordinate isotropic ridge rotation-equivalence check as an implementation control, with preprocessing transformed consistently. Per-coordinate scaling or dimension truncation changes that equivalence. Random controls require multiple frozen realizations; one favorable draw is inadequate.

## 5. Experiment register

The status column records progress as of 2026-10-15. The original questions remain in scope; completed controls and simulator pilots supply no empirical Traffic H1/H2 confirmation. The literature reviews have recorded access gaps.

| ID | Test | Main check / decision | Status |
| --- | --- | --- | --- |
| T01 | Numerical and causal fixtures | Orthonormality, Parseval/PSD scaling, eigenspace invariance, future-data mutation, and rotation-equivalence checks pass. | Completed; focused tests plus repository suite |
| T02 | Known graph-aligned linear dynamics | Recover known modal persistence/growth across frozen seeds; label this an oracle-aligned positive control. | Completed; initial and forcing-adaptive runs |
| T03 | Independent dynamics and common forcing | Measure spurious graph-specific gains when graph dependence is absent; distinguish changed forcing from changed damping. | Completed numerical controls; physical forcing study pending |
| T04 | Transport, directional dynamics, and operator mismatch | Constant-amplitude translation receives a no-growth label; measure cross-mode coupling and failures under mismatch. | Completed controls; real-road tracking pending |
| T05 | Physical traffic simulation | Reproduce plausible growing/decaying waves with a documented existing simulator or traffic model; compare all primary methods on held-out runs. | SUMO ring pilot and fresh-setting phase follow-up completed; irregular-road study pending |
| T06 | Public-data feasibility and label audit | Establish access, usable resolution, physical units, independent days, graph construction, and reliable amplitude tracking before selecting a primary real-data cohort. | Derived sample audited; causal inputs, independent days, and packet labels unresolved |
| T07 | Frozen real-data comparison | Evaluate Traffic H1/H2 on reserved days once; report paired losses, uncertainty, exclusions, and every primary comparator. | Planned |
| T08 | Sampling and transfer tests | Apply the frozen method under sensor thinning, coarser sampling, alternate days/geometry, and measured noise. | Planned |

T06 can begin after the plan is recorded; full-scale acquisition should wait for a bounded sample audit. T05 requires successful measurement controls in T01-T04. T07 requires both a reliable endpoint and a frozen protocol. Synthetic results support claims about the generating systems. Physical simulation supports claims about its documented model. Real-road conclusions require empirical evaluation.

### First bounded milestone

Implement T01-T04 with reproducible configuration, separate development and reserved seeds, and a comparison table. Audit one bounded real-data sample for T06. Use those results to decide whether to proceed to physical simulation and a real-data benchmark. Keep the initial model family small; add complexity only to test a recorded failure mechanism.

Progress: this milestone is complete. The [findings](traffic/findings.md) also include physical simulation and a phase-aware follow-up. Strengthening forcing estimation removed the initial GFT-feature advantage over the strongest comparator. A phase-aware Fourier model improves on weaker dynamical models in fresh SUMO settings; recent amplitude trend remains the strongest tested forecast there. The next study should test incremental information under heterogeneous dynamics and ongoing forcing before expanding to an irregular network.

## 6. Data candidates and acquisition gate

| Candidate | Intended role | Audit needed before selection |
| --- | --- | --- |
| I-24 MOTION trajectories [D1] | Preferred freeway candidate for resolving propagation and lane structure. Official documentation describes versioned, freely downloadable trajectories. | Obtain a bounded accessible sample; verify release/version, actual resolution, reconstruction artifacts, lane continuity, recording boundaries, distinct days, and feasible download size. |
| I-24 virtual trajectories [D2] | Possible smaller derived-data entry point. The paper describes trajectories derived from INCEPTION v1.0.0. | Determine whether processing preserves disturbance amplitudes, temporal phase, and the spatial coverage needed by the endpoint. Derived observations share the underlying source. |
| pNEUMA [D3] | Candidate urban geometry extension if graph structure warrants it. | Verify accessible files, terms, time coverage, intersection/lane map, and signal-phase confounding. The download page was inaccessible during the latest check; file suitability is unverified. |
| NGSIM, as used in [R4-R5] | Small replication or trajectory-measurement check. | Verify the exact public release and usable independent episodes. Short recording periods restrict confirmatory inference. |
| METR-LA / PEMS-BAY [D4] | Coarse-resolution sensitivity or longer-horizon forecasting comparison. | Five-minute aggregation cannot resolve a roughly hundred-second precursor at its original scale. Longer-horizon suitability needs a separate endpoint audit. |

Open access and open-source software are different properties. Record data-use terms and software licenses separately. For I-24, retain the documented citation and derivative-sharing conditions [D1]. Do not redistribute raw data until the applicable terms have been checked.

Each acquisition must record source URL, retrieval date, release/version, checksum, file size, units, schema, timestamp meaning, missingness, processing history, and use conditions. Preserve failed requests and rejected samples. Choose sample limits before download; avoid acquiring a large archive merely to inspect its format. Select cohorts by coverage and measurement quality without inspecting their future growth labels.

## 7. Validation, uncertainty, and decisions

Split whole simulation runs or traffic days, with episodes grouped to prevent shared disturbances across partitions. Purge windows whose feature or outcome intervals cross a split. Fit baselines, residual references, scalers, PCA, graph weights learned from data, and all hyperparameters using the allowed training history. Freeze map-derived graphs before evaluation. Online updates may use observations available at the forecast origin under an explicit rule.

The proposed primary score is held-out mean squared error for the continuous amplification endpoint, with equal episode weight and per-day results. Select the strongest comparator using development data. Report paired loss differences and relative changes against that frozen comparator. Aggregate dependent windows within episodes and estimate uncertainty by whole independent days/runs. Inspect sensitivity to grouping; a small number of days supports limited precision. Secondary measures include absolute error and calibrated growth probabilities if event counts permit them. Any alerting extension also needs natural event prevalence, false alarms per hour, and causal alert timestamps.

Before reserved outcomes are accessed, record:

- Cohort, graph, physical endpoint, forecast horizon, eligibility, censoring, and missing-data rules.
- Development/test allocation, seed manifest, sample-size or precision rationale, and fixed compute/download limits.
- Model sets, primary contrast, feature/tuning budgets, null realizations, uncertainty procedure, and a small sensitivity set.
- A practically relevant improvement threshold justified by measurement error and baseline loss; numerical values remain unset pending the label-blind audit and development study.

Treat the first synthetic and data-feasibility studies as development. A local protocol freeze documents analyst decisions; it is not external preregistration. Hash the frozen configuration and record which data/results have already been inspected. Any result-driven revision opens a new experiment entry and requires a fresh holdout for confirmation. Reserve explicit budgets for iterations; retain unsuccessful runs and avoid expanding the search until a positive result appears.

Support for Traffic H1 requires a stable, practically relevant held-out gain with uncertainty reported. Support for Traffic H2 also requires an advantage over appropriate basis/graph controls. Simulation-only success supports a mechanism under specified conditions. Failure of a physical measurement check stops predictive interpretation until repaired. Repeated parity with ordinary spatial transforms should redirect the work toward a resolution or equivalence result. If learned temporal modes work while GFT bands fail, investigate the measured operator mismatch as a separate, bounded question.

## 8. Outputs and presentation

Keep experiment code, frozen configurations, data manifests, validation results, and machine-readable scores together in a dedicated traffic research directory. Record the code commit and environment for each run. Raw third-party data and manuscripts stay outside tracked results unless redistribution is permitted.

The first report should include a physical-space disturbance plot paired with its spectrum; controlled forcing/decay/transport examples; paired held-out loss differences with uncertainty; and a sensing-resolution plot if T08 proceeds. Show the graph, operator convention, units, observation windows, and forecast boundary. Label fitted reconstructions and actual forecasts separately. Select illustrated episodes using a recorded rule and include representative failures. Spectral concentration alone supports a descriptive observation.

The public HTML article will be revised only after the evidence and its limits have been reviewed. Preserve its established voice and distinguish the earlier publication study from this traffic study.

## 9. Reference register

Reading depths below describe the preceding reviews, unless a current documentation check is explicitly stated. The linked review notes contain detailed overlap assessments and source provenance. Published performance figures have not been independently reproduced.

### Closest methods and traffic studies

- **R1. Takada, Azuma and Banno (2025).** [Prediction of Traffic Congestion Using Graph Fourier Transform](https://doi.org/10.11509/sci.SCI25.0_1017). Direct non-neural GFT congestion-warning prior art. All four pages and selected [author scripts](https://github.com/tkd01git/traffic-simulation-modified/tree/5da9642b6915f51238bdc92740b0c845ef93c412) inspected. Published-rule reproduction remains incomplete. Any replication must timestamp persistence-based warnings when the requirement is fulfilled; the audit's script checks do not establish an error in the published performance table.
- **R2. Li, Wang and Ouyang (2012).** [Prediction and field validation of traffic oscillation propagation under nonlinear car-following laws](https://doi.org/10.1016/j.trb.2011.11.003). Frequency-domain prediction of oscillation frequency and amplitude growth. Publisher abstract and institutional record inspected; obtain full methods before an implementation-level comparison.
- **R3. Zhao, Nie and Zhang (2014).** [Extended spectral envelope method for detecting and analyzing traffic oscillations](https://doi.org/10.1016/j.trb.2013.12.007). Close cross-spectral baseline for evolving oscillations. Abstract, introduction, and indexed section excerpts inspected; complete methods remain an access gap.
- **R4. Belletti et al. (2015).** [Prediction of traffic convective instability with spectral analysis of the Aw-Rascle-Zhang model](https://doi.org/10.1016/j.physleta.2015.05.019); [author manuscript](https://bayen.berkeley.edu/sites/default/files/prediction_of_traffic.pdf). Model-based gain, delay, and stability interpretation. Targeted full-text sections and relevant figures inspected; proofs and empirical results have not been independently validated.
- **R5. Avila and Mezic (2020).** [Data-driven analysis and forecasting of highway traffic dynamics](https://www.nature.com/articles/s41467-020-15582-5). Growing/decaying modes and moving-horizon forecasting with Hankel-DMD/Koopman methods. Main-text methods/results inspected; supplementary material and code remain unaudited.
- **R6. Ling et al. (2019 preprint; journal publication 2022).** [Koopman Operator Applications in Signalized Traffic Systems](https://arxiv.org/html/1902.11289v1); [author publication record](https://www.coogan.ece.gatech.edu/papers/ling2019koopman.html). Past-window DMD growth alerts are established. Section V, Algorithm 3, and surrounding definitions inspected; the signalized-queue endpoint differs from freeway packet amplification.

### Spectral foundations and adjacent work

- **R7. Loukas and Perraudin (2019 manuscript).** [Stationary time-vertex signal processing](https://arxiv.org/html/1611.00255v3). Joint stationarity and joint PSD foundation; relevant definitions inspected. Check assumptions before assigning physical meaning to independent graph-mode spectra.
- **R8. Isufi et al. (2019 manuscript).** [Forecasting Time Series with VARMA Recursions on Graphs](https://arxiv.org/html/1810.08581v2). Non-neural graph-temporal forecasting and modewise estimation. Modeling, fitting, and numerical-study sections inspected.
- **R9. Hasanzadeh et al. (2019 revision).** [Piecewise Stationary Modeling of Random Processes Over Graphs With an Application to Traffic Prediction](https://arxiv.org/html/1711.06954v2). Direct non-neural graph-frequency traffic forecasting overlap. Main text, algorithms, data, and results inspected.
- **R10. Mohan et al. (2014).** [Wavelets on Graphs with Application to Transportation Networks](https://doi.org/10.1109/ITSC.2014.6957939). Localization and observed congestion propagation through graph spectral filtering. All six pages read; relevant figures inspected. Useful if global modes fail to resolve localized disturbances.
- **R11. Cao et al. (NeurIPS 2020).** [StemGNN](https://arxiv.org/html/2103.07719v1). Explicit GFT/DFT neural forecasting on overlapping benchmark datasets. Architecture and reported ablations inspected. Relevant representation overlap; the proposed initial experiment uses non-neural estimators.
- **R12. Hollbeck et al. (2023 preprint; 2024 publication).** [Congestions and Spectral Transition in Time-Lagged Correlations of Motorway Traffic](https://arxiv.org/html/2312.12051v1). Correlation-spectrum traffic analysis. Empirical construction and results inspected; its spectral object differs from a road Laplacian.
- **R13. Deri and Moura (2017 manuscript).** [Agile Inexact Methods for Spectral Projector-Based Graph Fourier Transforms](https://arxiv.org/html/1701.02851v1). Directed graph spectral projectors and descriptive taxi activity. Relevant sections inspected; a possible directional-method reference if justified by operator diagnostics.
- **R14. Ji et al. (2025 revision; 2026 journal publication).** [Scalable analysis of stop-and-go waves: Representation, measurements and insights](https://arxiv.org/html/2409.00326v2); [author implementation and public demo](https://github.com/I24-MOTION/wave-analysis). Targeted methodology, data, and processing appendices reviewed on 2026-10-15. Graph-connected wave-boundary extraction supplies additional measurement prior art and a candidate label tool. Its graph and endpoint differ from a road-Laplacian amplification forecast.

### Data references

- **D1. Gloudemans et al.** [I-24 MOTION: An instrument for freeway traffic science](https://arxiv.org/abs/2301.11198v2); [official versioned documentation and data-use agreement](https://github.com/I24-MOTION/I24M_documentation). The documentation and paper abstract were checked on 2026-10-15. Exact release files and their suitability remain unaudited. The official documentation provides the required journal citation.
- **D2. Ji et al. (2023).** [Virtual trajectories for I-24 MOTION: data and tools](https://arxiv.org/abs/2311.10888v1). Abstract checked on 2026-10-15. Full processing details, downloadable products, and preservation of the intended endpoint require review.
- **D3. EPFL.** [pNEUMA project](https://open-traffic.epfl.ch/), [download portal](https://open-traffic.epfl.ch/index.php/downloads/). Candidate source only; download-page access timed out on 2026-10-15. Verify availability and terms before committing to the dataset.
- **D4. Li et al.** [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/html/1707.01926v3). Source description for METR-LA/PEMS-BAY and their five-minute aggregation, inspected in the precursor audit. This reference documents a data source; it does not specify our proposed estimator.
- **D5. Eclipse SUMO 1.28.0.** [Car-following parameters](https://sumo.dlr.de/docs/Car-Following-Models.html), [model definitions](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html), and [libsumo](https://sumo.dlr.de/docs/Libsumo.html). Official documentation checked for the physical pilot. The experiments use the existing IDM implementation; parameter sweeps have no real-road calibration.
- **D6. VT-tools, commit e51e3e98124ab2ef6aa2f2d478f64f82f445aa96.** [ASM source](https://github.com/I24-MOTION/VT_tools/blob/e51e3e98124ab2ef6aa2f2d478f64f82f445aa96/ASM.py). Static inspection finds symmetric temporal support. The default window permits 18 seconds of future observations at a target timestamp; exact demo-processing lineage remains unresolved. See the [audit](traffic/results/data_audit/smoothing.json).

### Reading queue and unresolved overlap

| Item | Why it matters | Next action |
| --- | --- | --- |
| R2 and R3 complete methods | Could narrow the claimed contribution and define the correct growth/spectral-envelope baselines. | Obtain accessible author manuscripts or equivalent complete sources before claiming exact replication. |
| [Incident-analysis preprint 1507.00224](https://arxiv.org/abs/1507.00224) and [1507.00804](https://arxiv.org/abs/1507.00804) | Adjacent spectral-envelope applications. | Full-text review; previous screening used abstracts only. |
| [OpenReview manuscript b6Py2zy0fK](https://openreview.net/forum?id=b6Py2zy0fK) | Indexed material mentions graph-time Fourier neural operators and frequency bands. | Retrieve and verify title, authors, methods, and results. Prior page/API access failed; detailed overlap is unresolved. |
| D1/D2 release documentation and measurement papers | Determines whether observable amplitude growth can be measured reliably. | Audit one bounded sample and record preprocessing effects. |
| Citation updates around R1-R6 | Newer work may duplicate a proposed contrast or supply a stronger baseline. | Repeat a focused citation search before the real-data protocol freeze and before a publication claim. |

## 10. Decision log

| Date | Decision or observation | Evidence / consequence |
| --- | --- | --- |
| 2026-10-15 | Record the plan before implementation. | No new traffic code, data acquisition, experiments, or article changes in this planning step. |
| 2026-10-15 | Study conditional disturbance amplification and representation value. | R1-R9 establish substantial prior art; broad GFT/PSD novelty claims are excluded. |
| 2026-10-15 | Start with falsification controls and a bounded data audit. | Measurement validity and accessible independent observations determine whether a real-data study is feasible. |
| 2026-10-15 | Complete numerical controls and strengthen forcing baselines on fresh seeds. | Initial GFT-feature gain fails to establish superiority against adaptive temporal dynamics; see `traffic/forcing-followup.md`. |
| 2026-10-15 | Advance to a physical SUMO pulse-response pilot and a frozen phase-aware follow-up. | Growing/decaying responses measured; Fourier AR(2) improves on weaker models and loses to recent amplitude trend. |
| 2026-10-15 | Keep the I-24 sample descriptive. | Missing endpoint rows and symmetric upstream smoothing are recorded; strict causal availability and independent-day validation remain unresolved. |

For each later entry, record the experiment ID, frozen configuration/commit, inspected data, result location, conclusion, and reason for any design change. Add references with their access date and reading depth. Preserve this initial plan as a dated record when assumptions or endpoints change.
