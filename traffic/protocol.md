# Initial traffic research protocol

Recorded on 2026-10-15 before running this benchmark. Parent design: [research plan](traffic-research-plan.md). Branch: `feature/traffic-spectral-amplification`.

## Scope

This milestone implements T01-T04 and audits a bounded public-data sample for T06. The numerical systems are controls with dimensionless state. Calibration to traffic is outside this numerical study. Their spatial RMS endpoint is measured over a fully observed domain. They do not implement real-road packet tracking or establish Traffic H1/H2 empirically. T05, T07, and T08 require separate protocols.

## Frozen synthetic design

- Two fixed 24-node geometries: uniform path and a weighted path with a six-node side branch. The operator is the unnormalized combinatorial Laplacian. Node order is documented in code. Spatial RMS uses equal node weights in these artificial systems.
- Five mechanisms: graph-aligned symmetric dynamics; a fixed hidden orthogonal evolution basis; independent isotropic AR dynamics; isotropic AR with shared external forcing; and a non-normal directed chain. Matrices are fixed within each case. Episode initial states and forcing strengths vary independently.
- Graph-aligned eigenvalues of the evolution operator are 0.985, 1.008, and 0.94 in three equal-count graph bands. The hidden-basis control rotates this same spectrum independently of road geometry. The directed chain has diagonal 0.92 and superdiagonal 0.15. It tests transient amplification despite asymptotically decaying eigenmodes.
- Each episode has 64 past samples and a 16-sample future. Time step is two nominal seconds; this is a sampling convention without a physical traffic calibration. Initial-state scale, process-noise scale, and initial coordinates are independent draws. Measurement-noise SD is 0.02.
- Training seeds 1000-1319; validation seeds 5000-5095; reserved evaluation seeds 10000-10191. One origin per episode prevents overlapping windows. Each mechanism/geometry uses a separate deterministic stream derived from the episode seed. Reusing seed identities across cases creates paired controls, rather than independent population replications.
- Amplitude is RMS over all nodes and the last four observations in the corresponding interval. Target: `log((future_amplitude + 1e-6) / (origin_amplitude + 1e-6))`. Future samples are used only for labels. The endpoint includes continuing forcing, so it measures realized amplitude change. It does not isolate a physical damping parameter.

The primary score is test MSE, with equal episode weights. Select the strongest non-GFT comparator on validation MSE. The primary contrasts are GFT power and GFT power-plus-dynamics against that frozen comparator. Also report all models against the state baseline. Use 2,000 paired episode-bootstrap replicates for descriptive 95% intervals. Ten artificial cases and multiple models are a development benchmark; intervals do not establish population-level significance or empirical novelty. No practical traffic improvement threshold is assigned to arbitrary-unit controls.

## Estimators

The state baseline contains physical RMS history, recent amplitude growth, mean level, spatial spread, ordinary coordinate differences, and graph-local differences. All ridge models use training-fitted standardization and the same penalties: 0.1, 1, 10, and 100. Validation chooses a penalty within each model. Test data never select it. Models remain fitted on training episodes after selection.

For GFT, DCT, and training-fitted PCA, compare the same three coordinate bands. Power features comprise three recent band-energy fractions and six integrated temporal PSD features. Dynamics adds three recent-to-early band amplitude changes and three pooled lag-one persistence estimates. Temporal PSD uses SciPy's one-sided Hann periodogram, constant detrending, and density scaling. Integrate by summing discrete-bin power times frequency spacing, with a split at one eighth of Nyquist. This is a windowed measurement; no joint-stationarity assumption is imposed on these transient simulations.

Three fixed random orthogonal bases and three fixed graph-label permutations receive identical power-plus-dynamics features and ridge grids. Report each realization and their mean loss. A permutation preserves graph spectrum and topology while changing the node assignment. Eigenvalue-band construction keeps repeated eigenspaces intact. On the uniform path, the chosen GFT and DCT should give equal subspaces and equal features up to numerical precision.

Additional comparators are a simplified Welch cross-spectral leading-eigenvalue summary, full-state DMD, delay-two Hankel linear dynamics, and band-shared AR in GFT/DCT coordinates. The cross-spectral summary is explicitly distinct from an exact reproduction of Zhao et al.'s extended spectral-envelope algorithm. DMD uses truncated-SVD regression with ranks 4, 8, and full dimension; Hankel uses ranks 8, 24, and full dimension. Their forecasts propagate estimated residual covariance and forecast mean. Validation chooses rank. Band AR fits one transition coefficient per spatial band and estimates full innovation covariance. These models share an evolution law across episodes; changing forcing strengths can violate their constant-covariance approximation.

## Required validation

Test PSD integration against the taper-weighted time-domain energy, orthogonality, path/DCT equivalence, sign/eigenspace invariance, past-only feature extraction, training-only basis/scaler fitting, run-disjoint splits, exact deterministic amplitude evolution, covariance propagation, and full-coordinate isotropic ridge rotation equivalence. A transport fixture shifts a periodic packet at constant amplitude: whole-domain amplitude stays fixed while a detector-window amplitude rises.

Independently recompute scores and paired contrasts from saved per-episode predictions. Preserve all cases and models. Failed sanity checks must be resolved before interpreting predictive comparisons. Record repairs and any design changes in `ledger.md`. Configuration/source hashes and package versions accompany outputs.

## Public-data audit

Download at most 64 MiB for a single version-pinned I-24 wave-analysis demo CSV. Preserve the accepted file on fetch failure, enforce size and optional checksum, and store source/size/hash/retrieval metadata. Audit schema, units, finite values, duplicate vehicle/time keys, monotonic trajectories, time increments, spatial coverage, and speed-position consistency. Record that virtual vehicles are derived trajectories and supply no independent vehicle count.

Inspect version-pinned VT-tools smoothing code without executing it. Establish temporal support before any forecasting use. A processed file with unresolved availability timing can support descriptive checks; it cannot supply a strict past-only forecasting result. The demo alone cannot establish independent-day replication. No raw dataset redistribution is planned.

## Proceed / stop

Proceed to a physical simulator only after the numerical and transport checks pass. Proceed to a real-road predictive protocol after accessible inputs, causal availability, and a physical endpoint are verified. If ordinary DCT is equivalent on a path, retain that result and focus graph-specific questions on geometry where the operators differ. A gain over a weak state model alone does not establish superiority over learned dynamical models. Preserve null or unfavorable results.
