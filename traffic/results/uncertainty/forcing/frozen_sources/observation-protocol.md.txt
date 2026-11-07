# Mode consistency under observation limits

Execution notes: the [clock correction](timing-correction.md) replaces the requested 0.0125 s step with 0.01 s. The [sampling correction](sampling-correction.md) aligns inversion and identifiability cutoffs. Initial outputs and source snapshots are retained; corrected outputs are in `results/observation`.

Recorded after the mechanism, heterogeneity, harmonic, and first NGSIM studies. Their outcomes and the q=3 autonomous comparison have been inspected. No outcomes of the experiments below have been collected. This is a controlled feasibility test with known dynamics and specified noise, without a real-road forecasting or novelty claim.

## New physical controls

Use the nominal 480 m, 24-vehicle ring, acceleration 0.3 m/s^2, and headway either 1.0 s or 1.8 s. The second headway was selected by the already derived equilibrium Jacobian: q=3 has positive linear growth there, whereas q=3 has negative linear growth at headway 1.0. This is an analytic positive-control selection, not a population sample.

For each headway, initialize either q=3 gap amplitude 0.02 m, q=3 speed amplitude 0.02 m/s, or q=1 speed amplitude 0.2 m/s. A separate q=11 speed-amplitude-0.2 control at headway 1.0 tests spatial aliasing. Use phases 0,pi/4,pi/2,3pi/4 and integration steps 0.025/0.0125 s. Record through 40 s every 0.1 s. Reuse a control per headway/step: 56 interventions plus four controls. Initial phase changes the actual wave, without shifting a graph-label interpretation.

## Observations and conditional identification

Compare all 24 vehicles, uniform 12, uniform 8, and 12 irregular vehicles selected with seed 26016. Each sampled vehicle supplies own speed and leader gap. The paired response is assumed available with an exact reference; actual empirical baseline error remains outside this experiment.

Compare a naive constant/cos(q=3)/sin(q=3) reconstruction with a declared spatial dictionary containing the constant and sine/cosine pairs q=1,2,3,11. A dictionary with incomplete spatial samples is an additional band-support assumption, not a guarantee. Compute its rank and whether q=3 coefficients are individually identifiable from its nullspace. A rank deficiency restricted to nuisance modes need not prevent q=3 identification. Report nonidentifiable targets as abstentions. Under an unrestricted 24-dimensional spatial signal, partial samples do not uniquely identify all modes; the conditional dictionary result must never be presented as unrestricted recovery.

Add independent, temporally white Gaussian measurement error to the paired gap/speed channels, with response-level standard deviations (gap m, speed m/s): (0,0), (0.01,0.001), (0.05,0.005), (0.2,0.02). These are explicit synthetic noise scenarios, without a fitted sensor specification. Propagate the spatial least-squares noise covariance exactly to the real/imaginary q=3 gap/speed state.

## Autonomous-consistency diagnostic

Subsample the recovered state at 0.5 s. Over samples 10-30 s, compare z[t+1] with the known discrete two-state q=3 transition. For nonzero noise use the mean squared Mahalanobis residual, with covariance Sigma + M Sigma M^T. Calibrate a 95th-percentile rejection threshold with 2,000 pure measurement-noise simulations, preserving the adjacent-residual temporal dependence. Use disjoint evaluation noise seeds and 100 noise replicates per physical response. The dynamics and noise covariance are known: this tests observation feasibility, not identification of an unknown traffic operator.

For zero noise use relative residual norm with a fixed 1% numerical-consistency threshold. Report q=3 signal/noise power and rejection frequency; a nonlinear response can be too small to detect at a given noise level. Direct q=3 initializations supply approximate autonomous controls. Fundamental-only responses test nonlinear excitation. q=11 supplies an alias control, without a genuine q=3 signal at linear order. No classifier, feature selection, or threshold tuning is allowed using evaluation outcomes. Retain complete parameter/noise/mask tables.

## Empirical resolution and quality check

Use the same accepted NGSIM blocks, with fixed supports [0,650) m for US-101 and [0,325) m for I-80. Compare temporal bins 1/2/5/10 s and spatial bins 25/50 m where the support divides exactly. Require at least eight spatial bins and complete cells across the full support; no interpolation or support reselection. Report every ineligible configuration. Compare retained and removed temporal mean as before.

As a secondary quality sensitivity, exclude entire vehicle traces with any position/speed discrepancy greater than 5 m/s in the selected block. This follows the earlier diagnostic, is recorded before filtered spectra, and changes the observation population. Keep unfiltered results primary. Reapply the same fixed-support coverage rule; no imputation or cell replacement. This is a retrospective sensitivity study, with no causal alert timestamp or empirical instability label.

## Validation

Check mode/phase initialization, nullspace identifiability, exact alias fixtures, covariance propagation, null false-rejection calibration, zero-noise linear residuals, and fixed support/coverage selection. Preserve all raw physical pairs and sources. Independently verify consequential results and update manuscript claims to the attained scope.
