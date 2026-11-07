# Operator, reference, and forcing validation

Recorded after the corrected observation studies and their 52.25% low-noise whole-window harmonic rejection result. None of the new physical outcomes or analyses below has been inspected. This is a bounded simulation/measurement study. Real-road instability labels, causal warnings, and publication priority remain unresolved.

## U1: fresh physical evaluation and operator error

Use the 24-vehicle nominal 480 m ring, acceleration 0.3 m/s^2, headways 1/1.8 s, and exact simulator steps 0.025/0.01 s. Four initial conditions: q=3 gap amplitude 0.02 m, q=3 speed amplitude 0.02 m/s, q=1 speed amplitude 0.2 or 0.4 m/s. Phases are pi/16 plus 0,pi/4,pi/2,3pi/4. Record every 0.1 s for 40 s and retain actual simulator timestamps. This gives 64 interventions plus four controls. Preserve every safe finite case and all failures. Require fresh raw/output directories.

Primary observations cover all vehicles. Use the same four-dimensional real q=3 gap-speed state and 41 samples over 10-30 s. Nonzero response noise standard deviations are the previously specified (gap m, speed m/s) pairs (0.01,0.001), (0.05,0.005), (0.2,0.02). Synthetic noise is independent of traffic dynamics and is uncalibrated to a real sensor.

Perturb assumed acceleration or headway by -20,-5,-1,0,+1,+5,+20 percent, one parameter at a time. Evaluate derivatives at the measured equilibrium gap/speed; keep the true equilibrium state fixed. Compare rejection against the nominal chi-square threshold. A direct-mode rejection under parameter error diagnoses model sensitivity; it does not establish a new physical instability.

## U2: independently learned operators

Training uses only the first observation study's direct q=3 gap/speed histories at 0-10 s, all four original phases, matched headway and step. Calibration uses only the second observation study's direct histories at 10-30 s, with its distinct phases. Those datasets have already been inspected and are development data. Fresh U1 phases supply evaluation only. Calibration and evaluation amplitudes for direct controls remain 0.02; the harmonic responses are absent from operator fitting and threshold calibration.

For each headway/step/noise level use 20 independent training-noise realizations. Fit complex two-state transitions by ordinary least squares, noise-whitened rank-two total least squares, and known-noise corrected moments. The last uses Y X* (X X* - n Sigma_complex)^-1. Report non-positive corrected Gram matrices and condition numbers above 1e8 as abstentions; do not silently regularize or select a method by evaluation. Known exact operators supply a reference. These estimators are established system-identification methods, with no new-estimator claim.

For each fit compare the plug-in exact-model chi-square cutoff with a 95th-percentile cutoff from separate direct-mode calibration responses, using 40 noise draws per calibration response. Evaluation uses 100 noise draws per physical response. Use paired noise across methods and disjoint training/calibration/evaluation streams (seed bases 310000/410000/510000). Save every fitted matrix, conditioning, growth-sign error, cutoff, and phase-specific rejection frequency. Conditional empirical calibration samples a specified mixture of direct responses; it supplies no distribution-free coverage or independent-road claim. Monte Carlo repetitions are not new physical environments.

## U3: reference and temporally correlated errors

With the known operator, use temporal AR(1) correlation rho=0,0.5,0.9. Compare exact reference, a constant reference error estimated from K=1 or 5 independent reference measurements, and a constant-plus-linear reference error with K=5. The latter has coefficients with the same variance as the K=5 mean; its linear time coordinate spans [-1,1] over the analysis window. These are specified observation scenarios, not inferred drift characteristics.

The separable joint covariance is R_time tensor Sigma_state, where R_time is the AR correlation plus 11^T/K and, for drift, tt^T/K. Compare an iid/exact-reference analysis, joint GLS with the correct covariance, and GLS using the known AR correlation while fitting constant and linear four-state nuisance terms. Fit nuisance coefficients on the complete retrospective window. Use chi-square degrees of freedom equal to 164 minus verified design rank. Evaluate all new physical cases with 100 noise histories per scenario/noise level, paired across methods. Use a separate seed base 610000. An exact-autonomous synthetic null independently checks calibration with at least 5,000 draws per observation scenario. No finite-window guarantee is claimed with unknown noise covariance.

## U4: forcing ambiguity and independent integration

Use SciPy solve_ivp (DOP853) to integrate the full 48-state continuous IDM system, with gaps obeying leader-speed minus own-speed. This supplies an independent integration path alongside SUMO. Headways 1/1.8, acceleration 0.3, and the four fresh phases remain fixed. Compare direct q=3 speed amplitude 0.02 and q=1 speed amplitude 0.2 with matched SUMO responses. Also integrate equilibrium initial states under spatial q=3 acceleration forcing of amplitude 0.002 m/s^2 or spatial q=1 forcing of amplitude 0.02 m/s^2, each multiplied by cos(2 pi 0.05 t). This gives 32 primary ODE responses. Primary tolerance rtol=1e-8, atol=1e-10, max_step=0.1 s; repeat at 1e-10/1e-12/0.05. Report maximum response difference and gap conservation. State collection starts at t=0, so compare SUMO at its actual first timestamp plus recorded elapsed times.

For q=3 forcing, use the exact continuous linear forced response as a known-input companion. Compare autonomous and forcing-adjusted whole-window diagnostics under the three noise levels. A rejection of autonomous dynamics can result from an external input even when the linear mode is stable. Report finite-amplitude departures of the known-input model as well.

Construct a separate exact linear counterexample with headway 1.8: the growing q=3 model at acceleration 0.3 and a decaying q=3 model at acceleration 1.5 share the same equilibrium and gap kinematics. Prescribe an acceleration-only input u(t)=(B_growing-B_decaying)z_growing(t). Independently integrate the decaying forced system and compare its state and spectral power with the growing autonomous trajectory. This is a constructive identifiability limit for unknown inputs; it establishes no observational equivalence under known forcing or unrestricted nonlinear IDM.

## Validation and stopping rule

Check direct physical safety, hashes, source snapshots, clocks, training/calibration/evaluation separation, whitening, complex-noise normalization, covariance propagation, projector rank, exact-linear false-rejection calibration, TLS equivalence to augmented-snapshot projection, and known-noise moment consistency. Independently recompute consequential tables. Repeat retained physical collection and analyses using fresh directories. Keep failures and subsequent repairs explicit.

Complete these four stages before selecting another expansion. If model/noise misspecification destroys calibration, report that result and its remedied/reference comparisons. Do not expand parameters until a favorable outcome appears. Irregular roads, driver-model transfer, and real-road labels stay in the research queue; this study does not claim to exhaust them.

## Method references checked 15 October 2026

- Dawson, Hemati, Williams, and Rowley (2016), [Characterizing and correcting for the effect of sensor noise in the dynamic mode decomposition](https://arxiv.org/html/1507.02264v3). Sections 2.1-2.5 inspected, including noise correction and TLS algorithms. Numerical examples were not replicated. The refined known-noise moment form and TLS block solution motivate the specified baselines.
- Hemati, Rowley, Deem, and Cattafesta (2017), [De-biasing the dynamic mode decomposition for applied Koopman spectral analysis](https://arxiv.org/html/1502.03854v2). Relevant least-squares, augmented-projection, and conditioning discussion inspected. The Princeton PDF fetch returned 401; the accessible arXiv methods supplied the implementation reference. No new noise-aware DMD method is claimed.
