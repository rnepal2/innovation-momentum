# Whole-window autonomous consistency follow-up

Execution notes: the [clock correction](timing-correction.md) applies a verified 0.01 s fine step; the [sampling correction](sampling-correction.md) repairs rank-deficient reconstruction. These are subsequent validation repairs. Original outputs remain preserved and corrected results are in `results/observation_window`.

Recorded after inspecting the observation study. Its 0.5 s innovation diagnostic rejects noiseless fundamental-driven q=3 histories and accepts direct q=3 controls. With nonzero specified noise, rejection frequencies for the full observation mask remain near the noise-null rate, including the lowest noise scenario. Whole-mode power can exceed its measurement-noise power even when the short-step evolution residual is poorly detected.

## Question and fixed comparison

Test whether fitting a complete autonomous trajectory preserves more information about model inconsistency than the adjacent-step innovation diagnostic. This is a retrospective, known-operator diagnostic. It estimates an initial four-dimensional real q=3 state using the entire analysis window. It supplies no causal warning lead time, parameter-identification result, or GFT-specific advantage.

For the 41 samples at 0.5 s spacing over 10-30 s, form the design with block rows I,M,...,M^40. Whiten each state by its known observation-noise covariance. Fit the initial state by least squares and project the whitened history onto the orthogonal complement of this four-column design. Use residual squared norm divided by 41. Under the exact autonomous linear model with independent Gaussian measurement errors, the unscaled statistic has chi-square distribution with 164-4=160 degrees of freedom. Calibrate its 95th percentile with 2,000 pure-noise histories as for the first diagnostic; independently check this against the chi-square quantile. For zero noise retain a 1% relative whole-history residual threshold and the same numerical power floor.

## Fresh physical responses

Repeat the full physical design in observation-protocol.md, replacing the previously inspected phases by pi/8,3pi/8,5pi/8,7pi/8. These phases are fixed here before collection. Keep headways, amplitudes, integration steps, duration, masks, band assumptions, and noise levels unchanged. This yields 56 interventions and four controls. Evaluate both diagnostics on these responses. Use 100 evaluation noise histories per response and independent seeds starting at 170000; calibration seeds start at 180000. No amplitude, time-window, threshold, or sampling-mask selection follows these outcomes.

## Interpretation and validation

Compare harmonic-response rejection with direct q=3 controls for the complete observation mask first. Report every mask and phase; preserve nonidentifiability abstentions. The analytic noise null and explicit covariance assumptions govern the rejection interpretation. Omitted harmonics under partial sensing can produce reconstruction error even when the declared q=3 coefficients are identifiable within the finite dictionary. A rejection is evidence against that autonomous observation model, without uniquely identifying a physical nonlinear mechanism.

Validate exact autonomous histories with unknown initial state, the residual projector rank, chi-square calibration, and independent physical-coordinate propagation. Record the adaptive sequence explicitly. A complete rerun must reproduce numerical tables. Real-road forecasting and publication priority remain unresolved.
