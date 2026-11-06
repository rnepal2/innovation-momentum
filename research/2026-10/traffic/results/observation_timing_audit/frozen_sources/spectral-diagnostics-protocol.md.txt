# Spectral scale selection and forcing dependence

Recorded 2026-10-15 before computing the spectral summaries below. Existing aggregate outcomes have been inspected. This is a descriptive follow-up to the recorded traffic pilots, with a new controlled pulse-amplitude experiment. It supplies simulation findings and a manuscript draft; empirical traffic validation and novelty remain unresolved.

## Questions

1. Which spatial scales amplify or decay in the paired traffic response? Does long-wavelength power concentration distinguish the observed response regimes?
2. Does integrating finite-window spectral density reproduce the corresponding time-domain mean-square fluctuation, with a useful allocation across spatial and temporal scales?
3. How do absolute power and normalized spatial shape change when pulse strength varies independently of driver-response parameters?

## Fixed spatial decomposition

Use the existing 24-vehicle cycle and its combinatorial Laplacian. The signal is paired speed difference `x = v_pulse - v_control`, in m/s. At each time, retain the spatial mean `m` separately and transform `r = x - m`. For each complete eigenspace indexed by ring wavenumber q, define `E_q = ||P_q r||^2 / N`. The exact identity is `mean(x^2) = m^2 + sum(E_q)`. This quadratic speed measure has units (m/s)^2; vehicle masses and physical kinetic energy are outside its definition.

Group conjugate spatial modes before calculating power. For q=1,...,11 there are two real dimensions; q=12 has one. The q=0 component is the separately reported uniform response. GFT and ordinary spatial FFT powers must agree. Never integrate over irregularly spaced Laplacian eigenvalues as though the mode masses were a continuous density.

Predefined spatial bands: long q=1-2 (12-24 vehicles per wavelength), intermediate q=3-6 (4-8 vehicles), and short q=7-12 (2 to approximately 3.43 vehicles). An isotropic, nonuniform signal allocates expected fractions 4/23, 8/23, and 11/23 by real dimension. Include that reference when interpreting concentration. Report absolute power and fractions together; a fraction can rise during absolute decay. Fractions are undefined when nonuniform power is below 1e-12 (m/s)^2. This is a numerical reporting floor, not measured sensor noise.

## Finite-window spectra and growth

Primary temporal estimator: SciPy one-sided Hann periodogram with density scaling, one-second sampling, and no temporal detrending after spatial demeaning. Sum frequency bins times df and sum graph modes, then divide by N. This recovers the taper-squared-weighted mean-square fluctuation. It is distinct from unweighted instantaneous power. Retain the temporal zero-frequency bin. Constant temporal detrending is a documented sensitivity with a different energy definition.

Use 32-sample early [4,36) and late [52,84) windows, whose centers differ by 48 seconds. Define finite-window modal amplitude growth `g_q = log(E_q_late/E_q_early)/(2*48)`, only when both powers exceed the floor. This is a response growth statistic, without an inferred intrinsic damping rate. Use 24- and 40-sample windows centered at the same times, plus constant detrending, as fixed sensitivities. Spectral peaks have limited temporal resolution in these short transients. No narrow-linewidth, resonance, or stationary-covariance claim follows.

Summarize all 108 saved grid responses and all 64 fresh-parameter responses. Primary grid comparisons use integration step 0.1 s; compare 0.2 s as a numerical sensitivity. Primary fresh-setting comparisons use 0.05 s, with 0.1 s as a sensitivity. Numerical replicas and repeated initializations are dependent. Report per-setting results and ranges; avoid treating them as independent real-road events.

Illustration rule fixed before spectral inspection: nominal 480 m, tau=1.0 s, seed=31, dt=0.1 s, comparing acceleration 0.3 and 1.5 m/s^2. Their aggregate growth/decay outcomes were already known. Show absolute and normalized time-mode power, early/late joint mode-frequency density, and all-cell band growth. Signed Fourier density is a companion for propagation direction; undirected GFT power alone supplies no direction. Spatial frequency uses cycles per vehicle index. Converting it into a road-frame wave velocity requires a separate positional measurement.

## Pulse-strength experiment

Retain 24 vehicles, nominal 480 m, tau=1.0 s, and acceleration in {0.3, 0.8, 1.5} m/s^2. These three settings are selected using the previous sweep and are a mechanism comparison. Request pulse drops {0.5, 1, 2} m/s over one second, then release external control. Use seeds {31,47,83} and integration steps {0.1,0.05} s. Record through 240 seconds after pulse initiation. Reuse the same unperturbed simulation within each acceleration/seed/step combination: 18 controls plus 54 pulse runs, or 72 simulator runs for 54 paired responses.

Measure the realized same-vehicle drop at the pulse endpoint; compare it with the command. Invalid retention, nonfinite values, collisions, or teleports prevent physical interpretation and must be reported. Normalize fluctuation power by realized pulse drop squared. Compare the same early/late windows as above and late [208,240) summaries for nonlinear evolution. The main checks are amplitude-scaling agreement and spatial-band concentration, without a new forecast metric. Report requested and realized drops, seed ranges, and integration-step sensitivity. Larger departures from quadratic scaling can arise through nonlinear response; attribution to a particular traffic mechanism requires further checks.

## Validation and manuscript

Required tests: spatial Parseval/mean decomposition, PSD integration versus direct taper-weighted power, eigenspace rotation invariance, exact GFT/FFT equivalence on a cycle, amplitude-scaling and zero-signal fixtures, known growing modes, signed traveling-wave orientation, and past-window isolation. Independently recompute key summaries using direct spatial FFT and raw response arrays.

Archive source and protocol content before the new runs, retain raw-input hashes, and preserve all parameter cells. Figures must share scales where compared and keep units/normalization explicit. The manuscript will report simulation scope, prior spectral work, unresolved priority, and the earlier negative forecasting comparison. No public-article revision or submission is part of this step.
