# Methods and secondary results

This supplement accompanies the [manuscript](manuscript.md). Complete historical outputs remain in their experiment directories. The tables below describe simulator conditions and conditional measurement tests; they are not independent-road estimates.

## Measurement definitions

The spatial mean is separated before nonuniform power is calculated. Modes q=1,...,11 have two real dimensions; q=12 has one. Full eigenspace powers equal grouped ordinary Fourier powers on the regular cycle. The spatial wavelength is 24/q vehicles. Changing intervehicle gaps prevent a fixed road-distance wavelength throughout a disturbance.

For a temporal window with taper w, integrated power is

\[
\Delta f\sum_j S_q(f_j)=\frac{\sum_t w_t^2 E_q(t)}{\sum_t w_t^2}.
\]

Spatial graph frequencies are discrete mode masses. Temporal frequency bins are summed with their spacing. The 32-sample pulse window has frequency spacing 0.03125 Hz. This coarse transient spectrum does not resolve fine linewidths or establish a stationary PSD. The pulse reporting floor is 10^-12 (m/s)^2. Modal amplitude growth is log(late/early power)/(2 x 48 s). Harmonic windows have centers 10 s apart and use log(late/early)/(2 x 10 s), with a 10^-14 floor.

Window sensitivities use 24 and 40 samples at fixed centers, plus constant temporal detrending. All primary pulse cases retain increasing long-band share under these sensitivities. The exact share changes with the baseline. Saved [mode](results/spectral/mode_summary.csv), [case](results/spectral/case_summary.csv), and [sensitivity](results/spectral/sensitivity.csv) tables retain every outcome.

## Equilibrium and numerical controls

IDM equilibrium satisfies

\[
s_i^*=\frac{2+\tau_i v^*}{\sqrt{1-(v^*/30)^4}},\qquad \sum_i(s_i^*+5)=C.
\]

For d=exp(ik)-1 and c=F_v+F_l exp(ik), the single-iteration discrete mode update used in the model comparison is

\[
T_q=\begin{pmatrix}
1+\Delta t^2 d F_s&\Delta t d(1+\Delta t c)\\
\Delta t F_s&1+\Delta t c
\end{pmatrix}.
\]

The 192 initial-condition responses span three accelerations, q=1,...,12, gap/speed initialization, and steps 0.05/0.025 s. Primary amplitude is 0.01; amplitude 0.1 is a sensitivity for q=1,3,6,12. The 72 finer-step primary cases meet the registered full-field 5% speed-error criterion. Maximum finite-difference versus analytic Jacobian error is 6.45 x 10^-11. Full-field errors include any generated off-mode response. See [validation](results/mechanism/independent_validation.json) and [full-field table](results/mechanism/full_response_validation.csv).

In the harmonic sweep at acceleration 1.5, the smallest-amplitude late q=3 power is below the reporting floor at both steps. Its growth is unreported, and the late scaling fits use three amplitudes. Early-window fits use all four amplitudes.

Three pulse-grid cases change growth sign under step refinement: nominal 480 m, headway 0.6 s, acceleration 0.8 m/s^2, at all three placement seeds. Late/early power ratios are about 1.050 at step 0.2 s and 0.934 at 0.1 s. None of the 32 additional settings changes sign between 0.1/0.05 s. Primary descriptive counts refer to the stated steps, without a convergence-order claim.

SUMO rounds a requested 0.0125 s step to 0.013 s. The affected harmonic and observation studies were preserved in `_timing_audit` directories and rerun at 0.025/0.01 s. The collector now checks actual integration and recording clocks. Original tables, repairs, and source snapshots remain documented in the [timing correction](timing-correction.md) and [clock receipt](results/observation_window/clock_validation.json). Repeatability alone had not detected that time-base error.

Independent continuous DOP853 integrations agree with selected matched SUMO speed responses within 0.642% at step 0.025 s and 0.256% at 0.01 s over 20 s. Actual simulator timestamps are used; insertion/update differences remain included. Continuous solver refinement changes relative speed response by at most 8.92 x 10^-13. These are numerical checks of selected conditions, not validation of IDM against roads.

## Pulse amplitude

Two sweeps fix nominal length 480 m and headway 1 s, with accelerations 0.3/0.8/1.5 m/s^2. Each uses three placement seeds, two integration steps, three pulse strengths, and 240 s recording. Each has 54 responses plus 18 shared controls. Realized same-vehicle endpoint drop D defines strength. Default-run compatibility checks preserve the earlier saved pair exactly.

| Acceleration (m/s^2) | Early exponent, drops 0.5-2 m/s | Early exponent, drops 0.05-0.2 m/s | Final exponent, smaller drops |
| --- | ---: | ---: | ---: |
| 0.3 | 2.903 | 2.088 | 1.433 |
| 0.8 | 2.636 | 2.059 | 2.026 |
| 1.5 | 2.477 | 2.044 | 2.046 |

Values are three-seed means at the finer step. Smaller drops approach quadratic early scaling; later behavior remains amplitude-dependent. Command/release effects have not been decomposed from traffic response, so these pulse slopes do not identify a universal nonlinear law. Full [larger-pulse](results/spectral_forcing/) and [small-pulse](results/spectral_small_pulse/) tables retain all windows and seeds.

## Heterogeneity

Four headway patterns have mean 1 s: homogeneous; alternating 0.7/1.3 s; two contiguous blocks; and one fixed random pattern. At acceleration 0.8, six zero-mean RMS-0.01 m/s initial speed disturbances per pattern and two steps give 48 responses. The known full operator is compared with band-diagonal approximations of matched dimensions 1/4/8/11 in uniform GFT, weighted GFT, and fixed random bases. Weights use equilibrium gap sensitivities, sqrt(F_s,i F_s,i+1), normalized to mean one.

| Headway pattern | Full operator error | Uniform bands | Weighted bands | Fixed random bands |
| --- | ---: | ---: | ---: | ---: |
| Homogeneous | 0.453% | 0.453% | 0.453% | 80.47% |
| Alternating | 0.446% | 24.30% | 24.30% | 82.50% |
| Clustered | 0.462% | 32.04% | 21.55% | 79.93% |
| Fixed random | 0.453% | 16.42% | 15.67% | 81.12% |

These are six-initialization mean relative speed-response errors over 20 s at step 0.025 s. The approximations use known coefficients without fitting. The weighted graph improves selected errors. Its dimensionless off-band Frobenius coupling can still be higher: in the clustered pattern, weighted coupling is higher despite lower response error. The metric does not rank finite-horizon accuracy. See [full tables](results/heterogeneity/).

## Observation design and eligibility

The initial and whole-window observation studies each have 56 interventions and four controls. The uncertainty cohort has 64 interventions/four controls; the adaptive restricted-fit cohort has 32/two. Headways are 1/1.8 s, with acceleration 0.3. Direct q=3 amplitudes are 0.02 m or m/s. Fundamental q=1 speed amplitudes are 0.2 m/s, with 0.4 added in the later cohorts. Source protocols record phases, selection timing, and seeds.

Full sensing, uniform-12, uniform-8, and fixed irregular-12 masks supply own speed and leader gap. Partial reconstruction assumes a finite constant-plus-q=1,2,3,11 dictionary. Uniform-12 q=3 remains identifiable despite two nuisance aliases; uniform-8 aliases q=11 with q=3 and abstains. Unmodeled harmonics can contaminate otherwise identifiable targets. Reconstruction and nullspace checks now share an SVD cutoff of 10^-10; a previous numerical-rank mismatch and its outputs remain in the [sampling correction](sampling-correction.md).

At headway 1 s, low noise, and the fine step, known-operator whole-window harmonic rejection is 52.25% with complete sensing, 23.75% for uniform-12, and 10% for irregular-12. These are conditioned on the stated dictionary, signal, exact paired reference, and specified covariance. They do not establish unrestricted sparse recovery.

An adjacent-innovation check uses residual covariance Sigma + M Sigma M^T. Its Monte Carlo threshold accounts for correlation between adjacent residuals. The whole-window exact-model null has 160 degrees of freedom; fitting four-state offset and drift leaves 152. Estimated-operator plug-in thresholds lack that exact-null guarantee. Empirical calibration uses a balanced mixture of direct-gap and direct-speed responses, which need not calibrate each subgroup.

High-noise fit eligibility in the unstructured study is:

| Method | Ineligible fits / attempted | Reason |
| --- | ---: | --- |
| Known operator | 0/80 | Reference condition |
| OLS | 0/80 | All reported fits eligible |
| TLS | 2/80 | Unidentified trajectory design |
| Known-noise corrected moments | 27/80 | Eight non-positive Gram matrices; 19 unidentified designs |

Thus 29/240 learned fits fail at the highest noise. Their rejection summaries are conditional on eligibility. Pooling them with 80 known-operator cases would obscure method-specific failure. All fitted matrices and statuses remain in [operators.csv](results/uncertainty/learned_operator/operators.csv).

The restricted fit infers headway from exact equilibrium and estimates only acceleration over [0.1,2] m/s^2, using covariance-weighted full-history fitting. TLS estimates an unrestricted complex transition from adjacent snapshots. The result combines stronger prior information, lower parameter dimension, and a different objective. No matched ablation separates those effects.

## Forcing and empirical sensitivities

Explicit q=3 sinusoidal acceleration forcing makes every tested low-noise response reject the autonomous model. Subtracting its known linear forced component reduces rejection to 9.5% at headway 1 s and 3.75% at 1.8 s. The former still exceeds the nominal rate; the finite-amplitude nonlinear response is not exactly the linear forced companion. The [forcing outputs](results/uncertainty/forcing/) retain inputs, numerical integration checks, and the algebraic equal-trajectory construction.

The initial NGSIM grid used 5 s/50 m cells, with 25/100 m sensitivities. US-101 has 13 complete 50 m bins over 650 m. I-80 has only seven and fails the primary eight-bin rule; its 25 m sensitivity has 13 over 325 m. Both 100 m grids fail the minimum. Temporal/spatial resolution tests on fixed supports retain 20 eligible configurations among 32 specified combinations. I-80 fails complete 1/2 s coverage; no imputation occurs. The [resolution tables](results/ngsim_resolution/) record every failure.

The six acquisition blocks belong to the same two recording contexts. Selected fields have complete values and unique selected keys; kinematic flags and trace exclusions remain distinct from a ground-truth cleaning operation. In the last I-80 block, field counts are 155,366 before and 85,492 after exclusion, with one newly empty cell. The [acquisition receipts](results/ngsim_expansion/) retain exact API queries, counts, licenses as reported by the source metadata, and hashes. Unknown preprocessing availability prevents a causal warning interpretation.

## Validation scope

Full repetition and independent recalculation answer different questions. Complete repeated simulation and analysis stages reproduce retained numerical tables. Independent FFT/Hann, physical-coordinate dynamics, GLS, and estimator checks cover the scope stated in each receipt. The uncertainty check independently verifies first-phase parameter/reference rows, all unstructured fitted matrices, and first/last training-noise fits for diagnostic replay. It is not exhaustive independent recomputation of every noise draw.

See [spectral validation](results/spectral/independent_validation.json), [mechanism validation](results/mechanism/independent_validation.json), [observation validation](results/observation_window/independent_validation.json), and [uncertainty validation](results/uncertainty/validation.json). Historical protocols, raw-array hashes, and frozen source text are preserved. No existing output table was regenerated for this editorial revision.
