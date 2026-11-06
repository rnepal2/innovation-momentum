# Follow-up: phase-aware traffic dynamics

Recorded after examining `results/sumo_placement`, before new parameter settings are simulated.

## Observation and hypothesis

The ring pilot contains growing and decaying pulse responses. Three-band scalar AR forecasts substantially overestimate decay, while an ordinary recent amplitude trend has lower error. A scalar lag-one coefficient can confound phase rotation with damping. A rotating constant-amplitude pair has lag-one inner-product persistence `cos(theta)`, although its norm stays fixed. Temporal phase and second-order dynamics deserve a direct check.

Test a complex AR(2) independently at each ordinary spatial Fourier frequency. This is an established spectral time-series construction, with no claim of methodological priority. It can approximate the two-state speed/gap dynamics of homogeneous following through a speed-only representation. Its validity for nonlinear perturbations is empirical within this simulator.

## Fixed model and evaluation

For each frequency, fit `z[t+1] = a z[t] + b z[t-1]` by complex ridge regression. Penalty is `1e-4 * trace(X^H X)/2`, with a 1e-12 numerical floor. Use only seconds 4-31, after the external pulse has ended. Forecast thirty seconds without future forcing. Preserve the complex coefficients and conjugate symmetry through real-input FFT/inverse FFT. No root clipping or test-selected penalty is allowed.

Compare zero growth, the prior last-eight-sample amplitude trend, the prior rank-four DMD, three-band GFT AR, a raw-coordinate ridge VAR(2) using the same post-pulse samples and relative penalty, and the phase-aware spectral model. The raw VAR comparator separates a spectral constraint from merely adding a second lag. Both new models use a fixed relative ridge rule and the same history. The spectral model constrains spatial translation symmetry on the ring; raw VAR has more coefficients. There is no equal-capacity claim.

Generate 32 new parameter settings with parameter seed 4261. Choose nominal circumference independently from {512, 640} metres; draw tau uniformly in [0.55, 1.9] seconds and acceleration uniformly in [0.25, 1.6] m/s^2. Use simulation seeds 101-132 and repeat at steps 0.1 and 0.05 seconds. Other vehicle/pulse settings and the endpoint are unchanged. This is 128 simulations (64 paired responses). Model choices are frozen before reading their outcomes.

Report mean loss over the two integration steps per setting, then equal-weight settings. Paired bootstrap intervals resample whole parameter settings, never individual integration-step replicas. These settings test transfer within an analyst-specified simulator distribution. They supply no empirical traffic validation. Report integration-step differences and all invalid scenarios. Treat near-zero growth cautiously: the earlier sweep had sign changes under integration-step refinement near zero.

The new method is ordinary Fourier AR(2) on a regular ring. Any benefit supports phase-aware dynamics under those conditions. A graph-specific or irregular-network claim requires another study.
