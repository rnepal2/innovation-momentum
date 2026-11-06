# Small-pulse scaling check

Recorded after inspecting the spectral diagnostic and 0.5/1/2 m/s pulse sweep, before running this check. The first sweep produced superquadratic early-window power exponents in all three acceleration settings, including the decaying response. Those pulses therefore do not establish a linear-response regime.

Repeat the same nominal 480 m, 24-vehicle, tau=1 s design at requested drops 0.05, 0.1, and 0.2 m/s. Use accelerations 0.3, 0.8, and 1.5 m/s^2, seeds 31, 47, and 83, and integration steps 0.05 and 0.025 s. Retain the 240 s observation horizon, paired control, one-second intervention, and existing windows. This is 18 controls plus 54 interventions. The finer integration step tests whether command discretization explains amplitude sensitivity.

The diagnostic is approach toward quadratic power scaling as drop decreases, with matched normalized power and spatial shape. A residual departure can involve the imposed speed command, its release, the simulator's response law, or nonlinear traffic evolution. This experiment alone cannot separate those sources. Retain all outcomes and compare both step sizes; select no new forecast endpoint. Do not infer intrinsic eigenvalues or a bifurcation boundary from the results.
