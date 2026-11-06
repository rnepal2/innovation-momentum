# SUMO impulse-response feasibility study

Recorded on 2026-10-15 before the physical simulation sweep. This advances T05's measurement checks. It supplies simulation evidence under specified IDM parameters; empirical Traffic H1/H2 remain open.

## Fixed design

Use SUMO/libsumo 1.28.0, its existing IDM car-following implementation, and a one-lane circular network with 24 homogeneous vehicles. Construct circles with nominal circumference 480 or 720 metres, then read actual lane lengths after network conversion. Set vehicle length 5 m, minimum gap 2 m, desired speed 30 m/s, comfortable deceleration 2 m/s^2, acceleration in {0.3, 0.8, 1.5} m/s^2, and desired time headway in {0.6, 1.0, 1.8} seconds. Tau is a desired headway parameter. Integration step and decision interval are recorded separately.

Run three fixed seeds {31, 47, 83}; randomize a common ring phase and add position jitter of at most 0.02 m. Start at the homogeneous IDM equilibrium speed computed for the actual mean gap. Allow 60 seconds to settle, then slow one vehicle by 2 m/s over one second and release external speed control. Run the paired unperturbed scenario with identical initialization. Preserve normal SUMO safety checks. Collect speeds by persistent vehicle identity at one-second intervals through 90 seconds after pulse initiation. Repeat at integration steps 0.2 and 0.1 seconds. The complete sweep has 216 simulations (108 paired responses).

## Endpoint and checks

The state is the speed difference between each perturbed vehicle and its same-ID unperturbed counterpart. This isolates the response to a known pulse in simulation. Such a counterfactual is unavailable in ordinary observational traffic data. Whole-platoon RMS measures disturbance strength as it moves around the complete observed ring. A fixed detector entry effect cannot produce this endpoint by itself.

Use samples 0-31 as past history and compare the last four samples with samples 58-61 for a 30-second-ahead log amplitude ratio, with floor 1e-6 m/s. Report the full amplitude trajectory and parameter-cell results, including decay and growth. Check collisions, teleports, vehicle retention, finite measurements, disturbance size before the pulse, and integration-step sensitivity. A failed physical check excludes predictive interpretation of that scenario and must be reported.

The primary purpose is endpoint/model feasibility. Three frozen illustrative forecasts are zero log growth, last-eight-sample log-amplitude trend extrapolation, and past-window rank-four DMD with zero future forcing. Add three-band GFT AR and its exactly matched ordinary real-Fourier counterpart, also with zero future forcing. No hyperparameter selection or empirical priority claim follows. Group summaries by parameter cell; repeated seeds and integration steps are numerical replications with dependence, rather than independent real-road events. A predictive benchmark with held-out traffic conditions remains a later protocol.

The vehicle-adjacency graph is a cycle. Its eigenspaces coincide with ordinary Fourier frequency subspaces. This pilot validates physical measurements and supplies an explicit equivalence check; it cannot establish an advantage specific to an irregular road graph.

Initialization repair before endpoint analysis: the first attempt failed the vehicle-retention check because sequential insertion did not establish all prescribed vehicles. Explicit `moveTo` placement now establishes the declared initial positions before simulation. The failed attempt's metadata are retained in `results/sumo`; the completed sweep will use `results/sumo_placement`. Scenario parameters and endpoints are unchanged.

## Sources

- [SUMO car-following parameters](https://sumo.dlr.de/docs/Car-Following-Models.html): tau, decision frequency, and integration steps.
- [SUMO vehicle/model definitions](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html): selection of the existing IDM implementation.
- [libsumo documentation](https://sumo.dlr.de/docs/Libsumo.html): simulation API and dependency.
- [Vehicle state control](https://sumo.dlr.de/userdoc/TraCI/Change_Vehicle_State.html): slowdown and return to normal speed control.
