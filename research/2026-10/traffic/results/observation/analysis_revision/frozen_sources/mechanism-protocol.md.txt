# Mode-response mechanism and empirical collection

Recorded 2026-10-15 before collecting the new mode-response outcomes. Earlier paired-pulse spectra and amplitude-scaling results have been inspected. These experiments are exploratory, with source snapshots and complete case retention. The public article stays unchanged.

## M1: homogeneous mechanism check

Use SUMO/libsumo 1.28.0 on the existing nominal 480 m ring, 24 identical vehicles, headway 1 s, acceleration 0.3/0.8/1.5 m/s^2, and existing other IDM parameters. Place vehicles at exact equilibrium with no placement jitter. Use the actual generated circumference. Record both bumper-to-bumper gaps and speeds, and verify leader order and conservation of total gap.

Initialize a single cosine mode q=1,...,12 either in gaps or speeds. Primary amplitude is 0.01 m or m/s, respectively. Repeat amplitude 0.1 for q=1,3,6,12 as a nonlinear sensitivity. Simulate 40 seconds without external speed commands at steps 0.05 and 0.025 s. Reuse an exact matched control per acceleration/step combination. This is 192 interventions plus six controls. Record every 0.1 s; define the first post-insertion sample as the initial state for response prediction. Initial gap perturbations have zero sum and positions are obtained by cumulative gaps.

Derive the coupled gap-speed Jacobian of continuous IDM at equilibrium. Derive its semi-implicit discrete update for a single SUMO IDM substep. The inspected pinned source uses one IDM iteration at these integration steps. At each wavenumber retain the complete two-state block; a speed-only coordinate can mix two temporal branches. Compare the analytic derivatives with independent finite differences and the spatial blocks with the full physical-coordinate Jacobian.

Primary measurements: relative L2 error of predicted speed response over 0-20 s and 0-40 s; mode leakage; conservation error; control departure from equilibrium; empirical two-state transition estimates pooled across gap/speed perturbations over the first second, with conditioning reported. Recover growth and frequency through eigenvalues of the sampled transition; compare with the continuous and discrete predictions. Use the actual first recorded gap/speed response for initialization. This is an initial-condition experiment with an observed initial state, not a held-out traffic forecast.

Report all modes. A relative error below 5% in the small-amplitude, finer-step, first-20-second response is a working agreement criterion, chosen before outcomes. Large errors stop mechanistic interpretation until the state definition, simulator controls, integration, or model implementation is checked. Numerical refinement, perturbation reduction, and alternate state initialization are permitted recorded follow-ups. Do not label a disagreement as a new physical effect before those checks.

## M2: heterogeneity gate

Proceed after M1 is understood. Keep a ring and replace common headway by a fixed heterogeneous headway pattern. Construct an equilibrium with common speed and individual gaps satisfying each driver's IDM law and the total circumference. Compare full linear prediction and cross-mode coupling in ordinary Fourier, a physically specified symmetric weighted Laplacian, and random-basis controls. A nonzero coupling statistic by itself establishes no benefit from a graph representation. Freeze exact patterns, graph weights, and comparisons in a follow-up before accessing outcomes.

## D1: bounded empirical acquisition

Inspect official US DOT metadata for NGSIM and TGSIM trajectory releases. Collect up to two eligible contiguous recording blocks with complete selected time ranges, selected by metadata/coverage before spectra. Prefer documented 0.1 s trajectories with positions, speeds, vehicle IDs, lane/run identifiers, and units. New downloads are capped at 150 MiB total, with per-response limits and atomic verified acquisition. Save exact query URLs, metadata, row coverage, hashes, retrieval times, license/use statements, and failures. Preserve prior accepted inputs.

Audit keys at recording/run + vehicle + time grain; duplicates, missingness, negative values, sampling intervals, coordinate/speed consistency, lane changes, and recording boundaries. Video-derived trajectories are processed observations; verify or flag smoothing and availability. Build only retrospective descriptive spectra unless causal availability, baseline construction, and independent episodes are established. Spatial analysis on an open road must use an appropriate nonperiodic graph; never silently wrap its endpoints into a cycle. Do not select blocks using apparent wave strength or desired findings.

## Validation and output

Unit tests cover analytic/finite-difference derivatives, full/block equivalence, gap conservation, equilibrium, and homogeneous Fourier decoupling. Independently recompute numerical claims and inspect figures. Archive failed and revised attempts. Update the manuscript with the achieved scope, retain prior results, and separate physical mechanism evidence from empirical measurement feasibility.
