# Simulator clock correction

The whole-window consistency follow-up exposed a systematic discrepancy in direct-mode controls. A live simulator check on 15 October 2026 established that SUMO 1.28.0 rounds a requested 0.0125 s integration step to 0.013 s. Eight simulation steps then span 0.104 s. The affected analysis had labeled that interval as 0.1 s and propagated an operator at the requested step.

Affected studies are the harmonic initial-condition sweep and the two observation studies. Their original result directories are preserved with `_timing_audit` suffixes. Their source archives and local raw arrays remain intact. Their 0.0125 s outputs are excluded from scientific claims. The 0.025 s outputs used exact milliseconds, but the complete affected studies are rerun for a consistent timing audit trail.

The replacement fine step is 0.01 s. The 0.025 s comparison step, all amplitudes, physical parameters, analysis windows, phases, masks, noise specifications, calibration methods, and thresholds remain fixed. This correction follows inspection of the affected outcomes. Whole-window held-out phases remain separate from the initial observation phases, while the timing correction is a subsequent validation repair.

The collector now rejects steps that are not exact milliseconds or do not divide the 0.1 s recording interval. After starting SUMO it verifies `getDeltaT()` against the request. Each recorded simulator timestamp is checked against the first actual timestamp plus integer multiples of 0.1 s. Output metadata record the effective integration step, first/last timestamps, and maximum recording-clock discrepancy.

The mechanism study used 0.05/0.025 s and the heterogeneity study used 0.05/0.025 s; they are unaffected by this rounding issue. Earlier pulse studies also used integer-millisecond steps that divide their recording interval. Existing bitwise reproduction checks establish repeatability, which alone did not detect the incorrect time-base assumption. The corrected studies require both reproduction and direct simulator-clock checks.

Corrected primary outputs retain the established `harmonics`, `observation`, and `observation_window` result names. Repeat runs use separate raw and output directories. A timing receipt records the live rounding probe and corrected clock checks. Manuscript numbers and figures are recalculated before use.
