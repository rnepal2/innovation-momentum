# Traffic experiment ledger

## 2026-10-15: setup and pre-evaluation checks

- Created branch `feature/traffic-spectral-amplification`; retained the research plan and both literature notes. The public article and earlier experiments remain unchanged.
- Recorded `protocol.md` before benchmark evaluation. T01-T04 use artificial numerical systems; empirical Traffic H1/H2 remain open.
- Numerical smoke check used seed 42 only. The first test pass found a test expectation that omitted the documented endpoint noise floor. Corrected that expectation; no benchmark outcome was inspected to make this change.
- Fourteen focused tests pass before the reserved-seed benchmark. Checks cover PSD units, transform equivalence, eigenspaces, transport, known evolution, covariance propagation, split/selection timing, reproducibility, and safe acquisition.
- Located a public 54,743,156-byte virtual-trajectory sample in the authors' wave-analysis repository. Acquisition is capped at 64 MiB. The sample is for measurement/provenance auditing, with no empirical forecast score planned.
- Inspected VT-tools' symmetric temporal smoothing expression. Its current default admits observations up to 18 seconds after the target timestamp. The exact demo-generation parameters and end-to-end availability delay are unresolved.
- Located Ji et al.'s additional wave-component identification study, arXiv:2409.00326v2, with a 2026 journal publication. Targeted methodology/data/processing review is part of T06; this adds prior art on graph-based wave measurement. It does not establish a GFT predictive result.

Append run locations, source hashes, results, failures, and any subsequent design changes below. The generated freeze and environment files identify the actual sources and working-tree state used by each run.

## Initial numerical run

- `results/initial`: 180 model/case scores and 34,560 reserved-seed predictions. Independent CSV/scalar recomputation verifies losses and validation-only comparator selection.
- An initial branched/aligned GFT-feature gain motivated examination of weak pooled-covariance dynamics baselines. Recorded `forcing-followup.md` before accessing new evaluation seeds 20000-20191. No GFT-feature or endpoint changes.
- `results/adaptive`: 220 scores and 42,240 fresh-seed predictions. The branched/aligned GFT feature comparison loses its advantage against the strengthened comparator. All cases and null realizations remain saved.

## Data audit and repair

- The initial audit rejected nonfinite physical values. Inspection found 1,943 missing position/speed rows, all at the last timestamp of their virtual trajectory. Changed the audit to retain and report missingness, with finite physical rows used only for kinematic/descriptive calculations. Added a missing-endpoint regression test. This repair changes no forecasting result.
- Persist per-source acquisition receipts before profiling so a later validation failure preserves provenance. The complete sample contains 1,137,588 rows, 2,857 virtual IDs, no duplicate keys, and 29 negative speeds. Raw data remain unchanged and ignored by Git.
- `results/data_audit`: versioned source URLs, SHA-256 hashes, profile, symmetric-smoothing inspection, and a descriptive field. Calendar date, independent days, and full availability delay remain unverified. No empirical predictor was fitted.

## Physical pilot and phase follow-up

- Recorded `sumo-protocol.md` before the simulator sweep. Used SUMO/libsumo 1.28.0 with its IDM model. The first attempt failed vehicle retention under sequential insertion. Preserved `results/sumo/freeze.json`; established the prescribed initial placement with the supported moveTo API. Endpoint/parameter design stayed fixed. Completed run: `results/sumo_placement`.
- The completed pilot has 216 simulator runs, 108 paired responses, zero collisions/teleports, and GFT/Fourier equivalence. Integration-step refinement produces three sign changes near zero; these are retained as a numerical limit.
- Recorded `phase-followup.md` after the pilot, before simulating 32 new settings. Added complex Fourier AR(2), a matching-history raw VAR(2), and an analytic traveling-wave unit test. `results/phase` contains 128 simulator runs and 64 paired responses. Fourier AR(2) improves on raw VAR(2) and loses to the recent amplitude trend. No hyperparameter was selected using those outcomes.
- Independent physical validation recalculates every endpoint from saved pulse/control arrays and checks all safety counts. The phase follow-up groups numerical-step replicas by parameter setting for uncertainty.
- The first figure-generation pass hit a pandas Series `.dt` accessor/name collision; changed the column lookup to bracket notation. Predictions and scores were unaffected.

## Next boundary

The current milestone ends with physical and measurement feasibility evidence. T07 real-data validation and T08 sensing-resolution/transfer tests remain pending. A heterogeneous/ongoing-forcing physical study needs its own frozen protocol. Current results do not justify a positive empirical forecasting claim or changes to the public article.

## Final validation

- Repository suite: 51 tests pass; scoped Ruff checks pass. Numerical and physical endpoints were independently recalculated. Figures were rendered and inspected; a clipped colorbar label was shortened before delivery.
- Repeated the complete initial numerical benchmark and the complete fresh-setting physical study from the final analysis code. Initial scores, per-episode predictions, paired contrasts, diagnostics, and selections are byte-identical; the physical scenario and comparison CSVs are also byte-identical. See `results/reproduction.json`. Reproduction copies stay in ignored `data/`.
- Both numerical studies retain their initial freeze/source hashes. The code now includes later adaptive/phase methods and additional CLI/validation support. Their original numerical outputs remain reproducible; later changes do not retroactively change what was frozen before each evaluation.
- The first runs recorded source/protocol hashes without content snapshots. Current code reproduces the checked results exactly, but the changed original source files cannot all be recovered through those hashes alone. Added automatic content snapshots to future freezes. Existing runs are not retroactively presented as having pre-evaluation source archives.
