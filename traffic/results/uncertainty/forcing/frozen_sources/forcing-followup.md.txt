# Follow-up: episode-specific forcing covariance

Recorded after inspecting `results/initial` on 2026-10-15, before evaluating the follow-up seeds.

## Reason

The initial constant-covariance DMD/AR forecasts perform poorly under independently varying episode forcing. Their pooled innovation covariance can overstate uncertainty for quiet episodes and understate it for noisy episodes. The feature regressions already observe per-episode variability. This is a baseline specification issue worth testing before interpreting a GFT advantage.

## Fixed change

Retain every original model and add adaptive-covariance versions of DMD, delay-two Hankel dynamics, GFT band AR, and DCT band AR. Evolution coefficients remain fitted on training episodes. Estimate each episode's innovation covariance using only its past window and the fitted operator. Propagate that covariance with the forecast mean to estimate future squared amplitude. There is no shrinkage search or extra covariance hyperparameter. DMD/Hankel use the same rank grids as before.

Retain the original training and validation seeds for model development. Evaluate once on fresh seeds 20000-20191, with the same ten cases and endpoint. The new predictions are a synthetic development replication. They are separate from empirical confirmation. Select the strongest comparator by validation loss, then recompute the original GFT contrasts on these fresh episodes.

Run with `--stage adaptive`; outputs go to `results/adaptive`. The initial results remain preserved. Numerical tests must verify that past-only covariance propagation agrees with an independent explicit recursion. No changes to the GFT feature set, graph, target, or tuning grid are planned.
