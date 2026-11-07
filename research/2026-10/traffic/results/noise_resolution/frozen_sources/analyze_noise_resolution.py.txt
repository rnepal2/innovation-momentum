"""Resolve the Figure 2 noise grid with fresh restricted fits and calibration."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_physics_fit import fit_physics
from analyze_uncertainty import compact, load_study, setup, state
from provenance import archive_sources
from run_mechanism import save_json
from uncertainty_core import JointGLS, joint_noise

HERE = Path(__file__).resolve().parent
SPEED_NOISE = (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02)
ANCHORS = (0.001, 0.005, 0.02)


def seeds(group_index, grid_index, replicate):
    speed_sd = SPEED_NOISE[grid_index]
    if speed_sd in ANCHORS:
        offset = group_index * 10000 + ANCHORS.index(speed_sd) * 1000 + replicate * 10
        bases = (1010000, 1110000, 1210000)
    else:
        offset = group_index * 100000 + grid_index * 1000 + replicate * 10
        bases = (2010000, 3010000, 4010000)
    return tuple(base + offset for base in bases)


def run(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose fresh output")
    out.mkdir(parents=True, exist_ok=True)
    meta, pairs, receipt = load_study("physics_fit_physical")
    tmeta, tpairs, treceipt = load_study("observation")
    cmeta, cpairs, creceipt = load_study("observation_window")
    assert set(meta["dt"]) == {0.01}
    assert sorted(meta.tau.unique()) == [1.0, 1.8]
    save_json(
        out / "freeze.json",
        {
            "protocol": "noise-resolution-followup.md",
            "sources": archive_sources(out, HERE),
            "speed_noise_sd_mps": SPEED_NOISE,
            "gap_to_speed_noise_ratio": 10,
            "training_replicates": 20,
            "calibration_draws_per_trajectory": 40,
            "evaluation_draws_per_trajectory": 100,
            "evaluation": receipt,
            "training": treceipt,
            "calibration": creceipt,
            "scenario_hashes": {
                name: hashlib.sha256(
                    (HERE / "results" / name / "scenarios.csv").read_bytes()
                ).hexdigest()
                for name in ["physics_fit_physical", "observation", "observation_window"]
            },
        },
    )
    records, operators = [], []
    for gi, ((tau, dt), group) in enumerate(meta.groupby(["tau", "dt"])):
        train = tmeta[
            (tmeta.tau == tau) & (tmeta["dt"] == dt) & tmeta.excitation.str.startswith("direct")
        ]
        cal = cmeta[
            (cmeta.tau == tau) & (cmeta["dt"] == dt) & cmeta.excitation.str.startswith("direct")
        ]
        evaluation = group.to_dict("records")
        assert len(train) == len(cal) == 8 and len(evaluation) == 16
        for ni, speed_sd in enumerate(SPEED_NOISE):
            truth, cov, scales, v, gaps = setup(evaluation[0], (10 * speed_sd, speed_sd))
            train_states = np.stack([state(tpairs[r.id], scales, True) for r in train.itertuples()])
            cal_states = np.stack([state(cpairs[r.id], scales) for r in cal.itertuples()])
            eval_states = np.stack([state(pairs[r["id"]], scales) for r in evaluation])
            for replicate in range(20):
                train_seed, cal_seed, eval_seed = seeds(gi, ni, replicate)
                training = train_states + joint_noise(cov, np.eye(21), 8, train_seed)
                calibration = (
                    cal_states[:, None]
                    + joint_noise(cov, np.eye(41), 320, cal_seed).reshape(8, 40, 41, 4)
                ).reshape(-1, 41, 4)
                measured = eval_states[:, None] + joint_noise(
                    cov, np.eye(41), 1600, eval_seed
                ).reshape(16, 100, 41, 4)
                operator, info = fit_physics(training, cov, scales, v, gaps[0], dt)
                identity = {
                    "tau": tau,
                    "dt": dt,
                    "speed_noise_sd_mps": speed_sd,
                    "gap_noise_sd_m": 10 * speed_sd,
                    "training_replicate": replicate,
                }
                model_info = {
                    **identity,
                    **info,
                    "training_seed": train_seed,
                    "calibration_seed": cal_seed,
                    "evaluation_seed": eval_seed,
                }
                if info["status"] != "eligible":
                    operators.append(model_info)
                    continue
                try:
                    gls = JointGLS(operator, cov)
                except ValueError:
                    operators.append({**model_info, "status": "unidentified_history_design"})
                    continue
                cutoff = float(np.quantile(gls.score(calibration), 0.95))
                operators.append(
                    {
                        **model_info,
                        "relative_operator_error": np.linalg.norm(operator - truth)
                        / np.linalg.norm(truth),
                        "empirical_cutoff": cutoff,
                        **{f"m{i}{j}": operator[i, j] for i in range(4) for j in range(4)},
                    }
                )
                for row, values in zip(evaluation, gls.score(measured)):
                    records.append(
                        {
                            **compact(row),
                            **identity,
                            "cutoff": cutoff,
                            "mean_score": values.mean(),
                            "rejection_fraction": np.mean(values > cutoff),
                        }
                    )
            print(f"Noise resolution: headway={tau:g}, speed SD={speed_sd:g}", flush=True)
    pd.DataFrame(operators).to_csv(out / "operators.csv", index=False)
    table = pd.DataFrame(records)
    table.to_csv(out / "diagnostics.csv", index=False)
    summary = table.groupby(
        ["tau", "dt", "excitation", "speed_noise_sd_mps"]
    ).rejection_fraction.agg(["mean", "min", "max", "count"])
    summary.to_csv(out / "summary.csv")
    original = pd.read_csv(HERE / "results/uncertainty/physics_fit/diagnostics.csv")
    original = original[
        (original.method == "restricted_idm") & (original.calibration == "direct_calibration")
    ].copy()
    original["speed_noise_sd_mps"] = original.noise_index.map(dict(enumerate(ANCHORS)))
    anchors = table[table.speed_noise_sd_mps.isin(ANCHORS)]
    key = ["id", "speed_noise_sd_mps", "training_replicate"]
    replay = anchors.merge(original, on=key, suffixes=("_new", "_old"), validate="one_to_one")
    assert len(replay) == len(anchors) == len(original) == 1920
    errors = {}
    for column in ["cutoff", "mean_score", "rejection_fraction"]:
        delta = abs(replay[column + "_new"] - replay[column + "_old"])
        errors[column] = float(delta.max())
        np.testing.assert_allclose(
            replay[column + "_new"], replay[column + "_old"], rtol=1e-12, atol=1e-10
        )
    save_json(out / "anchor_validation.json", {"rows": len(replay), "max_absolute_errors": errors})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/noise_resolution")
    run(parser.parse_args().out)
