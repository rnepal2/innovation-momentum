"""Operator misspecification, independent DMD fitting, and reference/noise tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mechanism_core import equilibrium, mode_operator
from mode_consistency import real_operator
from project_paths import recorded_path
from provenance import archive_sources
from run_mechanism import save_json
from uncertainty_core import JointGLS, fit_transition, joint_noise, mode_state, temporal_covariance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOISE = [(0.01, 0.001), (0.05, 0.005), (0.2, 0.02)]
REFERENCES = [
    ("exact", None, False),
    ("constant_K1", 1, False),
    ("constant_K5", 5, False),
    ("drift_K5", 5, True),
]


def load_study(name, root=None):
    out = Path(root).resolve() if root is not None else HERE / "results" / name
    meta = pd.read_csv(out / "scenarios.csv")
    manifest = json.loads((out / "manifest.json").read_text())
    pairs = {}
    for identity, entry in manifest.items():
        path = recorded_path(entry["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            pairs[identity] = arrays["perturbed"] - arrays["control"]
    return meta, pairs, {"result_directory": str(out.relative_to(ROOT)), "inputs": manifest}


def setup(row, noise):
    v, gaps = equilibrium(row["actual_ring_length"], row["tau"])
    scales = np.diag([1 / gaps[0], 1 / v, 1 / gaps[0], 1 / v])
    m = np.linalg.matrix_power(
        mode_operator(3, gaps[0], v, 0.3, row["tau"], dt=row["dt"]), round(0.5 / row["dt"])
    )
    operator = scales @ real_operator(m) @ np.linalg.inv(scales)
    g, s = noise
    cov = scales @ np.diag([g * g / 48, s * s / 48, g * g / 48, s * s / 48]) @ scales.T
    return operator, cov, scales, v, gaps


def state(pair, scales, training=False):
    history = mode_state(pair)[::5] @ scales.T
    return history[:21] if training else history[20:61]


def compact(row):
    return {k: row[k] for k in ["id", "tau", "dt", "excitation", "phase"]}


def operator_error(out, meta, pairs):
    rows, models = [], {}
    for case_index, row in enumerate(meta.to_dict("records")):
        for noise_index, noise in enumerate(NOISE):
            true_op, cov, scales, v, gaps = setup(row, noise)
            measured = state(pairs[row["id"]], scales)[None] + joint_noise(
                cov, np.eye(41), 100, 710000 + case_index * 100 + noise_index
            )
            for parameter in ["acceleration", "headway"]:
                for error in [-0.2, -0.05, -0.01, 0.0, 0.01, 0.05, 0.2]:
                    key = (row["tau"], row["dt"], noise_index, parameter, error)
                    if key not in models:
                        a = 0.3 * (1 + error) if parameter == "acceleration" else 0.3
                        tau = row["tau"] * (1 + error) if parameter == "headway" else row["tau"]
                        candidate = np.linalg.matrix_power(
                            mode_operator(3, gaps[0], v, a, tau, dt=row["dt"]),
                            round(0.5 / row["dt"]),
                        )
                        op = scales @ real_operator(candidate) @ np.linalg.inv(scales)
                        models[key] = (
                            JointGLS(op, cov),
                            np.linalg.norm(op - true_op) / np.linalg.norm(true_op),
                        )
                    gls, op_error = models[key]
                    scores = gls.score(measured)
                    rows.append(
                        {
                            **compact(row),
                            "noise_index": noise_index,
                            "parameter": parameter,
                            "relative_parameter_error": error,
                            "relative_operator_error": op_error,
                            "cutoff": gls.cutoff,
                            "mean_score": scores.mean(),
                            "rejection_fraction": np.mean(scores > gls.cutoff),
                        }
                    )
    pd.DataFrame(rows).to_csv(out / "diagnostics.csv", index=False)


def learned_operators(out, meta, pairs, train, calibration):
    train_meta, train_pairs, train_receipt = train
    cal_meta, cal_pairs, cal_receipt = calibration
    save_json(
        out / "development_inputs.json", {"training": train_receipt, "calibration": cal_receipt}
    )
    records, operators = [], []
    for group_index, ((tau, dt), evaluation) in enumerate(meta.groupby(["tau", "dt"])):
        training = train_meta[
            (train_meta.tau == tau)
            & (train_meta["dt"] == dt)
            & train_meta.excitation.str.startswith("direct")
        ]
        cal = cal_meta[
            (cal_meta.tau == tau)
            & (cal_meta["dt"] == dt)
            & cal_meta.excitation.str.startswith("direct")
        ]
        assert len(training) == len(cal) == 8
        evaluation_rows = evaluation.to_dict("records")
        for noise_index, noise in enumerate(NOISE):
            truth, cov, scales, _, _ = setup(evaluation_rows[0], noise)
            train_states = np.stack(
                [state(train_pairs[r["id"]], scales, True) for r in training.to_dict("records")]
            )
            cal_states = np.stack(
                [state(cal_pairs[r["id"]], scales) for r in cal.to_dict("records")]
            )
            eval_states = np.stack([state(pairs[r["id"]], scales) for r in evaluation_rows])
            true_growth = np.log(abs(np.linalg.eigvals(truth))).max() / 0.5
            for replicate in range(20):
                identity = group_index * 10000 + noise_index * 1000 + replicate * 10
                train_noise = joint_noise(cov, np.eye(21), len(training), 310000 + identity)
                noisy_training = train_states + train_noise
                cal_noise = joint_noise(cov, np.eye(41), len(cal) * 40, 410000 + identity)
                noisy_cal = (cal_states[:, None] + cal_noise.reshape(len(cal), 40, 41, 4)).reshape(
                    -1, 41, 4
                )
                eval_noise = joint_noise(cov, np.eye(41), len(evaluation) * 100, 510000 + identity)
                noisy_eval = eval_states[:, None] + eval_noise.reshape(len(evaluation), 100, 41, 4)
                for method in ["known", "ols", "tls", "noise_corrected"]:
                    estimate, condition, status = (
                        (truth, 1.0, "eligible")
                        if method == "known"
                        else fit_transition(noisy_training, cov, method)
                    )
                    info = {
                        "tau": tau,
                        "dt": dt,
                        "noise_index": noise_index,
                        "training_replicate": replicate,
                        "method": method,
                        "condition": condition,
                        "status": status,
                    }
                    if estimate is None:
                        operators.append(info)
                        continue
                    op_info = {
                        **info,
                        "relative_operator_error": np.linalg.norm(estimate - truth)
                        / np.linalg.norm(truth),
                        "estimated_growth": np.log(abs(np.linalg.eigvals(estimate))).max() / 0.5,
                        "true_growth": true_growth,
                    }
                    op_info["growth_sign_correct"] = bool(
                        (op_info["estimated_growth"] > 0) == (true_growth > 0)
                    )
                    op_info.update({f"m{i}{j}": estimate[i, j] for i in range(4) for j in range(4)})
                    try:
                        gls = JointGLS(estimate, cov)
                    except ValueError:
                        operators.append({**op_info, "status": "unidentified_history_design"})
                        continue
                    cal_scores = gls.score(noisy_cal)
                    empirical_cutoff = float(np.quantile(cal_scores, 0.95))
                    operators.append(
                        {
                            **op_info,
                            "empirical_cutoff": empirical_cutoff,
                            "plugin_cutoff": gls.cutoff,
                        }
                    )
                    scores = gls.score(noisy_eval)
                    for row, values in zip(evaluation_rows, scores):
                        for label, cutoff in [
                            ("plugin", gls.cutoff),
                            ("direct_calibration", empirical_cutoff),
                        ]:
                            records.append(
                                {
                                    **compact(row),
                                    "noise_index": noise_index,
                                    "training_replicate": replicate,
                                    "method": method,
                                    "calibration": label,
                                    "cutoff": cutoff,
                                    "mean_score": values.mean(),
                                    "rejection_fraction": np.mean(values > cutoff),
                                }
                            )
            print("Learned operators", tau, dt, noise_index, flush=True)
    pd.DataFrame(operators).to_csv(out / "operators.csv", index=False)
    pd.DataFrame(records).to_csv(out / "diagnostics.csv", index=False)


def reference_noise(out, meta, pairs):
    rows, models, nulls = [], {}, []
    for case_index, row in enumerate(meta.to_dict("records")):
        for noise_index, noise in enumerate(NOISE):
            operator, cov, scales, _, _ = setup(row, noise)
            for rho_index, rho in enumerate([0.0, 0.5, 0.9]):
                for ref_index, (reference, k, drift) in enumerate(REFERENCES):
                    tc = temporal_covariance(rho, k, drift)
                    noises = joint_noise(
                        cov,
                        tc,
                        100,
                        610000 + case_index * 1000 + noise_index * 100 + rho_index * 10 + ref_index,
                    )
                    measured = state(pairs[row["id"]], scales)[None] + noises
                    key = (row["tau"], row["dt"], noise_index, rho, reference)
                    if key not in models:
                        models[key] = {
                            "iid_exact_reference": JointGLS(operator, cov),
                            "correct_joint_covariance": JointGLS(operator, cov, tc),
                            "fit_offset_and_drift": JointGLS(
                                operator, cov, temporal_covariance(rho), nuisance=2
                            ),
                        }
                        if noise_index == 0:
                            exact = np.asarray(
                                [
                                    np.linalg.matrix_power(operator, t)
                                    @ np.array([0.001, 0.002, 0.003, 0.004])
                                    for t in range(41)
                                ]
                            )
                            pure = exact[None] + joint_noise(cov, tc, 5000, 810000 + len(nulls))
                            for method, gls in models[key].items():
                                values = gls.score(pure)
                                nulls.append(
                                    {
                                        "tau": row["tau"],
                                        "dt": row["dt"],
                                        "rho": rho,
                                        "reference": reference,
                                        "method": method,
                                        "dof": gls.dof,
                                        "rejection_fraction": np.mean(values > gls.cutoff),
                                        "mean_score": values.mean(),
                                    }
                                )
                    for method, gls in models[key].items():
                        values = gls.score(measured)
                        rows.append(
                            {
                                **compact(row),
                                "noise_index": noise_index,
                                "rho": rho,
                                "reference": reference,
                                "method": method,
                                "dof": gls.dof,
                                "cutoff": gls.cutoff,
                                "mean_score": values.mean(),
                                "rejection_fraction": np.mean(values > gls.cutoff),
                            }
                        )
        if case_index % 16 == 0:
            print("Reference/noise response", case_index + 1, flush=True)
    pd.DataFrame(rows).to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(nulls).to_csv(out / "exact_linear_nulls.csv", index=False)


def run(stage, out, physical=None):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose fresh output")
    out.mkdir(parents=True, exist_ok=True)
    meta, pairs, receipt = load_study("uncertainty_physical", physical)
    save_json(
        out / "freeze.json",
        {
            "sources": archive_sources(out, HERE),
            "protocol": "uncertainty-protocol.md",
            "physical_inputs": receipt,
            "stage": stage,
            "noise": NOISE,
            "evaluation_draws": 100,
            "training_replicates": 20,
            "calibration_draws_per_response": 40,
        },
    )
    if stage == "operator_error":
        operator_error(out, meta, pairs)
    elif stage == "learned_operator":
        learned_operators(
            out, meta, pairs, load_study("observation"), load_study("observation_window")
        )
    else:
        reference_noise(out, meta, pairs)
    table = pd.read_csv(out / "diagnostics.csv")
    group = [
        c
        for c in [
            "tau",
            "dt",
            "excitation",
            "noise_index",
            "method",
            "calibration",
            "parameter",
            "relative_parameter_error",
            "rho",
            "reference",
        ]
        if c in table
    ]
    table.groupby(group).rejection_fraction.agg(["mean", "min", "max"]).to_csv(out / "summary.csv")
    print(stage, "complete:", len(table), "diagnostic rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["operator_error", "learned_operator", "reference_noise"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--physical", type=Path)
    args = parser.parse_args()
    run(args.stage, args.out or HERE / "results/uncertainty" / args.stage, args.physical)
