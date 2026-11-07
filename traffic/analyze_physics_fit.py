"""Restricted IDM variable-projection fit on independent training responses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_uncertainty import NOISE, compact, load_study, setup, state
from mechanism_core import mode_operator
from mode_consistency import real_operator
from provenance import archive_sources
from run_mechanism import save_json
from scipy.optimize import minimize_scalar
from uncertainty_core import JointGLS, fit_transition, joint_noise

HERE = Path(__file__).resolve().parent


def fit_physics(training, cov, scales, v, gap, dt):
    inferred_tau = (gap * np.sqrt(1 - (v / 30) ** 4) - 2) / v

    def model(a):
        step = np.linalg.matrix_power(
            mode_operator(3, gap, v, a, inferred_tau, dt=dt), round(0.5 / dt)
        )
        return scales @ real_operator(step) @ np.linalg.inv(scales)

    def objective(a):
        return float(JointGLS(model(a), cov, np.eye(21)).score(training).sum() * 21)

    fit = minimize_scalar(objective, bounds=(0.1, 2.0), method="bounded", options={"xatol": 1e-7})
    status = "eligible" if fit.success and 0.10001 < fit.x < 1.99999 else "optimizer_or_boundary"
    return model(fit.x), {
        "estimated_acceleration": float(fit.x),
        "inferred_headway": float(inferred_tau),
        "training_objective": float(fit.fun),
        "optimizer_evaluations": fit.nfev,
        "status": status,
    }


def run(out, physical=None):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose fresh output")
    out.mkdir(parents=True, exist_ok=True)
    meta, pairs, receipt = load_study("physics_fit_physical", physical)
    tmeta, tpairs, treceipt = load_study("observation")
    cmeta, cpairs, creceipt = load_study("observation_window")
    save_json(
        out / "freeze.json",
        {
            "sources": archive_sources(out, HERE),
            "protocol": "physics-fit-followup.md",
            "evaluation": receipt,
            "training": treceipt,
            "calibration": creceipt,
        },
    )
    rows, models = [], []
    for group_index, ((tau, dt), group) in enumerate(meta.groupby(["tau", "dt"])):
        train = tmeta[
            (tmeta.tau == tau) & (tmeta["dt"] == dt) & tmeta.excitation.str.startswith("direct")
        ]
        cal = cmeta[
            (cmeta.tau == tau) & (cmeta["dt"] == dt) & cmeta.excitation.str.startswith("direct")
        ]
        evaluation = group.to_dict("records")
        assert len(train) == len(cal) == 8
        for ni, noise in enumerate(NOISE):
            truth, cov, scales, v, gaps = setup(evaluation[0], noise)
            train_states = np.stack(
                [state(tpairs[r["id"]], scales, True) for r in train.to_dict("records")]
            )
            cal_states = np.stack([state(cpairs[r["id"]], scales) for r in cal.to_dict("records")])
            eval_states = np.stack([state(pairs[r["id"]], scales) for r in evaluation])
            for replicate in range(20):
                seed = group_index * 10000 + ni * 1000 + replicate * 10
                training = train_states + joint_noise(cov, np.eye(21), 8, 1010000 + seed)
                calibration = (
                    cal_states[:, None]
                    + joint_noise(cov, np.eye(41), 320, 1110000 + seed).reshape(8, 40, 41, 4)
                ).reshape(-1, 41, 4)
                measured = eval_states[:, None] + joint_noise(
                    cov, np.eye(41), len(evaluation) * 100, 1210000 + seed
                ).reshape(len(evaluation), 100, 41, 4)
                restricted, fit_info = fit_physics(training, cov, scales, v, gaps[0], dt)
                tls, condition, status = fit_transition(training, cov, "tls")
                for method, operator, info in [
                    ("known", truth, {"status": "eligible"}),
                    ("tls", tls, {"status": status, "condition": condition}),
                    ("restricted_idm", restricted, fit_info),
                ]:
                    model_info = {
                        "tau": tau,
                        "dt": dt,
                        "noise_index": ni,
                        "training_replicate": replicate,
                        "method": method,
                        **info,
                    }
                    if operator is None or info["status"] != "eligible":
                        models.append(model_info)
                        continue
                    try:
                        gls = JointGLS(operator, cov)
                    except ValueError:
                        models.append({**model_info, "status": "unidentified_history_design"})
                        continue
                    empirical = float(np.quantile(gls.score(calibration), 0.95))
                    model_info.update(
                        {
                            "relative_operator_error": np.linalg.norm(operator - truth)
                            / np.linalg.norm(truth),
                            "empirical_cutoff": empirical,
                            "plugin_cutoff": gls.cutoff,
                            **{f"m{i}{j}": operator[i, j] for i in range(4) for j in range(4)},
                        }
                    )
                    models.append(model_info)
                    scores = gls.score(measured)
                    for row, values in zip(evaluation, scores):
                        for label, cutoff in [
                            ("plugin", gls.cutoff),
                            ("direct_calibration", empirical),
                        ]:
                            rows.append(
                                {
                                    **compact(row),
                                    "noise_index": ni,
                                    "training_replicate": replicate,
                                    "method": method,
                                    "calibration": label,
                                    "cutoff": cutoff,
                                    "mean_score": values.mean(),
                                    "rejection_fraction": np.mean(values > cutoff),
                                }
                            )
            print("Restricted fitting", tau, ni, flush=True)
    pd.DataFrame(models).to_csv(out / "operators.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "diagnostics.csv", index=False)
    table.groupby(
        ["tau", "dt", "excitation", "noise_index", "method", "calibration"]
    ).rejection_fraction.agg(["mean", "min", "max"]).to_csv(out / "summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/uncertainty/physics_fit")
    parser.add_argument("--physical", type=Path)
    args = parser.parse_args()
    run(args.out, args.physical)
