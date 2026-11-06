"""Full-state prediction and finite-difference validation of mechanism runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mechanism_core import acceleration, equilibrium, full_operator, relative_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def finite_jacobian(gaps, speed, accel, dt, taus=1.0):
    state = np.r_[gaps, np.full(24, speed)]

    def update(x):
        h, u = x[:24], x[24:]
        nxt = u + dt * acceleration(h, u, np.roll(u, -1), accel=accel, tau=taus)
        return np.r_[h + dt * (np.roll(nxt, -1) - nxt), nxt]

    return np.column_stack(
        [(update(state + e * 1e-5) - update(state - e * 1e-5)) / 2e-5 for e in np.eye(48)]
    )


def validate():
    out = HERE / "results/mechanism"
    meta = pd.read_csv(out / "scenarios.csv")
    manifest = json.loads((out / "manifest.json").read_text())
    records = []
    jacobian_errors = []
    cached = {}
    for row in meta.to_dict("records"):
        item = manifest[row["id"]]
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            x = arrays["perturbed"] - arrays["control"]
        speed, gaps = equilibrium(row["actual_ring_length"])
        key = (row["accel"], row["dt"])
        if key not in cached:
            numeric = finite_jacobian(gaps, speed, row["accel"], row["dt"])
            analytic = full_operator(gaps, speed, row["accel"], dt=row["dt"])
            jacobian_errors.append(float(abs(numeric - analytic).max()))
            cached[key] = (
                np.linalg.matrix_power(analytic, round(0.1 / row["dt"])),
                np.linalg.matrix_power(numeric, round(0.1 / row["dt"])),
            )
        step, numeric_step = cached[key]
        states = [np.r_[x[0, 0], x[0, 1]]]
        numeric_states = [states[0]]
        for _ in range(len(x) - 1):
            states.append(step @ states[-1])
            numeric_states.append(numeric_step @ numeric_states[-1])
        predicted = np.asarray(states)
        numeric_predicted = np.asarray(numeric_states)
        records.append(
            {
                "id": row["id"],
                "accel": row["accel"],
                "dt": row["dt"],
                "amplitude": row["amplitude"],
                "q": row["q"],
                "kind": row["kind"],
                "full_speed_error_20": relative_error(x[:201, 1], predicted[:201, 24:]),
                "full_speed_error_40": relative_error(x[:, 1], predicted[:, 24:]),
                "numeric_analytic_prediction_difference": relative_error(
                    predicted[:, 24:], numeric_predicted[:, 24:]
                ),
            }
        )
    table = pd.DataFrame(records)
    table.to_csv(out / "full_response_validation.csv", index=False)
    primary = table[(table.amplitude == 0.01) & (table["dt"] == 0.025)]
    validation = {
        "responses": len(table),
        "primary_responses": len(primary),
        "primary_max_full_speed_error_20": float(primary.full_speed_error_20.max()),
        "primary_median_full_speed_error_20": float(primary.full_speed_error_20.median()),
        "primary_criterion_failures": int((primary.full_speed_error_20 > 0.05).sum()),
        "max_jacobian_finite_difference_error": max(jacobian_errors),
        "max_numeric_analytic_prediction_difference": float(
            table.numeric_analytic_prediction_difference.max()
        ),
    }
    (out / "independent_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9.5, 4), layout="constrained")
    for accel, color in zip([0.3, 0.8, 1.5], ["#346d95", "#a26b18", "#3b806c"]):
        part = primary[primary.accel == accel]
        for kind, marker in [("gap", "^"), ("speed", "o")]:
            subset = part[part.kind == kind]
            ax.scatter(
                subset.q,
                subset.full_speed_error_20,
                label=f"a = {accel}; {kind}",
                color=color,
                marker=marker,
                s=22,
            )
    ax.axhline(0.05, color="#777777", ls="--", label="5% agreement criterion")
    ax.set(
        xlabel="Ring wavenumber q",
        ylabel="Relative full-field speed error (0-20 s)",
        yscale="log",
    )
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out / "full_response_errors.png", dpi=180)
    plt.close(fig)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    validate()
