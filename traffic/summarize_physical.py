"""Validate saved SUMO responses independently and produce research figures."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "traffic/data/sumo"


def physical_checks(directory, phase=False):
    data = pd.read_csv(directory / "scenarios.csv")
    assert np.isfinite(data.select_dtypes(include="number")).all().all()
    assert data.collisions.sum() == 0 and data.teleports.sum() == 0
    for row in data.itertuples():
        identity = f"phase_s{row.seed}_dt{row.dt}" if phase else row.id
        path = RAW / f"{identity}.npz"
        if phase:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row.raw_sha256
        with np.load(path, allow_pickle=False) as values:
            response = values["pulse"] - values["control"]
        # Explicit endpoint calculation; no experiment amplitude helpers are imported.
        now = np.sqrt(np.square(response[28:32]).sum() / (4 * 24))
        future = np.sqrt(np.square(response[58:62]).sum() / (4 * 24))
        actual = np.log((future + 1e-6) / (now + 1e-6))
        assert abs(actual - row.actual) < 1e-12
        assert np.max(np.abs(response[0])) == 0
    steps = data.pivot(
        index="seed" if phase else ["length", "tau", "accel", "seed"], columns="dt", values="actual"
    )
    difference = (steps.iloc[:, 0] - steps.iloc[:, 1]).abs()
    receipt = {
        "responses": len(data),
        "simulator_runs": 2 * len(data),
        "collisions": 0,
        "teleports": 0,
        "labels_recomputed_from_raw": len(data),
        "zero_pre_pulse_response": True,
        "step_difference_mean": float(difference.mean()),
        "step_difference_max": float(difference.max()),
        "step_sign_disagreements": int(((steps.iloc[:, 0] > 0) != (steps.iloc[:, 1] > 0)).sum()),
    }
    if not phase:
        receipt["max_gft_fourier_difference"] = float(np.max(np.abs(data.gft - data.fourier)))
        assert receipt["max_gft_fourier_difference"] < 1e-10
    else:
        compare = pd.read_csv(directory / "comparisons.csv")
        for row in compare.itertuples():
            candidate_loss = (
                data.assign(loss=(data.phase_ar2 - data.actual) ** 2).groupby("seed").loss.mean()
            )
            reference_loss = (
                data.assign(loss=(data[row.reference] - data.actual) ** 2)
                .groupby("seed")
                .loss.mean()
            )
            assert abs(candidate_loss.mean() - row.phase_mse) < 1e-12
            assert abs(reference_loss.mean() - row.reference_mse) < 1e-12
    (directory / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return data, receipt


def figures(pilot, phase):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, length in zip(axes, [480, 720]):
        table = pilot[(pilot.length == length) & (pilot.dt == 0.1)].pivot_table(
            index="tau", columns="accel", values="actual"
        )
        im = ax.imshow(table, vmin=-1.8, vmax=1.8, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(3), [str(a) for a in table.columns])
        ax.set_yticks(range(3), [str(a) for a in table.index])
        ax.set(
            xlabel="Maximum acceleration (m/s^2)",
            ylabel="Desired headway (s)",
            title=f"Nominal {length} m ring; 24 vehicles",
        )
        for i in range(3):
            for j in range(3):
                value = table.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 1.1 else "#222222",
                )
    fig.suptitle("30-second response; 0.1 s integration; mean of three seeds")
    fig.colorbar(im, ax=axes, label="Log RMS amplitude ratio")
    fig.savefig(HERE / "results/sumo_placement/response_grid.png", dpi=170)
    plt.close(fig)
    names = ["zero", "gft", "dmd", "raw_var2", "phase_ar2", "trend"]
    labels = [
        "Zero growth",
        "Three-band GFT AR",
        "Rank-four DMD",
        "Raw VAR(2)",
        "Fourier AR(2)",
        "Recent amplitude trend",
    ]
    mean_loss = [float(np.mean((phase[name] - phase.actual) ** 2)) for name in names]
    order = np.argsort(mean_loss)[::-1]
    labels = [labels[i] for i in order]
    mean_loss = [mean_loss[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    colors = {"Fourier AR(2)": "#346d95", "Recent amplitude trend": "#a26b18"}
    ax.barh(labels, mean_loss, color=[colors.get(label, "#929292") for label in labels])
    ax.set(
        xlabel="Mean squared error of log amplitude ratio", title="Fresh SUMO parameter settings"
    )
    ax.set_xlim(0, max(mean_loss) * 1.22)
    for y, value in enumerate(mean_loss):
        ax.text(value + max(mean_loss) * 0.012, y, f"{value:.4f}", va="center")
    fig.savefig(HERE / "results/phase/forecast_comparison.png", dpi=170)
    plt.close(fig)
    # Fixed scenario rule: first new setting and its finer integration step.
    row = phase[(phase.seed == phase.seed.min()) & (phase.dt == phase.dt.min())].iloc[0]
    path = RAW / f"phase_s{int(row.seed)}_dt{row['dt']}.npz"
    with np.load(path, allow_pickle=False) as values:
        response = values["pulse"] - values["control"]
    amplitude = np.sqrt(np.mean(response**2, axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    im = axes[0].imshow(
        response.T,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.max(np.abs(response)),
        vmax=np.max(np.abs(response)),
    )
    axes[0].axvline(31, color="#333333", ls="--", lw=1)
    axes[0].set(
        xlabel="Seconds after pulse initiation",
        ylabel="Persistent vehicle index",
        title="Paired speed response (simulation seed 101)",
    )
    fig.colorbar(im, ax=axes[0], label="Speed difference (m/s)")
    axes[1].plot(amplitude, color="#346d95", label="Observed paired response")
    axes[1].axvline(31, color="#333333", ls="--", lw=1)
    origin = np.sqrt(np.mean(response[28:32] ** 2))
    for name, color, marker in [("phase_ar2", "#346d95", "s"), ("trend", "#a26b18", "o")]:
        axes[1].scatter(
            61,
            (origin + 1e-6) * np.exp(row[name]) - 1e-6,
            color=color,
            marker=marker,
            label="Fourier AR(2) forecast" if name == "phase_ar2" else "Trend forecast",
        )
    axes[1].scatter(
        61,
        np.sqrt(np.mean(response[58:62] ** 2)),
        color="#333333",
        marker="x",
        label="Observed endpoint (4 s RMS)",
    )
    axes[1].set(
        xlabel="Seconds after pulse initiation",
        ylabel="Whole-platoon RMS (m/s)",
        title="Past window ends at 31 seconds",
    )
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(HERE / "results/phase/response_example.png", dpi=170)
    plt.close(fig)


def main():
    pilot, pcheck = physical_checks(HERE / "results/sumo_placement")
    phase, fcheck = physical_checks(HERE / "results/phase", phase=True)
    figures(pilot, phase)
    print(json.dumps({"pilot": pcheck, "phase": fcheck}, indent=2))


if __name__ == "__main__":
    main()
