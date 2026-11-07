"""Figures and numerical support for operator, reference, and forcing studies."""

import json
from pathlib import Path

import matplotlib
from project_paths import recorded_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results/uncertainty"


def parameter_figure():
    d = pd.read_csv(OUT / "operator_error/diagnostics.csv")
    d = d[(d["dt"] == 0.01) & (d.tau == 1) & (d.noise_index == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True, layout="constrained")
    for ax, parameter in zip(axes, ["acceleration", "headway"]):
        for excitation, color, label in [
            ("direct_gap", "#777777", "Direct q=3 gap"),
            ("direct_speed", "#346d95", "Direct q=3 speed"),
            ("harmonic_02", "#a26b18", "q=1 speed, 0.2 m/s"),
        ]:
            part = (
                d[(d.parameter == parameter) & (d.excitation == excitation)]
                .groupby("relative_parameter_error")
                .rejection_fraction.agg(["mean", "min", "max"])
            )
            ax.plot(100 * part.index, part["mean"], "o-", color=color, label=label)
            ax.vlines(100 * part.index, part["min"], part["max"], color=color, lw=1)
        ax.axhline(0.05, color="#777777", ls=":")
        ax.set(xlabel=f"Assumed {parameter} error (%)", ylim=(0, 1.02))
    axes[0].set_ylabel("Autonomous-model rejection fraction")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "operator_sensitivity.png", dpi=180)
    plt.close(fig)


def physics_figure():
    d = pd.read_csv(OUT / "physics_fit/diagnostics.csv")
    d = d[(d.tau == 1) & (d.noise_index == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    methods = ["known", "tls", "restricted_idm"]
    for ax, excitation, title in zip(
        axes, ["direct_speed", "harmonic_02"], ["Direct q=3 speed control", "q=1 speed, 0.2 m/s"]
    ):
        for calibration, offset, color, label in [
            ("plugin", -0.08, "#346d95", "Plug-in chi-square"),
            ("direct_calibration", 0.08, "#a26b18", "Direct-response calibration"),
        ]:
            group = (
                d[(d.excitation == excitation) & (d.calibration == calibration)]
                .groupby(["method", "training_replicate"])
                .rejection_fraction.mean()
                .groupby("method")
                .agg(["mean", "min", "max"])
                .reindex(methods)
            )
            x = np.arange(3) + offset
            ax.scatter(x, group["mean"], color=color, label=label, zorder=3)
            ax.vlines(x, group["min"], group["max"], color=color, lw=1)
        ax.axhline(0.05, color="#777777", ls=":")
        ax.set(
            title=title,
            ylim=(0, 1.02),
            xticks=range(3),
            xticklabels=["Known operator", "TLS fit", "Restricted IDM"],
        )
    axes[0].set_ylabel("Autonomous-model rejection fraction")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "physics_fit_comparison.png", dpi=180)
    plt.close(fig)


def noise_figure():
    d = pd.read_csv(OUT / "reference_noise/exact_linear_nulls.csv")
    d = d[(d["dt"] == 0.01) & (d.tau == 1)]
    index = [
        (rho, ref)
        for rho in [0.0, 0.5, 0.9]
        for ref in ["exact", "constant_K1", "constant_K5", "drift_K5"]
    ]
    methods = ["iid_exact_reference", "correct_joint_covariance", "fit_offset_and_drift"]
    table = d.pivot(
        index=["rho", "reference"], columns="method", values="rejection_fraction"
    ).reindex(index=index, columns=methods)
    fig, ax = plt.subplots(figsize=(8, 5.4), layout="constrained")
    im = ax.imshow(table, vmin=0, vmax=1, cmap="cividis", aspect="auto")
    for i in range(len(table)):
        for j in range(3):
            value = table.iloc[i, j]
            ax.text(
                j,
                i,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=9,
            )
    labels = {
        "exact": "exact reference",
        "constant_K1": "constant error; K=1",
        "constant_K5": "constant error; K=5",
        "drift_K5": "offset + drift; K=5",
    }
    ax.set(
        xticks=range(3),
        xticklabels=["IID / exact reference", "Joint covariance", "Fit offset + drift"],
        yticks=range(len(table)),
        yticklabels=[f"rho={rho:g}; {labels[ref]}" for rho, ref in index],
    )
    fig.colorbar(im, ax=ax, label="Exact-linear null rejection fraction")
    fig.savefig(OUT / "reference_calibration.png", dpi=180)
    plt.close(fig)


def ambiguity_figure():
    row = pd.read_csv(OUT / "forcing/ambiguity.csv").iloc[0]
    with np.load(recorded_path(row.path), allow_pickle=False) as a:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
        for key, color, style, label in [
            ("growing", "#346d95", "-", "Growing autonomous"),
            ("decaying_forced", "#a26b18", "--", "Decaying dynamics + input"),
            ("decaying_unforced", "#3b806c", ":", "Decaying, unforced"),
        ]:
            axes[0].plot(a["times"], 2 * abs(a[key][:, 1]) ** 2, color=color, ls=style, label=label)
        axes[0].set(xlabel="Time (s)", ylabel="q=3 speed power ((m/s)^2)", yscale="log")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(a["times"], 2 * abs(a["force"][:, 1]), color="#346d95")
        axes[1].set(
            xlabel="Time (s)", ylabel="Spatial acceleration-input amplitude (m/s^2)", ylim=(0, None)
        )
        fig.savefig(OUT / "forcing_ambiguity.png", dpi=180)
        plt.close(fig)


def empirical_figure():
    out = HERE / "results/ngsim_expansion"
    newer = pd.read_csv(out / "spectra.csv")
    earlier = pd.read_csv(HERE / "results/ngsim_resolution/spectra.csv")
    earlier = earlier[(earlier.width_m == 25) & (earlier.time_bin_s == 5)].copy()
    earlier["offset_s"] = 60
    d = pd.concat([earlier, newer], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), layout="constrained")
    for column, location in enumerate(["us-101", "i-80"]):
        for (quality, centered), part in d[(d.location == location) & d.eligible].groupby(
            ["quality", "centered"]
        ):
            part = part.sort_values("offset_s")
            color = "#a26b18" if centered else "#346d95"
            label = ("Mean removed" if centered else "Mean retained") + (
                "; exclusion" if quality != "unfiltered" else ""
            )
            for row, metric in enumerate(["power_mps2", "first_two_share"]):
                axes[row, column].plot(
                    part.offset_s,
                    part[metric],
                    "o--" if quality != "unfiltered" else "o-",
                    color=color,
                    label=label,
                )
        axes[0, column].set(title=location, yscale="log")
        axes[0, column].legend(frameon=False, fontsize=7)
        axes[1, column].set(ylim=(0, 1.02), xlabel="Block start since earliest timestamp (s)")
        for ax in axes[:, column]:
            ax.set_xticks([60, 240, 420])
    axes[0, 0].set_ylabel("Nonuniform power ((m/s)^2)")
    axes[1, 0].set_ylabel("First-two-mode share")
    fig.savefig(out / "nonoverlapping_blocks.png", dpi=180)
    plt.close(fig)
    d.to_csv(out / "combined_blocks.csv", index=False)


def support():
    summary = {}
    for study in ["operator_error", "learned_operator", "reference_noise", "physics_fit"]:
        d = pd.read_csv(OUT / study / "diagnostics.csv")
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
            if c in d
        ]
        summary[study] = d.groupby(group).rejection_fraction.mean().reset_index().to_dict("records")
    for name in ["responses", "sumo_comparison", "ambiguity", "diagnostics"]:
        summary["forcing_" + name] = pd.read_csv(OUT / "forcing" / (name + ".csv")).to_dict(
            "records"
        )
    summary["new_data"] = json.loads(
        (HERE / "results/ngsim_expansion/acquisition_summary.json").read_text()
    )
    (OUT / "manuscript_support.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parameter_figure()
    physics_figure()
    noise_figure()
    ambiguity_figure()
    empirical_figure()
    support()
