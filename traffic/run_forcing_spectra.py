"""Controlled pulse-strength sweep with paired SUMO responses and scale diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from provenance import archive_sources
from spectral_diagnostics import FLOOR, band_power, spatial_power, weighted_spatial_power
from sumo_pilot import circle_network, simulation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "traffic/data/spectral_forcing"
WINDOWS = {"early": 4, "late": 52, "final": 208}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize_response(x, realized):
    p = spatial_power(x)
    bands = band_power(p["energy"])
    rows = []
    for window, start in WINDOWS.items():
        e = weighted_spatial_power(x, start)
        for band, value in band_power(e).items():
            rows.append(
                {
                    "window": window,
                    "band": band,
                    "power": float(value),
                    "normalized_power": float(value / realized**2),
                    "share": float(value / e[1:].sum()),
                }
            )
        rows.append(
            {
                "window": window,
                "band": "all",
                "power": float(e[1:].sum()),
                "normalized_power": float(e[1:].sum() / realized**2),
                "share": 1.0,
            }
        )
    trajectory = pd.DataFrame(
        {
            "seconds": np.arange(len(x)),
            "normalized_power": p["total"] / realized**2,
            "mean_response": p["mean"],
            "long_share": np.divide(
                bands["long"], p["total"], out=np.full(len(x), np.nan), where=p["total"] > FLOOR
            ),
        }
    )
    return rows, trajectory


def default_reproduction(raw_root):
    network, lengths = circle_network(480, raw_root / "reproduction_network")
    path = ROOT / "traffic/data/sumo/L480_t1.0_a0.3_s31_dt0.1.npz"
    errors = {}
    with np.load(path, allow_pickle=False) as previous:
        for key, pulse in [("control", False), ("pulse", True)]:
            speed, _ = simulation(network, lengths, 1.0, 0.3, 31, 0.1, pulse)
            errors[key] = float(np.max(np.abs(speed - previous[key])))
    if max(errors.values()) != 0:
        raise AssertionError(f"Default simulation changed: {errors}")
    return errors


def scaling_table(windows):
    rows = []
    keys = ["accel", "seed", "dt", "window", "band"]
    for key, group in windows.groupby(keys):
        if len(group) != 3:
            raise ValueError("Incomplete amplitude comparison")
        valid = (group.power > FLOOR).all()
        exponent = (
            np.polyfit(np.log(group.realized_drop), np.log(group.power), 1)[0] if valid else np.nan
        )
        ratio = group.normalized_power.max() / group.normalized_power.min()
        rows.append(
            {**dict(zip(keys, key)), "power_exponent": exponent, "normalized_power_max_min": ratio}
        )
    return pd.DataFrame(rows)


def figures(out, windows, trajectories, scaling):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    primary_step = trajectories["dt"].min()
    colors = dict(
        zip(sorted(trajectories.requested_drop.unique()), ["#346d95", "#a26b18", "#3b806c"])
    )
    data = trajectories[trajectories["dt"] == primary_step]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharey="col", layout="constrained")
    for row, accel in enumerate([0.3, 0.8, 1.5]):
        for drop, color in colors.items():
            part = data[(data.accel == accel) & (data.requested_drop == drop)]
            for col, metric in enumerate(["normalized_power", "long_share"]):
                table = part.pivot(index="seconds", columns="seed", values=metric)
                ax = axes[row, col]
                ax.plot(table.index, table.mean(axis=1), color=color, label=f"{drop:g} m/s")
                ax.fill_between(
                    table.index, table.min(axis=1), table.max(axis=1), color=color, alpha=0.16
                )
                ax.set(
                    xlabel="Seconds after pulse initiation",
                    title=f"a = {accel} m/s^2",
                    xlim=(0, 240),
                )
        axes[row, 0].set(yscale="log", ylabel="Nonuniform power / realized drop^2")
        axes[row, 1].set(ylabel="Power fraction in q = 1-2", ylim=(0, 1.03))
    axes[0, 0].legend(title="Requested speed drop", frameon=False)
    fig.savefig(out / "forcing_dependence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True, layout="constrained")
    for ax, window in zip(axes, WINDOWS):
        for drop, color in colors.items():
            table = windows[
                (windows["dt"] == primary_step)
                & (windows.window == window)
                & (windows.band == "all")
                & (windows.requested_drop == drop)
            ]
            grouped = table.groupby("accel").normalized_power.agg(["mean", "min", "max"])
            ax.plot(grouped.index, grouped["mean"], "o-", color=color, label=f"{drop:g} m/s")
            ax.fill_between(grouped.index, grouped["min"], grouped["max"], color=color, alpha=0.16)
        ax.set(
            xlabel="Acceleration (m/s^2)",
            ylabel="Integrated power / realized drop^2",
            yscale="log",
            title=f"{WINDOWS[window]}-{WINDOWS[window] + 31} s",
        )
    axes[0].legend(frameon=False)
    fig.savefig(out / "integrated_forcing_power.png", dpi=180)
    plt.close(fig)

    table = scaling[(scaling["dt"] == primary_step) & (scaling.band == "all")]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for window, color in zip(WINDOWS, ["#346d95", "#a26b18", "#3b806c"]):
        group = (
            table[table.window == window]
            .groupby("accel")
            .power_exponent.agg(["mean", "min", "max"])
        )
        ax.plot(
            group.index,
            group["mean"],
            "o-",
            color=color,
            label=f"{WINDOWS[window]}-{WINDOWS[window] + 31} s",
        )
        ax.fill_between(group.index, group["min"], group["max"], color=color, alpha=0.16)
    ax.axhline(2, color="#777777", ls="--", label="Quadratic power scaling")
    ax.set(xlabel="Acceleration (m/s^2)", ylabel="Power exponent versus realized pulse drop")
    ax.legend(frameon=False)
    fig.savefig(out / "scaling_exponents.png", dpi=180)
    plt.close(fig)


def run(out, stage="initial", raw_root=RAW):
    import libsumo

    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists():
        raise FileExistsError("Preserve existing run; choose a new --out")
    freeze = {
        "simulator": libsumo.getVersion(),
        "sources": archive_sources(out, HERE),
        "length": 480,
        "tau": 1.0,
        "accel": [0.3, 0.8, 1.5],
        "drops": [0.5, 1.0, 2.0] if stage == "initial" else [0.05, 0.1, 0.2],
        "seeds": [31, 47, 83],
        "steps": [0.1, 0.05] if stage == "initial" else [0.05, 0.025],
        "stage": stage,
        "record_seconds": 240,
        "windows": WINDOWS,
        "scope": "Exploratory mechanism comparison; prior aggregate outcomes inspected",
    }
    save_json(out / "freeze.json", freeze)
    reproduction = default_reproduction(raw_root)
    cases, windows, trajectories, simulations, manifest = [], [], [], [], []
    network, lengths = circle_network(480, raw_root / "network")
    for accel in freeze["accel"]:
        for seed in freeze["seeds"]:
            for dt in freeze["steps"]:
                control, cmeta = simulation(
                    network, lengths, 1.0, accel, seed, dt, False, record_seconds=240
                )
                simulations.append({"accel": accel, "seed": seed, "dt": dt, "drop": 0.0, **cmeta})
                for drop in freeze["drops"]:
                    pulse, meta = simulation(
                        network,
                        lengths,
                        1.0,
                        accel,
                        seed,
                        dt,
                        True,
                        pulse_drop=drop,
                        record_seconds=240,
                    )
                    simulations.append(
                        {"accel": accel, "seed": seed, "dt": dt, "drop": drop, **meta}
                    )
                    x = pulse - control
                    if (
                        np.max(np.abs(x[0])) != 0
                        or meta["collisions"]
                        or meta["teleports"]
                        or cmeta["collisions"]
                        or cmeta["teleports"]
                    ):
                        raise ValueError("Invalid paired physical response; inspect simulator logs")
                    realized = float(-x[1, 0])
                    if realized <= 0:
                        raise ValueError("No realized speed drop")
                    identity = f"a{accel}_s{seed}_dt{dt}_drop{drop}"
                    info = {
                        "id": identity,
                        "accel": accel,
                        "seed": seed,
                        "dt": dt,
                        "requested_drop": drop,
                        "realized_drop": realized,
                    }
                    path = raw_root / f"{identity}.npz"
                    np.savez_compressed(path, control=control, pulse=pulse)
                    manifest.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                    rows, series = summarize_response(x, realized)
                    windows.extend({**info, **row} for row in rows)
                    for key, value in info.items():
                        series[key] = value
                    trajectories.append(series)
                    cases.append({**info, **meta, "pre_pulse_max": float(np.max(np.abs(x[0])))})
        print(f"Completed acceleration {accel}", flush=True)
    windows, trajectories = pd.DataFrame(windows), pd.concat(trajectories, ignore_index=True)
    scaling = scaling_table(windows)
    steps = windows.pivot(
        index=["accel", "seed", "requested_drop", "window", "band"],
        columns="dt",
        values="normalized_power",
    )
    fine, coarse = sorted(freeze["steps"])
    steps["log_power_step_difference"] = np.log(steps[fine] / steps[coarse])
    pd.DataFrame(cases).to_csv(out / "scenarios.csv", index=False)
    pd.DataFrame(simulations).to_csv(out / "simulations.csv", index=False)
    windows.to_csv(out / "window_power.csv", index=False)
    trajectories.to_csv(out / "trajectories.csv", index=False)
    scaling.to_csv(out / "scaling.csv", index=False)
    steps.to_csv(out / "step_comparison.csv")
    save_json(out / "input_manifest.json", manifest)
    save_json(
        out / "validation.json",
        {
            "simulator_runs": len(simulations),
            "paired_responses": len(cases),
            "collisions": sum(row["collisions"] for row in simulations),
            "teleports": sum(row["teleports"] for row in simulations),
            "default_reproduction_max_errors": reproduction,
            "realized_drop_min": min(row["realized_drop"] for row in cases),
            "realized_drop_max": max(row["realized_drop"] for row in cases),
        },
    )
    figures(out, windows, trajectories, scaling)
    print(
        scaling[(scaling["dt"] == fine) & (scaling.band == "all")]
        .groupby(["accel", "window"])
        .power_exponent.agg(["mean", "min", "max"])
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--stage", choices=["initial", "small"], default="initial")
    args = parser.parse_args()
    out = args.out or HERE / "results" / (
        "spectral_forcing" if args.stage == "initial" else "spectral_small_pulse"
    )
    run(out, args.stage, args.raw.resolve())


if __name__ == "__main__":
    main()
