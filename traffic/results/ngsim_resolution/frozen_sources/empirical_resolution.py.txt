"""Fixed-support NGSIM resolution and kinematic-exclusion sensitivities."""

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
from collect_ngsim import DATA, save_json
from provenance import archive_sources
from scipy.fft import dct
from scipy.signal import periodogram

HERE = Path(__file__).resolve().parent
SUPPORTS = {"us-101": 650, "i-80": 325}


def flagged_traces(frame):
    ordered = frame.sort_values(["vehicle_id", "global_time"])
    groups = ordered.groupby("vehicle_id")
    dt = groups.global_time.diff() / 1000
    dy = groups.local_y.diff() * 0.3048
    average_speed = (ordered.v_vel + groups.v_vel.shift()) * 0.3048 / 2
    error = abs(dy / dt - average_speed)
    return np.sort(ordered.loc[(dt > 0) & (error > 5), "vehicle_id"].unique())


def fixed_field(frame, start, length, width, seconds):
    info = {"support_m": length, "width_m": width, "time_bin_s": seconds}
    if length % width:
        return None, {**info, "eligible": False, "reason": "support_not_divisible"}
    n, t = length // width, 180 // seconds
    if n < 8:
        return None, {**info, "eligible": False, "reason": "fewer_than_eight_spatial_bins"}
    f = frame.assign(
        tbin=np.floor((frame.global_time - start) / (1000 * seconds)).astype(int),
        xbin=np.floor(frame.local_y * 0.3048 / width).astype(int),
        speed_mps=frame.v_vel * 0.3048,
    )
    f = f[(f.tbin >= 0) & (f.tbin < t) & (f.xbin >= 0) & (f.xbin < n)]
    field = (
        f.groupby(["tbin", "xbin"])
        .speed_mps.mean()
        .unstack()
        .reindex(index=range(t), columns=range(n))
    )
    counts = (
        f.groupby(["tbin", "xbin"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(t), columns=range(n), fill_value=0)
        .fillna(0)
    )
    missing = int(field.isna().to_numpy().sum())
    return field, {
        **info,
        "eligible": missing == 0,
        "reason": "complete" if missing == 0 else "missing_cells",
        "spatial_bins": n,
        "time_bins": t,
        "missing_cells": missing,
        "minimum_cell_observations": int(counts.min().min()),
        "field_rows": len(f),
    }


def spectrum(field, seconds, centered):
    x = np.asarray(field, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    coefficients = dct(x, type=2, norm="ortho", axis=1)
    frequencies, density = periodogram(
        coefficients,
        fs=1 / seconds,
        window="hann",
        detrend="constant" if centered else False,
        scaling="density",
        axis=0,
    )
    power = density.sum(axis=0) * (frequencies[1] - frequencies[0]) / x.shape[1]
    if centered:
        x -= x.mean(axis=0, keepdims=True)
    taper = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(len(x)) / len(x))
    independent = float((np.mean(x * x, axis=1) * taper**2).sum() / (taper**2).sum())
    assert abs(power[1:].sum() - independent) < 1e-10
    return power, abs(power[1:].sum() - independent)


def figure(table, out):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout="constrained")
    for column, location in enumerate(SUPPORTS):
        part = table[table.location == location]
        for (width, quality, centered), group in part.groupby(["width_m", "quality", "centered"]):
            if centered:
                continue
            for row, metric in enumerate(["power_mps2", "first_two_share"]):
                for center, color in [(False, "#346d95"), (True, "#a26b18")]:
                    selected = part[
                        (part.width_m == width)
                        & (part.quality == quality)
                        & (part.centered == center)
                    ]
                    label = f"{width} m, " + ("mean removed" if center else "mean retained")
                    if quality == "flagged_trace_exclusion":
                        label += ", exclusion"
                    axes[row, column].plot(
                        selected.time_bin_s,
                        selected[metric],
                        color=color,
                        ls="--" if quality == "flagged_trace_exclusion" else "-",
                        marker="s" if width == 50 else "o",
                        label=label,
                    )
        axes[0, column].set(title=f"{location}; fixed {SUPPORTS[location]} m support", yscale="log")
        axes[1, column].set(ylim=(0, 1.02), xlabel="Time-bin width (s)")
        for ax in axes[:, column]:
            ax.set_xticks([1, 2, 5, 10])
        axes[0, column].legend(frameon=False, fontsize=7)
    axes[0, 0].set_ylabel("Integrated nonuniform power ((m/s)^2)")
    axes[1, 0].set_ylabel("First-two-mode power fraction")
    fig.savefig(out / "resolution_sensitivity.png", dpi=180)
    plt.close(fig)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists():
        raise FileExistsError("Choose fresh output")
    source = HERE / "results/ngsim"
    manifest = json.loads((source / "manifest.json").read_text())
    for name, receipt in manifest.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == receipt["sha256"]
    save_json(
        out / "freeze.json",
        {
            "sources": archive_sources(out, HERE),
            "inputs": manifest,
            "supports_m": SUPPORTS,
            "time_bins_s": [1, 2, 5, 10],
            "widths_m": [25, 50],
            "quality_rule": "Exclude entire selected vehicle trace if any kinematic discrepancy >5 m/s",
            "protocol": "observation-protocol.md",
        },
    )
    profiles = json.loads((source / "profiles.json").read_text())
    eligibility, rows, quality_rows = [], [], []
    for location, length in SUPPORTS.items():
        raw = pd.read_csv(DATA / (location + ".csv"))
        flagged = flagged_traces(raw)
        filtered = raw[~raw.vehicle_id.isin(flagged)]
        quality_rows.append(
            {
                "location": location,
                "flagged_vehicle_ids": flagged.tolist(),
                "excluded_vehicles": len(flagged),
                "original_vehicles": raw.vehicle_id.nunique(),
                "excluded_rows": len(raw) - len(filtered),
                "original_rows": len(raw),
            }
        )
        for quality, frame in [("unfiltered", raw), ("flagged_trace_exclusion", filtered)]:
            for width in [25, 50]:
                for seconds in [1, 2, 5, 10]:
                    field, info = fixed_field(
                        frame, profiles[location]["start_epoch_ms"], length, width, seconds
                    )
                    base = {"location": location, "quality": quality, **info}
                    eligibility.append(base)
                    if not info["eligible"]:
                        continue
                    identity = f"{location}_{quality}_{width}m_{seconds}s"
                    field.to_csv(out / (identity + "_field.csv"), index_label="time_bin")
                    for centered in [False, True]:
                        power, error = spectrum(field, seconds, centered)
                        rows.append(
                            {
                                **base,
                                "centered": centered,
                                "power_mps2": power[1:].sum(),
                                "first_two_share": power[1:3].sum() / power[1:].sum(),
                                "integration_error": error,
                            }
                        )
                        pd.DataFrame({"mode": range(len(power)), "power_mps2": power}).to_csv(
                            out / f"{identity}_center{centered}_powers.csv", index=False
                        )
    results = pd.DataFrame(rows)
    results.to_csv(out / "spectra.csv", index=False)
    pd.DataFrame(eligibility).to_csv(out / "eligibility.csv", index=False)
    save_json(out / "quality_exclusions.json", quality_rows)
    figure(results, out)
    print(pd.DataFrame(eligibility).to_string(index=False))
    print(
        results.groupby(["location", "quality", "centered"])[["power_mps2", "first_two_share"]]
        .agg(["min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/ngsim_resolution")
    args = parser.parse_args()
    run(args.out)
