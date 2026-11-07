"""Three manuscript figures from saved evidence; no experiment is rerun."""

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "figures"
BLUE, GOLD, GREEN = "#27638a", "#a76b14", "#367968"
NEUTRAL = "#f2f4f5"
SIGNED_SPEED = LinearSegmentedColormap.from_list("signed_speed", [BLUE, NEUTRAL, GOLD])
ABSOLUTE_SPEED = LinearSegmentedColormap.from_list("absolute_speed", [NEUTRAL, BLUE])


def refresh_harmonic_field():
    identity = "a0.3_dt0.01_eps0.2"
    manifest = json.loads((HERE / "results/harmonics/manifest.json").read_text())
    item = manifest[identity]
    path = ROOT / item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    with np.load(path, allow_pickle=False) as a:
        field = a["perturbed"][:, 1] - a["control"][:, 1]
    field -= field.mean(axis=1, keepdims=True)
    table = pd.DataFrame(field, columns=[f"vehicle_{i}" for i in range(24)])
    table.insert(0, "elapsed_s", np.arange(len(field)) * 0.1)
    table.to_csv(OUT / "harmonic_field.csv", index=False)
    (OUT / "harmonic_field_source.json").write_text(
        json.dumps(
            {
                "id": identity,
                **item,
                "operation": "Paired speed response, spatial mean removed; no temporal filtering",
            },
            indent=2,
        )
        + "\n"
    )


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=200)
    fig.savefig(OUT / (name + ".svg"), metadata={"Date": None})
    if name == "observation_limits":
        fig.savefig(OUT / "observation_limits_preview.png", dpi=80)
    plt.close(fig)


def harmonics():
    field = pd.read_csv(OUT / "harmonic_field.csv")
    x = field.iloc[:, 1:].to_numpy()
    power = 2 * abs(np.fft.rfft(x, axis=1) / 24) ** 2
    table = pd.read_csv(HERE / "results/harmonics/responses.csv")
    table = table[(table.accel == 0.3) & (table["dt"] == 0.01) & (table.window == "early")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), layout="constrained")
    limit = float(abs(x).max())
    im = axes[0].imshow(
        x.T,
        origin="lower",
        aspect="auto",
        extent=[0, 40, -0.5, 23.5],
        cmap=SIGNED_SPEED,
        interpolation="nearest",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].set(
        title="a  Spatial response",
        xlabel="Elapsed time (s)",
        ylabel="Vehicle index",
        yticks=[0, 6, 12, 18, 23],
    )
    fig.colorbar(im, ax=axes[0], label="Speed deviation (m/s)", shrink=0.9)
    for q, color, slope in [(1, BLUE, 2), (2, GOLD, 4), (3, GREEN, 6)]:
        y = np.where(power[:, q] > 1e-14, power[:, q], np.nan)
        axes[1].plot(field.elapsed_s, y, color=color, label=f"q={q}", ls=["-", "--", "-."][q - 1])
        part = table[table.q == q].sort_values("amplitude")
        axes[2].plot(part.amplitude, part.power, "o", color=color, label=f"q={q}")
        reference = part.power.iloc[0] * (part.amplitude / part.amplitude.iloc[0]) ** slope
        axes[2].plot(part.amplitude, reference, color=color, ls=["-", "--", "-."][q - 1], lw=1)
        axes[2].annotate(
            rf"$\epsilon^{slope}$",
            (part.amplitude.iloc[-1], reference.iloc[-1]),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=8,
        )
    axes[1].set(
        title="b  Absolute modal power",
        xlabel="Elapsed time (s)",
        ylabel=r"Power (m$^2$ s$^{-2}$)",
        yscale="log",
        xlim=(0, 40),
        ylim=(1e-14, 1),
    )
    axes[1].axvspan(10, 20, color="0.92", zorder=-1)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set(
        title="c  Initial-amplitude scaling",
        xlabel="Initial q=1 speed amplitude (m/s)",
        ylabel=r"10-20 s mean power (m$^2$ s$^{-2}$)",
        xscale="log",
        yscale="log",
        xticks=[0.05, 0.1, 0.2, 0.4],
        xticklabels=["0.05", "0.1", "0.2", "0.4"],
    )
    axes[2].minorticks_off()
    save(fig, "harmonic_mechanism")


def detection():
    table = pd.read_csv(HERE / "results/noise_resolution/diagnostics.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.5), layout="constrained")
    for tau, color in [(1.0, BLUE), (1.8, GOLD)]:
        for excitation, style, amplitude in [("harmonic_02", "--", 0.2), ("harmonic_04", "-", 0.4)]:
            part = (
                table[(table.tau == tau) & (table.excitation == excitation)]
                .groupby("speed_noise_sd_mps")
                .rejection_fraction.mean()
            )
            ax.plot(
                part.index.to_numpy(),
                part.to_numpy(),
                marker="o" if tau == 1 else "s",
                markersize=4,
                ls=style,
                color=color,
                label=rf"$\tau={tau:g}\,\mathrm{{s}},\quad\epsilon={amplitude:g}\,\mathrm{{m\,s^{{-1}}}}$",
            )
    ax.axhline(0.05, color=".4", ls=":", lw=1)
    ax.set(
        xscale="log",
        xticks=[0.001, 0.002, 0.005, 0.01, 0.02],
        xticklabels=[
            r"$10^{-3}$",
            r"$2\times10^{-3}$",
            r"$5\times10^{-3}$",
            r"$10^{-2}$",
            r"$2\times10^{-2}$",
        ],
        xlabel=r"Speed-noise SD, $\sigma_v$ ($\mathrm{m\,s^{-1}}$)",
        ylabel="Autonomous-model rejection",
        ylim=(0, 1.04),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.minorticks_off()
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper right",
        labelspacing=0.65,
        handlelength=2.5,
    )
    save(fig, "observation_limits")


def empirical():
    field = pd.read_csv(HERE / "results/ngsim/us-101_25m_field.csv", index_col=0)
    first = pd.read_csv(HERE / "results/ngsim_resolution/spectra.csv")
    first = first[(first.width_m == 25) & (first.time_bin_s == 5)].copy()
    first["offset_s"] = 60
    later = pd.read_csv(HERE / "results/ngsim_expansion/spectra.csv")
    table = pd.concat([first, later], ignore_index=True)
    table = table[(table.quality == "unfiltered") & table.eligible]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), width_ratios=[1, 1.5], layout="constrained")
    im = axes[0].imshow(
        field.T,
        origin="lower",
        aspect="auto",
        extent=[0, 180, 0, 650],
        cmap=ABSOLUTE_SPEED,
        interpolation="nearest",
    )
    axes[0].set(
        title="a  US-101: first selected block",
        xlabel="Elapsed time within block (s)",
        ylabel="Longitudinal position (m)",
        xticks=[0, 60, 120, 180],
        yticks=[0, 150, 300, 450, 650],
    )
    fig.colorbar(im, ax=axes[0], label="Binned mean speed (m/s)", shrink=0.9)
    rows = []
    for location in ["us-101", "i-80"]:
        for offset in [60, 240, 420]:
            part = table[(table.location == location) & (table.offset_s == offset)]
            before, after = part[~part.centered].iloc[0], part[part.centered].iloc[0]
            rows.append((location, offset, before, after))
    for i, (location, offset, before, after) in enumerate(rows):
        axes[1].plot([before.first_two_share, after.first_two_share], [i, i], color=".65", lw=1.5)
        axes[1].scatter(
            before.first_two_share,
            i,
            color=BLUE,
            s=28,
            label="Temporal mean retained" if i == 0 else None,
        )
        axes[1].scatter(
            after.first_two_share,
            i,
            facecolors="white",
            edgecolors=GOLD,
            marker="s",
            s=30,
            label="Temporal mean removed" if i == 0 else None,
        )
        axes[1].text(
            1.025,
            i,
            f"{before.power_mps2:.2f} to {after.power_mps2:.2f}",
            transform=axes[1].get_yaxis_transform(),
            va="center",
            fontsize=9,
        )
    axes[1].text(1.025, 1.02, r"Power (m$^2$ s$^{-2}$)", transform=axes[1].transAxes, fontsize=9)
    axes[1].set(
        title="b  Same-field baseline comparison",
        yticks=range(6),
        yticklabels=[f"{location.upper()} / {offset} s" for location, offset, _, _ in rows],
        xlabel="First-two-mode share",
        xlim=(0.5, 1),
        ylim=(6.1, -0.8),
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "empirical_baseline")


def run(refresh=False):
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
            "svg.hashsalt": "traffic-manuscript-2026-11-06",
        }
    )
    if refresh:
        refresh_harmonic_field()
    harmonics()
    detection()
    empirical()
    inputs = [
        OUT / "harmonic_field.csv",
        HERE / "results/harmonics/responses.csv",
        HERE / "results/noise_resolution/diagnostics.csv",
        HERE / "results/ngsim/us-101_25m_field.csv",
        HERE / "results/ngsim_resolution/spectra.csv",
        HERE / "results/ngsim_expansion/spectra.csv",
    ]
    (OUT / "sources.json").write_text(
        json.dumps(
            {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            indent=2,
        )
        + "\n"
    )
    from render_manuscript import render

    render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-export the specified saved simulation, after checking its hash",
    )
    run(parser.parse_args().refresh_data)
