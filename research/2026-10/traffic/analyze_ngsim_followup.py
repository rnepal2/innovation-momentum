"""Fixed-support temporal-centering sensitivity and independent acquisition audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collect_ngsim import DATA, complete_strip, describe, profile
from scipy.fft import dct
from scipy.signal import periodogram

HERE = Path(__file__).resolve().parent


def run():
    out = HERE / "results/ngsim"
    manifest = json.loads((out / "manifest.json").read_text())
    for name, receipt in manifest.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == receipt["sha256"]
    profiles = json.loads((out / "profiles.json").read_text())
    rows, extremes = [], []
    for location in ["us-101", "i-80"]:
        raw = pd.read_csv(DATA / (location + ".csv"))
        expected = int(json.loads((DATA / (location + "_count.json")).read_text())[0]["rows"])
        assert len(raw) == expected
        result = profile(raw)
        profiles[location].update(result)
        start = profiles[location]["start_epoch_ms"]
        describe(raw, start, location, out)
        ordered = raw.sort_values(["vehicle_id", "global_time"])
        dt = ordered.groupby("vehicle_id").global_time.diff() / 1000
        dy = ordered.groupby("vehicle_id").local_y.diff() * 0.3048
        speed = (ordered.v_vel + ordered.groupby("vehicle_id").v_vel.shift()) * 0.3048 / 2
        error = abs(dy / dt - speed)
        for i in error.nlargest(5).index:
            extremes.append(
                {
                    "location": location,
                    **ordered.loc[
                        i, ["vehicle_id", "global_time", "lane_id", "local_y", "v_vel"]
                    ].to_dict(),
                    "position_speed_discrepancy_mps": float(error.loc[i]),
                }
            )
        for width in [50, 25, 100]:
            field, eligibility = complete_strip(raw, start, width)
            if not eligibility["eligible"]:
                rows.append({"location": location, "width_m": width, "eligible": False})
                continue
            x = field.to_numpy(copy=True)
            x -= x.mean(axis=1, keepdims=True)
            powers = {}
            for detrend in [False, "constant"]:
                coefficients = dct(x, type=2, norm="ortho", axis=1)
                f, p = periodogram(
                    coefficients, fs=0.2, window="hann", detrend=detrend, scaling="density", axis=0
                )
                power = p.sum(axis=0) * (f[1] - f[0]) / x.shape[1]
                centered = coefficients - coefficients.mean(axis=0) if detrend else coefficients
                w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(36) / 36)
                ft = np.fft.rfft(centered * w[:, None], axis=0)
                direct = abs(ft) ** 2
                direct[1:-1] *= 2
                direct = direct.sum(axis=0) / (36 * np.sum(w * w) * x.shape[1])
                error = float(abs(power - direct).max())
                assert error < 1e-10
                total = float(power[1:].sum())
                powers[str(detrend)] = power
                rows.append(
                    {
                        "location": location,
                        "width_m": width,
                        "eligible": True,
                        "temporal_detrend": str(detrend),
                        "power_mps2": total,
                        "first_two_share": float(power[1:3].sum() / total),
                        "independent_integration_error": error,
                        "spatial_bins": x.shape[1],
                    }
                )
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
            for name, power in powers.items():
                label = "Temporal mean retained" if name == "False" else "Temporal mean removed"
                axes[0].plot(np.arange(1, len(power)), power[1:], "o-", label=label)
                axes[1].step(
                    np.arange(len(power)),
                    np.r_[0, np.cumsum(power[1:]) / power[1:].sum()],
                    where="post",
                    label=label,
                )
            axes[0].set(
                xlabel="Open-path mode index", ylabel="Integrated power ((m/s)^2)", yscale="log"
            )
            axes[1].set(
                xlabel="Maximum included mode index",
                ylabel="Cumulative nonuniform power fraction",
                ylim=(0, 1.02),
            )
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{location}: {width} m bins; fixed complete spatial support")
            fig.savefig(out / f"{location}_{width}m_centering_sensitivity.png", dpi=180)
            plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "temporal_centering.csv", index=False)
    pd.DataFrame(extremes).to_csv(out / "kinematic_extremes.csv", index=False)
    (out / "profiles.json").write_text(json.dumps(profiles, indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
