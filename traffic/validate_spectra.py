"""Independent raw-FFT recalculation of saved spectral and forcing summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from project_paths import recorded_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BANDS = {
    "long": slice(1, 3),
    "intermediate": slice(3, 7),
    "short": slice(7, 13),
    "all": slice(1, 13),
}
WINDOWS = {"early": 4, "late": 52, "final": 208}


def direct_power(x):
    residual = x - x.mean(axis=1, keepdims=True)
    z = np.fft.fft(residual, axis=1) / 24
    p = np.abs(z[:, :13]) ** 2
    p[:, 1:12] += np.abs(z[:, -1:-12:-1]) ** 2
    return p


def direct_window(x, start, size=32):
    # Periodic Hann, independently of scipy.signal and the analysis helpers.
    w2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)) ** 2
    return (direct_power(x[start : start + size]) * w2[:, None]).sum(axis=0) / w2.sum()


def verify_sources(out):
    freeze = json.loads((out / "freeze.json").read_text())
    for name, expected in freeze["sources"].items():
        assert (
            hashlib.sha256((out / "frozen_sources" / (name + ".txt")).read_bytes()).hexdigest()
            == expected
        )
    manifest = json.loads((out / "input_manifest.json").read_text())
    for item in manifest:
        assert (
            hashlib.sha256((recorded_path(item["path"])).read_bytes()).hexdigest() == item["sha256"]
        )
    return manifest


def validate():
    errors = []
    counts = {}
    spectral = HERE / "results/spectral"
    manifest = verify_sources(spectral)
    modes = pd.read_csv(spectral / "mode_summary.csv")
    cases = pd.read_csv(spectral / "case_summary.csv")
    for item in manifest:
        path = recorded_path(item["path"])
        with np.load(path, allow_pickle=False) as raw:
            x = raw["pulse"] - raw["control"]
        early, late = direct_window(x, 4), direct_window(x, 52)
        rows = modes[modes.id == path.stem].sort_values("q")
        errors.extend(
            [
                np.max(np.abs(rows.early_power.to_numpy() - early[1:])),
                np.max(np.abs(rows.late_power.to_numpy() - late[1:])),
            ]
        )
        case = cases[cases.id == path.stem].iloc[0]
        for band, indices in BANDS.items():
            if band == "all":
                continue
            for window, expected in [("early", early), ("late", late)]:
                errors.append(abs(case[f"{band}_{window}_power"] - expected[indices].sum()))
                errors.append(
                    abs(
                        case[f"{band}_{window}_share"]
                        - expected[indices].sum() / expected[1:].sum()
                    )
                )
    counts["descriptive_responses"] = len(manifest)
    for study in ["spectral_forcing", "spectral_small_pulse"]:
        out = HERE / "results" / study
        manifest = verify_sources(out)
        windows = pd.read_csv(out / "window_power.csv")
        scenarios = pd.read_csv(out / "scenarios.csv")
        trajectory = pd.read_csv(out / "trajectories.csv")
        scaling = pd.read_csv(out / "scaling.csv")
        for item in manifest:
            path = recorded_path(item["path"])
            with np.load(path, allow_pickle=False) as raw:
                x = raw["pulse"] - raw["control"]
            assert x.shape == (241, 24) and np.isfinite(x).all() and np.max(np.abs(x[0])) == 0
            case = scenarios[scenarios.id == path.stem].iloc[0]
            realized = -x[1, 0]
            errors.append(abs(case.realized_drop - realized))
            for window, start in WINDOWS.items():
                e = direct_window(x, start)
                for band, indices in BANDS.items():
                    row = windows[
                        (windows.id == path.stem)
                        & (windows.window == window)
                        & (windows.band == band)
                    ].iloc[0]
                    errors.extend(
                        [
                            abs(row.power - e[indices].sum()),
                            abs(row.normalized_power - e[indices].sum() / realized**2),
                            abs(row.share - e[indices].sum() / e[1:].sum()),
                        ]
                    )
            series = trajectory[trajectory.id == path.stem].sort_values("seconds")
            p = direct_power(x)
            errors.append(
                np.max(
                    np.abs(series.normalized_power.to_numpy() - p[:, 1:].sum(axis=1) / realized**2)
                )
            )
        for row in scaling.to_dict("records"):
            group = windows[
                (windows.accel == row["accel"])
                & (windows.seed == row["seed"])
                & (windows["dt"] == row["dt"])
                & (windows.window == row["window"])
                & (windows.band == row["band"])
            ]
            if np.isfinite(row["power_exponent"]):
                log_d, log_p = (
                    np.log(group.realized_drop.to_numpy()),
                    np.log(group.power.to_numpy()),
                )
                expected = ((log_d - log_d.mean()) * (log_p - log_p.mean())).sum() / (
                    (log_d - log_d.mean()) ** 2
                ).sum()
                errors.append(abs(row["power_exponent"] - expected))
        sims = pd.read_csv(out / "simulations.csv")
        assert len(manifest) == 54 and len(sims) == 72
        assert sims.collisions.sum() == 0 and sims.teleports.sum() == 0
        counts[study] = len(manifest)
    maximum = float(max(errors))
    assert maximum < 1e-10, maximum
    receipt = {
        **counts,
        "comparisons": len(errors),
        "max_absolute_error": maximum,
        "method": "Raw full spatial FFT and explicit periodic-Hann weights; no spectral analysis helpers",
        "source_archives_and_raw_hashes": "verified",
    }
    (HERE / "results/spectral/independent_validation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    validate()
