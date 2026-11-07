"""Bounded official NGSIM acquisition and retrospective open-path spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audit_data import fetch_verified
from provenance import archive_sources
from scipy.fft import dct
from scipy.signal import periodogram

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "traffic/data/ngsim"
BASE = "https://data.transportation.gov/resource/8ect-6jqj"
COLUMNS = "vehicle_id,frame_id,global_time,local_x,local_y,v_vel,v_acc,lane_id,preceding,space_headway,v_length,location"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def url(query, extension="json"):
    return BASE + "." + extension + "?" + urlencode(query)


def acquire(url_, name, receipt, limit=2 * 1024**2, data_root=DATA):
    destination = data_root / name
    if name in receipt:
        import hashlib

        if receipt[name]["url"] != url_:
            raise ValueError("Accepted query changed; use a separate input directory")
        if (
            not destination.exists()
            or hashlib.sha256(destination.read_bytes()).hexdigest() != receipt[name]["sha256"]
        ):
            raise ValueError("Accepted input changed")
    else:
        if destination.exists():
            raise FileExistsError("Existing input has no accepted receipt in this run; preserve it")
        entry = fetch_verified(url_, destination, limit)
        if sum(item["bytes"] for item in receipt.values()) + entry["bytes"] > 150 * 1024**2:
            raise ValueError("Aggregate acquisition cap exceeded")
        receipt[name] = entry
    return destination


def profile(frame):
    numeric = [name for name in frame if name != "location"]
    if frame.empty or not np.isfinite(frame[numeric]).all().all():
        raise ValueError("Nonfinite or empty selected observations")
    ordered = frame.sort_values(["vehicle_id", "global_time"])
    dt = ordered.groupby("vehicle_id").global_time.diff() / 1000
    dy = ordered.groupby("vehicle_id").local_y.diff() * 0.3048
    avg_speed = (ordered.v_vel + ordered.groupby("vehicle_id").v_vel.shift()) * 0.3048 / 2
    valid = dt > 0
    mismatch = abs(dy[valid] / dt[valid] - avg_speed[valid])
    return {
        "rows": len(frame),
        "vehicles": int(frame.vehicle_id.nunique()),
        "duplicate_keys": int(frame.duplicated(["location", "vehicle_id", "global_time"]).sum()),
        "missing": frame.isna().sum().astype(int).to_dict(),
        "negative_speed_rows": int((frame.v_vel < 0).sum()),
        "time_range_epoch_ms": [int(frame.global_time.min()), int(frame.global_time.max())],
        "positive_dt_seconds": {str(p): float(dt[valid].quantile(p)) for p in [0, 0.5, 0.99, 1]},
        "speed_position_discrepancy_mps": {
            str(p): float(mismatch.quantile(p)) for p in [0.5, 0.95, 1]
        },
        "kinematic_mismatch_over_5mps": int((mismatch > 5).sum()),
        "kinematic_comparisons": int(valid.sum()),
        "lane_changes": int((ordered.groupby("vehicle_id").lane_id.diff().fillna(0) != 0).sum()),
        "position_range_m": [
            float(frame.local_y.min() * 0.3048),
            float(frame.local_y.max() * 0.3048),
        ],
        "units": {
            "local_y": "feet converted to meters",
            "speed": "feet/s converted to m/s",
            "global_time": "epoch milliseconds",
        },
        "causal_availability": "Unverified video processing and smoothing; retrospective analysis only",
    }


def complete_strip(frame, start, width):
    f = frame.assign(
        tbin=np.floor((frame.global_time - start) / 5000).astype(int),
        xbin=np.floor(frame.local_y * 0.3048 / width).astype(int),
        speed_mps=frame.v_vel * 0.3048,
    )
    field = f.groupby(["tbin", "xbin"]).speed_mps.mean().unstack()
    field = field.reindex(
        index=np.arange(36), columns=np.arange(field.columns.min(), field.columns.max() + 1)
    )
    complete = field.notna().all(axis=0).to_numpy()
    candidates = []
    left = 0
    for right in range(len(complete) + 1):
        if right == len(complete) or not complete[right]:
            if right > left:
                candidates.append((left, right))
            left = right + 1
    if not candidates:
        return None, {
            "bin_width_m": width,
            "eligible": False,
            "reason": "No complete spatial strip",
        }
    left, right = max(candidates, key=lambda item: item[1] - item[0])
    selected = field.iloc[:, left:right]
    info = {
        "bin_width_m": width,
        "eligible": selected.shape[1] >= 8,
        "spatial_bins": selected.shape[1],
        "spatial_range_m": [
            int(selected.columns[0] * width),
            int((selected.columns[-1] + 1) * width),
        ],
        "missing_full_grid_fraction": float(field.isna().to_numpy().mean()),
        "selection": "Longest contiguous fully observed strip, count-based; earliest strip breaks ties",
    }
    return selected, info


def describe(frame, start, location, out):
    summaries = []
    for width in [50, 25, 100]:
        field, info = complete_strip(frame, start, width)
        summaries.append(info)
        if not info["eligible"]:
            continue
        x = field.to_numpy()
        x = x - x.mean(axis=1, keepdims=True)
        n = x.shape[1]
        w = np.eye(n, k=1) + np.eye(n, k=-1)
        _, basis = np.linalg.eigh(np.diag(w.sum(axis=1)) - w)
        f, p = periodogram(
            x @ basis, fs=0.2, window="hann", detrend=False, axis=0, scaling="density"
        )
        integrated = p.sum(axis=0) * (f[1] - f[0]) / n
        cosine = dct(x, type=2, norm="ortho", axis=1)
        _, cp = periodogram(cosine, fs=0.2, window="hann", detrend=False, axis=0, scaling="density")
        info["max_gft_dct_power_error"] = float(abs(p - cp).max() / n)
        taper = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(36) / 36)
        expected = float((np.mean(x * x, axis=1) * taper**2).sum() / (taper**2).sum())
        info["integrated_nonuniform_power_mps2"] = float(integrated[1:].sum())
        info["direct_taper_power_mps2"] = expected
        info["integration_error"] = abs(expected - integrated[1:].sum())
        info["first_two_mode_share"] = float(integrated[1:3].sum() / integrated[1:].sum())
        pd.DataFrame(
            {
                "mode": np.arange(n),
                "integrated_power_mps2": integrated,
                "share": integrated / integrated[1:].sum(),
            }
        ).to_csv(out / f"{location}_{width}m_modes.csv", index=False)
        field.to_csv(out / f"{location}_{width}m_field.csv")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        im = axes[0].imshow(
            field.T,
            origin="lower",
            aspect="auto",
            cmap="cividis",
            extent=[0, 180, *info["spatial_range_m"]],
        )
        axes[0].set(
            xlabel="Seconds since selected block start",
            ylabel="Local longitudinal position (m)",
            title=location + ": binned mean speed",
        )
        fig.colorbar(im, ax=axes[0], label="Speed (m/s)")
        density = p[:, 1:] / n
        im = axes[1].imshow(
            density,
            origin="lower",
            aspect="auto",
            cmap="cividis",
            extent=[0.5, n - 0.5, f[0] - (f[1] - f[0]) / 2, f[-1] + (f[1] - f[0]) / 2],
        )
        axes[1].set(
            xlabel="Open-path mode index",
            ylabel="Temporal frequency (Hz)",
            title="Finite-window nonuniform power density",
        )
        fig.colorbar(im, ax=axes[1], label="Power density ((m/s)^2/Hz)")
        fig.savefig(out / f"{location}_{width}m_descriptive_spectrum.png", dpi=180)
        if width == 50:
            fig.savefig(out / f"{location}_descriptive_spectrum.png", dpi=180)
        plt.close(fig)
    return summaries


def run(out, data_root=DATA):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists():
        raise FileExistsError("Preserve run; choose a fresh output directory")
    save_json(
        out / "freeze.json",
        {
            "sources": archive_sources(out, HERE),
            "locations": ["us-101", "i-80"],
            "start_offset_seconds": 60,
            "duration_seconds": 180,
            "purpose": "Retrospective descriptive feasibility",
        },
    )
    receipt = {}
    failures = []
    profiles = {}
    try:
        for identity in ["8ect-6jqj", "6a6e-vfvi"]:
            acquire(
                f"https://data.transportation.gov/api/views/{identity}.json",
                identity + "_metadata.json",
                receipt,
                data_root=data_root,
            )
            save_json(out / "manifest.json", receipt)
        path = acquire(
            url(
                {
                    "$select": "location,min(global_time) as first_time,max(global_time) as last_time,count(*) as rows",
                    "$group": "location",
                }
            ),
            "locations.json",
            receipt,
            data_root=data_root,
        )
        ranges = {row["location"]: row for row in json.loads(path.read_text())}
        for location in ["us-101", "i-80"]:
            start = int(ranges[location]["first_time"]) + 60000
            where = f"location='{location}' AND global_time >= {start} AND global_time < {start + 180000}"
            count_path = acquire(
                url({"$select": "count(*) as rows", "$where": where}),
                location + "_count.json",
                receipt,
                data_root=data_root,
            )
            expected = int(json.loads(count_path.read_text())[0]["rows"])
            if not 0 < expected < 500000:
                raise ValueError("Row cap or empty coverage")
            data_path = acquire(
                url(
                    {
                        "$select": COLUMNS,
                        "$where": where,
                        "$order": "global_time,vehicle_id,frame_id",
                        "$limit": 500000,
                    },
                    "csv",
                ),
                location + ".csv",
                receipt,
                64 * 1024**2,
                data_root=data_root,
            )
            save_json(out / "manifest.json", receipt)
            frame = pd.read_csv(data_path)
            if len(frame) != expected:
                raise ValueError(f"Incomplete block: {len(frame)} vs {expected}")
            result = profile(frame)
            result["start_epoch_ms"] = start
            result["license_id"] = "CC_30_BY_SA"
            if result["duplicate_keys"]:
                result["spectral_status"] = "Stopped: duplicate trajectory keys"
            else:
                result["spectra"] = describe(frame, start, location, out)
            profiles[location] = result
            save_json(out / "profiles.json", profiles)
            print(json.dumps({location: result}, indent=2), flush=True)
    except Exception as exc:
        failures.append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        save_json(out / "manifest.json", receipt)
        save_json(out / "failures.json", failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/ngsim")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    run(args.out, args.data.resolve())
