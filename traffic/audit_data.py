"""Bounded acquisition and measurement audit of a public derived I-24 sample."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "traffic/data"
WAVE_COMMIT = "dbe85b4ad5fbb592840036fa4f5edde2d427f28a"
VT_COMMIT = "e51e3e98124ab2ef6aa2f2d478f64f82f445aa96"
CSV_URL = (
    f"https://raw.githubusercontent.com/I24-MOTION/wave-analysis/{WAVE_COMMIT}"
    "/demo_data/vt_lane_1.csv"
)
ASM_URL = f"https://raw.githubusercontent.com/I24-MOTION/VT_tools/{VT_COMMIT}/ASM.py"
CSV_BYTES = 54_743_156


def fetch_verified(url, destination, max_bytes, expected_size=None, expected_sha=None):
    """Bound and verify before replacing an accepted local input."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "innovation-momentum-research/0.1"})
        with urlopen(request, timeout=60) as response:
            announced = response.headers.get("Content-Length")
            if announced and int(announced) > max_bytes:
                raise ValueError("Announced size exceeds acquisition limit")
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                digest, count = hashlib.sha256(), 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > max_bytes:
                        raise ValueError("Stream exceeds acquisition limit")
                    stream.write(block)
                    digest.update(block)
            if expected_size is not None and count != expected_size:
                raise ValueError(f"Incomplete input: expected {expected_size}, received {count}")
            checksum = digest.hexdigest()
            if expected_sha is not None and checksum != expected_sha:
                raise ValueError("Checksum mismatch")
        os.replace(temporary, destination)
        temporary = None
        return {
            "url": url,
            "bytes": count,
            "sha256": checksum,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def profile_trajectories(frame: pd.DataFrame):
    required = ["time", "space", "speed", "v_id"]
    if list(frame.columns) != required:
        raise ValueError(f"Unexpected schema: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("Empty input")
    if not np.isfinite(frame[["time", "v_id"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite trajectory keys")
    finite = np.isfinite(frame[required].to_numpy(dtype=float)).all(axis=1)
    usable = frame.loc[finite]
    if usable.empty:
        raise ValueError("No finite physical observations")
    original_dt = usable.groupby("v_id", sort=False).time.diff()
    ordered = usable.sort_values(["v_id", "time"])
    dt = ordered.groupby("v_id", sort=False).time.diff()
    dx = ordered.groupby("v_id", sort=False).space.diff()
    previous_speed = ordered.groupby("v_id", sort=False).speed.shift()
    ok = dt > 0
    # The source convention is seconds, miles, and mph; check it against kinematics.
    implied = np.abs(dx[ok]) / dt[ok] * 3600
    discrepancy = np.abs(implied - previous_speed[ok])
    return {
        "rows": len(frame),
        "virtual_trajectory_ids": int(frame.v_id.nunique()),
        "schema": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "units": {"time": "relative seconds", "space": "mile marker", "speed": "mph"},
        "finite_fraction": float(finite.mean()),
        "missing_by_column": {name: int(value) for name, value in frame.isna().sum().items()},
        "nonfinite_physical_rows": int((~finite).sum()),
        "nonfinite_at_last_timestamp": int(
            (~finite & frame.time.eq(frame.groupby("v_id").time.transform("max"))).sum()
        ),
        "duplicate_vehicle_time": int(frame.duplicated(["v_id", "time"]).sum()),
        "nonpositive_dt_in_source_order": int((original_dt.dropna() <= 0).sum()),
        "time_range_seconds": [float(frame.time.min()), float(frame.time.max())],
        "space_range_miles": [float(frame.space.min()), float(frame.space.max())],
        "speed_quantiles_mph": {
            str(q): float(frame.speed.quantile(q)) for q in [0, 0.01, 0.5, 0.99, 1]
        },
        "positive_dt_quantiles_seconds": {str(q): float(dt[ok].quantile(q)) for q in [0, 0.5, 1]},
        "kinematic_abs_error_mph": {str(q): float(discrepancy.quantile(q)) for q in [0.5, 0.95, 1]},
        "negative_speed_rows": int((frame.speed < 0).sum()),
        "date_and_timezone": "No calendar date or timezone column; unverified from this file alone",
        "independence": "Generated virtual IDs are not independently observed vehicles or days",
        "forecast_eligibility": "Not established: processing availability and independent days unresolved",
    }


def inspect_smoothing(source: str):
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    egtf = functions["EGTF"]
    filters = [
        node
        for node in ast.walk(egtf)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "speed" for t in node.targets)
    ]
    expression = next(
        ast.unparse(node.value) for node in filters if isinstance(node.value, ast.Subscript)
    )
    signature = functions["smooth_raw_data"].args
    defaults = dict(
        zip(
            [a.arg for a in signature.args][-len(signature.defaults) :],
            [ast.literal_eval(v) for v in signature.defaults],
        )
    )
    expected = "np.abs(speed_raw_df.t - t) <= smooth_t_window / 2"
    if expected not in expression:
        raise ValueError("Upstream temporal filter changed; manual review required")
    return {
        "filter_expression": expression,
        "default_temporal_window_seconds": defaults["smooth_t_window"],
        "default_future_support_seconds": defaults["smooth_t_window"] / 2,
        "interpretation": "Symmetric filter admits future observations at its target timestamp",
        "scope": "Inspected source implementation; demo generation parameters/lineage not verified",
        "remedy": "Use raw causal inputs, regenerate with past-only support, or verify full availability delay",
        "third_party_code_executed": False,
    }


def figure(frame, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.floor((frame.time - frame.time.min()) / 10).astype(int)
    x = np.floor((frame.space - frame.space.min()) / 0.05).astype(int)
    field = frame.groupby([x, t]).speed.mean().unstack().sort_index().sort_index(axis=1)
    field = field.reindex(index=np.arange(x.max() + 1), columns=np.arange(t.max() + 1))
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        cmap="cividis",
        vmin=0,
        vmax=80,
        extent=[
            0,
            (field.columns.max() + 1) * 10 / 60,
            frame.space.min(),
            frame.space.min() + (field.index.max() + 1) * 0.05,
        ],
    )
    ax.set(
        xlabel="Minutes since sample start",
        ylabel="Mile marker",
        title="I-24 virtual-trajectory demo: descriptive binned mean speed",
    )
    fig.colorbar(im, ax=ax, label="Speed (mph)", extend="both")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    out = HERE / "results/data_audit"
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        (CSV_URL, DATA / "vt_lane_1.csv", 64 * 1024**2, CSV_BYTES),
        (ASM_URL, DATA / "ASM.py", 128 * 1024, None),
    ]
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for url, path, limit, expected_size in sources:
        accepted = previous.get(path.name)
        if path.exists() and accepted:
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum != accepted["sha256"]:
                raise ValueError(f"Accepted input changed: {path.name}")
            manifest[path.name] = accepted
        elif args.fetch:
            manifest[path.name] = fetch_verified(url, path, limit, expected_size)
            manifest_path.write_text(json.dumps(previous | manifest, indent=2) + "\n")
        else:
            raise FileNotFoundError(f"Run with --fetch for verified acquisition: {path}")
    frame = pd.read_csv(DATA / "vt_lane_1.csv")
    profile = profile_trajectories(frame)
    smoothing = inspect_smoothing((DATA / "ASM.py").read_text())
    for name, value in [("manifest", manifest), ("profile", profile), ("smoothing", smoothing)]:
        (out / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    figure(frame.dropna(subset=["space", "speed"]), out / "speed_field.png")
    print(json.dumps(profile, indent=2))
    print(json.dumps(smoothing, indent=2))


if __name__ == "__main__":
    main()
