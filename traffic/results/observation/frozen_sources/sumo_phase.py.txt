"""Fresh-scenario test of fixed phase-aware spectral dynamics in SUMO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import libsumo
import numpy as np
import pandas as pd
from provenance import archive_sources
from run_experiments import bootstrap_delta
from sumo_pilot import RAW, circle_network, simulation
from traffic_core import (
    amplification,
    coordinate_bands,
    fit_band_transition,
    fit_transition,
    forecast_amplitude,
    graph_basis,
    phase_forecast,
    raw_var2_forecast,
)

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/phase")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists():
        raise FileExistsError("Preserve the existing phase run")
    rng = np.random.default_rng(4261)
    parameters = [
        {
            "length": int(rng.choice([512, 640])),
            "tau": float(rng.uniform(0.55, 1.9)),
            "accel": float(rng.uniform(0.25, 1.6)),
            "seed": 101 + i,
        }
        for i in range(32)
    ]
    freeze = {
        "parameters": parameters,
        "integration_steps": [0.1, 0.05],
        "sumo": libsumo.getVersion(),
        "protocol_sha256": hashlib.sha256((HERE / "phase-followup.md").read_bytes()).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), HERE / "traffic_core.py", HERE / "sumo_pilot.py"]
        },
    }
    freeze["archived_sources_sha256"] = archive_sources(out, HERE)
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    networks = {length: circle_network(length, RAW / str(length)) for length in [512, 640]}
    adjacency = np.roll(np.eye(24), 1, axis=0) + np.roll(np.eye(24), -1, axis=0)
    e, u = graph_basis(adjacency)
    bands = coordinate_bands(24, e)
    rows = []
    for setting in parameters:
        network, lengths = networks[setting["length"]]
        for dt in freeze["integration_steps"]:
            control, cm = simulation(
                network, lengths, setting["tau"], setting["accel"], setting["seed"], dt, False
            )
            pulse, pm = simulation(
                network, lengths, setting["tau"], setting["accel"], setting["seed"], dt, True
            )
            x = pulse - control
            identity = f"phase_s{setting['seed']}_dt{dt}"
            path = RAW / f"{identity}.npz"
            np.savez_compressed(path, control=control, pulse=pulse)
            row = {
                **setting,
                "dt": dt,
                "actual": amplification(x, 31, 30),
                "collisions": cm["collisions"] + pm["collisions"],
                "teleports": cm["teleports"] + pm["teleports"],
                "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "zero": 0.0,
            }
            logamp = np.log(np.sqrt(np.mean(x * x, axis=1)) + 1e-6)
            row["trend"] = float(np.polyfit(np.arange(8), logamp[24:32], 1)[0] * 30)
            row["phase_ar2"] = phase_forecast(x[4:32], 30)
            row["raw_var2"] = raw_var2_forecast(x[4:32], 30)
            a, q = fit_transition(x[None, :32], 4)
            row["dmd"] = float(forecast_amplitude(x[None, :32], a, np.zeros_like(q), 30)[0])
            a, q = fit_band_transition(x[None, :32], u, bands)
            row["gft"] = float(forecast_amplitude(x[None, :32], a, np.zeros_like(q), 30)[0])
            rows.append(row)
        print(f"Completed physical setting {setting['seed']}", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(out / "scenarios.csv", index=False)
    losses = pd.DataFrame(
        {
            m: (data[m] - data.actual) ** 2
            for m in ["zero", "trend", "dmd", "gft", "raw_var2", "phase_ar2"]
        }
    )
    losses["seed"] = data.seed
    grouped = losses.groupby("seed").mean()
    comparisons = []
    for reference in ["zero", "trend", "dmd", "gft", "raw_var2"]:
        delta, lo, hi = bootstrap_delta(grouped.phase_ar2.to_numpy(), grouped[reference].to_numpy())
        comparisons.append(
            {
                "reference": reference,
                "phase_mse": float(grouped.phase_ar2.mean()),
                "reference_mse": float(grouped[reference].mean()),
                "delta_mse": delta,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    pd.DataFrame(comparisons).to_csv(out / "comparisons.csv", index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
