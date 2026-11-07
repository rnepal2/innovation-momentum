"""Fresh, phase-separated SUMO controls for operator/reference validation."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from provenance import archive_sources
from run_mechanism import collect, save_json
from sumo_pilot import circle_network

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASES = np.pi / 16 + np.arange(4) * np.pi / 4


def run(out, raw, physics_followup=False):
    import libsumo

    if (out.exists() and any(out.iterdir())) or (raw.exists() and any(raw.iterdir())):
        raise FileExistsError("Choose fresh output and raw directories")
    out.mkdir(parents=True, exist_ok=True)
    phases = 3 * np.pi / 32 + np.arange(4) * np.pi / 4 if physics_followup else PHASES
    steps = [0.01] if physics_followup else [0.025, 0.01]
    save_json(
        out / "freeze.json",
        {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "physics-fit-followup.md"
            if physics_followup
            else "uncertainty-protocol.md",
            "sources": archive_sources(out, HERE),
            "simulator": libsumo.getVersion(),
            "phases": phases.tolist(),
            "steps": steps,
            "duration_s": 40,
        },
    )
    network, lengths = circle_network(480, raw / "network")
    rows, controls, manifest = [], [], {}
    for tau in [1.0, 1.8]:
        for dt in steps:
            control, meta = collect(network, lengths, 0.3, dt, taus=tau)
            controls.append({"tau": tau, "dt": dt, **meta})
            for name, q, amplitude, kind in [
                ("direct_gap", 3, 0.02, "gap"),
                ("direct_speed", 3, 0.02, "speed"),
                ("harmonic_02", 1, 0.2, "speed"),
                ("harmonic_04", 1, 0.4, "speed"),
            ]:
                for phase_index, phase in enumerate(phases):
                    state, meta = collect(
                        network, lengths, 0.3, dt, q, amplitude, kind, taus=tau, phase=phase
                    )
                    identity = f"t{tau}_dt{dt}_{name}_p{phase_index}"
                    path = raw / (identity + ".npz")
                    np.savez_compressed(path, perturbed=state, control=control)
                    manifest[identity] = {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    rows.append(
                        {
                            "id": identity,
                            "tau": tau,
                            "dt": dt,
                            "excitation": name,
                            "q": q,
                            "amplitude": amplitude,
                            "phase": float(phase),
                            "kind": kind,
                            **meta,
                        }
                    )
            print("Collected", tau, dt, flush=True)
    pd.DataFrame(rows).to_csv(out / "scenarios.csv", index=False)
    save_json(out / "manifest.json", manifest)
    save_json(out / "control_checks.json", controls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/uncertainty_physical")
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "traffic/data/uncertainty_physical"
    )
    parser.add_argument("--physics-followup", action="store_true")
    args = parser.parse_args()
    run(args.out, args.raw.resolve(), args.physics_followup)
