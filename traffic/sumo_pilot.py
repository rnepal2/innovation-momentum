"""Paired pulse responses using SUMO's IDM, without custom traffic physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from provenance import archive_sources
from scipy.optimize import brentq
from traffic_core import (
    amplification,
    coordinate_bands,
    fit_band_transition,
    fit_transition,
    forecast_amplitude,
    graph_basis,
)

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "traffic/data/sumo"


def circle_network(circumference, directory):
    directory.mkdir(parents=True, exist_ok=True)
    radius = circumference / (2 * np.pi)
    nodes, edges = ET.Element("nodes"), ET.Element("edges")
    for i in range(4):
        angle = i * np.pi / 2
        ET.SubElement(
            nodes,
            "node",
            id=f"n{i}",
            x=str(radius * np.cos(angle)),
            y=str(radius * np.sin(angle)),
            type="priority",
        )
        angles = np.linspace(angle, angle + np.pi / 2, 25)
        shape = " ".join(f"{radius * np.cos(t)},{radius * np.sin(t)}" for t in angles)
        ET.SubElement(
            edges,
            "edge",
            id=f"e{i}",
            **{
                "from": f"n{i}",
                "to": f"n{(i + 1) % 4}",
                "numLanes": "1",
                "speed": "30",
                "shape": shape,
            },
        )
    for name, value in [("nodes", nodes), ("edges", edges)]:
        ET.ElementTree(value).write(
            directory / f"{name}.xml", encoding="utf-8", xml_declaration=True
        )
    network = directory / "ring.net.xml"
    subprocess.run(
        [
            shutil.which("netconvert"),
            "--node-files",
            str(directory / "nodes.xml"),
            "--edge-files",
            str(directory / "edges.xml"),
            "--output-file",
            str(network),
            "--no-internal-links",
            "true",
            "--junctions.corner-detail",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lengths = [float(node.attrib["length"]) for node in ET.parse(network).findall("edge/lane")]
    if len(lengths) != 4:
        raise ValueError("Unexpected ring geometry")
    return network, np.asarray(lengths)


def real_fourier_basis(n):
    positions = np.arange(n)
    columns = [np.ones(n) / np.sqrt(n)]
    for k in range(1, (n + 1) // 2):
        columns += [
            np.sqrt(2 / n) * np.cos(2 * np.pi * k * positions / n),
            np.sqrt(2 / n) * np.sin(2 * np.pi * k * positions / n),
        ]
    if n % 2 == 0:
        columns.append((-1.0) ** positions / np.sqrt(n))
    return np.column_stack(columns)


def simulation(network, lengths, tau, accel, seed, dt, pulse, pulse_drop=2.0, record_seconds=90):
    import libsumo as sim

    n = 24
    gap = lengths.sum() / n - 5
    equilibrium = brentq(lambda v: 1 - (v / 30) ** 4 - ((2 + tau * v) / gap) ** 2, 0, 29.999)
    rng = np.random.default_rng(seed)
    positions = (
        np.arange(n) * lengths.sum() / n
        + rng.uniform(0, lengths.sum() / n)
        + rng.uniform(-0.02, 0.02, n)
    ) % lengths.sum()
    collisions = teleports = 0
    speeds = []
    try:
        # Select the actual simulator model through vType XML, not a Python substitute.
        routes = ET.Element("routes")
        ET.SubElement(
            routes,
            "vType",
            id="idm",
            carFollowModel="IDM",
            tau=str(tau),
            accel=str(accel),
            decel="2",
            emergencyDecel="9",
            length="5",
            minGap="2",
            maxSpeed="30",
            speedFactor="1",
            speedDev="0",
            delta="4",
        )
        routefile = network.parent / "types.rou.xml"
        ET.ElementTree(routes).write(routefile, encoding="utf-8", xml_declaration=True)
        sim.start(
            [
                "sumo",
                "-n",
                str(network),
                "-r",
                str(routefile),
                "--step-length",
                str(dt),
                "--seed",
                str(seed),
                "--no-step-log",
                "true",
                "--duration-log.disable",
                "true",
                "--time-to-teleport",
                "-1",
                "--collision.action",
                "warn",
                "--no-warnings",
                "true",
            ]
        )
        bounds = np.r_[0, np.cumsum(lengths)]
        for i, pos in enumerate(positions):
            edge = min(int(np.searchsorted(bounds, pos, side="right") - 1), 3)
            route = [f"e{(edge + j) % 4}" for j in range(400)]
            sim.route.add(f"r{i}", route)
            sim.vehicle.add(
                f"v{i}",
                f"r{i}",
                typeID="idm",
                departPos=str(pos - bounds[edge]),
                departSpeed=str(equilibrium),
            )
            sim.vehicle.moveTo(f"v{i}", f"e{edge}_0", float(pos - bounds[edge]))
        for step in range(round((61 + record_seconds) / dt)):
            time = step * dt
            if abs(time - 60) < dt / 2 and pulse:
                sim.vehicle.slowDown("v0", max(0, sim.vehicle.getSpeed("v0") - pulse_drop), 1.0)
            if abs(time - 61) < dt / 2 and pulse:
                sim.vehicle.setSpeed("v0", -1.0)
            sim.simulationStep()
            collisions += sim.simulation.getCollidingVehiclesNumber()
            teleports += sim.simulation.getStartingTeleportNumber()
            now = sim.simulation.getTime()
            if now >= 60 - 1e-6 and abs(now - round(now)) < 1e-6:
                if sim.vehicle.getIDCount() != n:
                    raise ValueError(f"Vehicle retention failed: {sim.vehicle.getIDCount()} of {n}")
                speeds.append([sim.vehicle.getSpeed(f"v{i}") for i in range(n)])
        speed = np.asarray(speeds)[: record_seconds + 1]
        if speed.shape != (record_seconds + 1, n) or not np.isfinite(speed).all():
            raise ValueError("Incomplete physical time series")
        return speed, {
            "collisions": collisions,
            "teleports": teleports,
            "equilibrium_speed": equilibrium,
            "actual_ring_length": float(lengths.sum()),
        }
    finally:
        sim.close()


def main():
    import libsumo

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/sumo")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists():
        raise FileExistsError("Preserve existing runs; use a new --out")
    freeze = {
        "simulator": libsumo.getVersion(),
        "circumferences": [480, 720],
        "vehicles": 24,
        "tau": [0.6, 1.0, 1.8],
        "accel": [0.3, 0.8, 1.5],
        "seeds": [31, 47, 83],
        "steps": [0.2, 0.1],
        "protocol_sha256": hashlib.sha256((HERE / "sumo-protocol.md").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    freeze["archived_sources_sha256"] = archive_sources(out, HERE)
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    adjacency = np.roll(np.eye(24), 1, axis=0) + np.roll(np.eye(24), -1, axis=0)
    e, u = graph_basis(adjacency)
    bands = coordinate_bands(24, e)
    fourier = real_fourier_basis(24)
    cases, trajectories = [], []
    for length in freeze["circumferences"]:
        network, lengths = circle_network(length, RAW / str(length))
        for tau in freeze["tau"]:
            for accel in freeze["accel"]:
                for seed in freeze["seeds"]:
                    for dt in freeze["steps"]:
                        control, cmeta = simulation(network, lengths, tau, accel, seed, dt, False)
                        perturbed, pmeta = simulation(network, lengths, tau, accel, seed, dt, True)
                        x = perturbed - control
                        identifier = f"L{length}_t{tau}_a{accel}_s{seed}_dt{dt}"
                        RAW.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            RAW / f"{identifier}.npz", control=control, pulse=perturbed
                        )
                        actual = amplification(x, 31, 30)
                        logamp = np.log(np.sqrt(np.mean(x * x, axis=1)) + 1e-6)
                        slope = np.polyfit(np.arange(8), logamp[24:32], 1)[0]
                        record = {
                            "id": identifier,
                            "length": length,
                            "tau": tau,
                            "accel": accel,
                            "seed": seed,
                            "dt": dt,
                            "actual": actual,
                            "zero": 0.0,
                            "trend": slope * 30,
                            "initial_amplitude": float(np.exp(logamp[31]) - 1e-6),
                            "pre_pulse_amplitude": float(np.sqrt(np.mean(x[0] ** 2))),
                            "collisions": cmeta["collisions"] + pmeta["collisions"],
                            "teleports": cmeta["teleports"] + pmeta["teleports"],
                            **pmeta,
                        }
                        record["collisions"] = cmeta["collisions"] + pmeta["collisions"]
                        record["teleports"] = cmeta["teleports"] + pmeta["teleports"]
                        windows = x[None, :32]
                        a, q = fit_transition(windows, rank=4)
                        record["dmd"] = float(
                            forecast_amplitude(windows, a, np.zeros_like(q), 30)[0]
                        )
                        for name, basis in [("gft", u), ("fourier", fourier)]:
                            a, q = fit_band_transition(windows, basis, bands)
                            record[name] = float(
                                forecast_amplitude(windows, a, np.zeros_like(q), 30)[0]
                            )
                        cases.append(record)
                        for t, value in enumerate(np.exp(logamp) - 1e-6):
                            trajectories.append(
                                {"id": identifier, "seconds_after_pulse": t, "rms_mps": value}
                            )
        print(f"Completed nominal {length} m ring", flush=True)
    result = pd.DataFrame(cases)
    result.to_csv(out / "scenarios.csv", index=False)
    pd.DataFrame(trajectories).to_csv(out / "amplitudes.csv", index=False)
    print(result.groupby(["length", "tau", "accel"]).actual.agg(["min", "mean", "max"]).to_string())
    print("Maximum GFT/Fourier forecast difference:", np.max(np.abs(result.gft - result.fourier)))


if __name__ == "__main__":
    main()
