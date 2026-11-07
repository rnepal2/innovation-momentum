"""Independent physical-coordinate checks, harmonic reconstruction, and repeat receipts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from audit_data import fetch_verified
from mechanism_core import derivatives, equilibrium, full_operator
from project_paths import recorded_path
from run_heterogeneity import basis_for, patterns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def band_projectors(u):
    projectors = []
    for low, high in [(0, 1), (1, 5), (5, 13), (13, 24)]:
        p = u[:, low:high] @ u[:, low:high].T
        projectors.append(
            np.block(
                [[p if low else np.zeros((24, 24)), np.zeros((24, 24))], [np.zeros((24, 24)), p]]
            )
        )
    return projectors


def validate():
    errors = []
    counts = {}
    control_count = 0
    for study in ["mechanism", "heterogeneity", "harmonics"]:
        out = HERE / "results" / study
        freeze = json.loads((out / "freeze.json").read_text())
        for name, digest in freeze["sources"].items():
            assert (
                hashlib.sha256((out / "frozen_sources" / (name + ".txt")).read_bytes()).hexdigest()
                == digest
            )
        manifest = json.loads((out / "manifest.json").read_text())
        counts[study] = len(manifest)
        for item in manifest.values():
            path = recorded_path(item["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            with np.load(path, allow_pickle=False) as raw:
                c, p = raw["control"], raw["perturbed"]
            assert (
                c.shape == p.shape == (401, 2, 24) and np.isfinite(c).all() and np.isfinite(p).all()
            )
            assert (p[:, 0] > 0).all() and (p[:, 1] >= 0).all()
        control = json.loads((out / "control_checks.json").read_text())
        control_count += len(control)
        assert all(r["collisions"] == r["teleports"] == r["leader_errors"] == 0 for r in control)
    out = HERE / "results/heterogeneity"
    table = pd.read_csv(out / "responses.csv")
    coupling_table = pd.read_csv(out / "coupling.csv")
    for row in coupling_table.to_dict("records"):
        taus = patterns()[row["pattern"]]
        speed, gaps = equilibrium(479.92, taus)
        a = full_operator(gaps, speed, 0.8, taus)
        if row["basis"] == "weighted_gft":
            fs, _, _ = derivatives(gaps, speed, 0.8, taus)
            weights = np.sqrt(fs * np.roll(fs, -1))
            u = basis_for(weights / weights.mean())
        else:
            u = basis_for(seed=int(row["seed"]) if row["seed"] >= 0 else None)
        scales = np.r_[np.full(24, gaps.mean()), np.full(24, speed)]
        a = a * scales[None, :] / scales[:, None]
        projectors = band_projectors(u)
        physical = sum(projectors)
        full = physical @ a @ physical
        block = sum(p @ a @ p for p in projectors)
        expected = float(np.sum((full - block) ** 2) / np.sum(full**2))
        errors.append(abs(expected - row["coupling_fraction"]))
    manifest = json.loads((out / "manifest.json").read_text())
    cache = {}
    for identity, group in table.groupby("id"):
        item = manifest[identity]
        with np.load(recorded_path(item["path"]), allow_pickle=False) as raw:
            x = raw["perturbed"] - raw["control"]
            taus = raw["taus"]
        row = group.iloc[0]
        key = (row["pattern"], row["dt"])
        if key not in cache:
            speed, gaps = equilibrium(row.actual_ring_length, taus)
            a = full_operator(gaps, speed, 0.8, taus, dt=row["dt"])
            fs, _, _ = derivatives(gaps, speed, 0.8, taus)
            weights = np.sqrt(fs * np.roll(fs, -1))
            weights /= weights.mean()
            operators = {"full_operator": a}
            for name, u in [
                ("uniform_gft", basis_for()),
                ("weighted_gft", basis_for(weights)),
                ("random_basis", basis_for(seed=2000)),
            ]:
                operators[name] = sum(p @ a @ p for p in band_projectors(u))
            cache[key] = {
                name: np.linalg.matrix_power(op, round(0.1 / row["dt"]))
                for name, op in operators.items()
            }
        for record in group.to_dict("records"):
            state = np.r_[x[0, 0], x[0, 1]]
            history = [state]
            for _ in range(400):
                history.append(cache[key][record["basis"]] @ history[-1])
            speed = np.asarray(history)[:, 24:]
            for size, label in [(201, "speed_error_20"), (401, "speed_error_40")]:
                expected = np.sqrt(
                    np.sum((x[:size, 1] - speed[:size]) ** 2) / np.sum(x[:size, 1] ** 2)
                )
                errors.append(abs(expected - record[label]))
    out = HERE / "results/harmonics"
    table = pd.read_csv(out / "responses.csv")
    manifest = json.loads((out / "manifest.json").read_text())
    for identity, group in table.groupby("id"):
        with np.load(recorded_path(manifest[identity]["path"]), allow_pickle=False) as raw:
            x = raw["perturbed"][:, 1] - raw["control"][:, 1]
        power = abs(np.fft.rfft(x, axis=1, norm="ortho")) ** 2 / 24
        power[:, 1:-1] *= 2
        for row in group.to_dict("records"):
            start = 100 if row["window"] == "early" else 200
            errors.append(abs(power[start : start + 100, row["q"]].mean() - row["power"]))
    exponents = pd.read_csv(out / "exponents.csv")
    for row in exponents.to_dict("records"):
        selected = table[
            (table.accel == row["accel"])
            & (table["dt"] == row["dt"])
            & (table.q == row["q"])
            & (table.window == row["window"])
            & (table.power > 1e-14)
        ]
        assert len(selected) == row["eligible_amplitudes"]
        if len(selected) >= 3:
            x, y = np.log(selected.amplitude.to_numpy()), np.log(selected.power.to_numpy())
            slope = np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean())
            errors.append(abs(slope - row["power_exponent"]))
    autonomous = pd.read_csv(out / "autonomous_comparison.csv")
    coordinate = 2 * np.pi * 3 * np.arange(24) / 24
    p3 = (
        2
        * (
            np.outer(np.cos(coordinate), np.cos(coordinate))
            + np.outer(np.sin(coordinate), np.sin(coordinate))
        )
        / 24
    )
    for identity, group in autonomous.groupby("id"):
        with np.load(recorded_path(manifest[identity]["path"]), allow_pickle=False) as raw:
            state = raw["perturbed"] - raw["control"]
        info = table[table.id == identity].iloc[0]
        speed, gaps = equilibrium(info.actual_ring_length)
        step = np.linalg.matrix_power(
            full_operator(gaps, speed, info.accel, dt=info["dt"]), round(0.1 / info["dt"])
        )
        history = [np.r_[p3 @ state[0, 0], p3 @ state[0, 1]]]
        for _ in range(400):
            history.append(step @ history[-1])
        power = np.mean(np.asarray(history)[:, 24:] ** 2, axis=1)
        for row in group.to_dict("records"):
            start = 100 if row["window"] == "early" else 200
            prediction = float(power[start : start + 100].mean())
            relative = abs(prediction - row["autonomous_linear_power"]) / max(prediction, 1e-30)
            assert relative < 1e-5, (identity, relative)
            actual = (
                table[(table.id == identity) & (table.q == 3) & (table.window == row["window"])]
                .iloc[0]
                .power
            )
            errors.extend(
                [
                    abs(actual - row["observed_power"]),
                    abs(prediction - row["autonomous_linear_power"]),
                ]
            )
    growth = table.pivot(index=["accel", "dt", "amplitude", "q"], columns="window", values="power")
    growth["eligible"] = (growth.early > 1e-14) & (growth.late > 1e-14)
    growth["finite_window_growth"] = np.where(
        growth.eligible, np.log(growth.late / growth.early) / 20, np.nan
    )
    saved_growth = pd.read_csv(out / "growth.csv")
    needs_floor_correction = (
        saved_growth.loc[~saved_growth.eligible, "finite_window_growth"].notna().any()
    )
    if needs_floor_correction and not (out / "growth_before_floor.csv").exists():
        shutil.copyfile(out / "growth.csv", out / "growth_before_floor.csv")
    growth.to_csv(out / "growth.csv")
    assert max(errors) < 1e-7, max(errors)
    repeats = {}
    for study, files in [
        ("mechanism", ["scenarios.csv", "responses.csv", "operators.csv"]),
        ("heterogeneity", ["responses.csv", "coupling.csv"]),
        ("harmonics", ["responses.csv", "exponents.csv"]),
    ]:
        original = HERE / "results" / study
        repeat_name = (
            "repeat_harmonics_clock_corrected" if study == "harmonics" else "repeat_" + study
        )
        repeat = ROOT / "traffic/data" / repeat_name
        verified = {}
        for name in files:
            a, b = (original / name).read_bytes(), (repeat / name).read_bytes()
            assert a == b, (study, name)
            verified[name] = hashlib.sha256(b).hexdigest()
        repeats[study] = verified
    source = ROOT / "traffic/data/IDM_v1_28_0.cpp"
    receipt_path = HERE / "results/mechanism/simulator_source.json"
    if not receipt_path.exists():
        receipt = fetch_verified(
            "https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_28_0/src/microsim/cfmodels/MSCFModel_IDM.cpp",
            source,
            128 * 1024,
        )
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == json.loads(receipt_path.read_text())["sha256"]
    )
    result = {
        "responses": counts,
        "independent_comparisons": len(errors),
        "max_absolute_discrepancy": float(max(errors)),
        "full_production_runs": sum(counts.values()) + control_count,
        "full_rerun_byte_identical": repeats,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "libsumo", "eclipse-sumo"]
        },
        "harmonic_below_floor_growth_rows": int((~growth.eligible).sum()),
    }
    (HERE / "results/mechanism/extension_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    validate()
