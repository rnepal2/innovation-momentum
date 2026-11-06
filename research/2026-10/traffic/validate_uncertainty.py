"""Independent GLS/estimator checks, frozen inputs, and repeat-run receipts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_uncertainty import NOISE, REFERENCES, load_study, setup, state
from mechanism_core import equilibrium, full_operator
from scipy.linalg import expm, toeplitz
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results/uncertainty"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(out):
    value = json.loads((out / "freeze.json").read_text())
    for name, digest in value["sources"].items():
        assert sha(out / "frozen_sources" / (name + ".txt")) == digest


def full_operator_projection(row, accel=0.3, tau=None):
    v, gaps = equilibrium(row["actual_ring_length"], row["tau"])
    angles = 2 * np.pi * 3 * np.arange(24) / 24
    p = np.zeros((4, 48))
    p[0, :24] = np.cos(angles) / 24
    p[1, 24:] = np.cos(angles) / 24
    p[2, :24] = -np.sin(angles) / 24
    p[3, 24:] = -np.sin(angles) / 24
    a = full_operator(gaps, v, accel, row["tau"] if tau is None else tau, row["dt"])
    op = p @ np.linalg.matrix_power(a, round(0.5 / row["dt"])) @ (48 * p.T)
    s = np.diag([1 / gaps[0], 1 / v, 1 / gaps[0], 1 / v])
    return s @ op @ np.linalg.inv(s)


def covariance_time(rho=0, k=None, drift=False, n=41):
    c = toeplitz(float(rho) ** np.arange(n))
    if k is not None:
        c += np.ones((n, n)) / k
        if drift:
            t = np.linspace(-1, 1, n)
            c += np.outer(t, t) / k
    return c


def independent_noise(cov, tc, draws, seed):
    full = np.kron(tc, cov)
    return (
        np.random.default_rng(seed).normal(size=(draws, len(tc) * 4)) @ np.linalg.cholesky(full).T
    ).reshape(draws, len(tc), 4)


def independent_score(y, op, cov, tc=None, nuisance=False):
    n = y.shape[-2]
    if tc is None:
        tc = np.eye(n)
    d = np.concatenate([np.linalg.matrix_power(op, t) for t in range(n)])
    if nuisance:
        d = np.column_stack(
            [d, np.tile(np.eye(4), (n, 1)), np.kron(np.linspace(-1, 1, n)[:, None], np.eye(4))]
        )
    w = np.kron(np.linalg.inv(np.linalg.cholesky(tc)), np.linalg.inv(np.linalg.cholesky(cov)))
    q, _ = np.linalg.qr(w @ d, mode="reduced")
    rank = d.shape[1]
    flat = y.reshape(-1, n * 4) @ w.T
    error = flat - (flat @ q) @ q.T
    return np.sum(error * error, axis=1) / n, float(chi2.ppf(0.95, n * 4 - rank) / n)


def difference(errors, actual, expected):
    errors.append(
        float(
            np.max(
                abs(np.asarray(actual) - np.asarray(expected))
                / np.maximum(1, abs(np.asarray(expected)))
            )
        )
    )


def physical_checks():
    report = {}
    for study, expected, controls in [
        ("uncertainty_physical", 64, 4),
        ("physics_fit_physical", 32, 2),
    ]:
        out = HERE / "results" / study
        frozen(out)
        meta, pairs, _ = load_study(study)
        assert len(meta) == len(pairs) == expected
        assert len(json.loads((out / "control_checks.json").read_text())) == controls
        assert meta[["collisions", "teleports", "leader_errors"]].to_numpy().max() == 0
        assert np.array_equal(meta["effective_dt"], meta["dt"])
        assert meta.recording_clock_error.max() < 1e-9
        assert np.max(abs(meta.last_record_time - meta.first_record_time - 40)) < 1e-9
        for x in pairs.values():
            assert x.shape == (401, 2, 24) and np.isfinite(x).all()
        for entry in json.loads((out / "manifest.json").read_text()).values():
            with np.load(ROOT / entry["path"], allow_pickle=False) as arrays:
                assert (arrays["perturbed"][:, 0] > 0).all()
                assert (arrays["perturbed"][:, 1] >= 0).all()
        repeat = ROOT / "data/research_2026/traffic" / ("repeat_" + study)
        assert (out / "scenarios.csv").read_bytes() == (repeat / "scenarios.csv").read_bytes()
        report[study] = {
            "paired_responses": expected,
            "controls": controls,
            "scenarios_repeat_sha256": sha(out / "scenarios.csv"),
        }
    return report


def scores_check(errors):
    meta, pairs, _ = load_study("uncertainty_physical")
    index = meta.set_index("id")
    for stage in ["operator_error", "reference_noise"]:
        out = OUT / stage
        frozen(out)
        d = pd.read_csv(out / "diagnostics.csv")
        selected = d[d.id.str.endswith("_p0")]
        for r in selected.to_dict("records"):
            info = index.loc[r["id"]].to_dict()
            _, cov, scales, _, _ = setup(info, NOISE[r["noise_index"]])
            case_index = int(meta.index[meta.id == r["id"]][0])
            if stage == "operator_error":
                error = r["relative_parameter_error"]
                a = 0.3 * (1 + error) if r["parameter"] == "acceleration" else 0.3
                tau = info["tau"] * (1 + error) if r["parameter"] == "headway" else info["tau"]
                op = full_operator_projection(info, a, tau)
                tc = np.eye(41)
                noise = independent_noise(
                    cov, tc, 100, 710000 + case_index * 100 + r["noise_index"]
                )
                values, cutoff = independent_score(
                    state(pairs[r["id"]], scales)[None] + noise, op, cov
                )
            else:
                rho = r["rho"]
                reference = r["reference"]
                ri = [x[0] for x in REFERENCES].index(reference)
                _, k, drift = REFERENCES[ri]
                tc = covariance_time(rho, k, drift)
                seed = (
                    610000
                    + case_index * 1000
                    + r["noise_index"] * 100
                    + [0.0, 0.5, 0.9].index(rho) * 10
                    + ri
                )
                noise = independent_noise(cov, tc, 100, seed)
                assumed = (
                    np.eye(41)
                    if r["method"] == "iid_exact_reference"
                    else tc
                    if r["method"] == "correct_joint_covariance"
                    else covariance_time(rho)
                )
                values, cutoff = independent_score(
                    state(pairs[r["id"]], scales)[None] + noise,
                    full_operator_projection(info),
                    cov,
                    assumed,
                    r["method"] == "fit_offset_and_drift",
                )
            difference(errors, values.mean(), r["mean_score"])
            difference(errors, cutoff, r["cutoff"])
            assert abs(np.mean(values > cutoff) - r["rejection_fraction"]) < 1e-12
        print("Independent scores", stage, len(selected), flush=True)


def independent_estimate(h, cov, method):
    sd = np.sqrt(2 * np.diag(cov)[:2])
    z = (h[..., :2] + 1j * h[..., 2:]) / sd
    x = z[:, :-1].reshape(-1, 2).T
    y = z[:, 1:].reshape(-1, 2).T
    if method == "ols":
        a = np.linalg.lstsq(x.T, y.T, rcond=1e-10)[0].T
    elif method == "tls":
        _, _, vh = np.linalg.svd(np.vstack([x, y]), full_matrices=False)
        v = vh[:2].conj().T
        a = (y @ v) @ np.linalg.pinv(x @ v)
    else:
        gram = x @ x.conj().T - x.shape[1] * np.eye(2)
        if np.linalg.eigvalsh(gram).min() <= 0:
            return None
        a = (y @ x.conj().T) @ np.linalg.inv(gram)
    a = sd[:, None] * a / sd[None, :]
    return np.block([[a.real, -a.imag], [a.imag, a.real]])


def learned_checks(errors, restricted=False):
    stage = "physics_fit" if restricted else "learned_operator"
    out = OUT / stage
    frozen(out)
    meta, pairs, _ = load_study("physics_fit_physical" if restricted else "uncertainty_physical")
    tm, tp, _ = load_study("observation")
    cm, cp, _ = load_study("observation_window")
    ops = pd.read_csv(out / "operators.csv")
    table = pd.read_csv(out / "diagnostics.csv")
    for gi, ((tau, dt), evaluation) in enumerate(meta.groupby(["tau", "dt"])):
        training = tm[(tm.tau == tau) & (tm["dt"] == dt) & tm.excitation.str.startswith("direct")]
        calibration = cm[
            (cm.tau == tau) & (cm["dt"] == dt) & cm.excitation.str.startswith("direct")
        ]
        assert not set(training.phase) & set(calibration.phase)
        assert not set(training.phase) & set(evaluation.phase)
        assert not set(calibration.phase) & set(evaluation.phase)
        for ni, noise in enumerate(NOISE):
            info = evaluation.iloc[0].to_dict()
            _, cov, scales, v, gaps = setup(info, noise)
            ts = np.stack([state(tp[r.id], scales, True) for r in training.itertuples()])
            cs = np.stack([state(cp[r.id], scales) for r in calibration.itertuples()])
            for rep in range(20):
                seed = gi * 10000 + ni * 1000 + rep * 10
                bases = [1010000, 1110000, 1210000] if restricted else [310000, 410000, 510000]
                train = ts + independent_noise(cov, np.eye(21), 8, bases[0] + seed)
                group = ops[
                    (ops.tau == tau)
                    & (ops["dt"] == dt)
                    & (ops.noise_index == ni)
                    & (ops.training_replicate == rep)
                ]
                for record in group.to_dict("records"):
                    method = record["method"]
                    if method in ["ols", "tls", "noise_corrected"]:
                        alternative = independent_estimate(train, cov, method)
                        if alternative is None:
                            assert record["status"] == "nonpositive_corrected_gram"
                            continue
                        saved = np.array([[record[f"m{i}{j}"] for j in range(4)] for i in range(4)])
                        difference(errors, saved, alternative)
                    elif method == "known":
                        difference(
                            errors,
                            [[record[f"m{i}{j}"] for j in range(4)] for i in range(4)],
                            full_operator_projection(info),
                        )
                    if rep not in [0, 19] or record["status"] != "eligible":
                        continue
                    estimate = np.array([[record[f"m{i}{j}"] for j in range(4)] for i in range(4)])
                    if method == "restricted_idm":
                        from analyze_physics_fit import fit_physics

                        def objective(a, info=info, record=record, train=train, cov=cov):
                            candidate = full_operator_projection(
                                info, a, record["inferred_headway"]
                            )
                            return independent_score(train, candidate, cov)[0].sum() * 21

                        grid = np.linspace(0.1, 2, 31)
                        values = np.array([objective(a) for a in grid])
                        i = int(values.argmin())
                        assert 0 < i < len(grid) - 1
                        fitted = minimize_scalar(
                            objective,
                            bracket=tuple(grid[i - 1 : i + 2]),
                            method="brent",
                            options={"xtol": 1e-9},
                        )
                        assert abs(fitted.x - record["estimated_acceleration"]) < 2e-6
                        direct, _ = fit_physics(train, cov, scales, v, gaps[0], dt)
                        difference(errors, estimate, direct)
                    cal_noise = independent_noise(cov, np.eye(41), 320, bases[1] + seed)
                    measured = (cs[:, None] + cal_noise.reshape(8, 40, 41, 4)).reshape(-1, 41, 4)
                    cal_values, plugin = independent_score(measured, estimate, cov)
                    empirical = float(np.quantile(cal_values, 0.95))
                    difference(errors, empirical, record["empirical_cutoff"])
                    eval_noise = independent_noise(
                        cov, np.eye(41), len(evaluation) * 100, bases[2] + seed
                    ).reshape(len(evaluation), 100, 41, 4)
                    for j, row in enumerate(evaluation.to_dict("records")):
                        if not row["id"].endswith("_p0"):
                            continue
                        actual, _ = independent_score(
                            state(pairs[row["id"]], scales)[None] + eval_noise[j], estimate, cov
                        )
                        selected = table[
                            (table.id == row["id"])
                            & (table.noise_index == ni)
                            & (table.training_replicate == rep)
                            & (table.method == method)
                        ]
                        for value in selected.to_dict("records"):
                            cutoff = plugin if value["calibration"] == "plugin" else empirical
                            difference(errors, actual.mean(), value["mean_score"])
                            assert (
                                abs(np.mean(actual > cutoff) - value["rejection_fraction"]) < 1e-12
                            )
    print("Independent fitted operators", stage, len(ops), flush=True)


def new_data_check(errors):
    out = HERE / "results/ngsim_expansion"
    frozen(out)
    summary = json.loads((out / "acquisition_summary.json").read_text())
    raw = ROOT / summary["raw_directory"]
    manifest = json.loads((out / "manifest.json").read_text())
    for name, receipt in manifest.items():
        assert sha(raw / name) == receipt["sha256"]
    table = pd.read_csv(out / "spectra.csv")
    profiles = json.loads((out / "profiles.json").read_text())
    for identity, profile in profiles.items():
        frame = pd.read_csv(raw / (identity + ".csv"))
        assert (
            len(frame)
            == int(json.loads((raw / (identity + "_count.json")).read_text())[0]["rows"])
            == profile["rows"]
        )
        assert frame.global_time.min() >= profile["start_epoch_ms"]
        assert frame.global_time.max() < profile["start_epoch_ms"] + 180000
        assert frame.location.eq(profile["location"]).all()
        assert not frame.duplicated(["location", "vehicle_id", "global_time"]).any()
        bad = []
        for vehicle, group in frame.groupby("vehicle_id"):
            group = group.sort_values("global_time")
            dt = np.diff(group.global_time.to_numpy()) / 1000
            position = np.diff(group.local_y.to_numpy()) * 0.3048 / dt
            v = group.v_vel.to_numpy() * 0.3048
            if (abs(position - (v[1:] + v[:-1]) / 2) > 5).any():
                bad.append(int(vehicle))
        assert bad == profile["flagged_vehicle_ids"]
        for quality in ["unfiltered", "flagged_trace_exclusion"]:
            f = frame if quality == "unfiltered" else frame[~frame.vehicle_id.isin(bad)]
            nt = 36
            nx = 26 if profile["location"] == "us-101" else 13
            t = ((f.global_time.to_numpy() - profile["start_epoch_ms"]) // 5000).astype(int)
            x = np.floor(f.local_y.to_numpy() * 0.3048 / 25).astype(int)
            valid = (t >= 0) & (t < nt) & (x >= 0) & (x < nx)
            cell = t[valid] * nx + x[valid]
            counts = np.bincount(cell, minlength=nt * nx)
            selected = table[(table.id == identity) & (table.quality == quality)]
            assert bool((counts > 0).all()) == bool(selected.iloc[0].eligible)
            if (counts == 0).any():
                continue
            field = (
                np.bincount(cell, weights=f.v_vel.to_numpy()[valid] * 0.3048, minlength=nt * nx)
                / counts
            ).reshape(nt, nx)
            field -= field.mean(axis=1, keepdims=True)
            basis = np.sqrt(2 / nx) * np.cos(
                np.pi * np.outer(np.arange(nx) + 0.5, np.arange(nx)) / nx
            )
            basis[:, 0] /= np.sqrt(2)
            z = field @ basis
            taper = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nt) / nt)
            for row in selected.to_dict("records"):
                values = z - z.mean(axis=0) if row["centered"] else z
                power = abs(np.fft.rfft(values * taper[:, None], axis=0)) ** 2
                power[1:-1] *= 2
                integrated = power.sum(axis=0) / (nt * (taper * taper).sum() * nx)
                difference(errors, integrated[1:].sum(), row["power_mps2"])
                difference(
                    errors, integrated[1:3].sum() / integrated[1:].sum(), row["first_two_share"]
                )
    assert sum(p["rows"] for p in profiles.values()) == summary["new_rows"]
    return summary


def repeat_checks():
    report = {}
    for study, names in [
        ("operator_error", ["diagnostics.csv", "summary.csv"]),
        ("learned_operator", ["operators.csv", "diagnostics.csv", "summary.csv"]),
        ("reference_noise", ["diagnostics.csv", "summary.csv", "exact_linear_nulls.csv"]),
        ("physics_fit", ["operators.csv", "diagnostics.csv", "summary.csv"]),
        ("forcing", ["responses.csv", "diagnostics.csv", "sumo_comparison.csv"]),
    ]:
        original = OUT / study
        repeat = ROOT / "data/research_2026/traffic" / ("repeat_uncertainty_" + study)
        report[study] = {}
        for name in names:
            assert (original / name).read_bytes() == (repeat / name).read_bytes(), (study, name)
            report[study][name] = sha(original / name)
    return report


def forcing_check(errors):
    out = OUT / "forcing"
    frozen(out)
    for entry in json.loads((out / "manifest.json").read_text()).values():
        assert sha(ROOT / entry["path"]) == entry["sha256"]
        with np.load(ROOT / entry["path"], allow_pickle=False) as a:
            assert a["response"].shape == a["refined"].shape == (401, 2, 24)
    table = pd.read_csv(out / "responses.csv")
    assert table.refinement_relative_speed_error.max() < 1e-8
    assert table.gap_conservation_error.max() < 1e-9
    for row in pd.read_csv(out / "ambiguity.csv").to_dict("records"):
        assert sha(ROOT / row["path"]) == row["sha256"]
        with np.load(ROOT / row["path"], allow_pickle=False) as a:
            error = np.linalg.norm(a["growing"] - a["decaying_forced"]) / np.linalg.norm(
                a["growing"]
            )
            difference(errors, error, row["relative_state_error"])
            assert error < 1e-8
            assert abs(a["force"][:, 0]).max() == 0
            speed, gaps = equilibrium(479.92, 1.8)
            angle = 2 * np.pi * 3 * np.arange(24) / 24
            p = np.zeros((4, 48))
            p[0, :24], p[1, 24:] = np.cos(angle) / 24, np.cos(angle) / 24
            p[2, :24], p[3, 24:] = -np.sin(angle) / 24, -np.sin(angle) / 24
            growing = p @ full_operator(gaps, speed, 0.3, 1.8) @ (48 * p.T)
            decaying = p @ full_operator(gaps, speed, 1.5, 1.8) @ (48 * p.T)
            initial = np.array([0, 0.01 * np.cos(row["phase"]), 0, 0.01 * np.sin(row["phase"])])
            expected = np.array([expm(growing * t) @ initial for t in a["times"]])
            observed = np.c_[a["growing"].real, a["growing"].imag]
            difference(errors, observed, expected)
            difference(
                errors, np.c_[a["force"].real, a["force"].imag], expected @ (growing - decaying).T
            )
    return {
        "ode_responses": len(table),
        "maximum_refinement_relative_speed_error": float(
            table.refinement_relative_speed_error.max()
        ),
        "maximum_sumo_ode_relative_speed_error_20": float(
            pd.read_csv(out / "sumo_comparison.csv").relative_speed_error_20.max()
        ),
    }


def run():
    errors = []
    report = {"physical": physical_checks(), "reproduction": repeat_checks()}
    scores_check(errors)
    learned_checks(errors)
    learned_checks(errors, True)
    report["ngsim_expansion"] = new_data_check(errors)
    report["forcing"] = forcing_check(errors)
    assert max(errors) < 1e-6, max(errors)
    nulls = pd.read_csv(OUT / "reference_noise/exact_linear_nulls.csv")
    calibrated = nulls[nulls.method.isin(["correct_joint_covariance", "fit_offset_and_drift"])]
    assert calibrated.rejection_fraction.between(0.025, 0.08).all()
    report.update(
        {
            "independent_comparisons": len(errors),
            "maximum_error_scaled_by_max_1_expected": max(errors),
            "score_scope": "All first-phase parameter/reference rows; first/last training-noise fits for diagnostic replay; all unstructured fitted matrices checked independently",
            "exact_linear_calibrated_null_rate_range": [
                float(calibrated.rejection_fraction.min()),
                float(calibrated.rejection_fraction.max()),
            ],
        }
    )
    report["environment"] = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "matplotlib"]
        },
        "simulator": json.loads((HERE / "results/uncertainty_physical/freeze.json").read_text())[
            "simulator"
        ],
    }
    (OUT / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
