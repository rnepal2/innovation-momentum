"""Independently replay one training-noise fit per headway and noise level."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_uncertainty import load_study, setup
from run_mechanism import save_json

HERE = Path(__file__).resolve().parent


def modal_history(pair, scales):
    z = np.fft.fft(pair[:, :2], axis=-1)[:, :, 3] / 24
    return np.column_stack([z.real, z.imag])[100:301:5] @ scales.T


def scores(histories, operator, covariance):
    # Explicit whitened least squares avoids the production SVD/projection shortcut.
    design = np.vstack([np.linalg.matrix_power(operator, t) for t in range(41)])
    sd = np.tile(np.sqrt(np.diag(covariance)), 41)
    design = design / sd[:, None]
    observed = histories.reshape(-1, 164).T / sd[:, None]
    coefficients = np.linalg.lstsq(design, observed, rcond=1e-10)[0]
    residual = observed - design @ coefficients
    return np.sum(residual**2, axis=0) / 41


def validate(out):
    freeze = json.loads((out / "freeze.json").read_text())
    for name, expected in freeze["sources"].items():
        actual = hashlib.sha256((out / "frozen_sources" / (name + ".txt")).read_bytes()).hexdigest()
        assert actual == expected
    for name, expected in freeze["scenario_hashes"].items():
        assert (
            hashlib.sha256((HERE / "results" / name / "scenarios.csv").read_bytes()).hexdigest()
            == expected
        )
    meta, pairs, evaluation_receipt = load_study("physics_fit_physical")
    cmeta, cpairs, calibration_receipt = load_study("observation_window")
    _, _, training_receipt = load_study("observation")
    assert evaluation_receipt == freeze["evaluation"]
    assert calibration_receipt == freeze["calibration"]
    assert training_receipt == freeze["training"]
    operators = pd.read_csv(out / "operators.csv")
    diagnostics = pd.read_csv(out / "diagnostics.csv")
    max_error, checks = 0.0, 0
    for model in operators[operators.training_replicate == 0].to_dict("records"):
        evaluation = meta[(meta.tau == model["tau"]) & (meta["dt"] == model["dt"])]
        calibration = cmeta[
            (cmeta.tau == model["tau"])
            & (cmeta["dt"] == model["dt"])
            & cmeta.excitation.str.startswith("direct")
        ]
        _, covariance, scales, _, _ = setup(
            evaluation.iloc[0], (model["gap_noise_sd_m"], model["speed_noise_sd_mps"])
        )
        operator = np.array([[model[f"m{i}{j}"] for j in range(4)] for i in range(4)])
        sd = np.sqrt(np.diag(covariance))
        cal_states = np.stack(
            [modal_history(cpairs[r.id], scales) for r in calibration.itertuples()]
        )
        cal_noise = np.random.default_rng(model["calibration_seed"]).normal(size=(320, 41, 4)) * sd
        cal_samples = (cal_states[:, None] + cal_noise.reshape(8, 40, 41, 4)).reshape(-1, 41, 4)
        cutoff = np.quantile(scores(cal_samples, operator, covariance), 0.95)
        np.testing.assert_allclose(cutoff, model["empirical_cutoff"], rtol=1e-10, atol=1e-10)
        max_error = max(max_error, abs(cutoff - model["empirical_cutoff"]))
        eval_states = np.stack(
            [modal_history(pairs[r.id], scales) for r in evaluation.itertuples()]
        )
        eval_noise = np.random.default_rng(model["evaluation_seed"]).normal(size=(1600, 41, 4)) * sd
        measured = eval_states[:, None] + eval_noise.reshape(16, 100, 41, 4)
        values = scores(measured, operator, covariance).reshape(16, 100)
        saved = (
            diagnostics[
                (diagnostics.tau == model["tau"])
                & (diagnostics.speed_noise_sd_mps == model["speed_noise_sd_mps"])
                & (diagnostics.training_replicate == 0)
            ]
            .set_index("id")
            .loc[evaluation.id]
        )
        np.testing.assert_allclose(values.mean(axis=1), saved.mean_score, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            (values > cutoff).mean(axis=1), saved.rejection_fraction, atol=1e-14
        )
        max_error = max(max_error, float(abs(values.mean(axis=1) - saved.mean_score).max()))
        checks += 1 + 2 * len(evaluation)
    save_json(
        out / "independent_validation.json",
        {
            "fits_replayed": 22,
            "scalar_comparisons": checks,
            "max_absolute_error": max_error,
            "scope": "First training realization at every headway/noise level; explicit least-squares calibration and all evaluation means/rates. All source/raw/scenario hashes verified. Optimizer is not independently rerun here.",
        },
    )
    print(f"Independent noise-grid validation: {checks} comparisons; max error {max_error:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results/noise_resolution")
    validate(parser.parse_args().out)
