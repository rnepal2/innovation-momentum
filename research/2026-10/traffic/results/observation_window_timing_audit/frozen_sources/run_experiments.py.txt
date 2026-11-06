"""Run the frozen numerical benchmark. No traffic observations enter these scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from provenance import archive_sources
from traffic_core import (
    GEOMETRIES,
    MECHANISMS,
    Design,
    amplification,
    coordinate_bands,
    coupling_fraction,
    cross_spectral_features,
    dct_basis,
    evolution,
    fit_band_transition,
    fit_pca,
    fit_transition,
    forecast_amplitude,
    graph_basis,
    random_basis,
    ridge_predict,
    road_graph,
    simulate,
    spectral_features,
    state_features,
    transport_fixture,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_groups(d: Design, stage="initial"):
    test_start = 20000 if stage == "adaptive" else 10000
    return {
        "train": list(range(1000, 1000 + d.train)),
        "validation": list(range(5000, 5000 + d.validation)),
        "test": list(range(test_start, test_start + d.test)),
    }


def bootstrap_delta(candidate_loss, reference_loss, seed=731):
    delta = np.asarray(candidate_loss) - np.asarray(reference_loss)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
    return float(delta.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()


def run_case(geometry, mechanism, design, case_id, stage="initial"):
    w = road_graph(geometry, design.nodes)
    eigenvalues, u = graph_basis(w)
    bands = coordinate_bands(design.nodes, eigenvalues)
    a = evolution(mechanism, u, bands)
    groups = seed_groups(design, stage)
    observations, windows, labels = {}, {}, {}
    for split, seeds in groups.items():
        observations[split] = np.stack(
            [
                simulate(
                    a,
                    np.random.SeedSequence([seed, case_id]),
                    design,
                    shared=mechanism == "shared_forcing",
                )
                for seed in seeds
            ]
        )
        windows[split] = observations[split][:, : design.window]
        labels[split] = np.array(
            [amplification(x, design.window - 1, design.horizon) for x in observations[split]]
        )
    bases = {"gft": u, "dct": dct_basis(design.nodes), "pca": fit_pca(windows["train"])}
    for j in range(3):
        bases[f"random{j}"] = random_basis(design.nodes, 901 + j)
        bases[f"permuted{j}"] = u[np.random.default_rng(701 + j).permutation(design.nodes)]

    common = {
        split: np.stack([state_features(x, w) for x in data]) for split, data in windows.items()
    }
    features = {"state": common}
    for name, basis in bases.items():
        power, dynamics = {}, {}
        for split, data in windows.items():
            values = [spectral_features(x, basis, bands, design.dt) for x in data]
            power[split] = np.c_[common[split], np.stack([v[0] for v in values])]
            dynamics[split] = np.c_[common[split], np.stack([v[1] for v in values])]
        if name in ("gft", "dct", "pca"):
            features[f"{name}_power"] = power
        features[f"{name}_dynamics"] = dynamics
    features["cpsd_summary"] = {
        split: np.c_[common[split], np.stack([cross_spectral_features(x, design.dt) for x in data])]
        for split, data in windows.items()
    }
    predictions, selections = {}, {}
    for name, values in features.items():
        p, selection, _ = ridge_predict(
            values["train"],
            labels["train"],
            values["validation"],
            labels["validation"],
            values["test"],
        )
        predictions[name], selections[name] = (
            p,
            selection
            | {
                "features": values["train"].shape[1],
                "family": "ridge",
            },
        )

    transitions = [(1, "dmd", False), (2, "hankel", False)]
    if stage == "adaptive":
        transitions += [(1, "dmd_adaptive", True), (2, "hankel_adaptive", True)]
    for delay, name, adaptive in transitions:
        options = []
        ranks = (4, 8, design.nodes) if delay == 1 else (8, 24, 2 * design.nodes)
        for rank in sorted(set(ranks)):
            operator, covariance = fit_transition(windows["train"], rank, delay)
            val = forecast_amplitude(
                windows["validation"], operator, covariance, design.horizon, adaptive
            )
            loss = float(np.mean((val - labels["validation"]) ** 2))
            options.append((loss, rank, operator, covariance))
        loss, rank, operator, covariance = min(options, key=lambda item: item[:2])
        predictions[name] = forecast_amplitude(
            windows["test"], operator, covariance, design.horizon, adaptive
        )
        selections[name] = {"rank": rank, "validation_mse": loss, "family": "transition"}
    band_models = [("gft", False), ("dct", False)]
    if stage == "adaptive":
        band_models += [("gft", True), ("dct", True)]
    for basis_name, adaptive in band_models:
        name = f"{basis_name}_band_ar" + ("_adaptive" if adaptive else "")
        operator, covariance = fit_band_transition(windows["train"], bases[basis_name], bands)
        val = forecast_amplitude(
            windows["validation"], operator, covariance, design.horizon, adaptive
        )
        predictions[name] = forecast_amplitude(
            windows["test"], operator, covariance, design.horizon, adaptive
        )
        selections[name] = {
            "validation_mse": float(np.mean((val - labels["validation"]) ** 2)),
            "family": "transition",
            "rates": len(bands),
        }

    comparators = [name for name in predictions if not name.startswith("gft")]
    # Comparator selection reads validation losses exclusively.
    best = min(comparators, key=lambda name: (selections[name]["validation_mse"], name))
    losses = {name: (p - labels["test"]) ** 2 for name, p in predictions.items()}
    for group in ("random", "permuted"):
        losses[f"{group}_mean_loss"] = np.mean(
            [losses[f"{group}{j}_dynamics"] for j in range(3)], axis=0
        )
    paired, scores, rows = [], [], []
    for name, prediction in predictions.items():
        scores.append(
            {
                "geometry": geometry,
                "mechanism": mechanism,
                "model": name,
                "mse": float(losses[name].mean()),
                "mae": float(np.mean(np.abs(prediction - labels["test"]))),
                "validation_mse": selections[name]["validation_mse"],
                "n_test": design.test,
                "strongest_comparator": best,
            }
        )
        for seed, actual, predicted in zip(groups["test"], labels["test"], prediction):
            rows.append(
                {
                    "geometry": geometry,
                    "mechanism": mechanism,
                    "model": name,
                    "seed": seed,
                    "actual": actual,
                    "predicted": predicted,
                }
            )
    candidates = ["gft_power", "gft_dynamics", "gft_band_ar"]
    if stage == "adaptive":
        candidates.append("gft_band_ar_adaptive")
    for candidate in candidates:
        for reference in dict.fromkeys(
            [
                best,
                "state",
                "dct_dynamics",
                "pca_dynamics",
                "random_mean_loss",
                "permuted_mean_loss",
            ]
        ):
            delta, low, high = bootstrap_delta(losses[candidate], losses[reference])
            paired.append(
                {
                    "geometry": geometry,
                    "mechanism": mechanism,
                    "candidate": candidate,
                    "reference": reference,
                    "delta_mse": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "relative_improvement": float(-delta / losses[reference].mean()),
                    "primary_reference": reference == best,
                }
            )
    dct_distance = max(
        float(np.linalg.norm(u[:, b] @ u[:, b].T - bases["dct"][:, b] @ bases["dct"][:, b].T))
        for b in bands
    )
    diagnostic = {
        "geometry": geometry,
        "mechanism": mechanism,
        "graph_coupling": coupling_fraction(a, u),
        "dct_coupling": coupling_fraction(a, bases["dct"]),
        "max_gft_dct_projector_distance": dct_distance,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a)))),
        "operator_norm": float(np.linalg.norm(a, 2)),
        "test_target_sd": float(np.std(labels["test"])),
        "test_growth_fraction": float(np.mean(labels["test"] > 0)),
    }
    return scores, rows, paired, selections, diagnostic


def figures(out, pairs):
    colors = {"path": "#346d95", "branch": "#a26b18"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    primary = pairs[(pairs.candidate == "gft_dynamics") & pairs.primary_reference].copy()
    for y, row in enumerate(primary.itertuples()):
        ax.errorbar(
            row.delta_mse,
            y,
            xerr=[[row.delta_mse - row.ci_low], [row.ci_high - row.delta_mse]],
            fmt="o" if row.geometry == "path" else "s",
            color=colors[row.geometry],
            capsize=3,
        )
    ax.axvline(0, color="#444444", lw=1)
    ax.set_yticks(
        range(len(primary)),
        [f"{r.geometry} / {r.mechanism}\nvs {r.reference}" for r in primary.itertuples()],
    )
    ax.set_xlabel("GFT dynamics MSE minus validation-selected comparator MSE")
    ax.set_title("Independent synthetic episodes; paired 95% bootstrap intervals")
    fig.savefig(out / "predictive_comparison.png", dpi=170)
    plt.close(fig)
    x, local = transport_fixture()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    im = axes[0].imshow(x.T, origin="lower", aspect="auto", cmap="cividis")
    axes[0].axhline(48, color="white", ls="--", lw=1)
    axes[0].axhline(64, color="white", ls="--", lw=1)
    axes[0].set(
        xlabel="Sample", ylabel="Position index", title="Constant-amplitude translating packet"
    )
    fig.colorbar(im, ax=axes[0], label="State (arbitrary units)")
    axes[1].plot(np.sqrt(np.mean(x * x, axis=1)), label="Whole domain", color="#346d95")
    axes[1].plot(local, label="Fixed region 48:64", color="#a26b18", ls="--")
    axes[1].set(
        xlabel="Sample", ylabel="Spatial RMS (arbitrary units)", title="Observation-region effect"
    )
    axes[1].legend(frameon=False)
    fig.savefig(out / "transport_control.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["initial", "adaptive"], default="initial")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or HERE / "results" / args.stage
    if (out / "freeze.json").exists():
        raise FileExistsError("Preserve existing runs: choose a new --out directory")
    out.mkdir(parents=True, exist_ok=True)
    design = Design()
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))
    }
    freeze = {
        "design": asdict(design),
        "seeds": seed_groups(design, args.stage),
        "stage": args.stage,
        "mechanisms": MECHANISMS,
        "geometries": GEOMETRIES,
        "source_sha256": source_hashes,
        "protocol_sha256": hashlib.sha256((HERE / "protocol.md").read_bytes()).hexdigest(),
    }
    if args.stage == "adaptive":
        freeze["followup_sha256"] = hashlib.sha256(
            (HERE / "forcing-followup.md").read_bytes()
        ).hexdigest()
    freeze["archived_sources_sha256"] = archive_sources(out, HERE)
    save_json(out / "freeze.json", freeze)
    save_json(
        out / "environment.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ),
            "scope": "Synthetic development controls, no empirical traffic inference",
        },
    )
    scores, predictions, pairs, diagnostics, selections = [], [], [], [], {}
    for case_id, (geometry, mechanism) in enumerate((g, m) for g in GEOMETRIES for m in MECHANISMS):
        result = run_case(geometry, mechanism, design, case_id, args.stage)
        s, p, contrasts, selected, diagnostic = result
        scores.extend(s)
        predictions.extend(p)
        pairs.extend(contrasts)
        diagnostics.append(diagnostic)
        selections[f"{geometry}/{mechanism}"] = selected
        print(f"Completed {geometry}/{mechanism}", flush=True)
    for name, values in [
        ("scores", scores),
        ("predictions", predictions),
        ("paired", pairs),
        ("diagnostics", diagnostics),
    ]:
        pd.DataFrame(values).to_csv(out / f"{name}.csv", index=False)
    save_json(out / "selections.json", selections)
    figures(out, pd.DataFrame(pairs))
    print(f"Saved {len(scores)} model/case scores and {len(predictions)} predictions to {out}")


if __name__ == "__main__":
    main()
