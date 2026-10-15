"""Generate numerical summaries, dependence-aware sensitivity intervals and figures."""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import DATA, OUT, save_json

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
COLORS = {
    "base": "#4b5563",
    "spectral": "#007f73",
    "null": "#b7bac0",
    "own": "#4b5563",
    "factor": "#be3c65",
    "graph": "#007f73",
    "shuffled_graph": "#b7bac0",
    "transition": "#bc7100",
}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 170,
        "axes.titlesize": 11,
    }
)


def read(path):
    return pd.read_csv(path, keep_default_na=False)


def block_interval(base, candidate, block, repeats=1000):
    rng = np.random.default_rng(610)
    n = len(base)
    effects = []
    for _ in range(repeats):
        starts = rng.integers(n, size=int(np.ceil(n / block)))
        index = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
        effects.append(100 * (candidate[index].mean() / base[index].mean() - 1))
    return np.quantile(effects, [0.025, 0.975])


def weighted_ap(y, p, w):
    order = np.argsort(-p, kind="stable")
    y, p, w = y[order], p[order], w[order]
    end = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y * w)[end]
    fp = np.cumsum((1 - y) * w)[end]
    if tp[-1] == 0:
        return np.nan
    precision = tp / np.maximum(tp + fp, 1e-20)
    return np.sum(np.diff(np.r_[0, tp]) / tp[-1] * precision)


def node_interval(y, base, spectral, ii, jj, repeats=300):
    if y.sum() == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(731)
    n = max(ii.max(), jj.max()) + 1
    effects = []
    for _ in range(repeats):
        weights = rng.multinomial(n, np.full(n, 1 / n))
        w = weights[ii] * weights[jj]
        effects.append(100 * (weighted_ap(y, spectral, w) - weighted_ap(y, base, w)))
    return np.nanquantile(effects, [0.025, 0.975])


def h1_summary():
    totals, intervals = [], []
    for source, suffix in [("openalex", ""), ("openalex", "_no_january1"), ("arxiv", "")]:
        df = read(OUT / f"{source}_h1{suffix}.csv")
        for (n, h), group in df.groupby(["n", "horizon"]):
            base = group[group.model == "baseline"].sort_values("target")
            for model, part in group.groupby("model"):
                totals.append(
                    dict(
                        source=source,
                        sensitivity=suffix,
                        n=n,
                        horizon=h,
                        model=model,
                        months=len(base),
                        mse=part.mse.mean(),
                        deviance=part.deviance.mean(),
                        mse_change_pct=100 * (part.mse.mean() / base.mse.mean() - 1),
                    )
                )
                candidate = part.groupby("target").mse.mean().reindex(base.target).to_numpy()
                if model != "baseline":
                    for block in [6, 12]:
                        low, high = block_interval(base.mse.to_numpy(), candidate, block)
                        intervals.append(
                            dict(
                                source=source,
                                sensitivity=suffix,
                                n=n,
                                horizon=h,
                                model=model,
                                block_months=block,
                                low_pct=low,
                                high_pct=high,
                            )
                        )
    pd.DataFrame(totals).to_csv(OUT / "h1_totals.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "h1_block_resampling.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    models = [
        "factor",
        "cooccurrence",
        "pooled_cooccurrence",
        "references",
        "pooled_references",
        "transition",
    ]
    labels = [
        "Factors",
        "Graph",
        "Pooled graph",
        "References",
        "Pooled references",
        "Learned transition",
    ]
    for ax, (source, suffix, title) in zip(
        axes,
        [
            ("openalex", "", "OpenAlex"),
            ("openalex", "_no_january1", "OpenAlex: January-1 dates excluded"),
            ("arxiv", "", "arXiv: 38 phrases"),
        ],
    ):
        df = read(OUT / f"{source}_h1{suffix}.csv")
        df = df[(df.n == df.n.max()) & (df.horizon == 3)]
        pivot = df.groupby(["model", "block"]).mse.mean().unstack()
        change = 100 * (pivot.div(pivot.loc["baseline"]) - 1)
        chosen = [m for m in models if m in change.index]
        values = change.reindex(chosen).to_numpy()
        ax.imshow(values, vmin=-6, vmax=6, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(change.columns)), change.columns)
        ax.set_yticks(range(len(chosen)), [labels[models.index(m)] for m in chosen])
        for i in range(len(chosen)):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:+.1f}%", ha="center", va="center", fontsize=9)
        ax.set_title(title)
    fig.suptitle(
        "Activity forecasts: change in log-count MSE against the temporal baseline\nThree-month horizon; negative values are improvements",
        fontsize=12,
    )
    fig.savefig(FIG / "activity.png")
    plt.close(fig)


def h2_summary():
    rows = []
    for source in ["openalex", "arxiv"]:
        for suffix in ["", "_rolling"]:
            df = read(OUT / f"{source}_h2{suffix}.csv")
            for c in ["ap", "roc"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            pred = np.load(DATA / f"{source}_h2{suffix}_predictions.npz")
            for (risk, year), part in df[df.target == "independent"].groupby(["risk", "year"]):
                base = part[part.feature == "base"].iloc[0]
                spec = part[part.feature == "spectral"].iloc[0]
                null = part[part.feature == "null"]
                prefix = f"{year}_{risk}_"
                low, high = node_interval(
                    pred[prefix + "y"],
                    pred[prefix + "base"],
                    pred[prefix + "spectral"],
                    pred[prefix + "i"],
                    pred[prefix + "j"],
                )
                rows.append(
                    dict(
                        source=source,
                        vocabulary="rolling" if suffix else "fixed",
                        risk=risk,
                        origin=year,
                        n=base.n,
                        events=base.events,
                        prevalence=base.prevalence,
                        base_ap=base.ap,
                        spectral_ap=spec.ap,
                        delta_ap_pp=100 * (spec.ap - base.ap),
                        null_ap_min=null.ap.min(),
                        null_ap_max=null.ap.max(),
                        null_ap_mean=null.ap.mean(),
                        node_resampling_low_pp=low,
                        node_resampling_high_pp=high,
                        base_p20=base.p20,
                        spectral_p20=spec.p20,
                        base_p50=base.p50,
                        spectral_p50=spec.p50,
                        base_brier=base.brier,
                        spectral_brier=spec.brier,
                    )
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "h2_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, risk in enumerate(["zero", "weak"]):
        for col, vocabulary in enumerate(["fixed", "rolling"]):
            ax = axes[row, col]
            part = result[
                (result.source == "openalex")
                & (result.risk == risk)
                & (result.vocabulary == vocabulary)
            ]
            x = np.arange(len(part))
            ax.fill_between(
                x,
                part.null_ap_min,
                part.null_ap_max,
                color=COLORS["null"],
                alpha=0.5,
                label="10 activity-stratified graph nulls",
            )
            ax.plot(
                x,
                part.base_ap,
                "o-",
                color=COLORS["base"],
                label="Temporal + local network controls",
            )
            ax.plot(
                x,
                part.spectral_ap,
                "o-",
                color=COLORS["spectral"],
                label="Controls + spectral features",
            )
            ax.set_xticks(x, [f"{y + 1}-{str(y + 2)[2:]}" for y in part.origin])
            ax.set_ylabel("Average precision")
            ax.set_ylim(bottom=0)
            ax.set_title(
                f"{'Previously absent' if risk == 'zero' else '1-3 prior papers'}; {vocabulary} vocabulary"
            )
            ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1, frameon=False)
    fig.suptitle(
        "Sustained joint use in OpenAlex: three papers with disjoint listed authors\nNatural class prevalence; outcome windows shown on the horizontal axis",
        fontsize=12,
    )
    fig.savefig(FIG / "recombination.png")
    plt.close(fig)
    # Calibration of the fixed-vocabulary rankings, with quantile bins.
    pred = np.load(DATA / "openalex_h2_predictions.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, risk in zip(axes, ["zero", "weak"]):
        for year, style in [(2019, "-"), (2023, "--")]:
            y, p = pred[f"{year}_{risk}_y"], pred[f"{year}_{risk}_spectral"]
            order = np.argsort(p)
            groups = np.array_split(order, 10)
            ax.plot(
                [p[g].mean() for g in groups],
                [y[g].mean() for g in groups],
                style,
                marker="o",
                label=f"Origin {year}",
            )
        ax.plot([0, 1], [0, 1], color="#888888", lw=1)
        ax.set(
            xlabel="Mean predicted probability",
            ylabel="Observed fraction",
            title=f"{risk.capitalize()} pairs",
        )
        ax.legend(frameon=False)
    fig.suptitle("Probability calibration: retrospective joint-use events", fontsize=12)
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)


def synthetic_figure():
    df = read(OUT / "synthetic.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), layout="constrained")
    for ax, process in zip(axes, ["independent", "shared_shock", "graph", "directed"]):
        for model in ["own", "factor", "graph", "shuffled_graph", "transition"]:
            part = df[(df.process == process) & (df.model == model)]
            mean = part.groupby("length").mse.mean()
            se = part.groupby("length").mse.std() / np.sqrt(20)
            ax.errorbar(
                [0, 1],
                mean,
                yerr=se,
                marker="o",
                color=COLORS[model],
                label=model.replace("_", " "),
            )
        ax.set_title(process.replace("_", " ").capitalize())
        ax.set_xticks([0, 1], ["96", "240"])
        ax.set_xlabel("Time observations")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Later-period MSE")
    fig.suptitle(
        "Known-process checks: 20 seeds per condition; bars show seed standard errors", fontsize=12
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False
    )
    fig.savefig(FIG / "synthetic.png")
    plt.close(fig)


def main():
    h1_summary()
    h2_summary()
    synthetic_figure()
    save_json(
        OUT / "acquisition_manifest.json",
        {
            "openalex": json.loads((DATA / "manifest.json").read_text()),
            "arxiv": json.loads((DATA / "arxiv_manifest.json").read_text()),
            "resampling": "H1 circular time blocks; H2 multinomial vertex weights, pair weight wi*wj. Sensitivity intervals conditional on the fitted rankings, not population coverage guarantees.",
        },
    )


if __name__ == "__main__":
    main()
