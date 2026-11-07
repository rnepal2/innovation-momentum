"""Expanding-origin forecasts with causal residuals and past-only graph construction."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from common import OUT, graph, permutation_within_activity, prepare, save_json
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def local_predict(x, y, train, test, alpha):
    """Independent standardized ridge models, solved as a batch over concepts."""
    z = x[train]
    mean, scale = z.mean(axis=0), z.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1)
    z = (z - mean) / scale
    target = y[train]
    ym = target.mean(axis=0)
    gram = np.einsum("tnd,tne->nde", z, z)
    gram += alpha * np.eye(z.shape[-1])[None, :, :]
    rhs = np.einsum("tnd,tn->nd", z, target - ym)
    coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
    return np.einsum("tnd,nd->tn", (x[test] - mean) / scale, coefficients) + ym


def baseline_features(x, volume, horizon):
    t, n = x.shape
    feats = np.zeros((t, n, 14))
    for origin in range(12, t):
        own = [x[origin - lag] for lag in [0, 1, 2, 3, 6, 12]]
        own += [
            x[origin - 2 : origin + 1].mean(axis=0),
            x[origin - 11 : origin + 1].mean(axis=0),
            x[origin - 2 : origin + 1].mean(axis=0) - x[origin - 5 : origin - 2].mean(axis=0),
        ]
        common = [
            volume[origin],
            volume[origin - 3],
            volume[origin - 11 : origin + 1].mean(),
            np.sin(2 * np.pi * (origin + horizon) / 12),
            np.cos(2 * np.pi * (origin + horizon) / 12),
        ]
        feats[origin] = np.column_stack(own + [np.full(n, v) for v in common])
    return feats


def causal_baselines(x, volume, horizon):
    f = baseline_features(x, volume, horizon)
    y = np.roll(x, -horizon, axis=0)
    output = np.full_like(x, np.nan)
    for t in range(36, len(x) - horizon):
        train = np.arange(12, t - horizon + 1)
        output[t] = local_predict(f, y, train, [t], 10)[0]
    return output


def correction_features(residual, s, fit_end, kind):
    n = residual.shape[1]
    own = np.stack([residual, np.roll(residual, 1, axis=0), np.roll(residual, 2, axis=0)], axis=-1)
    if kind == "own":
        return own
    pca = PCA(n_components=min(4, n)).fit(residual[38 : fit_end + 1])
    factors = pca.transform(residual)
    common = np.column_stack([factors, np.roll(factors, 1, axis=0)])
    f = np.concatenate(
        [own, np.broadcast_to(common[:, None, :], (len(own), n, common.shape[1]))], axis=-1
    )
    if kind == "factor":
        return f
    extra = np.stack(
        [
            residual @ s.T,
            residual @ (s @ s).T,
            np.roll(residual, 1, axis=0) @ s.T,
            np.roll(residual, 1, axis=0) @ (s @ s).T,
        ],
        axis=-1,
    )
    return np.concatenate([f, extra], axis=-1)


def predict_correction(features, error, train, test, alpha, kind):
    if kind.startswith("pooled"):
        own = features[:, :, :-4]
        fitted = local_predict(own, error, train, train, alpha)
        base = local_predict(own, error, train, test, alpha)
        extra = features[:, :, -4:]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(extra[train].reshape(-1, 4), (error[train] - fitted).ravel())
        return base + model.predict(extra[test].reshape(-1, 4)).reshape(len(test), -1)
    if kind == "transition":
        x = features.reshape(len(features), -1)
        return (
            make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            .fit(x[train], error[train])
            .predict(x[test])
        )
    return local_predict(features, error, train, test, alpha)


def poisson_deviance(y, mu):
    mu = np.maximum(mu, 1e-8)
    term = np.zeros_like(mu)
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    return 2 * (term - y + mu)


def run(data, source, n, horizon, nulls):
    records, months, incidence, monthly, vocab, semantic = data
    counts = monthly[:, :n]
    x = np.log1p(counts)
    volume = np.log1p(np.bincount(months, minlength=192))
    b1 = causal_baselines(x, volume, 1)
    bh = b1 if horizon == 1 else causal_baselines(x, volume, horizon)
    residual = np.zeros_like(x)
    residual[37:] = x[37:] - b1[36:-1]
    # The final baseline row is unavailable; never use it in a scored origin.
    residual = np.nan_to_num(residual)
    error = np.roll(x, -horizon, axis=0) - bh
    kinds = ["own", "factor", "cooccurrence", "pooled_cooccurrence", "transition"]
    if source == "openalex" and any(p["references"] for p in records):
        kinds += ["references", "pooled_references"]
    alphas = [1.0, 10.0, 100.0, 1000.0]
    development = {kind: np.zeros(len(alphas)) for kind in kinds}
    choices, rows = {}, []
    for t in range(84, 192 - horizon, 3):
        tests = np.arange(t, min(t + 3, 192 - horizon))
        train = np.arange(49, t - horizon + 1)
        # Hold the origin fixed within each three-month batch. Later baseline forecasts
        # still use their own past; correction parameters and graph update quarterly.
        s = graph(incidence, months, t, n)
        matrices = {"cooccurrence": s}
        if "references" in kinds:
            matrices["references"] = graph(incidence, months, t, n, records, "references")
        for kind in kinds:
            key = kind.removeprefix("pooled_")
            f = correction_features(
                residual, matrices.get(key, s), t, kind if kind in ["own", "factor"] else "graph"
            )
            if kind == "transition":
                f = np.stack([residual, np.roll(residual, 1, axis=0)], axis=-1)
            if t < 108:
                valid_tests = tests[tests + horizon < 108]
                if not len(valid_tests):
                    continue
                for a, alpha in enumerate(alphas):
                    pred = bh[valid_tests] + predict_correction(
                        f, error, train, valid_tests, alpha, kind
                    )
                    development[kind][a] += np.mean(
                        (pred - np.roll(x, -horizon, axis=0)[valid_tests]) ** 2
                    )
                continue
            alpha = choices.setdefault(kind, alphas[int(np.argmin(development[kind]))])
            pred = bh[tests] + predict_correction(f, error, train, tests, alpha, kind)
            append_losses(rows, source, n, horizon, kind, None, tests, pred, counts)
        if t >= 108:
            append_losses(rows, source, n, horizon, "baseline", None, tests, bh[tests], counts)
            count = counts[: t + 1].sum(axis=0)
            for seed in range(nulls):
                p = permutation_within_activity(count, seed + 19000)
                null = s[np.ix_(p, p)]
                f = correction_features(residual, null, t, "graph")
                pred = bh[tests] + predict_correction(
                    f, error, train, tests, choices["cooccurrence"], "graph"
                )
                append_losses(rows, source, n, horizon, "null", seed, tests, pred, counts)
                pred = bh[tests] + predict_correction(
                    f, error, train, tests, choices["pooled_cooccurrence"], "pooled"
                )
                append_losses(rows, source, n, horizon, "pooled_null", seed, tests, pred, counts)
        if t % 12 == 0:
            print(f"{source} H1 n={n} h={horizon}, origin {2010 + t // 12} complete", flush=True)
    return rows, dict(
        source=source,
        n=n,
        horizon=horizon,
        selected=choices,
        development_loss={k: v.tolist() for k, v in development.items()},
    )


def append_losses(rows, source, n, horizon, kind, seed, tests, pred, counts):
    for row, t in enumerate(tests):
        target_t = t + horizon
        year = 2010 + target_t // 12
        if year < 2019:
            continue
        actual = counts[target_t]
        mu = np.maximum(np.expm1(np.clip(pred[row], 0, 12)), 1e-8)
        rows.append(
            dict(
                source=source,
                n=n,
                horizon=horizon,
                model=kind,
                seed=seed,
                origin=int(t),
                target=int(target_t),
                year=int(year),
                block="2019-20" if year < 2021 else "2021-22" if year < 2023 else "2023-25",
                mse=float(np.mean((pred[row] - np.log1p(actual)) ** 2)),
                deviance=float(poisson_deviance(actual, mu).mean()),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["openalex", "arxiv"], default="openalex")
    parser.add_argument("--nulls", type=int, default=10)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--exclude-january1", action="store_true")
    args = parser.parse_args()
    data = prepare(args.source)
    suffix = ""
    if args.exclude_january1:
        records, months, incidence, monthly, vocab, semantic = data
        keep = np.array([p["date"][5:] != "01-01" for p in records])
        records = [p for p, ok in zip(records, keep) if ok]
        months, incidence = months[keep], incidence[keep]
        monthly = np.array(
            [np.asarray(incidence[months == t].sum(axis=0)).ravel() for t in range(192)]
        )
        data = records, months, incidence, monthly, vocab, semantic
        suffix = "_no_january1"
    rows, choices = [], []
    specs = [(128, 3)] + ([(128, 1), (64, 3)] if args.sensitivity else [])
    for n, horizon in dict.fromkeys((min(n, len(data[4])), h) for n, h in specs):
        result, selected = run(data, args.source, n, horizon, args.nulls)
        rows.extend(result)
        choices.append(selected)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"{args.source}_h1{suffix}.csv", index=False)
    frame.groupby(["n", "horizon", "model", "block"], dropna=False)[
        ["mse", "deviance"]
    ].mean().to_csv(OUT / f"{args.source}_h1{suffix}_summary.csv")
    save_json(OUT / f"{args.source}_h1{suffix}_selection.json", choices)


if __name__ == "__main__":
    main()
