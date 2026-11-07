"""Exploratory held-out cross-lag covariance in fixed, past-only bases."""

import numpy as np
import pandas as pd
from common import OUT, graph, permutation_within_activity, prepare
from h1 import causal_baselines
from sklearn.decomposition import PCA


def main():
    rows = []
    for source in ["openalex", "arxiv"]:
        _records, months, incidence, monthly, vocab, _ = prepare(source)
        n = min(128, len(vocab))
        x = np.log1p(monthly[:, :n])
        volume = np.log1p(np.bincount(months, minlength=192))
        baseline = causal_baselines(x, volume, 1)
        residual = x[37:] - baseline[36:-1]
        train_end = 108 - 37
        mean, scale = residual[:train_end].mean(axis=0), residual[:train_end].std(axis=0)
        z = (residual - mean) / np.maximum(scale, 1e-8)
        pca = PCA(n_components=4).fit(z[:train_end])
        cleaned = z - pca.inverse_transform(pca.transform(z))
        s = graph(incidence, months, 107, n)
        _, u = np.linalg.eigh(np.eye(n) - s)
        bases = {"node": np.eye(n), "graph": u}
        # A full covariance basis is available by direct symmetric decomposition at any sample size.
        _, bases["pca"] = np.linalg.eigh(np.cov(z[:train_end], rowvar=False))
        for seed in range(10):
            p = permutation_within_activity(monthly[:108, :n].sum(axis=0), seed + 19000)
            _, bases[f"null_{seed}"] = np.linalg.eigh(np.eye(n) - s[np.ix_(p, p)])
        for factor_removed, state in [(False, z), (True, cleaned)]:
            for horizon in [1, 3]:
                left = state[train_end:-horizon]
                right = state[train_end + horizon :]
                cross = right.T @ left / len(left)
                for name, basis in bases.items():
                    transformed = basis.T @ cross @ basis
                    rows.append(
                        dict(
                            source=source,
                            n=n,
                            factor_removed=factor_removed,
                            horizon=horizon,
                            basis=name,
                            diagonal_energy_fraction=float(
                                np.sum(np.diag(transformed) ** 2) / np.sum(cross**2)
                            ),
                        )
                    )
    pd.DataFrame(rows).to_csv(OUT / "basis_diagnostic.csv", index=False)


if __name__ == "__main__":
    main()
