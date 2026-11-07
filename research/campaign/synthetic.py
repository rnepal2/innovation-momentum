"""Known-process validation for representation comparisons."""

import numpy as np
import pandas as pd
from common import OUT, save_json
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def poly_features(x, s):
    return np.stack([x, x @ s.T, x @ (s @ s).T], axis=-1).reshape(-1, 3)


def run(seed, process, length):
    rng = np.random.default_rng(seed)
    n = 48
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if rng.random() < (0.3 if i // 12 == j // 12 else 0.025):
                w[i, j] = w[j, i] = 1
    w += np.eye(n) * 1e-6
    d = w.sum(axis=1)
    s = w / np.sqrt(d[:, None] * d[None, :])
    a = 0.65 * np.eye(n)
    if process == "graph":
        a = 0.25 * np.eye(n) + 0.65 * s
    elif process == "directed":
        a = 0.25 * np.eye(n) + 0.65 * np.roll(np.eye(n), 1, axis=1)
    load = rng.normal(size=(n, 2))
    state = np.zeros((length + 100, n))
    latent = np.zeros(2)
    for t in range(1, len(state)):
        latent = 0.8 * latent + rng.normal(size=2)
        shock = latent @ load.T if process == "shared_shock" else 0
        state[t] = a @ state[t - 1] + rng.normal(size=n) + shock
    state = state[100:]
    x, y = state[:-1], state[1:]
    tr, va = int(len(x) * 0.6), int(len(x) * 0.8)
    p = rng.permutation(n)
    null = s[np.ix_(p, p)]
    outputs = []
    for name in ["own", "factor", "graph", "shuffled_graph", "transition"]:

        def predict(stop, target, alpha):
            if name == "own":
                coef = (x[:stop] * y[:stop]).sum(axis=0) / ((x[:stop] ** 2).sum(axis=0) + alpha)
                return x[target] * coef
            if name == "factor":
                pca = PCA(n_components=4).fit(x[:stop])
                common = pca.transform(x)
                pred = np.empty_like(x[target])
                for j in range(n):
                    f = np.column_stack([x[:, j], common])
                    pred[:, j] = Ridge(alpha=alpha).fit(f[:stop], y[:stop, j]).predict(f[target])
                return pred
            if name in ("graph", "shuffled_graph"):
                ss = s if name == "graph" else null
                f = poly_features(x, ss).reshape(len(x), n, 3)
                fit = Ridge(alpha=alpha).fit(f[:stop].reshape(-1, 3), y[:stop].ravel())
                return fit.predict(f[target].reshape(-1, 3)).reshape(-1, n)
            return Ridge(alpha=alpha).fit(x[:stop], y[:stop]).predict(x[target])

        alphas = [0.1, 1, 10, 100]
        losses = [np.mean((predict(tr, slice(tr, va), a) - y[tr:va]) ** 2) for a in alphas]
        alpha = alphas[int(np.argmin(losses))]
        mse = np.mean((predict(va, slice(va, None), alpha) - y[va:]) ** 2)
        outputs.append(
            dict(seed=seed, process=process, length=length, model=name, mse=mse, alpha=alpha)
        )
    # Full orthogonal graph coordinates must leave isotropic ridge predictions unchanged.
    _, u = np.linalg.eigh(np.eye(n) - s)
    direct = Ridge(alpha=10).fit(x[:va], y[:va]).predict(x[va:])
    rotated = Ridge(alpha=10).fit(x[:va] @ u, y[:va] @ u).predict(x[va:] @ u) @ u.T
    assert np.allclose(direct, rotated, atol=1e-9)
    return outputs


def main():
    rows = []
    for length in [96, 240]:
        for process in ["independent", "shared_shock", "graph", "directed"]:
            for seed in range(20):
                rows.extend(run(seed, process, length))
            print(f"Synthetic {process}, T={length} complete", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic.csv", index=False)
    df.groupby(["process", "length", "model"]).mse.agg(["mean", "std"]).to_csv(
        OUT / "synthetic_summary.csv"
    )
    save_json(
        OUT / "synthetic_validation.json",
        {
            "orthogonal_ridge_invariance": True,
            "seeds": 20,
            "processes": 4,
            "lengths": [96, 240],
            "nodes": 48,
        },
    )


if __name__ == "__main__":
    main()
