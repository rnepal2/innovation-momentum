"""Chronological, natural-prevalence prediction of sustained concept-pair adoption."""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from common import DATA, OUT, graph, permutation_within_activity, prepare, save_json
from scipy.linalg import eigh
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def independent_three(author_sets):
    """Exact existence test: three nonempty, pairwise author-disjoint paper sets."""
    groups = [set(a) for a in author_sets if a]
    for i in range(len(groups) - 2):
        compatible = [j for j in range(i + 1, len(groups)) if groups[i].isdisjoint(groups[j])]
        for j, k in combinations(compatible, 2):
            if groups[j].isdisjoint(groups[k]):
                return True
    return False


def heat_features(s, ii, jj):
    values, vectors = eigh(np.eye(len(s)) - s, check_finite=False)
    output = []
    for scale in [0.5, 2.0, 8.0]:
        kernel = (vectors * np.exp(-scale * values)) @ vectors.T
        output.extend(
            [kernel[ii, jj], kernel.diagonal()[ii] + kernel.diagonal()[jj] - 2 * kernel[ii, jj]]
        )
    return np.column_stack(output)


def snapshot(records, months, incidence, monthly, vocab, semantic, year, nulls=10):
    cutoff = (year - 2010 + 1) * 12 - 1
    n = len(vocab)
    past = incidence[months <= cutoff]
    co = (past.T @ past).toarray().astype(float)
    count = np.diag(co).copy()
    ii, jj = np.triu_indices(n, 1)
    nonoverlap = np.array(
        [not set(vocab[i].split()) & set(vocab[j].split()) for i, j in zip(ii, jj)]
    )
    keep = (co[ii, jj] <= 3) & (count[ii] >= 20) & (count[jj] >= 20) & nonoverlap
    ii, jj = ii[keep], jj[keep]
    prior = co[ii, jj]
    recent = monthly[cutoff - 11 : cutoff + 1].sum(axis=0)
    older = monthly[cutoff - 23 : cutoff - 11].sum(axis=0)
    growth = np.log1p(recent) - np.log1p(older)
    co_recent = (
        incidence[(months > cutoff - 12) & (months <= cutoff)].T
        @ incidence[(months > cutoff - 12) & (months <= cutoff)]
    ).toarray()
    adjacency = (co > 0).astype(float)
    np.fill_diagonal(adjacency, 0)
    degree = adjacency.sum(axis=1)
    cn = adjacency @ adjacency
    aa = (adjacency / np.maximum(np.log1p(degree), 1)) @ adjacency
    weighted = co / np.sqrt(np.maximum(count[:, None] * count[None, :], 1))
    np.fill_diagonal(weighted, 0)
    wcn = weighted @ weighted
    old = incidence[months <= cutoff - 24]
    old_co = (old.T @ old).toarray().astype(float)
    old_count = np.diag(old_co).copy()
    old_adjacency = (old_co > 0).astype(float)
    np.fill_diagonal(old_adjacency, 0)
    old_degree = old_adjacency.sum(axis=1)
    old_cn = old_adjacency @ old_adjacency
    old_weighted = old_co / np.sqrt(np.maximum(old_count[:, None] * old_count[None, :], 1))
    np.fill_diagonal(old_weighted, 0)
    old_wcn = old_weighted @ old_weighted
    joint_first = np.full((n, n), cutoff + 1, dtype=int)
    for year_start in range(0, cutoff + 1, 12):
        x = incidence[(months >= year_start) & (months < year_start + 12)]
        present = (x.T @ x).toarray() > 0
        joint_first[present] = np.minimum(joint_first[present], year_start)
    s = graph(incidence, months, cutoff, n)
    old_s = graph(incidence, months, cutoff - 24, n)
    heat = heat_features(s, ii, jj)
    spectral = np.column_stack([heat, heat - heat_features(old_s, ii, jj)])
    first = np.array([np.flatnonzero(monthly[:, j] > 0)[0] for j in range(n)])
    base, names = [], []
    for name, x in [
        ("support", np.log1p(count)),
        ("recent", np.log1p(recent)),
        ("growth", growth),
        ("age", (cutoff - first) / 12),
        ("degree", np.log1p(degree)),
        ("neighbor_growth", s @ growth),
        ("degree_change", np.log1p(degree) - np.log1p(old_degree)),
    ]:
        base.extend([np.minimum(x[ii], x[jj]), np.maximum(x[ii], x[jj])])
        names.extend([name + "_min", name + "_max"])
    base.extend(
        [
            prior,
            co_recent[ii, jj],
            semantic[ii, jj],
            np.log1p(cn[ii, jj]),
            aa[ii, jj],
            wcn[ii, jj],
            np.log1p(degree[ii] * degree[jj]),
            np.where(prior > 0, (cutoff - joint_first[ii, jj]) // 12 + 1, 0),
            np.log1p(cn[ii, jj]) - np.log1p(old_cn[ii, jj]),
            wcn[ii, jj] - old_wcn[ii, jj],
        ]
    )
    names.extend(
        [
            "prior_joint",
            "recent_joint",
            "context_cosine",
            "log_common_neighbors",
            "adamic_adar",
            "weighted_common_neighbors",
            "log_preferential_attachment",
            "joint_age_year_bin",
            "common_neighbor_change",
            "weighted_neighbor_change",
        ]
    )
    base = np.column_stack(base)
    assert np.isfinite(base).all() and np.isfinite(spectral).all()
    future_mask = (months > cutoff) & (months <= cutoff + 24)
    future = incidence[future_mask]
    future_records = [p for p, keep in zip(records, future_mask) if keep]
    fco = (future.T @ future).toarray()
    bins = np.zeros(len(ii), int)
    for b in range(4):
        x = incidence[(months > cutoff + 6 * b) & (months <= cutoff + 6 * (b + 1))]
        bins += (x.T @ x).toarray()[ii, jj] > 0
    repeated = (fco[ii, jj] >= 3) & (bins >= 2)
    independent = np.zeros(len(ii), bool)
    csc = future.tocsc()
    for index in np.flatnonzero(repeated):
        i, j = ii[index], jj[index]
        rows = np.intersect1d(
            csc.indices[csc.indptr[i] : csc.indptr[i + 1]],
            csc.indices[csc.indptr[j] : csc.indptr[j + 1]],
            assume_unique=True,
        )
        independent[index] = independent_three([future_records[r]["authors"] for r in rows])
    null_features = []
    for seed in range(nulls):
        p = permutation_within_activity(count, seed + 29000)
        null_heat = heat_features(s[np.ix_(p, p)], ii, jj)
        null_old = heat_features(old_s[np.ix_(p, p)], ii, jj)
        null_features.append(np.column_stack([null_heat, null_heat - null_old]))
    return dict(
        year=year,
        base=base,
        spectral=spectral,
        null=null_features,
        vocab=vocab,
        independent=independent.astype(int),
        repeated=repeated.astype(int),
        risk=np.where(prior == 0, "zero", "weak"),
        ii=ii,
        jj=jj,
        future_count=fco[ii, jj],
        names=names,
    )


SPECS = [("logistic", c) for c in [0.1, 1.0, 10.0]] + [("boost", d) for d in [3, 6]]


def estimator(spec):
    if spec[0] == "logistic":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=spec[1], max_iter=500, solver="lbfgs")
        )
    return HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=2 ** spec[1],
        max_depth=spec[1],
        min_samples_leaf=30,
        learning_rate=0.06,
        l2_regularization=10,
        early_stopping=False,
        random_state=7,
    )


def features(snap, feature, seed=None):
    if feature == "base":
        return snap["base"]
    extra = snap["spectral"] if seed is None else snap["null"][seed]
    return np.column_stack([snap["base"], extra])


def metrics(y, p):
    return dict(
        n=len(y),
        events=int(y.sum()),
        prevalence=float(y.mean()),
        ap=float(average_precision_score(y, p)) if y.sum() else None,
        roc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        brier=float(brier_score_loss(y, p)),
        p20=float(y[np.argsort(-p, kind="stable")[:20]].mean()),
        p50=float(y[np.argsort(-p, kind="stable")[:50]].mean()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["openalex", "arxiv"], default="openalex")
    parser.add_argument("--nulls", type=int, default=10)
    parser.add_argument("--rolling-vocabulary", action="store_true")
    args = parser.parse_args()
    data = prepare(args.source)
    records, months, incidence, monthly, vocab, semantic = data
    snaps = []
    for year in [2015, 2017, 2019, 2021, 2023]:
        current = (
            prepare(args.source, vocabulary_end=(year - 2010 + 1) * 12 - 1, records=records)
            if args.rolling_vocabulary
            else data
        )
        snaps.append(snapshot(*current, year, args.nulls))
    suffix = "_rolling" if args.rolling_vocabulary else ""
    print(f"{args.source}: H2 snapshots ready", flush=True)
    result, choices, predictions, full = [], [], [], {}
    for target in ["independent", "repeated"]:
        for risk in ["zero", "weak"]:
            train, valid = snaps[:2]
            mt, mv = train["risk"] == risk, valid["risk"] == risk
            if len(np.unique(train[target][mt])) < 2 or not valid[target][mv].sum():
                choices.append(
                    dict(target=target, risk=risk, status="insufficient_development_events")
                )
                continue
            selected = {}
            for feature in ["base", "spectral"] + [f"null_{s}" for s in range(args.nulls)]:
                seed = int(feature.split("_")[1]) if feature.startswith("null_") else None
                f = "spectral" if seed is not None else feature
                scores = []
                for spec in SPECS:
                    model = estimator(spec).fit(features(train, f, seed)[mt], train[target][mt])
                    pred = model.predict_proba(features(valid, f, seed)[mv])[:, 1]
                    scores.append(average_precision_score(valid[target][mv], pred))
                selected[feature] = SPECS[int(np.argmax(scores))]
                choices.append(
                    dict(
                        target=target,
                        risk=risk,
                        feature=feature,
                        spec=selected[feature],
                        validation_ap=max(scores),
                        scores=scores,
                    )
                )
            for index in range(2, len(snaps)):
                test = snaps[index]
                mask = test["risk"] == risk
                y = test[target][mask]
                for feature in ["base", "spectral", "null"]:
                    seeds = range(args.nulls) if feature == "null" else [None]
                    for seed in seeds:
                        f = "spectral" if feature == "null" else feature
                        x_train = np.concatenate(
                            [features(s, f, seed)[s["risk"] == risk] for s in snaps[:index]]
                        )
                        y_train = np.concatenate(
                            [s[target][s["risk"] == risk] for s in snaps[:index]]
                        )
                        spec_key = f"null_{seed}" if seed is not None else f
                        model = estimator(selected[spec_key]).fit(x_train, y_train)
                        pred = model.predict_proba(features(test, f, seed)[mask])[:, 1]
                        result.append(
                            dict(
                                source=args.source,
                                target=target,
                                risk=risk,
                                year=test["year"],
                                feature=feature,
                                seed=seed,
                                **metrics(y, pred),
                            )
                        )
                        if feature != "null" and target == "independent":
                            prefix = f"{test['year']}_{risk}"
                            full[prefix + "_" + feature] = pred
                            full[prefix + "_y"] = y
                            full[prefix + "_i"] = test["ii"][mask]
                            full[prefix + "_j"] = test["jj"][mask]
                            eligible = np.flatnonzero(mask)
                            for pos in np.argsort(-pred, kind="stable")[:50]:
                                row = eligible[pos]
                                predictions.append(
                                    dict(
                                        year=test["year"],
                                        risk=risk,
                                        model=feature,
                                        concept_a=test["vocab"][test["ii"][row]],
                                        concept_b=test["vocab"][test["jj"][row]],
                                        probability=pred[pos],
                                        event=int(y[pos]),
                                        future_papers=int(test["future_count"][row]),
                                    )
                                )
                print(f"{args.source} H2 {target}/{risk} origin {test['year']} done", flush=True)
    pd.DataFrame(result).to_csv(OUT / f"{args.source}_h2{suffix}.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / f"{args.source}_h2{suffix}_top_pairs.csv", index=False)
    save_json(OUT / f"{args.source}_h2{suffix}_selection.json", choices)
    save_json(OUT / f"{args.source}_h2{suffix}_features.json", snaps[0]["names"])
    np.savez_compressed(DATA / f"{args.source}_h2{suffix}_predictions.npz", **full)
    pd.DataFrame(
        [
            dict(
                year=s["year"],
                risk=risk,
                target=target,
                n=int((s["risk"] == risk).sum()),
                events=int(s[target][s["risk"] == risk].sum()),
            )
            for s in snaps
            for risk in ["zero", "weak"]
            for target in ["independent", "repeated"]
        ]
    ).to_csv(OUT / f"{args.source}_h2{suffix}_prevalence.csv", index=False)


if __name__ == "__main__":
    main()
