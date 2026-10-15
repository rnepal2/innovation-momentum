"""Exploratory attribution of the positive fixed-vocabulary OpenAlex result."""

import numpy as np
import pandas as pd
from common import OUT, prepare, save_json
from h2 import SPECS, estimator, metrics, snapshot
from sklearn.metrics import average_precision_score


def select(snap, model):
    names = snap["names"]
    dynamic = [
        i
        for i, name in enumerate(names)
        if any(token in name for token in ["neighbor_growth", "degree_change", "neighbor_change"])
    ]
    network = [
        i
        for i, name in enumerate(names)
        if any(token in name for token in ["neighbor", "degree", "adamic", "preferential"])
    ]
    base = snap["base"]
    if model == "popularity_context":
        return np.delete(base, network, axis=1)
    if model == "static_network":
        return np.delete(base, dynamic, axis=1)
    if model == "local_dynamic":
        return base
    extra = snap["spectral"]
    if model == "static_spectral":
        extra = extra[:, :6]
    elif model == "change_spectral":
        extra = extra[:, 6:]
    return np.column_stack([base, extra])


def main():
    data = prepare("openalex")
    snaps = [snapshot(*data, year, nulls=0) for year in [2015, 2017, 2019, 2021, 2023]]
    rows, selection = [], []
    train, valid = snaps[:2]
    mt, mv = train["risk"] == "weak", valid["risk"] == "weak"
    for name in [
        "popularity_context",
        "static_network",
        "local_dynamic",
        "static_spectral",
        "change_spectral",
        "full_spectral",
    ]:
        scores = []
        for spec in SPECS:
            model = estimator(spec).fit(select(train, name)[mt], train["independent"][mt])
            pred = model.predict_proba(select(valid, name)[mv])[:, 1]
            scores.append(average_precision_score(valid["independent"][mv], pred))
        chosen = SPECS[int(np.argmax(scores))]
        selection.append(dict(model=name, spec=chosen, validation_ap=max(scores)))
        for index in range(2, len(snaps)):
            test = snaps[index]
            mask = test["risk"] == "weak"
            x = np.concatenate([select(s, name)[s["risk"] == "weak"] for s in snaps[:index]])
            y = np.concatenate([s["independent"][s["risk"] == "weak"] for s in snaps[:index]])
            fit = estimator(chosen).fit(x, y)
            pred = fit.predict_proba(select(test, name)[mask])[:, 1]
            rows.append(
                dict(model=name, origin=test["year"], **metrics(test["independent"][mask], pred))
            )
        print(f"Ablation {name} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "h2_ablation.csv", index=False)
    save_json(OUT / "h2_ablation_selection.json", selection)


if __name__ == "__main__":
    main()
