"""Chronological development, held-out population tests and a frozen prediction ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from adoption_data import DATA, HERE, OUT, load, save
from adoption_features import snapshot
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPECS = [("logistic", c) for c in [0.05, 0.2, 1.0]] + [("boost", 2)]
STAGES = ["temporal", "static", "exposure", "breadth", "undirected", "directed", "reverse"] + [
    f"null_{i}" for i in range(10)
]


def classifier(spec):
    if spec[0] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=spec[1], max_iter=1000))
    return HistGradientBoostingClassifier(
        max_iter=60,
        max_depth=2,
        max_leaf_nodes=4,
        min_samples_leaf=10,
        learning_rate=0.05,
        l2_regularization=10,
        early_stopping=False,
        random_state=42,
    )


def features(rows, stage):
    output = []
    for r in rows:
        x = r["base"].copy()
        if stage != "temporal":
            x += r["static"]
        if stage not in ["temporal", "static"]:
            x += r["exposure"]
        if stage not in ["temporal", "static", "exposure"]:
            x += r["null"][int(stage.split("_")[1])] if stage.startswith("null_") else r["breadth"]
        if stage in ["undirected", "directed", "reverse"]:
            x += r["undirected"]
        if stage in ["directed", "reverse"]:
            x += r[stage]
        output.append(x)
    return np.array(output, float)


def metrics(y, p):
    return dict(
        n=len(y),
        events=int(y.sum()),
        prevalence=float(y.mean()),
        ap=float(average_precision_score(y, p)) if y.sum() else None,
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        roc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        reviewed20=min(len(y), 20),
        p20=float(y[np.argsort(-p, kind="stable")[:20]].mean()),
    )


def development():
    frozen = OUT / "development_selection.json"
    if frozen.exists():
        settings = json.loads(frozen.read_text())
        if (
            settings["ontology_sha256"]
            != hashlib.sha256((HERE / "ontology.json").read_bytes()).hexdigest()
        ):
            raise RuntimeError(
                "Frozen development ontology differs; use a separate study directory"
            )
        print("Existing development settings preserved.")
        return
    records = load("robotics")
    snaps = [snapshot(records, "robotics", f"{year}-12-31") for year in [2015, 2017]]
    train, valid = [s["rows"] for s in snaps]
    yt, yv = [np.array([r["label"]["event"] for r in rows]) for rows in [train, valid]]
    weak_t = np.array([r["prior"] > 0 for r in train])
    weak_v = np.array([r["prior"] > 0 for r in valid])
    power = int(yt[weak_t].sum()) >= 10 and int(yv[weak_v].sum()) >= 10 and len(np.unique(yt)) == 2
    selected = {}
    for stage in STAGES:
        if not power:
            selected[stage] = dict(
                spec=["logistic", 0.2],
                validation_ap=None,
                scores=[],
                reason="frozen low-power fallback",
            )
            continue
        scores = []
        specs = SPECS if power else [("logistic", 0.2)]
        for spec in specs:
            if len(np.unique(yt)) < 2:
                raise RuntimeError(
                    "No two-class development training set; do not inspect held-out outcomes"
                )
            fit = classifier(spec).fit(features(train, stage), yt)
            p = fit.predict_proba(features(valid, stage))[:, 1]
            scores.append(
                float(average_precision_score(yv[weak_v], p[weak_v])) if yv[weak_v].sum() else 0.0
            )
        selected[stage] = dict(
            spec=specs[int(np.argmax(scores))], validation_ap=max(scores), scores=scores
        )
    save(
        OUT / "development_selection.json",
        dict(
            selected=selected,
            primary_training_events=int(yt[weak_t].sum()),
            primary_validation_events=int(yv[weak_v].sum()),
            sufficient_development_events=power,
            rows=[len(train), len(valid)],
            ontology_sha256=hashlib.sha256((HERE / "ontology.json").read_bytes()).hexdigest(),
            frozen_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    print(
        f"Development frozen: {len(train)}/{len(valid)} rows; primary events {yt[weak_t].sum()}/{yv[weak_v].sum()}",
        flush=True,
    )


def evaluate():
    settings = json.loads((OUT / "development_selection.json").read_text())
    assert (
        settings["ontology_sha256"]
        == hashlib.sha256((HERE / "ontology.json").read_bytes()).hexdigest()
    )
    records = {domain: load(domain) for domain in ["robotics", "medical"]}
    snapshots = {
        domain: [
            snapshot(records[domain], domain, f"{year}-12-31")
            for year in [2015, 2017, 2019, 2021, 2023]
        ]
        for domain in records
    }
    result, predictions = [], []
    for index in range(2, 5):
        current = snapshots["medical"][index]
        for domain in records:
            for historical in snapshots[domain][:index]:
                assert pd.Timestamp(historical["cutoff"]) + pd.DateOffset(
                    months=24
                ) <= pd.Timestamp(current["cutoff"])
        train = [r for domain in records for s in snapshots[domain][:index] for r in s["rows"]]
        yt = np.array([r["label"]["event"] for r in train])
        test = current["rows"]
        for stage in STAGES:
            fit = classifier(settings["selected"][stage]["spec"]).fit(features(train, stage), yt)
            p = fit.predict_proba(features(test, stage))[:, 1]
            for risk, mask in [
                ("weak", np.array([r["prior"] > 0 for r in test])),
                ("zero", np.array([r["prior"] == 0 for r in test])),
            ]:
                if not mask.sum():
                    continue
                for target in ["event", "one_group", "collaborator_excluded"]:
                    y = np.array(
                        [
                            r["strict"]["event"]
                            if target == "collaborator_excluded"
                            else r["label"][target]
                            for r in test
                        ]
                    )
                    alert_indices = np.flatnonzero(mask)[np.argsort(-p[mask], kind="stable")[:20]]
                    leads = [
                        (
                            pd.Timestamp(test[i]["label"]["event_day"])
                            - pd.Timestamp(current["cutoff"])
                        ).days
                        / 30.4375
                        for i in alert_indices
                        if test[i]["label"]["event_day"]
                    ]
                    result.append(
                        dict(
                            origin=current["cutoff"],
                            stage=stage,
                            risk=risk,
                            target=target,
                            **metrics(y[mask], p[mask]),
                            false_positive_rate=float(
                                (1 - y[alert_indices]).sum() / max((1 - y[mask]).sum(), 1)
                            ),
                            true_alert_median_lead_months=float(np.median(leads))
                            if leads and target == "event"
                            else None,
                        )
                    )
            for r, prob in zip(test, p):
                predictions.append(
                    dict(
                        origin=current["cutoff"],
                        stage=stage,
                        method=r["method"],
                        application=r["application"],
                        method_index=r["method_index"],
                        app_index=r["app_index"],
                        prior=r["prior"],
                        event=r["label"]["event"],
                        one_group=r["label"]["one_group"],
                        strict_event=r["strict"]["event"],
                        probability=prob,
                    )
                )
        for risk in ["weak", "zero"]:
            chosen = [r for r in test if (r["prior"] > 0) == (risk == "weak")]
            if chosen:
                y = np.array([r["label"]["event"] for r in chosen])
                score = np.array([r["burst_score"] for r in chosen])
                top = np.argsort(-score, kind="stable")[:20]
                result.append(
                    dict(
                        origin=current["cutoff"],
                        stage="burst",
                        risk=risk,
                        target="event",
                        n=len(y),
                        events=int(y.sum()),
                        prevalence=float(y.mean()),
                        ap=float(average_precision_score(y, score)) if y.sum() else None,
                        p20=float(y[top].mean()),
                        reviewed20=min(20, len(y)),
                        brier=None,
                        log_loss=None,
                        roc=None,
                        false_positive_rate=float((1 - y[top]).sum() / max((1 - y).sum(), 1)),
                    )
                )
        print(f"Medical evaluation {current['cutoff']}: {len(test)} pairs complete", flush=True)
    pd.DataFrame(result).to_csv(OUT / "evaluation.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv", index=False)
    if (OUT / "prospective_predictions.json").exists():
        print("Existing prospective ledger preserved; a new freeze requires a separate study file.")
        return
    # This ledger uses today's actually observed records, with an entirely later date window.
    prospective_cutoff = pd.Timestamp(datetime.now(timezone.utc).date())
    current = snapshot(records["medical"], "medical", prospective_cutoff, include_outcomes=False)
    train = [r for domain in records for s in snapshots[domain] for r in s["rows"]]
    yt = np.array([r["label"]["event"] for r in train])
    prospective = []
    for stage in STAGES[:7]:
        fit = classifier(settings["selected"][stage]["spec"]).fit(features(train, stage), yt)
        p = fit.predict_proba(features(current["rows"], stage))[:, 1]
        for r, prob in zip(current["rows"], p):
            prospective.append(
                dict(
                    stage=stage,
                    method=r["method"],
                    application=r["application"],
                    prior=r["prior"],
                    probability=float(prob),
                    seed_authors=r["seed_authors"],
                )
            )
    save(
        OUT / "prospective_predictions.json",
        dict(
            frozen_at=datetime.now(timezone.utc).isoformat(),
            feature_publication_cutoff="2026-10-14",
            model_feature_origin=str(prospective_cutoff.date()),
            target_start=str((prospective_cutoff + pd.Timedelta(days=1)).date()),
            target_end=str((prospective_cutoff + pd.DateOffset(months=24)).date()),
            status="local prediction ledger, not externally registered; no completed outcomes",
            source_manifest_sha256=hashlib.sha256(
                (OUT / "medical_acquisition.json").read_bytes()
            ).hexdigest(),
            seen_ids_sha256=hashlib.sha256(
                "\n".join(p["id"] for p in records["medical"]).encode()
            ).hexdigest(),
            predictions=prospective,
        ),
    )
    (DATA / "prospective_seen_ids.txt").write_text(
        "\n".join(p["id"] for p in records["medical"]) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    if args.development_only:
        development()
    else:
        evaluate()
