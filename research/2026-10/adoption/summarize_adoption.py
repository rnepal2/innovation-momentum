"""Reviewed metrics, dependence sensitivities and figures for the held-out pilot."""

import hashlib
import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adoption_data import DATA, METHODS, ONTOLOGY, OUT, load, save
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

STAGES = ["temporal", "static", "exposure", "breadth", "undirected", "directed", "reverse"]
LABELS = [
    "History + context",
    "Static relationships",
    "Exposure counts",
    "Exposure breadth",
    "Undirected paths",
    "Directed paths",
    "Reversed paths",
]
PAIRS = [
    ("static", "exposure"),
    ("exposure", "breadth"),
    ("undirected", "directed"),
    ("reverse", "directed"),
]


def weighted_ap(y, p, w):
    order = np.argsort(-p, kind="stable")
    y, p, w = y[order], p[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y * w)[ends]
    denominator = np.cumsum(w)[ends]
    if not tp[-1]:
        return np.nan
    return np.sum(np.diff(np.r_[0, tp]) / tp[-1] * tp / np.maximum(denominator, 1e-12))


def summarize():
    results = pd.read_csv(OUT / "evaluation.csv", keep_default_na=False)
    for field in ["ap", "brier", "log_loss", "true_alert_median_lead_months"]:
        if field in results:
            results[field] = pd.to_numeric(results[field], errors="coerce")
    pred = pd.read_csv(OUT / "predictions.csv")
    rows = []
    for (origin, risk), part in pred.assign(risk=np.where(pred.prior > 0, "weak", "zero")).groupby(
        ["origin", "risk"]
    ):
        frames = {
            stage: d.sort_values(["method_index", "app_index"])
            for stage, d in part.groupby("stage")
        }
        common = frames["static"]
        y = common.event.to_numpy()
        ii, jj = common.method_index.to_numpy(), common.app_index.to_numpy()
        for baseline, stage in PAIRS:
            p, q = frames[baseline].probability.to_numpy(), frames[stage].probability.to_numpy()
            rng = np.random.default_rng(19)
            effects = []
            for _ in range(500):
                wm = rng.multinomial(len(METHODS), np.full(len(METHODS), 1 / len(METHODS)))
                wa = rng.multinomial(
                    len(ONTOLOGY["medical"]),
                    np.full(len(ONTOLOGY["medical"]), 1 / len(ONTOLOGY["medical"])),
                )
                w = wm[ii] * wa[jj]
                effects.append(100 * (weighted_ap(y, q, w) - weighted_ap(y, p, w)))
            low, high = np.nanquantile(effects, [0.025, 0.975]) if y.sum() else (np.nan, np.nan)

            def ap(prob):
                return average_precision_score(y, prob) if y.sum() else np.nan

            def top(prob):
                return y[np.argsort(-prob, kind="stable")[:20]].mean()

            rows.append(
                dict(
                    origin=origin,
                    risk=risk,
                    baseline=baseline,
                    stage=stage,
                    n=len(y),
                    events=int(y.sum()),
                    delta_ap_pp=100 * (ap(q) - ap(p)),
                    resampling_low_pp=low,
                    resampling_high_pp=high,
                    delta_p20=top(q) - top(p),
                    reviewed20=min(20, len(y)),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "comparisons.csv", index=False)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    primary = results[(results.risk == "weak") & (results.target == "event")]
    for ax, (origin, group) in zip(axes, primary.groupby("origin")):
        actual = group.set_index("stage").reindex(STAGES)
        ax.barh(
            range(len(STAGES)),
            actual.ap,
            color=["#767b82", "#767b82", "#007f73", "#007f73", "#b47a00", "#ab3a63", "#9a9fa7"],
        )
        ax.axvline(
            actual.prevalence.iloc[0], color="#111111", linestyle=":", label="Event prevalence"
        )
        null = group[group.stage.str.startswith("null_")].ap
        ax.axvspan(
            null.min(), null.max(), color="#aaaaaa", alpha=0.25, label="Community-null AP range"
        )
        ax.set_yticks(range(len(STAGES)), LABELS)
        ax.invert_yaxis()
        ax.set_xlabel("Average precision")
        year = int(origin[:4])
        ax.set_title(
            f"{year + 1}-{year + 2}: {int(actual.events.iloc[0])}/{int(actual.n.iloc[0])} events"
        )
        ax.set_xlim(0, 1)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Sustained asserted method use by new authors in medical imaging\nOne to ten prior joint-use papers; configurations fixed on robotics",
        fontsize=12,
    )
    fig.savefig(OUT / "adoption_ap.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6.3), layout="constrained")
    comparisons = pd.DataFrame(rows)
    offsets, tick_labels = [], []
    offset = 0
    part = comparisons[comparisons.risk == "weak"]
    for (baseline, stage), group in part.groupby(["baseline", "stage"], sort=False):
        for _, r in group.iterrows():
            ax.hlines(offset, r.resampling_low_pp, r.resampling_high_pp, color="#007f73")
            ax.plot(r.delta_ap_pp, offset, "o", color="#007f73")
            year = int(str(r.origin)[:4])
            offsets.append(offset)
            tick_labels.append(f"{year + 1}-{year + 2}   {stage} vs {baseline}")
            offset += 1
        offset += 0.6
    ax.axvline(0, color="#777777", lw=1)
    ax.set_yticks(offsets, tick_labels)
    ax.set_xlabel("AP increment (percentage points)")
    ax.invert_yaxis()
    fig.suptitle(
        "Incremental performance: 1-10 prior joint-use papers\n95% conditional identity-resampling ranges; 500 draws, models held fixed",
        fontsize=12,
    )
    fig.savefig(OUT / "adoption_increments.png")
    plt.close(fig)
    matching = []
    for year in [2019, 2021, 2023]:
        snap = json.loads((DATA / f"medical_{year}-12-31_snapshot.json").read_text())
        data = snap["rows"]
        eligible = [r for r in data if r["prior"] > 0 and r["exposure"][0] > 0]
        matched = []
        used = set()
        for mi in range(len(METHODS)):
            subset = [r for r in eligible if r["method_index"] == mi]
            for high in [r for r in subset if np.expm1(r["breadth"][0]) >= 1.99]:
                controls = [
                    r
                    for r in subset
                    if 0.99 <= np.expm1(r["breadth"][0]) < 1.99
                    and (r["method"], r["application"]) not in used
                    and int(np.log2(r["prior"])) == int(np.log2(high["prior"]))
                    and int(np.log2(max(np.expm1(r["exposure"][0]), 1)))
                    == int(np.log2(max(np.expm1(high["exposure"][0]), 1)))
                ]
                if not controls:
                    continue
                x = np.array([r["base"][:6] + r["base"][8:11] + r["exposure"][:2] for r in subset])
                scale = np.maximum(x.std(axis=0), 1)

                def cov(r):
                    return np.array(r["base"][:6] + r["base"][8:11] + r["exposure"][:2]) / scale

                distances = [np.linalg.norm(cov(high) - cov(r)) for r in controls]
                best = int(np.argmin(distances))
                if distances[best] <= 2:
                    low = controls[best]
                    used.add((low["method"], low["application"]))
                    matched.append((high, low))
        high_x = np.array([a["base"][:6] + a["base"][8:11] + a["exposure"][:2] for a, b in matched])
        low_x = np.array([b["base"][:6] + b["base"][8:11] + b["exposure"][:2] for a, b in matched])
        max_smd = (
            float(
                np.max(
                    np.abs(high_x.mean(axis=0) - low_x.mean(axis=0))
                    / np.maximum(np.sqrt((high_x.var(axis=0) + low_x.var(axis=0)) / 2), 1e-12)
                )
            )
            if len(matched) >= 2
            else None
        )
        matching.append(
            dict(
                origin=year,
                eligible=len(eligible),
                single_community_candidates=sum(
                    bool(0.99 <= np.expm1(r["breadth"][0]) < 1.99) for r in eligible
                ),
                multi_community_candidates=sum(
                    bool(np.expm1(r["breadth"][0]) >= 1.99) for r in eligible
                ),
                nonzero_directed_path_candidates=sum(
                    any(value > 0 for value in r["directed"]) for r in data if r["prior"] > 0
                ),
                matches=len(matched),
                max_abs_standardized_difference=max_smd,
                balance_gate_passed=bool(
                    len(matched) >= 20 and max_smd is not None and max_smd <= 0.2
                ),
                high_rate=np.mean([a["label"]["event"] for a, b in matched]) if matched else None,
                low_rate=np.mean([b["label"]["event"] for a, b in matched]) if matched else None,
                conclusion="Descriptive only; no causal identification. Few matches cannot establish a breadth effect.",
            )
        )
    save(OUT / "matching.json", matching)


def validate():
    result = pd.read_csv(OUT / "evaluation.csv", keep_default_na=False)
    pred = pd.read_csv(OUT / "predictions.csv")
    checks = 0
    for (origin, stage), g in pred.groupby(["origin", "stage"]):
        for risk in ["weak", "zero"]:
            rows = g[g.prior > 0] if risk == "weak" else g[g.prior == 0]
            if not len(rows):
                continue
            reference = result[
                (result.origin == origin)
                & (result.stage == stage)
                & (result.risk == risk)
                & (result.target == "event")
            ].iloc[0]
            if rows.event.sum():
                assert np.isclose(
                    average_precision_score(rows.event, rows.probability), float(reference.ap)
                )
            assert np.isclose(
                brier_score_loss(rows.event, rows.probability), float(reference.brier)
            )
            assert rows.event.sum() == reference.events and len(rows) == reference.n
            top = rows.event.to_numpy()[
                np.argsort(-rows.probability.to_numpy(), kind="stable")[:20]
            ]
            assert np.isclose(top.mean(), float(reference.p20))
            assert np.isclose(
                log_loss(rows.event, rows.probability, labels=[0, 1]), float(reference.log_loss)
            )
            checks += 1
    records = load("medical")
    by_id = {p["id"]: p for p in records}
    cases = []
    feature_checks = 0
    for year in [2019, 2021, 2023]:
        snap = json.loads((DATA / f"medical_{year}-12-31_snapshot.json").read_text())
        assert snap["audit"]["past_papers"] == sum(
            p["day"] <= pd.Timestamp(snap["cutoff"]) for p in records
        )
        for r in snap["rows"]:
            assert 0 <= r["prior"] <= 10
            for group in [
                "base",
                "static",
                "exposure",
                "breadth",
                "undirected",
                "directed",
                "reverse",
            ]:
                assert np.isfinite(r[group]).all()
            feature_checks += 1
        for r in [x for x in snap["rows"] if x["label"]["event"]][:5]:
            cutoff = pd.Timestamp(snap["cutoff"])
            seeds = {
                a
                for p in records
                if p["day"] <= cutoff
                and r["method_index"] in p["methods"]
                and r["app_index"] in p["applications"]
                for a in p["authors"]
            }
            witness = [by_id[i] for i in r["label"]["witness"]]
            assert len(witness) == 3
            for p in witness:
                assert cutoff < p["day"] <= cutoff + pd.DateOffset(months=24)
                assert p["authors"] and not set(p["authors"]) & seeds and len(p["authors"]) <= 50
                assert r["method_index"] in p["methods"] and r["app_index"] in p["applications"]
            assert all(
                not set(a["authors"]) & set(b["authors"]) for a, b in combinations(witness, 2)
            )
            eligible_future = [
                p
                for p in records
                if cutoff < p["day"] <= cutoff + pd.DateOffset(months=24)
                and r["method_index"] in p["methods"]
                and r["app_index"] in p["applications"]
                and p["authors"]
                and len(p["authors"]) <= 50
                and not set(p["authors"]) & seeds
            ]
            bins = {
                next(i for i in range(4) if p["day"] <= cutoff + pd.DateOffset(months=6 * (i + 1)))
                for p in eligible_future
            }
            assert len(bins) >= 2
            cases.append(
                dict(
                    origin=year,
                    method=r["method"],
                    application=r["application"],
                    witness=[
                        dict(id=p["id"], title=p["title"], date=p["date"], authors=p["authors"])
                        for p in witness
                    ],
                )
            )
    prospective_path = OUT / "prospective_predictions.json"
    prospective = json.loads(prospective_path.read_text())
    assert (
        pd.Timestamp(prospective["target_start"]).date()
        > pd.Timestamp(prospective["frozen_at"]).date()
    )
    assert all(
        0 <= p["probability"] <= 1 and 0 <= p["prior"] <= 10 for p in prospective["predictions"]
    )
    assert (
        prospective["source_manifest_sha256"]
        == hashlib.sha256((OUT / "medical_acquisition.json").read_bytes()).hexdigest()
    )
    assert (
        prospective["seen_ids_sha256"]
        == hashlib.sha256(
            (DATA / "prospective_seen_ids.txt").read_text().rstrip("\n").encode()
        ).hexdigest()
    )
    save(OUT / "validated_cases.json", cases)
    save(
        OUT / "validation.json",
        dict(
            status="passed",
            metric_checks=checks,
            witness_checks=len(cases),
            feature_row_checks=feature_checks,
            prospective_ledger_sha256=hashlib.sha256(prospective_path.read_bytes()).hexdigest(),
            prospective_date_and_source_checks="passed",
            scope="All primary saved model metrics recomputed; sampled date/author witnesses rechecked. Text meaning remains agent-reviewed; metadata vintage and author identity are not independently established.",
        ),
    )


if __name__ == "__main__":
    summarize()
    validate()
