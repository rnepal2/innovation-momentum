"""Independent arithmetic and raw-text spot checks for the reported experiments."""

import hashlib
import itertools
import json
import re

import numpy as np
import pandas as pd
from common import DATA, OUT, ROOT, load_corpus, save_json
from sklearn.metrics import average_precision_score, brier_score_loss


def main():
    checks, cases = [], []
    for filename in ["manifest.json", "arxiv_manifest.json"]:
        manifest = json.loads((DATA / filename).read_text())
        assert (
            hashlib.sha256((ROOT / manifest["path"]).read_bytes()).hexdigest() == manifest["sha256"]
        )
        checks.append(dict(check="frozen_input_sha256", manifest=filename))
    for source in ["openalex", "arxiv"]:
        records = load_corpus(source)
        for suffix in ["", "_rolling"]:
            frame = pd.read_csv(OUT / f"{source}_h2{suffix}.csv", keep_default_na=False)
            probabilities = np.load(DATA / f"{source}_h2{suffix}_predictions.npz")
            for _, row in frame[
                (frame.target == "independent") & frame.feature.isin(["base", "spectral"])
            ].iterrows():
                prefix = f"{row.year}_{row.risk}_"
                y, pred = probabilities[prefix + "y"], probabilities[prefix + row.feature]
                assert len(y) == row.n and y.sum() == row.events
                if y.sum():
                    assert np.isclose(average_precision_score(y, pred), float(row.ap))
                assert np.isclose(brier_score_loss(y, pred), row.brier)
                assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
                checks.append(
                    dict(
                        check="metric_recomputation",
                        source=source,
                        vocabulary=suffix or "fixed",
                        year=row.year,
                        risk=row.risk,
                        model=row.feature,
                    )
                )
            ranked = pd.read_csv(OUT / f"{source}_h2{suffix}_top_pairs.csv")
            samples = []
            for _, group in ranked.groupby(["year", "risk", "model"]):
                samples.extend(group.head(1).to_dict("records"))
                negative = group[group.event == 0]
                if len(negative):
                    samples.extend(negative.head(1).to_dict("records"))
            needed = set(
                p for sample in samples for p in [sample["concept_a"], sample["concept_b"]]
            )
            index = {phrase: [] for phrase in needed}
            # Re-tokenize original abstracts independently of sklearn's incidence matrix.
            for i, paper in enumerate(records):
                words = re.findall(r"\b[a-z][a-z]+\b", paper["abstract"].lower())
                observed = {a + " " + b for a, b in zip(words, words[1:])}
                for phrase in needed & observed:
                    index[phrase].append(i)
            for sample in samples:
                cutoff = (sample["year"] - 2010 + 1) * 12 - 1
                overlap = set(index[sample["concept_a"]]) & set(index[sample["concept_b"]])
                future = [
                    records[i]
                    for i in sorted(overlap)
                    if cutoff < records[i]["month"] <= cutoff + 24
                ]
                assert len(future) == sample["future_papers"]
                bins = {(p["month"] - cutoff - 1) // 6 for p in future}
                witness = None
                if len(future) >= 3 and len(bins) >= 2:
                    for triple in itertools.combinations(future, 3):
                        a, b, c = [set(p["authors"]) for p in triple]
                        if a and b and c and not (a & b or a & c or b & c):
                            witness = triple
                            break
                assert int(witness is not None) == sample["event"]
                cases.append(
                    dict(
                        source=source,
                        vocabulary=suffix or "fixed",
                        **sample,
                        witness=[
                            dict(id=p["id"], title=p["title"], date=p["date"], authors=p["authors"])
                            for p in witness
                        ]
                        if witness
                        else [],
                    )
                )
    # Numerical checks on the forecast table: unique months, horizons and finite losses.
    ablation = pd.read_csv(OUT / "h2_ablation.csv")
    primary = pd.read_csv(OUT / "openalex_h2.csv", keep_default_na=False)
    for model, feature in [("local_dynamic", "base"), ("full_spectral", "spectral")]:
        for year in [2019, 2021, 2023]:
            a = ablation[(ablation.model == model) & (ablation.origin == year)].iloc[0]
            b = primary[
                (primary.target == "independent")
                & (primary.risk == "weak")
                & (primary.feature == feature)
                & (primary.year == year)
            ].iloc[0]
            assert a.n == b.n and a.events == b.events
            assert np.allclose([a.ap, a.brier, a.p20, a.p50], [float(b.ap), b.brier, b.p20, b.p50])
            checks.append(dict(check="ablation_reproduces_primary", model=model, origin=year))
    for source, suffix in [("openalex", ""), ("openalex", "_no_january1"), ("arxiv", "")]:
        df = pd.read_csv(OUT / f"{source}_h1{suffix}.csv", keep_default_na=False)
        assert ((df.target - df.origin) == df.horizon).all()
        assert (df.year >= 2019).all() and (df.target <= 191).all()
        assert np.isfinite(df[["mse", "deviance"]].to_numpy()).all()
        assert (df.deviance >= 0).all()
        assert not df.duplicated(["n", "horizon", "model", "seed", "target"]).any()
        checks.append(
            dict(check="forecast_table_integrity", source=source, sensitivity=suffix, rows=len(df))
        )
    oa, ar = load_corpus("openalex"), load_corpus("arxiv")

    def title(p):
        return re.sub(r"[^a-z0-9]", "", p["title"].lower())

    titles = {title(p) for p in oa if len(title(p)) >= 40}
    overlap = sum(title(p) in titles for p in ar)
    save_json(
        OUT / "source_overlap.json",
        dict(
            arxiv_papers=len(ar),
            openalex_papers=len(oa),
            exact_normalized_title_matches=overlap,
            interpretation="Incomplete exact-title matching diagnostic with possible ambiguous matches; source samples overlap.",
        ),
    )
    save_json(
        OUT / "validation.json",
        dict(
            status="passed",
            checks=checks,
            raw_text_case_checks=len(cases),
            scope="All saved baseline/spectral AP and Brier values; sampled ranked-pair counts and author witnesses; forecast row integrity. This does not validate metadata vintages or scientific meaning.",
        ),
    )
    save_json(OUT / "validated_cases.json", cases)
    print(
        f"Validated {len(checks)} metric/table checks and {len(cases)} raw-text cases", flush=True
    )


if __name__ == "__main__":
    main()
