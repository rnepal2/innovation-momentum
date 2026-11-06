"""Recompute scores from CSV rows without the experiment's numerical helpers."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import chain
from pathlib import Path
from statistics import fmean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results/initial")
    args = parser.parse_args()
    root = args.out
    freeze = json.loads((root / "freeze.json").read_text())
    seeds = freeze["seeds"]
    assert len(set(chain.from_iterable(seeds.values()))) == sum(map(len, seeds.values()))
    groups = defaultdict(list)
    labels = {}
    with (root / "predictions.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = row["geometry"], row["mechanism"], row["model"]
            seed = int(row["seed"])
            actual, prediction = float(row["actual"]), float(row["predicted"])
            assert math.isfinite(actual) and math.isfinite(prediction)
            identity = key[:2] + (seed,)
            assert identity not in labels or actual == labels[identity]
            labels[identity] = actual
            groups[key].append((seed, (actual - prediction) ** 2, abs(actual - prediction)))
    scores = {}
    with (root / "scores.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = row["geometry"], row["mechanism"], row["model"]
            records = groups[key]
            assert sorted(r[0] for r in records) == seeds["test"]
            assert math.isclose(fmean(r[1] for r in records), float(row["mse"]), abs_tol=1e-12)
            assert math.isclose(fmean(r[2] for r in records), float(row["mae"]), abs_tol=1e-12)
            scores[key] = float(row["mse"])
    assert len(scores) == len(groups)
    with (root / "paired.csv").open() as stream:
        for row in csv.DictReader(stream):
            case = row["geometry"], row["mechanism"]
            candidate, reference = row["candidate"], row["reference"]
            if reference.endswith("_mean_loss"):
                prefix = reference.removesuffix("_mean_loss")
                reference_loss = fmean(scores[case + (f"{prefix}{j}_dynamics",)] for j in range(3))
            else:
                reference_loss = scores[case + (reference,)]
            delta = scores[case + (candidate,)] - reference_loss
            assert math.isclose(delta, float(row["delta_mse"]), abs_tol=1e-12)
    selections = json.loads((root / "selections.json").read_text())
    with (root / "scores.csv").open() as stream:
        for row in csv.DictReader(stream):
            choices = selections[f"{row['geometry']}/{row['mechanism']}"]
            best = min(
                (m for m in choices if not m.startswith("gft")),
                key=lambda m: (choices[m]["validation_mse"], m),
            )
            assert row["strongest_comparator"] == best
    receipt = {
        "validated_model_cases": len(scores),
        "validated_predictions": sum(map(len, groups.values())),
        "score_recalculation": "stdlib csv, statistics.fmean, and scalar arithmetic",
        "checks": [
            "seed separation",
            "finite predictions",
            "label consistency",
            "MSE",
            "MAE",
            "paired loss differences",
            "validation-only comparator selection",
        ],
        "predictions_sha256": hashlib.sha256((root / "predictions.csv").read_bytes()).hexdigest(),
    }
    (root / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
