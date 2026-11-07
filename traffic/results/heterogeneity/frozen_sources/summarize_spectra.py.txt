"""Export manuscript claim support directly from the checked experiment tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def summarize():
    cases = pd.read_csv(RESULTS / "spectral/case_summary.csv")
    support = {}
    for study, step in [("sumo_placement", 0.1), ("phase", 0.05)]:
        rows = cases[(cases.study == study) & (cases["dt"] == step)]
        support[study] = {
            "responses": len(rows),
            "power_increases": int(
                (rows.late_nonuniform_power > rows.early_nonuniform_power).sum()
            ),
            "power_decreases": int(
                (rows.late_nonuniform_power < rows.early_nonuniform_power).sum()
            ),
            "long_share_increases": int((rows.long_late_share > rows.long_early_share).sum()),
            "late_long_share_range": [
                float(rows.long_late_share.min()),
                float(rows.long_late_share.max()),
            ],
        }
    support["examples"] = cases[
        cases.id.isin(["L480_t1.0_a0.3_s31_dt0.1", "L480_t1.0_a1.5_s31_dt0.1"])
    ].to_dict("records")
    for row in support["examples"]:
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    exponents = []
    for study in ["spectral_forcing", "spectral_small_pulse"]:
        scaling = pd.read_csv(RESULTS / study / "scaling.csv")
        primary = scaling["dt"].min()
        rows = scaling[(scaling["dt"] == primary) & (scaling.band == "all")]
        table = (
            rows.groupby(["accel", "window"])
            .power_exponent.agg(["mean", "min", "max"])
            .reset_index()
        )
        table["study"] = study
        exponents.extend(table.to_dict("records"))
        scenarios = pd.read_csv(RESULTS / study / "scenarios.csv")
        steps = pd.read_csv(RESULTS / study / "step_comparison.csv")
        steps = steps[steps.band == "all"]
        support[study] = {
            "max_realized_drop_error": float(
                abs(scenarios.realized_drop - scenarios.requested_drop).max()
            ),
            "max_absolute_log_step_difference": steps.groupby("window")
            .log_power_step_difference.apply(lambda x: float(abs(x).max()))
            .to_dict(),
        }
    support["exponents"] = exponents
    support["validation"] = json.loads(
        (RESULTS / "spectral/independent_validation.json").read_text()
    )
    (RESULTS / "spectral/manuscript_support.json").write_text(
        json.dumps(support, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(exponents).to_csv(RESULTS / "spectral/exponent_summary.csv", index=False)
    print(
        json.dumps(
            {key: value for key, value in support.items() if key not in ["examples", "exponents"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    summarize()
