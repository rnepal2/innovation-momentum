"""Preserve pre-correction tables and freeze the corrected spatial recovery."""

import hashlib
import json
import shutil
from pathlib import Path

from provenance import archive_sources
from run_observation import analyze

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run():
    for repeated in [False, True]:
        for study in ["observation", "observation_window"]:
            out = (
                ROOT / "traffic/data" / ("repeat_" + study + "_clock_corrected")
                if repeated
                else HERE / "results" / study
            )
            revision = out / "analysis_revision"
            revision.mkdir()
            for name in [
                "diagnostics.csv",
                "rejection_summary.csv",
                "sampling_diagnostics.json",
                "noise_detection.png",
            ]:
                shutil.copyfile(out / name, revision / ("before_" + name))
            if (out / "whole_window_comparison.png").exists():
                shutil.copyfile(
                    out / "whole_window_comparison.png",
                    revision / "before_whole_window_comparison.png",
                )
            freeze = {
                "sources": archive_sources(revision, HERE),
                "correction_sha256": hashlib.sha256(
                    (HERE / "sampling-correction.md").read_bytes()
                ).hexdigest(),
                "raw_inputs": "Unchanged; use the original manifest.json",
            }
            (revision / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
            analyze(out, None, window=study == "observation_window")


if __name__ == "__main__":
    run()
