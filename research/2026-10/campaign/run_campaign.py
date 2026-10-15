"""Rerun the bounded campaign, or verify existing outputs without refitting."""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import OUT, save_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Refresh/resume corpus files from cached pages and APIs",
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Recompute summaries and validation from existing forecasts",
    )
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    commands = []
    if args.download:
        commands += [["fetch_corpus.py"], ["fetch_arxiv.py"]]
    if not args.checks_only:
        commands += [["synthetic.py"]]
        for source in ["openalex", "arxiv"]:
            commands += [
                ["h1.py", "--source", source, "--sensitivity"],
                ["h2.py", "--source", source],
                ["h2.py", "--source", source, "--rolling-vocabulary"],
            ]
        commands += [
            ["h1.py", "--source", "openalex", "--exclude-january1"],
            ["basis_diagnostic.py"],
            ["h2_ablation.py"],
        ]
    commands += [["summarize.py"], ["sensitivity.py"], ["validate_results.py"]]
    for filename, *arguments in commands:
        subprocess.run([sys.executable, str(HERE / filename), *arguments], check=True, env=env)
    save_json(
        OUT / "execution_manifest.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libraries": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
            },
            "script_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HERE.glob("*.py"))
            },
            "last_action": "summary and validation"
            if args.checks_only
            else "full experiment campaign",
            "data_manifest": "acquisition_manifest.json",
            "credential_policy": "OPENALEX_API_KEY environment only; not saved",
        },
    )


if __name__ == "__main__":
    main()
