"""Compare full repeated pulse sweeps with their recorded numerical outputs."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def verify():
    receipt = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "matplotlib", "eclipse-sumo", "libsumo"]
        },
        "studies": {},
    }
    for study, repeat in [
        ("spectral_forcing", "repeat_spectral_forcing"),
        ("spectral_small_pulse", "repeat_spectral_small"),
    ]:
        original = HERE / "results" / study
        repeated = ROOT / "traffic/data" / repeat
        files = [
            "scenarios.csv",
            "simulations.csv",
            "window_power.csv",
            "trajectories.csv",
            "scaling.csv",
            "step_comparison.csv",
        ]
        checks = {}
        for name in files:
            expected, actual = (original / name).read_bytes(), (repeated / name).read_bytes()
            assert expected == actual, (study, name)
            checks[name] = hashlib.sha256(actual).hexdigest()
        receipt["studies"][study] = {
            "byte_identical_files": checks,
            "repeat_path": str(repeated.relative_to(ROOT)),
        }
    (HERE / "results/spectral/reproduction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    verify()
