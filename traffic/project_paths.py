"""Resolve input paths while leaving historical experiment receipts unchanged."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"


def recorded_path(value):
    """Map repository-relative receipt paths to the current layout."""
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path
    for old, new in [
        ("data/research_2026/traffic", DATA),
        ("data/research_2026/traffic_novelty", DATA / "literature"),
        ("research/2026-10/traffic", HERE),
        ("research/2026-10", ROOT / "research"),
        ("artifacts/traffic_noise_resolution_repeat", HERE / "artifacts/noise_resolution_repeat"),
        ("artifacts/traffic_manuscript_layout", HERE / "artifacts/manuscript_layout"),
    ]:
        if path.is_relative_to(old):
            return new / path.relative_to(old)
    return ROOT / path
