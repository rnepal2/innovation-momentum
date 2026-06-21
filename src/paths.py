"""Project path helpers for local research runs."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the workspace root used for data, configs, reports, and artifacts."""
    return Path(os.environ.get("INNOVATION_DYNAMICS_ROOT", Path.cwd())).expanduser().resolve()
