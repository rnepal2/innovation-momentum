"""Preserve source content alongside future experiment freezes."""

import hashlib
import shutil
from pathlib import Path


def archive_sources(output, source_directory):
    output, source_directory = Path(output), Path(source_directory)
    destination = output / "frozen_sources"
    destination.mkdir()
    paths = sorted(source_directory.glob("*.py")) + sorted(source_directory.glob("*protocol.md"))
    paths += sorted(source_directory.glob("*followup.md"))
    hashes = {}
    for path in paths:
        target = destination / (path.name + ".txt")
        shutil.copyfile(path, target)
        hashes[path.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes
