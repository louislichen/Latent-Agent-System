"""Save the executable source package and its hashes alongside a run."""

import hashlib
from pathlib import Path
import shutil


def snapshot_sources(destination):
    package = Path(__file__).resolve().parent
    frozen = destination / package.name
    shutil.copytree(package, frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return {
        path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(frozen.rglob("*.py"))
    }
