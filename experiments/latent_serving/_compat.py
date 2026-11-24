"""Load the source package for legacy file-based benchmark commands."""

from importlib import import_module
from pathlib import Path
import sys


def run(module):
    src = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(src))
    import_module(module).main()
