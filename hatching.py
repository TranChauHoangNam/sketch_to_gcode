"""Compatibility wrapper for ``sketch_to_gcode.hatching``."""

from __future__ import annotations

import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sketch_to_gcode import hatching as _impl

for _name in dir(_impl):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_impl, _name)

__all__ = [
    _name
    for _name in dir(_impl)
    if not (_name.startswith("__") and _name.endswith("__"))
]


if __name__ == "__main__":
    _impl._demo_hatching()

del _name, _impl, Path, sys
