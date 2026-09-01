"""Compatibility wrapper for ``sketch_to_gcode.reporting``."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_IMPL_DIR = _SRC / "sketch_to_gcode" / "reporting"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__path__ = [str(_IMPL_DIR)]

from sketch_to_gcode.reporting import *  # noqa: F401,F403

del Path, sys
