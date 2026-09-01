"""Compatibility entry point for the src-layout package.

Run the app with ``python sketch_to_gcode.py`` or import this module as before.
New code should prefer ``python -m sketch_to_gcode.app`` or the console script
defined in ``pyproject.toml``.
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_PACKAGE_DIR = _SRC / "sketch_to_gcode"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


if __name__ == "__main__":
    from sketch_to_gcode.app import main

    main()
else:
    __path__ = [str(_PACKAGE_DIR)]
    __package__ = __name__
    _app_path = _PACKAGE_DIR / "app.py"
    globals()["__file__"] = str(_app_path)
    exec(compile(_app_path.read_text(encoding="utf-8"), str(_app_path), "exec"), globals(), globals())
