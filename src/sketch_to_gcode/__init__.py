"""Sketch-to-G-Code package.

The app started as a single module. This package keeps that public surface
available while allowing the source to live in a standard ``src`` layout.
"""

from __future__ import annotations

import sys
from types import ModuleType

from . import app as _app


class _AppProxyModule(ModuleType):
    def __getattr__(self, name: str):
        try:
            return getattr(_app, name)
        except AttributeError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    def __setattr__(self, name: str, value):
        ModuleType.__setattr__(self, name, value)
        if not name.startswith("__") and hasattr(_app, name):
            setattr(_app, name, value)

    def __delattr__(self, name: str):
        ModuleType.__delattr__(self, name)
        if not name.startswith("__") and hasattr(_app, name):
            delattr(_app, name)


_module = sys.modules[__name__]
_module.__class__ = _AppProxyModule

for _name in dir(_app):
    if not (_name.startswith("__") and _name.endswith("__")):
        setattr(_module, _name, getattr(_app, _name))

__all__ = [
    _name
    for _name in dir(_app)
    if not (_name.startswith("__") and _name.endswith("__"))
]

del _name, _module
