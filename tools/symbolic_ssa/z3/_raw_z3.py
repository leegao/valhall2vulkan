#!/usr/bin/env python3
"""
Safe loader for the upstream `z3` Python package (`z3-solver`).

When test runners such as `unittest discover -s tools/symbolic_ssa` prepend
`tools/symbolic_ssa/` onto `sys.path`, a bare `import z3` would shadow the
upstream `z3` wheel with this package (`tools/symbolic_ssa/z3/`). This helper
loads and caches the genuine upstream `z3` module unambiguously.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

_THIS_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_PKG_DIR)


def _load_upstream_z3() -> ModuleType:
    existing = sys.modules.get("z3")
    if existing is not None:
        ef = getattr(existing, "__file__", "") or ""
        if not os.path.abspath(ef).startswith(_THIS_PKG_DIR):
            return existing
        # Evict shadowed entry so upstream z3 can be imported cleanly
        sys.modules.pop("z3", None)

    saved_path = list(sys.path)
    try:
        sys.path = [
            p for p in sys.path
            if os.path.abspath(p or ".") not in (_PARENT_DIR, _THIS_PKG_DIR)
        ]
        mod = importlib.import_module("z3")
        return mod
    finally:
        sys.path = saved_path


raw_z3: ModuleType = _load_upstream_z3()
