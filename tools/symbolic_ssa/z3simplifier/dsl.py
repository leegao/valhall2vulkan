#!/usr/bin/env python3
"""Compatibility re-export of `tools.symbolic_ssa.z3.dsl`."""

import sys as _sys
from tools.symbolic_ssa.z3 import dsl as _real_dsl

_sys.modules[__name__] = _real_dsl
