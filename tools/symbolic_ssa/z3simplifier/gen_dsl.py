#!/usr/bin/env python3
"""Compatibility forwarding entrypoint to `tools.symbolic_ssa.z3.gen_dsl`."""

import sys

from tools.symbolic_ssa.z3.gen_dsl import generate_dsl_source, main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
