#!/usr/bin/env python3
"""Module entrypoint for python3 -m tools.coverage."""

import sys
from tools.coverage.cli import main

if __name__ == "__main__":
    sys.exit(main())
