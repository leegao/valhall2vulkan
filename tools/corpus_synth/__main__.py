"""Allow running `python3 -m tools.corpus_synth`."""

import sys

from tools.corpus_synth.cli import main

if __name__ == "__main__":
    sys.exit(main())
