"""Entry point: ``python -m mobz``."""

import sys

from .pipeline import run

if __name__ == "__main__":
    sys.exit(run())
