"""Structured, timestamped logging that writes to stderr.

stdout is reserved for anything the harness might parse; all human/diagnostic
logging goes to stderr so it never contaminates results.
"""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "mobz") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.environ.get("MOBZ_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
