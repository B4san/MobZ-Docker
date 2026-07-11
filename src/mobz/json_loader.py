"""Module 1 — JSON Loader.

Reads ``/input/tasks.json``, validates its shape and builds an in-memory queue
of :class:`~mobz.models.Task` objects. No AI, no tokens consumed.

Accepted input formats
----------------------
1. A top-level JSON array (the format described by the hackathon)::

       [ { "task_id": "t1", "prompt": "..." }, ... ]

2. A wrapper object ``{ "tasks": [ ... ] }`` (tolerated for convenience).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque

from .logging_conf import get_logger
from .models import Task

log = get_logger(__name__)


class TaskLoadError(RuntimeError):
    """Raised when the input file is missing or malformed."""


def load_tasks(input_path: str) -> Deque[Task]:
    """Load, validate and enqueue tasks from ``input_path``.

    Raises :class:`TaskLoadError` on any structural problem so the orchestrator
    can exit with a non-zero status.
    """
    path = Path(input_path)
    if not path.is_file():
        raise TaskLoadError(f"Input file not found: {input_path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskLoadError(f"Input file is not valid JSON: {exc}") from exc

    if isinstance(raw, dict) and "tasks" in raw:
        raw = raw["tasks"]

    if not isinstance(raw, list):
        raise TaskLoadError(
            "Input JSON must be an array of tasks (or an object with a 'tasks' array)"
        )

    if not raw:
        raise TaskLoadError("Input JSON contained zero tasks")

    queue: Deque[Task] = deque()
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        try:
            task = Task.from_dict(entry)
        except ValueError as exc:
            raise TaskLoadError(f"Invalid task at index {index}: {exc}") from exc
        if task.task_id in seen:
            raise TaskLoadError(f"Duplicate task_id detected: '{task.task_id}'")
        seen.add(task.task_id)
        queue.append(task)

    log.info("Loaded %d task(s) from %s", len(queue), input_path)
    return queue
