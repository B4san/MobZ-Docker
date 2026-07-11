"""Module 6 — Response Validator + results writer.

Checks each inference result for integrity and writes the final, always-valid
``/output/results.json``. The output schema required by the harness is minimal::

    [ { "task_id": "t1", "answer": "..." }, ... ]

Validation guarantees:
* the file is always valid JSON (malformed output scores zero);
* every task_id from the input appears exactly once;
* answers are non-empty strings (failed/empty inferences get a safe fallback);
* when a structured (JSON) output was expected, we best-effort verify/repair it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .logging_conf import get_logger
from .models import CognitiveProfile, InferenceResult, TaskResult

log = get_logger(__name__)

# Fallback answer for tasks that produced no usable output. Non-empty so the
# result stays schema-valid even in the worst case.
_FALLBACK_ANSWER = "N/A"


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return False


def _extract_json_block(text: str) -> str | None:
    """Best-effort extraction of a JSON object/array embedded in prose."""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            if _looks_like_json(candidate):
                return candidate
    return None


def validate_result(
    result: InferenceResult,
    cognitive: CognitiveProfile | None = None,
) -> TaskResult:
    """Validate one inference result, repairing/annotating where possible."""
    answer = (result.answer or "").strip()

    if result.error and not answer:
        log.error("Task %s failed with no answer: %s", result.task_id, result.error)
        return TaskResult(task_id=result.task_id, answer=_FALLBACK_ANSWER)

    if not answer:
        log.warning("Task %s produced an empty answer; using fallback", result.task_id)
        return TaskResult(task_id=result.task_id, answer=_FALLBACK_ANSWER)

    # When JSON output was expected, try to ensure the answer is valid JSON.
    if cognitive and cognitive.expected_output == "json" and not _looks_like_json(answer):
        block = _extract_json_block(answer)
        if block is not None:
            log.info("Task %s: extracted embedded JSON from response", result.task_id)
            answer = block
        else:
            log.warning(
                "Task %s: JSON expected but response is not valid JSON; "
                "returning raw text",
                result.task_id,
            )

    return TaskResult(task_id=result.task_id, answer=answer)


def write_results(output_path: str, results: Iterable[TaskResult]) -> int:
    """Atomically write the results array to ``output_path``. Returns count."""
    payload = [r.to_output_dict() for r in results]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file then rename so a crash never leaves a partial file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

    log.info("Wrote %d result(s) to %s", len(payload), output_path)
    return len(payload)
