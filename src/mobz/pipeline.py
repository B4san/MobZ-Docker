"""Pipeline orchestrator.

Wires every module together and enforces the hackathon operational rules:

    JSON Loader -> Cognitive Analyzer -> Inference Policy Engine
                -> Fireworks Client -> Response Validator -> results.json

* Exit code 0 on success, non-zero on failure (see :func:`run`).
* Hard runtime budget (< 10 min) enforced with an asyncio timeout; whatever has
  completed is still written so we never produce malformed/empty output.
* Tasks run concurrently (bounded by ``config.concurrency``).
"""

from __future__ import annotations

import asyncio
import time

from .cognitive_analyzer import CognitiveAnalyzer, load_cognitive_analyzer
from .config import Config
from .fireworks_client import FireworksClient
from .json_loader import load_tasks
from .logging_conf import get_logger
from .models import CognitiveProfile, InferenceResult, Task, TaskResult
from .policy_engine import InferencePolicyEngine
from .profile_store import load_profile_store
from .prompt_compressor import PromptCompressor, load_prompt_compressor
from .validator import validate_result, write_results

log = get_logger(__name__)


async def _process_task(
    task: Task,
    compressor: PromptCompressor,
    analyzer: CognitiveAnalyzer,
    engine: InferencePolicyEngine,
    client: FireworksClient,
    semaphore: asyncio.Semaphore,
) -> tuple[TaskResult, CognitiveProfile]:
    async with semaphore:
        # Module 2 — compress the prompt (local, zero tokens). The compressed
        # prompt flows through the rest of the pipeline, cutting Fireworks input
        # tokens. Safety checks inside the compressor keep the original if the
        # rewrite would lose critical info or not actually be shorter.
        compressed = await asyncio.to_thread(compressor.compress, task.prompt)
        if compressed != task.prompt:
            log.info("Task %s compressed: %d -> %d words", task.task_id,
                     len(task.prompt.split()), len(compressed.split()))
        ctask = task if compressed == task.prompt else Task(task_id=task.task_id,
                                                            prompt=compressed)
        # Module 3 — local analysis of the compressed prompt (no tokens).
        cognitive = await asyncio.to_thread(analyzer.analyze, ctask.prompt)
        # Module 4 — pure knowledge routing (no tokens).
        decision = engine.select(ctask.task_id, cognitive)
        # Module 5 — the single inference call (uses the compressed prompt).
        result: InferenceResult = await client.infer(ctask, decision)
        # Module 6 — validation.
        return validate_result(result, cognitive), cognitive


async def _run_async(config: Config) -> int:
    # Module 1 — load & validate input (raises on malformed input).
    tasks = list(load_tasks(config.input_path))

    # Load local models (compressor + analyzer) and the profile DB.
    compressor = load_prompt_compressor(config)
    analyzer = load_cognitive_analyzer(config)
    profiles = load_profile_store(config).load_profiles()
    engine = InferencePolicyEngine(config, profiles)
    client = FireworksClient(config)

    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    # Preserve input order in the output.
    order = {task.task_id: i for i, task in enumerate(tasks)}
    results: dict[str, TaskResult] = {}

    coros = [
        _process_task(task, compressor, analyzer, engine, client, semaphore)
        for task in tasks
    ]

    try:
        gathered = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=True),
            timeout=config.max_runtime_seconds,
        )
    except asyncio.TimeoutError:
        log.error(
            "Runtime budget of %ds exceeded; writing whatever completed",
            config.max_runtime_seconds,
        )
        gathered = []
    finally:
        await client.aclose()

    for task, outcome in zip(tasks, gathered):
        if isinstance(outcome, Exception):
            log.error("Task %s raised: %s", task.task_id, outcome)
            results[task.task_id] = TaskResult(task_id=task.task_id, answer="N/A")
        else:
            task_result, _ = outcome
            results[task.task_id] = task_result

    # Guarantee every input task has a result (covers timeout gaps).
    for task in tasks:
        results.setdefault(task.task_id, TaskResult(task_id=task.task_id, answer="N/A"))

    ordered = sorted(results.values(), key=lambda r: order.get(r.task_id, 1_000_000))
    written = write_results(config.output_path, ordered)

    if written != len(tasks):
        log.error("Wrote %d results for %d tasks", written, len(tasks))
        return 1
    return 0


def run() -> int:
    """Synchronous entry point. Returns a process exit code."""
    start = time.perf_counter()
    try:
        config = Config.from_env()
    except Exception as exc:  # noqa: BLE001
        log.error("Configuration error: %s", exc)
        return 2

    log.info(
        "MobZ starting | models=%s | concurrency=%d | budget=%ds",
        config.allowed_models, config.concurrency, config.max_runtime_seconds,
    )
    try:
        exit_code = asyncio.run(_run_async(config))
    except Exception as exc:  # noqa: BLE001
        log.exception("Fatal pipeline error: %s", exc)
        return 1

    elapsed = time.perf_counter() - start
    log.info("MobZ finished in %.1fs with exit code %d", elapsed, exit_code)
    return exit_code
