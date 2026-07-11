"""Module 5 — Fireworks Client.

The single point of inference in MobZ. Every call goes through
``FIREWORKS_BASE_URL`` with ``FIREWORKS_API_KEY`` (both injected by the harness).
Bypassing this URL would make tokens unrecorded and score zero, so there is
exactly one code path to the network and it always uses the configured client.

Uses the OpenAI-compatible Chat Completions API that Fireworks exposes. The
client is async so tasks run concurrently within the 10-minute budget.
"""

from __future__ import annotations

import asyncio
import time

from openai import AsyncOpenAI

from .config import Config
from .logging_conf import get_logger
from .models import InferenceResult, RoutingDecision, Task

log = get_logger(__name__)


class FireworksClient:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        # ALL Fireworks traffic is configured to go through base_url. This is
        # the only client instance in the process.
        self._client = AsyncOpenAI(
            api_key=config.fireworks_api_key,
            base_url=config.fireworks_base_url,
            timeout=config.request_timeout_seconds,
            max_retries=0,  # we implement our own bounded retry below
        )

    async def infer(self, task: Task, decision: RoutingDecision) -> InferenceResult:
        """Run one inference call for ``task`` using the routed model."""
        attempt = 0
        last_error: str | None = None
        start = time.perf_counter()
        # Budget can grow if the model truncates before emitting its answer.
        budget = decision.max_output_tokens
        grew_for_truncation = False

        while attempt <= self._cfg.max_retries:
            attempt += 1
            try:
                messages = []
                if self._cfg.system_prompt.strip():
                    messages.append({"role": "system", "content": self._cfg.system_prompt})
                messages.append({"role": "user", "content": task.prompt})
                completion = await self._client.chat.completions.create(
                    model=decision.selected_model,
                    messages=messages,
                    max_tokens=budget,
                    temperature=0.0,  # deterministic, minimal wasted tokens
                )
                latency_ms = (time.perf_counter() - start) * 1000.0

                choice = completion.choices[0] if completion.choices else None
                answer = (choice.message.content if choice and choice.message else "") or ""
                finish = getattr(choice, "finish_reason", None) if choice else None

                # Truncated before finishing: a reasoning model likely ran out of
                # room mid-answer. Grow the cap once and retry so the answer can
                # complete (a truncated answer fails the accuracy gate = 0 score).
                if (finish == "length" and not grew_for_truncation
                        and budget < self._cfg.max_output_tokens):
                    grew_for_truncation = True
                    budget = min(self._cfg.max_output_tokens, max(budget * 4, 1024))
                    log.info("Task %s truncated (finish=length); retrying with "
                             "max_tokens=%d", task.task_id, budget)
                    attempt -= 1  # this retry does not count against error retries
                    continue

                usage = completion.usage
                return InferenceResult(
                    task_id=task.task_id,
                    model=decision.selected_model,
                    answer=answer.strip(),
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    latency_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001 — bounded retry on any error
                last_error = f"{type(exc).__name__}: {exc}"
                backoff = min(2.0 * attempt, 8.0)
                log.warning(
                    "Task %s attempt %d/%d failed: %s",
                    task.task_id, attempt, self._cfg.max_retries + 1, last_error,
                )
                if attempt <= self._cfg.max_retries:
                    await asyncio.sleep(backoff)

        return InferenceResult(
            task_id=task.task_id,
            model=decision.selected_model,
            answer="",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            error=last_error or "unknown inference error",
        )

    async def aclose(self) -> None:
        await self._client.close()
