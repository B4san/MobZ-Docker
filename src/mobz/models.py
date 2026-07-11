"""Data models shared across the MobZ pipeline.

These are plain dataclasses with light validation so that every module speaks
the same language. No business logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Module 1 — JSON Loader
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """A single unit of work read from ``/input/tasks.json``."""

    task_id: str
    prompt: str

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Task":
        if not isinstance(raw, dict):
            raise ValueError(f"Task entry must be an object, got {type(raw).__name__}")
        task_id = raw.get("task_id")
        prompt = raw.get("prompt")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Each task requires a non-empty string 'task_id'")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Task '{task_id}' requires a non-empty string 'prompt'")
        return Task(task_id=task_id, prompt=prompt)


# --------------------------------------------------------------------------- #
# Module 3 — Cognitive Analyzer
# --------------------------------------------------------------------------- #
@dataclass
class CognitiveProfile:
    """Estimation of the cognitive load a prompt demands.

    Produced by the (placeholder) local Cognitive Analyzer model. It never
    answers the prompt; it only characterises it.
    """

    task: str                       # capability required, e.g. "math", "coding"
    difficulty: float               # 0.0 (trivial) .. 1.0 (very hard)
    reasoning_depth: float          # 0.0 (none) .. 1.0 (deep multi-step)
    expected_output: str            # "plain_text" | "json" | "code" | ...
    expected_output_tokens: int     # rough size of the expected answer
    confidence: float               # 0.0 .. 1.0 analyzer self-confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "difficulty": self.difficulty,
            "reasoning_depth": self.reasoning_depth,
            "expected_output": self.expected_output,
            "expected_output_tokens": self.expected_output_tokens,
            "confidence": self.confidence,
        }


# --------------------------------------------------------------------------- #
# Non-relational DB — Model profiles (the "memory" of MobZ)
# --------------------------------------------------------------------------- #
@dataclass
class ModelProfile:
    """A cognitive profile for one Fireworks model.

    Capability fields are competence scores in ``[0, 1]``. Operational fields
    (avg_tokens, latency, cost) are used by the utility function to break ties
    towards the cheapest / most token-efficient option.
    """

    model: str
    # Capability scores per task category.
    capabilities: dict[str, float] = field(default_factory=dict)
    # Operational statistics.
    avg_tokens: float = 0.0         # mean tokens the model spends per call
    latency: float = 0.0            # mean latency in ms
    cost: float = 0.0               # cost per 1K tokens (or per call; see engine)
    json_reliability: float = 0.0   # 0..1 probability of well-formed JSON

    # Common capability keys (documented, not enforced).
    KNOWN_CAPABILITIES = (
        "knowledge",
        "math",
        "coding",
        "reasoning",
        "summary",
        "ner",
        "sentiment",
    )

    def capability(self, task: str) -> float:
        """Return the competence score for ``task`` (0.0 if unknown)."""
        return float(self.capabilities.get(task, 0.0))

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ModelProfile":
        if not isinstance(raw, dict):
            raise ValueError("Model profile must be an object")
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model profile requires a non-empty 'model' id")

        operational = {"model", "avg_tokens", "latency", "cost", "json_reliability"}
        capabilities = {
            key: float(value)
            for key, value in raw.items()
            if key not in operational and isinstance(value, (int, float))
        }
        return ModelProfile(
            model=model,
            capabilities=capabilities,
            avg_tokens=float(raw.get("avg_tokens", 0.0)),
            latency=float(raw.get("latency", 0.0)),
            cost=float(raw.get("cost", 0.0)),
            json_reliability=float(raw.get("json_reliability", 0.0)),
        )


# --------------------------------------------------------------------------- #
# Module 4 — Inference Policy Engine
# --------------------------------------------------------------------------- #
@dataclass
class RoutingDecision:
    """The outcome of the utility function for a task."""

    task_id: str
    selected_model: str
    utility: float
    max_output_tokens: int
    reason: str
    considered: dict[str, float] = field(default_factory=dict)  # model -> utility


# --------------------------------------------------------------------------- #
# Module 5 — Fireworks Client
# --------------------------------------------------------------------------- #
@dataclass
class InferenceResult:
    """Raw result of a single Fireworks inference call."""

    task_id: str
    model: str
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------- #
# Module 6 — Response Validator / final output
# --------------------------------------------------------------------------- #
@dataclass
class TaskResult:
    """Final, validated result written to ``/output/results.json``.

    The submission schema only requires ``task_id`` and ``answer``. Extra
    fields are kept internal and stripped before writing.
    """

    task_id: str
    answer: str

    def to_output_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "answer": self.answer}
