"""Module 4 — Inference Policy Engine.

Replaces the idea of an LLM-based router. **No AI, no tokens, no model calls.**
It only consults knowledge:

    Prompt -> Cognitive load -> Profile lookup -> Utility function -> Optimal model

Given a :class:`~mobz.models.CognitiveProfile`, the ``ALLOWED_MODELS`` list and
the model profiles from the (placeholder) non-relational DB, it:

1. Derives the *quality bar* required by the task's difficulty / reasoning depth.
2. Filters candidates to those expected to clear that bar (accuracy gate).
3. Scores survivors with a utility function that, matching the hackathon scoring
   (accuracy gate first, then *fewer tokens = higher rank*), strongly favours the
   most token- and cost-efficient model that still passes quality.
4. Returns a :class:`~mobz.models.RoutingDecision` including the ``max_tokens``
   budget to send to Fireworks (derived from ``expected_output_tokens``).

The engine is robust to missing profiles: unknown models get neutral defaults so
routing never crashes, though a real profile always yields better decisions.
"""

from __future__ import annotations

from .config import Config
from .logging_conf import get_logger
from .models import CognitiveProfile, ModelProfile, RoutingDecision
from .profile_store import normalise_id

log = get_logger(__name__)

# Neutral defaults used when a model has no profile in the DB.
_NEUTRAL_CAPABILITY = 0.75
_NEUTRAL_JSON_RELIABILITY = 0.8


class PolicyError(RuntimeError):
    """Raised when no viable model can be selected."""


class InferencePolicyEngine:
    def __init__(self, config: Config, profiles: dict[str, ModelProfile]) -> None:
        self._cfg = config
        self._profiles = profiles

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def select(self, task_id: str, cog: CognitiveProfile) -> RoutingDecision:
        """Select the optimal model for one task."""
        candidates = self._cfg.allowed_models
        if not candidates:
            raise PolicyError("ALLOWED_MODELS is empty")

        quality_bar = self._required_quality(cog)
        max_tokens = self._token_budget(cog)

        scored: dict[str, float] = {}
        eligible: dict[str, float] = {}

        for model in candidates:
            profile = self._profile_for(model)
            capability = self._capability(profile, cog.task)
            utility = self._utility(profile, cog, capability, quality_bar)
            scored[model] = round(utility, 5)
            if capability >= quality_bar:
                eligible[model] = utility

        # Prefer models that pass the accuracy gate; if none do, fall back to
        # the highest-capability model so we still attempt an answer.
        if eligible:
            best = max(eligible, key=eligible.get)
            reason = (
                f"task={cog.task} quality_bar={quality_bar:.2f}: selected best "
                f"utility among {len(eligible)} model(s) passing the gate"
            )
        else:
            best = max(
                candidates,
                key=lambda m: self._capability(self._profile_for(m), cog.task),
            )
            reason = (
                f"task={cog.task} quality_bar={quality_bar:.2f}: no model cleared "
                f"the gate, falling back to highest-capability model"
            )
            log.warning("Task %s: %s -> %s", task_id, reason, best)

        decision = RoutingDecision(
            task_id=task_id,
            selected_model=best,
            utility=scored[best],
            max_output_tokens=max_tokens,
            reason=reason,
            considered=scored,
        )
        log.info(
            "Task %s routed to %s (utility=%.4f, max_tokens=%d)",
            task_id, best, decision.utility, max_tokens,
        )
        return decision

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #
    def _profile_for(self, model: str) -> ModelProfile | None:
            profile = self._profiles.get(model)
            if profile is None:
                profile = self._profiles.get(normalise_id(model))
            return profile

    def _capability(self, profile: ModelProfile | None, task: str) -> float:
        if profile is None:
            return _NEUTRAL_CAPABILITY
        value = profile.capability(task)
        return value if value > 0 else _NEUTRAL_CAPABILITY

    def _json_reliability(self, profile: ModelProfile | None) -> float:
        if profile is None or profile.json_reliability <= 0:
            return _NEUTRAL_JSON_RELIABILITY
        return profile.json_reliability

    def _required_quality(self, cog: CognitiveProfile) -> float:
        """Quality bar rises with difficulty and reasoning depth."""
        base = self._cfg.quality_threshold
        demand = 0.5 * cog.difficulty + 0.5 * cog.reasoning_depth
        # Scale the remaining headroom above the base threshold by demand.
        bar = base + (1.0 - base) * demand
        return min(0.99, bar)

    def _token_budget(self, cog: CognitiveProfile) -> int:
        raw = int(round(cog.expected_output_tokens * self._cfg.output_token_margin))
        return max(self._cfg.min_output_tokens, min(self._cfg.max_output_tokens, raw))

    def _utility(
        self,
        profile: ModelProfile | None,
        cog: CognitiveProfile,
        capability: float,
        quality_bar: float,
    ) -> float:
        """Higher is better.

        Rewards capability **only up to the quality bar** (a small margin above
        it), then penalises expected token spend, cost and latency. Capping the
        capability reward is the key to token efficiency: once several models
        clear the accuracy gate, the *cheapest / least verbose* one wins instead
        of always over-provisioning to the single most capable model.
        """
        cfg = self._cfg

        # Capability beyond the gate (+ small safety margin) gives no extra
        # reward — so easy tasks route to the cheapest qualifying model and only
        # genuinely hard tasks pull in the most capable one.
        effective_cap = min(capability, quality_bar + 0.03)

        # Expected token spend for this task (normalised to [0, ~1]).
        est_tokens = profile.avg_tokens if (profile and profile.avg_tokens > 0) else float(
            cog.expected_output_tokens
        )
        norm_tokens = min(1.0, est_tokens / float(cfg.max_output_tokens))

        # Cost = output price in $/1M tokens; normalise against a $5/1M reference.
        est_cost = (profile.cost if profile else 0.0)
        norm_cost = min(1.0, est_cost / 5.0)

        # Latency (normalised against a 5s reference).
        latency = profile.latency if profile else 0.0
        norm_latency = min(1.0, latency / 5000.0)

        # Format reliability matters only when a structured output is expected.
        fmt = self._json_reliability(profile) if cog.expected_output == "json" else 1.0

        return (
            cfg.weight_quality * effective_cap
            + cfg.weight_format * fmt
            - cfg.weight_tokens * norm_tokens
            - cfg.weight_cost * norm_cost
            - cfg.weight_latency * norm_latency
        )
