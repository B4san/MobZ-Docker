"""Runtime configuration for MobZ.

Every value comes from environment variables. The evaluation harness injects
``FIREWORKS_API_KEY``, ``FIREWORKS_BASE_URL`` and ``ALLOWED_MODELS`` at runtime;
we never hardcode them or bundle a ``.env`` file inside the image.

For local development a ``.env`` file may be loaded (see ``load_dotenv``), but
it must never be part of the built image (see ``.dockerignore``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


def _get_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable '{name}'. "
            "The harness injects it at runtime; for local dev set it in your .env."
        )
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable '{name}' must be a number") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable '{name}' must be an integer") from exc


@dataclass
class Config:
    # --- Harness-injected (required) -------------------------------------- #
    fireworks_api_key: str
    fireworks_base_url: str
    allowed_models: list[str]

    # --- I/O paths -------------------------------------------------------- #
    input_path: str = "/input/tasks.json"
    output_path: str = "/output/results.json"

    # --- Placeholder integration points ---------------------------------- #
    # Path/URI the user will point at their local Cognitive Analyzer model.
    cognitive_model_path: str = ""
    # Path to the local Gemma prompt-compressor model.
    compressor_model_path: str = ""
    # Whether to compress prompts before analysis + inference.
    compression_enabled: bool = True
    # URI/path the user will point at their non-relational profiles database.
    profiles_db_uri: str = ""

    # --- Pipeline tuning -------------------------------------------------- #
    max_runtime_seconds: int = 570          # hard budget < 10 min hackathon cap
    concurrency: int = 8                    # parallel Fireworks calls
    request_timeout_seconds: float = 60.0
    max_retries: int = 2
    # Minimum capability score a model must clear to pass the accuracy gate.
    quality_threshold: float = 0.70
    # Safety multiplier applied to expected_output_tokens for max_tokens.
    output_token_margin: float = 1.5
    # Absolute floor/ceiling for the max_tokens sent to Fireworks. The floor is
    # generous because max_tokens is a CAP, not a target: you are billed for
    # tokens actually generated, so a too-low cap only truncates good answers
    # (failing the accuracy gate) without saving tokens. Reasoning-capable
    # models need headroom to emit their answer.
    min_output_tokens: int = 256
    max_output_tokens: int = 4096

    # System instruction prepended to every call. Steers models to answer
    # directly (fewer tokens, and avoids reasoning models truncating before the
    # final answer). Set MOBZ_SYSTEM_PROMPT="" to disable.
    system_prompt: str = (
        "You are a precise assistant. Provide only the final answer, as "
        "concisely as possible. Do not show your reasoning or add explanations "
        "unless the user explicitly asks for them. If a specific output format "
        "is requested, follow it exactly."
    )

    # Utility-function weights (see policy_engine).
    weight_quality: float = 1.0
    weight_tokens: float = 1.0
    weight_cost: float = 0.5
    weight_latency: float = 0.2
    weight_format: float = 0.4

    @staticmethod
    def from_env() -> "Config":
        allowed = [
            m.strip() for m in _get_required("ALLOWED_MODELS").split(",") if m.strip()
        ]
        if not allowed:
            raise ConfigError("ALLOWED_MODELS did not contain any model id")

        return Config(
            fireworks_api_key=_get_required("FIREWORKS_API_KEY"),
            fireworks_base_url=_get_required("FIREWORKS_BASE_URL"),
            allowed_models=allowed,
            input_path=os.environ.get("MOBZ_INPUT_PATH", "/input/tasks.json"),
            output_path=os.environ.get("MOBZ_OUTPUT_PATH", "/output/results.json"),
            cognitive_model_path=os.environ.get("MOBZ_COGNITIVE_MODEL_PATH", ""),
            compressor_model_path=os.environ.get("MOBZ_COMPRESSOR_MODEL_PATH", ""),
            compression_enabled=os.environ.get("MOBZ_COMPRESSION", "on").lower()
            not in ("off", "0", "false", "no"),
            profiles_db_uri=os.environ.get("MOBZ_PROFILES_DB_URI", ""),
            max_runtime_seconds=_get_int("MOBZ_MAX_RUNTIME_SECONDS", 570),
            concurrency=_get_int("MOBZ_CONCURRENCY", 8),
            request_timeout_seconds=_get_float("MOBZ_REQUEST_TIMEOUT", 60.0),
            max_retries=_get_int("MOBZ_MAX_RETRIES", 2),
            quality_threshold=_get_float("MOBZ_QUALITY_THRESHOLD", 0.70),
            output_token_margin=_get_float("MOBZ_OUTPUT_TOKEN_MARGIN", 1.5),
            min_output_tokens=_get_int("MOBZ_MIN_OUTPUT_TOKENS", 256),
            max_output_tokens=_get_int("MOBZ_MAX_OUTPUT_TOKENS", 4096),
            system_prompt=os.environ.get("MOBZ_SYSTEM_PROMPT", Config.system_prompt),
        )
