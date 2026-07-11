"""Model profile store  (NON-RELATIONAL DB).

These profiles are the "memory" of MobZ. Two schemas are supported:

1. **Flat** (legacy / simple): one object per model with capability fields at the
   top level — read by :class:`FileProfileStore`.

2. **Rich / gold** (``mobz_model_profiles.json``): nested documents with
   ``capability_scores``, ``composite_indices``, ``cost``, ``cognitive_profile``,
   etc. — read by :class:`RichProfileStore`, which maps them onto
   :class:`~mobz.models.ModelProfile`.

``load_profile_store`` auto-detects the schema. Model ids are normalised so a
profile authored as ``fireworks/glm-5p2`` matches the API id
``accounts/fireworks/models/glm-5p2`` (and vice-versa) — lookups are keyed by the
trailing slug as well as the full id.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any

from .config import Config
from .logging_conf import get_logger
from .models import ModelProfile

log = get_logger(__name__)


class ProfileStoreError(RuntimeError):
    """Raised when profiles cannot be loaded."""


def normalise_id(model_id: str) -> str:
    """Return the canonical trailing slug for a model id.

    ``accounts/fireworks/models/glm-5p2`` -> ``glm-5p2``
    ``fireworks/glm-5p2``                 -> ``glm-5p2``
    ``glm-5p2``                           -> ``glm-5p2``
    """
    return model_id.rsplit("/", 1)[-1].strip().lower()


class ProfileStore(abc.ABC):
    """Abstract non-relational store of model profiles."""

    @abc.abstractmethod
    def load_profiles(self) -> dict[str, ModelProfile]:
        """Return a mapping ``lookup_key -> ModelProfile``.

        Implementations index each profile under both its full id and its
        normalised slug so the policy engine can resolve either form.
        """
        raise NotImplementedError

    @staticmethod
    def _index(profiles: list[ModelProfile]) -> dict[str, ModelProfile]:
        index: dict[str, ModelProfile] = {}
        for p in profiles:
            index[p.model] = p
            index.setdefault(normalise_id(p.model), p)
        return index


# --------------------------------------------------------------------------- #
#  Flat schema (simple).
# --------------------------------------------------------------------------- #
class FileProfileStore(ProfileStore):
    """Reads flat profiles from a local JSON document (array or {id: profile})."""

    def __init__(self, path: str) -> None:
        self._path = path

    def load_profiles(self) -> dict[str, ModelProfile]:
        path = Path(self._path)
        if not path.is_file():
            log.warning("No profiles document at %s; using neutral defaults.", self._path)
            return {}
        raw = _read_json(path)
        entries = list(raw.values()) if isinstance(raw, dict) else raw
        profiles = [ModelProfile.from_dict(e) for e in entries]
        log.info("Loaded %d flat model profile(s) from %s", len(profiles), self._path)
        return self._index(profiles)


# --------------------------------------------------------------------------- #
#  Rich / gold schema (mobz_model_profiles.json).
# --------------------------------------------------------------------------- #
class RichProfileStore(ProfileStore):
    """Maps the rich, enriched profile documents onto :class:`ModelProfile`.

    Consumes the numeric ``capability_scores`` block produced by the enrichment
    step (with graceful fallbacks to ``composite_indices`` if absent), plus
    operational fields (cost, verbosity, json reliability, context length).
    """

    # Reference output price ($/1M tokens) used to normalise cost to ~[0,1].
    _COST_REFERENCE = 5.0

    def __init__(self, path: str) -> None:
        self._path = path

    def load_profiles(self) -> dict[str, ModelProfile]:
        path = Path(self._path)
        if not path.is_file():
            raise ProfileStoreError(f"Profiles DB not found: {self._path}")
        raw = _read_json(path)
        entries = list(raw.values()) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ProfileStoreError("Rich profiles DB must be a JSON array/object")

        profiles = [self._to_profile(e) for e in entries if isinstance(e, dict)]
        log.info("Loaded %d rich model profile(s) from %s", len(profiles), self._path)
        return self._index(profiles)

    def _to_profile(self, doc: dict[str, Any]) -> ModelProfile:
        model_id = doc.get("model_id") or doc.get("model") or "unknown"
        caps = self._capabilities(doc)
        cost = doc.get("cost") or {}
        out_cost = _num(cost.get("output_per_million"))

        return ModelProfile(
            model=model_id,
            capabilities=caps,
            # avg output tokens the model tends to spend (verbosity-derived).
            avg_tokens=float(doc.get("estimated_output_tokens") or 0.0),
            latency=self._latency_estimate(doc),
            # cost normalised to a per-1M-output-token dollar figure.
            cost=out_cost if out_cost is not None else 0.0,
            json_reliability=float(doc.get("json_reliability_score") or 0.0),
        )

    def _capabilities(self, doc: dict[str, Any]) -> dict[str, float]:
        caps = doc.get("capability_scores")
        if isinstance(caps, dict) and caps:
            return {k: float(v) for k, v in caps.items() if isinstance(v, (int, float))}

        # Fallback: derive from composite_indices when capability_scores absent.
        ci = doc.get("composite_indices") or {}
        m = {
            "knowledge": ci.get("knowledge_index"),
            "math": ci.get("math_index"),
            "coding": ci.get("coding_index"),
            "reasoning": ci.get("reasoning_index"),
            "instruction_following": ci.get("instruction_index"),
        }
        return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}

    def _latency_estimate(self, doc: dict[str, Any]) -> float:
        """Coarse latency proxy (ms). Larger context / verbosity → slower.

        The DB has no measured latency; this keeps the utility function's latency
        term meaningful without dominating it. Returns 0 (neutral) if unknown.
        """
        factor = _num(doc.get("verbosity_factor")) or 1.0
        return round(600.0 * factor, 1)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileStoreError(f"Profiles file is not valid JSON: {exc}") from exc


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _detect_rich(path: Path) -> bool:
    """Heuristically decide whether a JSON doc uses the rich schema."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    sample = (raw[0] if isinstance(raw, list) and raw
              else next(iter(raw.values()), {}) if isinstance(raw, dict) else {})
    return isinstance(sample, dict) and (
        "capability_scores" in sample or "composite_indices" in sample
        or "cognitive_profile" in sample or "model_id" in sample
    )


def load_profile_store(config: Config) -> ProfileStore:
    """Factory that returns the active profile store.

    Resolution order:
    1. ``MOBZ_PROFILES_DB_URI`` if set (auto-detects flat vs rich schema).
    2. ``mobz_model_profiles.json`` at the project root (rich gold DB).
    3. ``data/profiles.json`` (flat placeholder).

    To connect a real non-relational database (MongoDB, DynamoDB, ...), implement
    a :class:`ProfileStore` subclass and return it here based on the URI scheme.
    """
    root = Path(__file__).resolve().parents[2]

    if config.profiles_db_uri:
        uri = config.profiles_db_uri
        # (Plug real DB clients here, e.g. uri.startswith("mongodb://").)
        path = Path(uri)
        if path.is_file():
            if _detect_rich(path):
                log.info("Profiles: rich schema at %s", uri)
                return RichProfileStore(uri)
            log.info("Profiles: flat schema at %s", uri)
            return FileProfileStore(uri)
        raise ProfileStoreError(f"MOBZ_PROFILES_DB_URI path not found: {uri}")

    gold = root / "mobz_model_profiles.json"
    if gold.is_file():
        log.info("Profiles: rich gold DB at %s", gold)
        return RichProfileStore(str(gold))

    default_path = str(root / "data" / "profiles.json")
    log.warning("Falling back to flat placeholder profile store: %s", default_path)
    return FileProfileStore(default_path)
