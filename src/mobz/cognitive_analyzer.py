"""Module 3 — Cognitive Analyzer  (LOCAL MODEL).

The only local model in MobZ. It does **not** answer prompts, generate text or
reason like an LLM. Its single job is to estimate the cognitive load of a prompt
and emit a :class:`~mobz.models.CognitiveProfile`. Local tokens count as zero for
scoring, so this model is "free".

The trained model shipped by the user is a **ModernBERT-base multi-head**
regressor/classifier (``mobz_cognitive_analyzer/.../model.safetensors``):

    encoder        : ModernBERT-base  (hidden=768, vocab=50368, 22 layers, RoPE)
    task_head      : Linear(768, 8)   -> task category      (classification)
    format_head    : Linear(768, 10)  -> expected_output    (classification)
    difficulty_head: Linear(768, 1)   -> difficulty         (regression 0..1)
    reasoning_head : Linear(768, 1)   -> reasoning_depth     (regression 0..1)
    tokens_head    : Linear(768, 1)   -> expected_output_tokens (regression)

:class:`TransformersCognitiveAnalyzer` loads and runs it. If any required piece
is unavailable (torch/transformers not installed, or the model dir is missing
``config.json`` / tokenizer files / a ``labels.json`` label map) it logs exactly
what is missing and falls back to :class:`PlaceholderCognitiveAnalyzer` so the
pipeline always runs.

WHAT THE MODEL DIR STILL NEEDS FOR A CORRECT LOAD
-------------------------------------------------
The safetensors weights alone are not enough to interpret the outputs. Drop
these next to ``model.safetensors`` (all standard HF artifacts from training):
  * ``config.json``               — encoder config (ModernBERT-base).
  * tokenizer files               — ``tokenizer.json`` / ``tokenizer_config.json``
                                     / ``special_tokens_map.json``.
  * ``labels.json`` (optional)    — ``{"task": [...8 labels...],
                                        "format": [...10 labels...],
                                        "tokens_transform": "raw|expm1",
                                        "pooling": "cls|mean"}``.
Without ``labels.json`` the analyzer uses the documented DEFAULT_LABELS below;
confirm they match your training label order or predictions will be mislabelled.
"""

from __future__ import annotations

import abc
import json
import math
import os
import re
from pathlib import Path

from .config import Config
from .logging_conf import get_logger
from .models import CognitiveProfile

log = get_logger(__name__)


# Documented defaults — OVERRIDE via <model_dir>/labels.json to match training.
DEFAULT_TASK_LABELS = [
    "knowledge", "math", "coding", "reasoning",
    "summary", "ner", "sentiment", "classification",
]
DEFAULT_FORMAT_LABELS = [
    "plain_text", "json", "code", "list", "markdown",
    "table", "number", "boolean", "yaml", "xml",
]

# Map the trained model's task vocabulary onto the capability keys used by the
# profile DB / policy engine (capability_scores). Unmapped labels pass through.
TASK_ALIASES = {
    "code_generation": "coding",
    "debug": "coding",
    "logic": "reasoning",
}

# Format labels that imply source code -> validator treats them uniformly.
_CODE_FORMATS = {"code", "python", "rust", "csharp", "go", "java", "cpp", "javascript"}


def _labels_to_list(spec, default: list[str]) -> list[str]:
    """Normalise a label spec into an ordered list indexed by class id.

    Accepts an ordered list, or an index dict like ``{"0": "knowledge", ...}``.
    """
    if isinstance(spec, list) and spec:
        return spec
    if isinstance(spec, dict) and spec:
        try:
            return [spec[str(i)] for i in range(len(spec))]
        except KeyError:
            # Non-contiguous keys: sort by integer index.
            return [spec[k] for k in sorted(spec, key=lambda x: int(x))]
    return default


class CognitiveAnalyzer(abc.ABC):
    """Abstract interface for the local Cognitive Analyzer."""

    @abc.abstractmethod
    def analyze(self, prompt: str) -> CognitiveProfile:
        """Estimate the cognitive load of ``prompt``. Never answers it."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Heuristic fallback (deterministic, no dependencies).
# --------------------------------------------------------------------------- #
class PlaceholderCognitiveAnalyzer(CognitiveAnalyzer):
    """Deterministic heuristic stand-in used when the trained model can't load."""

    _CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
        "coding": ("code", "function", "python", "javascript", "bug", "compile",
                   "algorithm", "class ", "def ", "sql", "regex"),
        "math": ("calculate", "solve", "equation", "integral", "derivative",
                 "sum", "product", "probability", "matrix"),
        "reasoning": ("why", "explain", "reason", "deduce", "infer", "prove",
                      "step by step", "logic"),
        "summary": ("summarise", "summarize", "summary", "tl;dr", "condense",
                    "shorten"),
        "ner": ("extract", "entities", "named entity", "identify names",
                "list the people", "organizations"),
        "sentiment": ("sentiment", "positive or negative", "emotion", "tone",
                      "opinion"),
        "knowledge": ("what is", "who is", "when did", "define", "capital of",
                      "history of", "fact"),
    }
    _JSON_HINTS = ("json", "as an object", "key-value", "schema", "fields")
    _CODE_HINTS = ("code", "function", "script", "snippet", "program")

    def analyze(self, prompt: str) -> CognitiveProfile:
        text = prompt.lower()
        tokens_estimate = max(1, len(re.findall(r"\w+", prompt)))
        scores = {
            category: sum(1 for kw in keywords if kw in text)
            for category, keywords in self._CATEGORY_KEYWORDS.items()
        }
        task = max(scores, key=scores.get) if any(scores.values()) else "knowledge"

        difficulty = min(1.0, tokens_estimate / 400.0)
        if any(kw in text for kw in ("step by step", "prove", "derive", "complex")):
            difficulty = min(1.0, difficulty + 0.25)
        reasoning_depth = min(
            1.0,
            (text.count("?") * 0.1) + (difficulty * 0.6)
            + (0.3 if task in ("reasoning", "math", "coding") else 0.0),
        )
        if any(h in text for h in self._JSON_HINTS):
            expected_output = "json"
        elif any(h in text for h in self._CODE_HINTS):
            expected_output = "code"
        else:
            expected_output = "plain_text"
        if any(h in text for h in ("one sentence", "one word", "briefly", "tl;dr")):
            expected_output_tokens = 32
        elif expected_output in ("code", "json"):
            expected_output_tokens = 256
        else:
            expected_output_tokens = max(32, min(512, tokens_estimate))

        return CognitiveProfile(
            task=task,
            difficulty=round(difficulty, 3),
            reasoning_depth=round(reasoning_depth, 3),
            expected_output=expected_output,
            expected_output_tokens=expected_output_tokens,
            confidence=0.5,
        )


# --------------------------------------------------------------------------- #
#  Real model — ModernBERT-base multi-head.
# --------------------------------------------------------------------------- #
class TransformersCognitiveAnalyzer(CognitiveAnalyzer):
    """Loads and runs the trained ModernBERT multi-head model.

    Construction may raise; use :meth:`try_load` for a safe factory that returns
    ``None`` on any failure so callers can fall back to the placeholder.
    """

    def __init__(self, model_dir: str, encoder_name: str = "answerdotai/ModernBERT-base"):
        import torch  # noqa: F401  (lazy, heavy import)
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from safetensors.torch import load_file

        self._torch = torch
        self._dir = Path(model_dir)
        weights_path = self._dir / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"model.safetensors not found in {model_dir}")

        # Config/tokenizer/labels may live in the model dir OR its parent.
        def _find(name: str) -> Path | None:
            for base in (self._dir, self._dir.parent):
                p = base / name
                if p.exists():
                    return p
            return None

        # Label maps / interpretation knobs.
        labels: dict = {}
        labels_path = _find("labels.json")
        if labels_path is not None:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            log.info("Loaded label maps from %s", labels_path)
        else:
            log.warning("No labels.json found; using DEFAULT_LABELS (confirm order!)")

        # Accept either ordered lists ("task"/"format") or index dicts
        # ("id2task"/"id2format", e.g. {"0": "knowledge", ...}).
        self._task_labels = _labels_to_list(
            labels.get("task") or labels.get("id2task"), DEFAULT_TASK_LABELS)
        self._format_labels = _labels_to_list(
            labels.get("format") or labels.get("id2format"), DEFAULT_FORMAT_LABELS)
        # tokens_head outputs the expected token count directly (verified: raw
        # values ~50-140 track expected answer length). Override via labels.json
        # ("tokens_transform": "expm1") if your training used a log1p target.
        self._tokens_transform = labels.get("tokens_transform", "raw")

        # Optional training sidecar (mobz_config.json) with pooling / max_length.
        mobz_cfg: dict = {}
        mobz_cfg_path = _find("mobz_config.json")
        if mobz_cfg_path is not None:
            mobz_cfg = json.loads(mobz_cfg_path.read_text(encoding="utf-8"))
            log.info("Loaded training sidecar %s", mobz_cfg_path)
        self._max_length = int(mobz_cfg.get("max_length", 512))

        # Encoder config + tokenizer: prefer local files, else the base model.
        cfg_path = _find("config.json")
        tok_path = _find("tokenizer.json")
        cfg_src = str(cfg_path.parent) if cfg_path else encoder_name
        tok_src = str(tok_path.parent) if tok_path else encoder_name
        log.info("Cognitive model: encoder cfg from '%s', tokenizer from '%s'",
                 cfg_src, tok_src)
        config = AutoConfig.from_pretrained(cfg_src)
        self._tokenizer = AutoTokenizer.from_pretrained(tok_src)
        encoder = AutoModel.from_config(config)
        hidden = getattr(config, "hidden_size", 768)

        # Pooling: labels.json / mobz_config.json win. Default is CLS — verified
        # empirically for this multi-head checkpoint (the tokens_head scales
        # sensibly under CLS: short≈30 / long≈134 tokens, but is dead under mean
        # pooling). The encoder config's classifier_pooling ("mean") is a
        # ModernBertForMaskedLM default and does NOT reflect how these task heads
        # were trained.
        self._pooling = labels.get("pooling") or mobz_cfg.get("pooling") or "cls"
        log.info("Cognitive model pooling=%s, tokens_transform=%s",
                 self._pooling, self._tokens_transform)

        # Reconstruct the multi-head model and load weights.
        state = load_file(str(weights_path))
        n_task = state["task_head.weight"].shape[0]
        n_format = state["format_head.weight"].shape[0]

        class MultiHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.task_head = nn.Linear(hidden, n_task)
                self.format_head = nn.Linear(hidden, n_format)
                self.difficulty_head = nn.Linear(hidden, 1)
                self.reasoning_head = nn.Linear(hidden, 1)
                self.tokens_head = nn.Linear(hidden, 1)

        self._model = MultiHead()
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Cognitive model missing keys on load: %s", missing[:6])
        self._model.eval()

    @classmethod
    def try_load(cls, model_dir: str) -> "TransformersCognitiveAnalyzer | None":
        try:
            encoder_name = os.environ.get("MOBZ_ENCODER_NAME", "answerdotai/ModernBERT-base")
            analyzer = cls(model_dir, encoder_name=encoder_name)
            log.info("Loaded trained ModernBERT cognitive analyzer from %s", model_dir)
            return analyzer
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load trained cognitive model (%s): %s. "
                        "Falling back to heuristic analyzer.", type(exc).__name__, exc)
            return None

    def analyze(self, prompt: str) -> CognitiveProfile:
        torch = self._torch
        enc = self._tokenizer(
            prompt, truncation=True, max_length=self._max_length, return_tensors="pt",
        )
        with torch.no_grad():
            out = self._model.encoder(**enc)
            hidden_states = out.last_hidden_state
            if self._pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:  # cls
                pooled = hidden_states[:, 0]

            task_logits = self._model.task_head(pooled)
            fmt_logits = self._model.format_head(pooled)
            difficulty = torch.sigmoid(self._model.difficulty_head(pooled)).item()
            reasoning = torch.sigmoid(self._model.reasoning_head(pooled)).item()
            tokens_raw = self._model.tokens_head(pooled).item()

        task_idx = int(task_logits.argmax(-1).item())
        fmt_idx = int(fmt_logits.argmax(-1).item())
        raw_task = self._task_labels[task_idx] if task_idx < len(self._task_labels) else "knowledge"
        fmt = self._format_labels[fmt_idx] if fmt_idx < len(self._format_labels) else "plain_text"
        # Map the model's task vocabulary onto capability keys the engine uses.
        task = TASK_ALIASES.get(raw_task, raw_task)
        # Collapse source-code format variants (python/rust/...) into "code".
        if fmt in _CODE_FORMATS:
            fmt = "code"

        if self._tokens_transform == "expm1":
            # Clamp before expm1 to avoid float overflow (e^8.4 ≈ 4400 > cap).
            clamped = max(0.0, min(8.4, float(tokens_raw)))
            tokens = int(round(math.expm1(clamped)))
        else:
            tokens = int(round(float(tokens_raw)))
        if tokens != tokens or tokens < 0:  # NaN / negative guard
            tokens = 256
        tokens = max(1, min(4096, tokens))

        # Softmax confidence of the task head.
        conf = float(torch.softmax(task_logits, -1).max().item())

        return CognitiveProfile(
            task=task,
            difficulty=round(float(difficulty), 3),
            reasoning_depth=round(float(reasoning), 3),
            expected_output=fmt,
            expected_output_tokens=tokens,
            confidence=round(conf, 3),
        )


def load_cognitive_analyzer(config: Config) -> CognitiveAnalyzer:
    """Return the active Cognitive Analyzer.

    Tries the trained ModernBERT model at ``MOBZ_COGNITIVE_MODEL_PATH``; on any
    failure (missing deps or artifacts) falls back to the heuristic placeholder
    so the pipeline always runs.
    """
    model_path = config.cognitive_model_path
    if model_path and Path(model_path).exists():
        analyzer = TransformersCognitiveAnalyzer.try_load(model_path)
        if analyzer is not None:
            return analyzer
    else:
        if model_path:
            log.warning("MOBZ_COGNITIVE_MODEL_PATH does not exist: %s", model_path)
        log.warning("Using PLACEHOLDER Cognitive Analyzer (heuristic, not trained).")
    return PlaceholderCognitiveAnalyzer()
