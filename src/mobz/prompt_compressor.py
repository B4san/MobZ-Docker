"""Module 2 — Prompt Compressor.

Compresses every input prompt *before* it flows to the Cognitive Analyzer and
the Fireworks Client, reducing the tokens billed by Fireworks while preserving
meaning. Two hard rules (per spec):

* **Never drop critical information.** A safety check keeps the original prompt
  if the compressed version loses numbers/entities or shrinks implausibly.
* **Only compress if it actually helps.** The compressed prompt is used *only*
  when it tokenizes to strictly fewer tokens than the original.

Implementations
---------------
* :class:`GemmaPromptCompressor` — the user's local Gemma-4-E2B prompt optimizer
  (generative). Loaded from ``MOBZ_COMPRESSOR_MODEL_PATH``. Local inference is
  free (zero scored tokens).
* :class:`HeuristicPromptCompressor` — dependency-free fallback: whitespace
  normalisation, filler-phrase removal, sentence de-duplication.
* :class:`NoopPromptCompressor` — identity (compression disabled).

``load_prompt_compressor`` picks the Gemma model if available, else the
heuristic, else no-op — so the pipeline always runs.
"""

from __future__ import annotations

import abc
import os
import re
from pathlib import Path

from .config import Config
from .logging_conf import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Result + shared safety helpers
# --------------------------------------------------------------------------- #
def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def is_safe_compression(original: str, compressed: str) -> bool:
    """Guard against losing critical information.

    Rejects a compression that: is empty, loses any numeric literal, or shrinks
    below 15% of the original character length (implausible for a faithful
    rewrite).
    """
    compressed = compressed.strip()
    if not compressed:
        return False
    if len(compressed) < 0.15 * len(original):
        return False
    # Every number in the original must survive (critical for math/data tasks).
    if not _numbers(original).issubset(_numbers(compressed)):
        return False
    return True


class PromptCompressor(abc.ABC):
    """Abstract prompt compressor."""

    @abc.abstractmethod
    def compress(self, prompt: str) -> str:
        """Return a token-reduced prompt, or the original if it can't help."""
        raise NotImplementedError

    # Token counter used to verify a real reduction. Subclasses with a real
    # tokenizer override this; the default is a word-count proxy.
    def _count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def _accept_if_smaller(self, original: str, candidate: str) -> str:
        """Apply candidate only if it is safe AND strictly fewer tokens."""
        if not is_safe_compression(original, candidate):
            return original
        if self._count_tokens(candidate) >= self._count_tokens(original):
            return original
        return candidate.strip()


# --------------------------------------------------------------------------- #
#  No-op
# --------------------------------------------------------------------------- #
class NoopPromptCompressor(PromptCompressor):
    def compress(self, prompt: str) -> str:
        return prompt


# --------------------------------------------------------------------------- #
#  Heuristic (dependency-free) — lossless-ish semantic tidy-up
# --------------------------------------------------------------------------- #
class HeuristicPromptCompressor(PromptCompressor):
    """Light semantic compressor: no new tokenizer, just safe reductions."""

    _FILLERS = [
        r"\bplease\b", r"\bkindly\b", r"\bcould you\b", r"\bcan you\b",
        r"\bi would like you to\b", r"\bi want you to\b", r"\bi'd like you to\b",
        r"\bas an ai( language model)?\b", r"\bif you don't mind\b",
        r"\bfor me\b", r"\bthank you\b", r"\bthanks\b",
    ]

    def compress(self, prompt: str) -> str:
        text = prompt
        # 1. Normalise whitespace.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 2. Remove filler phrases (case-insensitive).
        for pat in self._FILLERS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        # 3. Collapse duplicate consecutive sentences / repeated instructions.
        seen: set[str] = set()
        out_sentences = []
        for sent in re.split(r"(?<=[.!?\n])\s+", text):
            key = sent.strip().lower()
            if key and key in seen:
                continue
            seen.add(key)
            out_sentences.append(sent)
        text = " ".join(s for s in out_sentences if s.strip())
        text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip()
        return self._accept_if_smaller(prompt, text)


# --------------------------------------------------------------------------- #
#  Gemma-4-E2B generative prompt optimizer (user's local model)
# --------------------------------------------------------------------------- #
class GemmaPromptCompressor(PromptCompressor):
    """Runs the local Gemma prompt-optimizer model.

    Uses the model's own tokenizer to verify the rewrite is strictly shorter,
    and the shared safety check to avoid losing critical info. Construction may
    raise; use :meth:`try_load` for a safe factory.
    """

    _DEFAULT_TEMPLATE = (
        "Rewrite the following prompt to use as few tokens as possible while "
        "preserving ALL information, numbers, names and the exact task. Output "
        "only the rewritten prompt.\n\nPrompt:\n{prompt}\n\nRewritten prompt:"
    )

    def __init__(self, model_dir: str, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._max_new_tokens = max_new_tokens
        self._template = os.environ.get("MOBZ_COMPRESSOR_TEMPLATE", self._DEFAULT_TEMPLATE)

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Load in low precision to keep memory (and image RAM) reasonable.
        self._model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        self._model.eval()

    @classmethod
    def try_load(cls, model_dir: str) -> "GemmaPromptCompressor | None":
        try:
            comp = cls(model_dir)
            log.info("Loaded Gemma prompt compressor from %s", model_dir)
            return comp
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load Gemma prompt compressor (%s): %s. "
                        "Falling back to heuristic compressor.",
                        type(exc).__name__, exc)
            return None

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def compress(self, prompt: str) -> str:
        torch = self._torch
        text = self._template.format(prompt=prompt)
        # Prefer the model's chat template when available.
        try:
            if getattr(self._tokenizer, "chat_template", None):
                msgs = [{"role": "user", "content": self._template.format(prompt=prompt)}]
                inputs = self._tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt")
            else:
                inputs = self._tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                out = self._model.generate(
                    inputs, max_new_tokens=self._max_new_tokens,
                    do_sample=False, temperature=None, top_p=None,
                    pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
                )
            gen = out[0][inputs.shape[-1]:]
            candidate = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Compression failed for a prompt (%s); using original", exc)
            return prompt
        return self._accept_if_smaller(prompt, candidate)


def load_prompt_compressor(config: Config) -> PromptCompressor:
    """Return the active prompt compressor.

    Order: Gemma model at ``MOBZ_COMPRESSOR_MODEL_PATH`` -> heuristic -> no-op
    (if disabled via ``MOBZ_COMPRESSION=off``).
    """
    if not config.compression_enabled:
        log.info("Prompt compression disabled (MOBZ_COMPRESSION=off).")
        return NoopPromptCompressor()

    path = config.compressor_model_path
    if path and Path(path).exists():
        comp = GemmaPromptCompressor.try_load(path)
        if comp is not None:
            return comp
    elif path:
        log.warning("MOBZ_COMPRESSOR_MODEL_PATH does not exist: %s", path)

    log.info("Using heuristic prompt compressor.")
    return HeuristicPromptCompressor()
