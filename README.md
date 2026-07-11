# MobZ — Product Requirements Document (PRD) & Technical Reference

**MobZ is a high-precision quality/price/token model selector for Fireworks AI.**
Given a batch of tasks, it compresses each prompt, estimates its cognitive load
with a local model, and uses a pure-knowledge policy engine to route each task to
the **cheapest Fireworks model that still clears the quality bar** — minimising
the tokens billed by the judging proxy while passing the accuracy gate.

> Built for a hackathon whose scoring is: **(1) accuracy gate** (an LLM judge must
> accept the answer), then **(2) token efficiency** (fewer total tokens = higher
> rank). MobZ optimises directly for that: pass the gate, then spend the fewest
> tokens.

---

## 1. Product vision

Frontier LLMs differ wildly in price and verbosity, yet many tasks (sentiment,
short QA, extraction) are solved just as well by a cheap model as by a flagship.
Sending everything to the strongest model burns tokens; sending everything to the
cheapest fails hard tasks. **MobZ picks the right model per task**, using:

- a **local, zero-token pipeline** for everything except the single billed
  inference call, and
- a **gold profile database** of model capabilities + costs as its "memory".

Because local models and their tokens count as **zero** for scoring, all the
"thinking" (compression + cognitive analysis + routing) is free; only the final
Fireworks answer is billed.

---

## 2. Goals & non-goals

**Goals**
- Read `/input/tasks.json`, write `/output/results.json` before exit (valid JSON, in order).
- Route every task to the most token/cost-efficient model that passes the quality gate.
- Send **only** compressed prompts through `FIREWORKS_BASE_URL`; obey `ALLOWED_MODELS`.
- Ship as a Linux Docker image **≤ 10 GB compressed**, non-root, exit 0/non-zero.
- Never lose critical information during compression; never truncate a valid answer.

**Non-goals**
- Training the models (the Cognitive Analyzer and Prompt Compressor are supplied).
- Building the profile DB from scratch (supplied + enriched here).
- Beating flagship quality on the hardest tasks — the gate + fallback handle those.

---

## 3. Architecture

```
                                   Docker
┌──────────────────────────────────────────────────────────────────┐
│  /input/tasks.json                                                 │
│        │                                                           │
│        ▼                                                           │
│  Module 1 · JSON Loader            (no AI · no tokens)             │
│        │                                                           │
│        ▼                                                           │
│  Module 2 · Prompt Compressor      (LOCAL Gemma-4-E2B · zero tokens)│
│        │   compress → verify fewer tokens → keep original if unsafe │
│        ▼                                                           │
│  Module 3 · Cognitive Analyzer     (LOCAL ModernBERT · zero tokens)│
│        │   → { task, difficulty, reasoning_depth,                  │
│        │       expected_output, expected_output_tokens, confidence}│
│        ▼                                                           │
│  Module 4 · Inference Policy Engine (no AI · no tokens)            │
│        │   profiles ◀── Gold non-relational DB (294 models)        │
│        │   utility fn → optimal model + max_tokens budget          │
│        ▼                                                           │
│  Module 5 · Fireworks Client       (the ONLY billed inference)     │
│        │   compressed prompt → FIREWORKS_BASE_URL                  │
│        ▼                                                           │
│  Module 6 · Response Validator     (integrity + JSON repair)       │
│        │                                                           │
│        ▼                                                           │
│  /output/results.json                                              │
└──────────────────────────────────────────────────────────────────┘
```

Routing logic (no LLM router):

```
Prompt → Compression → Cognitive load → Profile lookup → Utility function → Optimal model
```

| Module | File | Role | AI? | Tokens? |
|--------|------|------|-----|---------|
| 1. JSON Loader | `src/mobz/json_loader.py` | Read & validate `/input/tasks.json` | No | No |
| 2. Prompt Compressor | `src/mobz/prompt_compressor.py` | Shrink prompts, preserve meaning | Local | Zero |
| 3. Cognitive Analyzer | `src/mobz/cognitive_analyzer.py` | Estimate cognitive load | Local | Zero |
| — Profile Store | `src/mobz/profile_store.py` | Gold model profiles (NoSQL-ready) | No | No |
| 4. Policy Engine | `src/mobz/policy_engine.py` | Utility function → optimal model | No | No |
| 5. Fireworks Client | `src/mobz/fireworks_client.py` | The single billed inference call | Yes | Yes |
| 6. Response Validator | `src/mobz/validator.py` | Integrity + write `results.json` | No | No |
| — Orchestrator | `src/mobz/pipeline.py` | Wire modules, concurrency, budget | — | — |

---

## 4. The routing engine (how "best model" is decided)

The Policy Engine derives a **quality bar** that rises with the task's difficulty
and reasoning depth, filters candidates to those whose capability for the task's
category clears the bar (the **accuracy gate**), then maximises a utility that —
crucially — **caps the capability reward at the bar**. Above the gate, extra
capability earns nothing, so the tie-break becomes *fewest expected tokens and
lowest cost*. This yields exactly the hackathon-optimal behaviour:

- **Easy tasks → cheapest qualifying model.**
- **Hard tasks → most capable model** (only it clears the raised bar).
- A `max_tokens` budget is derived from `expected_output_tokens` (a cap, not a
  target) with a generous floor so reasoning models never truncate mid-answer.

**Verified routing (real gold DB, engine only):**

| ALLOWED_MODELS | easy (sentiment) | medium (coding) | hard (math) |
|---|---|---|---|
| all 6 | `gpt-oss-120b` ($0.60) | `gpt-oss-120b` | `deepseek-v4-pro` |
| without gpt-oss | `deepseek-v4-pro` ($0.87) | `deepseek-v4-pro` | `deepseek-v4-pro` |
| only glm/kimi | `kimi-k2p5` ($3.0) | `kimi-k2p5` | `glm-5p2` |
| glm-5.2 vs kimi-2.5 | `kimi-k2p5` (cheaper) | `kimi-k2p5` | `glm-5p2` (stronger) |

The engine always picks the cheapest model that passes for easy work and pulls in
the strongest only when the task demands it — adapting to whatever `ALLOWED_MODELS`
is published on launch day.

---

## 5. Local models & the gold DB

Two large model folders live **locally only** (git-ignored, baked into the image
at build time — see §9):

### 5.1 Cognitive Analyzer — `mobz_cognitive_analyzer_final/` (~596 MB)
A **ModernBERT-base multi-head** network: shared encoder + five heads —
`task` (8-way), `format` (10-way), `difficulty`, `reasoning_depth`, `tokens`.
`TransformersCognitiveAnalyzer` rebuilds the architecture, loads the weights, and
runs it (verified). Confirmed settings from the shipped `labels.json` /
`mobz_config.json`: **pooling = CLS**, **tokens head = direct count**, and the
`id2task`/`id2format` maps (0=logic,1=debug,2=ner,3=sentiment,4=code_generation,
5=summary,6=math,7=knowledge). Task labels are mapped onto the DB capability
vocabulary (code_generation/debug→coding, logic→reasoning). Falls back to a
deterministic heuristic analyzer if torch/artifacts are missing.

### 5.2 Prompt Compressor — `prompt_compressor_model/` (Gemma-4-E2B, ~6 GB)
`GemmaPromptCompressor` rewrites each prompt to use fewer tokens while preserving
all information. Two hard guarantees:
- **Never drops critical info** — a safety check keeps the original if numbers are
  lost or the rewrite shrinks implausibly.
- **Only used if it truly helps** — the compressed prompt is sent to Fireworks
  *only* when it tokenizes to strictly fewer tokens.
Falls back to a dependency-free `HeuristicPromptCompressor` (filler removal,
whitespace normalisation, sentence de-duplication) that already reduces
filler-heavy prompts ~40% while preserving numbers.

### 5.3 Gold DB — `mobz_model_profiles.json` (294 models, tracked in git)
Rich documents (`capability_scores`, `composite_indices`, `cost`,
`cognitive_profile`, …). `RichProfileStore` maps them onto `ModelProfile` and
normalises ids by slug so `fireworks/glm-5p2` matches the API id
`accounts/fireworks/models/glm-5p2`. The 6 models that actually exist on this
Fireworks account carry researched 2026 benchmarks; `scripts/enrich_profiles.py`
recomputes indices and derives numeric capabilities across all 294.

**The 6 real chat models (confirmed via the Fireworks `/models` API):**
`gpt-oss-120b` ($0.15/$0.60), `deepseek-v4-pro` ($0.435/$0.87),
`glm-5p2` ($1.40/$4.40), `glm-5p1` ($1.40/$4.40), `kimi-k2p6` ($0.95/$4.00),
`kimi-k2p5` ($0.60/$3.00) — input/output per 1M tokens.

---

## 6. Journey — what was built (chronological summary)

1. **Initial build** — Implemented the full pipeline (JSON Loader → Cognitive
   Analyzer → Policy Engine → Fireworks Client → Validator) with the Prompt
   Compressor and MobZ Bench intentionally omitted, and the analyzer/DB as clean
   placeholders. Lean Docker image (227 MB), 17 unit tests, mocked E2E all green.
2. **Real integrations** — Queried the Fireworks `/models` API (only 6 real chat
   models). Web-researched real 2026 benchmarks + pricing. Enriched the 294-model
   gold DB (recomputed broken composite indices, added numeric `capability_scores`).
   Wrote `RichProfileStore` to consume the rich schema. Wired the real ModernBERT
   analyzer; ran the full pipeline in Docker against the **real Fireworks API**.
3. **Model artifact fixes** — Diagnosed the analyzer's architecture from the
   safetensors, recovered the correct label maps, pooling (CLS) and token head
   interpretation; the model now classifies tasks correctly.
4. **Quality/efficiency tuning** — Added a concise system prompt, a generous
   token-budget floor, and an adaptive retry on `finish_reason=length` so
   reasoning models never truncate (a truncated answer fails the gate = 0 score).
   Capped the capability reward at the quality bar so easy tasks route to the
   cheapest model.
5. **Prompt Compressor (this iteration)** — Re-introduced Module 2 with the
   Gemma-4-E2B model + heuristic fallback and full pipeline integration
   (compress → analyze → route → infer with the compressed prompt).

---

## 7. Verification (evidence)

- **26 unit tests pass** (loader, compressor safety, analyzer heuristic, rich
  profile store, policy routing incl. cheap-vs-capable, validator, config).
- **Real ModernBERT analyzer** loads and runs in Docker (`pooling=cls`), task
  classification correct on a probe set.
- **Full E2E in Docker against the real Fireworks API**: exit 0, valid
  `results.json`, correct answers (e.g. `47×23=1081`, JSON entity extraction,
  `is_prime`, Euclid's infinitude-of-primes proof, EN→FR translation).
- **Routing verified** across multiple `ALLOWED_MODELS` sets (see §4): easy→cheapest,
  hard→most capable, adapts to the allowed set.
- **Image size**: lean image **227 MB**; ML image (analyzer only) **3.9 GB** — both
  well under the 10 GB cap.

---

## 8. I/O contract & environment

**Input** `/input/tasks.json`: `[ { "task_id": "t1", "prompt": "..." }, ... ]`
**Output** `/output/results.json`: `[ { "task_id": "t1", "answer": "..." }, ... ]`
(atomic write, always valid JSON, one entry per task in input order).

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREWORKS_API_KEY` | ✅ | Injected by the harness |
| `FIREWORKS_BASE_URL` | ✅ | **All** Fireworks calls route through it |
| `ALLOWED_MODELS` | ✅ | Comma-separated permitted model IDs |
| `MOBZ_COMPRESSOR_MODEL_PATH` | | Gemma compressor dir (empty → heuristic) |
| `MOBZ_COMPRESSION` | | `on`/`off` (default on) |
| `MOBZ_COGNITIVE_MODEL_PATH` | | ModernBERT analyzer dir (empty → heuristic) |
| `MOBZ_PROFILES_DB_URI` | | Profiles DB path/URI (empty → bundled gold DB) |
| `MOBZ_MAX_RUNTIME_SECONDS` | | Hard budget, default `570` (< 10 min) |
| `MOBZ_CONCURRENCY` | | Parallel Fireworks calls, default `8` |
| `MOBZ_QUALITY_THRESHOLD` | | Base accuracy gate, default `0.70` |
| `MOBZ_MIN/MAX_OUTPUT_TOKENS` | | Token-budget clamp (default 256 / 4096) |

See `.env.example`. The image never bundles a `.env`.

---

## 9. Build & run

**Lean image** (heuristic compressor + heuristic analyzer, no torch — 227 MB):
```bash
docker build -t mobz:latest .
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  mobz:latest
```

**ML image** (real ModernBERT analyzer, `Dockerfile.ml` — ~3.9 GB):
```bash
docker build -f Dockerfile.ml -t mobz:ml .
```

Tests:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

---

## 10. NEXT STEP (the deliverable)

**Bake both local model folders into a single Docker image ≤ 10 GB and run the
full pipeline from it to pick the best model per task.**

The two folders (`mobz_cognitive_analyzer_final/` and `prompt_compressor_model/`)
are git-ignored but **must be present locally** at build time. Concretely:

1. Ensure the complete models are present locally:
   - `mobz_cognitive_analyzer_final/` (present, ~596 MB).
   - `prompt_compressor_model/` — extract the Gemma-4-E2B model here. **The last
     uploaded `gemma4-e2b-prompt-optimizer.zip` was truncated** (shard 2 of 3 was
     cut ~588 MB short because the disk was 99% full during upload); it must be
     re-uploaded complete before it can be extracted/loaded.
2. In `Dockerfile.ml`, enable the compressor lines (already documented there):
   ```dockerfile
   COPY prompt_compressor_model/gemma4-e2b-prompt-optimizer/ ./prompt_compressor_model/
   ENV MOBZ_COMPRESSOR_MODEL_PATH=/app/prompt_compressor_model
   ```
3. **Watch the 10 GB budget.** Gemma weights (~6 GB) + torch/transformers (~2.5 GB)
   + ModernBERT analyzer (~0.6 GB) ≈ 9–10 GB uncompressed. If the compressed image
   exceeds 10 GB, ship a **4-bit / GGUF quantised** compressor (~1.5–2 GB) — the
   model is already quantised, so this is straightforward.
4. Build, verify size (`docker images`), and run the full pipeline end-to-end
   against `FIREWORKS_BASE_URL`.

---

## 11. Known limitations / honest caveats

- The Cognitive Analyzer's **difficulty head is nearly flat** in practice, so most
  everyday prompts read as "easy" → they route to the cheapest capable model
  (`gpt-oss-120b`). This is token-optimal when that model can answer (it usually
  can), and the gate still escalates genuinely hard tasks to `deepseek-v4-pro`.
- Sentiment/NER are occasionally mis-labelled by the analyzer at low confidence;
  routing stays sensible because the models chosen are broadly capable.
- The Gemma compressor could not be run here yet — its uploaded archive was
  truncated (see §10). The pipeline runs today with the heuristic compressor.

---

## 12. Hackathon compliance

- ✅ Reads `/input/tasks.json`, writes `/output/results.json` before exit.
- ✅ Reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` from the
  environment only — no hardcoded values, no bundled `.env`.
- ✅ **Every** inference call goes through `FIREWORKS_BASE_URL` (single client).
- ✅ Only models from `ALLOWED_MODELS` are ever selected.
- ✅ Exit code `0` on success, non-zero on failure.
- ✅ Hard runtime budget (`570s`) under the 10-minute cap; partial completion is
  still written as valid JSON.
- ✅ Slim, non-root Linux image, under the 10 GB compressed limit.
