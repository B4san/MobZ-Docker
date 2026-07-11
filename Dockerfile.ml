# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------- #
# MobZ — ML image that runs the REAL local Cognitive Analyzer (ModernBERT
# multi-head). Heavier than the lean image but still well under the 10GB cap.
# Local inference counts as ZERO tokens for scoring.
#
# Build:  docker build -f Dockerfile.ml -t mobz:ml .
# ----------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

RUN groupadd --system mobz && useradd --system --gid mobz --home /app mobz

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MOBZ_COGNITIVE_MODEL_PATH=/app/mobz_cognitive_analyzer_final

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt -r requirements-ml.txt

# Pre-cache the ModernBERT-base config + tokenizer so the analyzer can rebuild
# the encoder fully offline at runtime (no base weights are downloaded — we load
# the trained weights from model.safetensors via AutoModel.from_config).
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from transformers import AutoConfig, AutoTokenizer; \
    AutoConfig.from_pretrained('answerdotai/ModernBERT-base'); \
    AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')" \
 && mkdir -p /app/.hf

COPY src/ ./src/
COPY data/ ./data/
COPY mobz_model_profiles.json ./
COPY mobz_cognitive_analyzer_final/ ./mobz_cognitive_analyzer_final/

# --- Prompt Compressor (Module 2), OPTIONAL ---------------------------------
# Enable once the Gemma-4-E2B model is extracted to prompt_compressor_model/.
# WARNING: the Gemma weights are ~6GB; verify the final COMPRESSED image stays
# under 10GB (consider a 4-bit/GGUF quant if it doesn't). Then uncomment:
#   COPY prompt_compressor_model/gemma4-e2b-prompt-optimizer/ ./prompt_compressor_model/
# and set the env below.
# ENV MOBZ_COMPRESSOR_MODEL_PATH=/app/prompt_compressor_model
# ----------------------------------------------------------------------------

RUN mkdir -p /input /output && chown -R mobz:mobz /app /input /output
USER mobz

ENTRYPOINT ["python", "-m", "mobz"]
