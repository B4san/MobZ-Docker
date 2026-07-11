# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------- #
# MobZ — model selector for Fireworks AI.
# Multi-stage, slim, Linux image. Compressed size stays far below the 10GB cap.
# ----------------------------------------------------------------------------- #

# ---- Stage 1: build wheels ------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip wheel --wheel-dir /wheels -r requirements.txt


# ---- Stage 2: runtime ------------------------------------------------------ #
FROM python:3.11-slim AS runtime

# Non-root user for safety.
RUN groupadd --system mobz && useradd --system --gid mobz --home /app mobz

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies from prebuilt wheels (no build toolchain in final image).
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# Application code and default placeholder data.
COPY src/ ./src/
COPY data/ ./data/
# The "gold" non-relational profile DB (auto-detected by load_profile_store).
COPY mobz_model_profiles.json ./

# The harness mounts these; create them so a non-root user can always write.
RUN mkdir -p /input /output \
 && chown -R mobz:mobz /app /input /output

USER mobz

# Read tasks from /input/tasks.json, write /output/results.json, then exit.
ENTRYPOINT ["python", "-m", "mobz"]
