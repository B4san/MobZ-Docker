#!/usr/bin/env bash
# ============================================================================
# judge_run.sh — Simulate the hackathon harness against a PUBLISHED image.
#
# Reproduces exactly what a judge does:
#   1. Pull the Docker image from the public registry.
#   2. Report its COMPRESSED size (the ≤10GB gate).
#   3. Mount /input (read-only) + /output and inject FIREWORKS_API_KEY,
#      FIREWORKS_BASE_URL, ALLOWED_MODELS — the harness contract.
#   4. Run the container (10-minute budget) and capture exit code + timing.
#   5. Validate /output/results.json (valid JSON, one entry per task, in order)
#      and print a per-task answer summary for the accuracy review.
#
# Usage:
#   scripts/judge_run.sh [IMAGE] [TASKS_JSON]
#     IMAGE       default: b4san/mobz:latest
#     TASKS_JSON  default: scripts/judge_tasks.json
#
# Credentials/config are read from the environment (or a local .env):
#   FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
# ============================================================================
set -euo pipefail

IMAGE="${1:-b4san/mobz:latest}"
TASKS_JSON="${2:-$(dirname "$0")/judge_tasks.json}"
BUDGET_SECONDS="${MOBZ_MAX_RUNTIME_SECONDS:-600}"

# --- Load .env for local runs (harness injects these itself) ----------------
if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  . ./.env; set +a
fi

: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (env or .env)}"
: "${FIREWORKS_BASE_URL:?set FIREWORKS_BASE_URL (env or .env)}"
: "${ALLOWED_MODELS:?set ALLOWED_MODELS (env or .env)}"

WORKDIR="$(mktemp -d -t mobz-judge-XXXXXX)"
mkdir -p "$WORKDIR/input" "$WORKDIR/output"
cp "$TASKS_JSON" "$WORKDIR/input/tasks.json"
# The container runs as a non-root user; make the mounted input world-readable
# and the output world-writable so that user can read tasks and write results.
chmod 755 "$WORKDIR"
chmod -R a+rX "$WORKDIR/input"
chmod 777 "$WORKDIR/output"
NTASKS="$(python3 -c "import json;print(len(json.load(open('$WORKDIR/input/tasks.json'))))")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "==================================================================="
echo " MobZ judge harness"
echo "   image  : $IMAGE"
echo "   tasks  : $TASKS_JSON ($NTASKS tasks)"
echo "   models : $ALLOWED_MODELS"
echo "==================================================================="

# --- 1. Pull ----------------------------------------------------------------
echo "[1/4] Pulling image ..."
docker pull "$IMAGE"

# --- 2. Compressed size gate (≤10GB) ----------------------------------------
echo "[2/4] Checking compressed image size (10GB limit) ..."
COMPRESSED_BYTES="$(docker manifest inspect "$IMAGE" 2>/dev/null \
  | python3 -c "import json,sys; m=json.load(sys.stdin); print(sum(l['size'] for l in m.get('layers',[])))" 2>/dev/null || echo 0)"
if [[ "$COMPRESSED_BYTES" -gt 0 ]]; then
  python3 - "$COMPRESSED_BYTES" <<'PY'
import sys
gb = int(sys.argv[1]) / 1e9
verdict = "PASS ✅" if gb <= 10 else "FAIL ❌ (> 10GB)"
print(f"      compressed size: {gb:.2f} GB  -> {verdict}")
PY
else
  echo "      (could not read manifest; skipping size check)"
fi

# --- 3. Run like the harness ------------------------------------------------
echo "[3/4] Running container (budget ${BUDGET_SECONDS}s) ..."
START="$(date +%s)"
set +e
timeout "$BUDGET_SECONDS" docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$WORKDIR/input:/input:ro" \
  -v "$WORKDIR/output:/output" \
  "$IMAGE"
EXIT_CODE=$?
set -e
ELAPSED=$(( $(date +%s) - START ))
echo "      exit code: $EXIT_CODE | elapsed: ${ELAPSED}s"

# --- 4. Validate & summarise outputs ----------------------------------------
echo "[4/4] Validating /output/results.json ..."
RESULTS="$WORKDIR/output/results.json" TASKS="$WORKDIR/input/tasks.json" \
EXIT_CODE="$EXIT_CODE" python3 <<'PY'
import json, os, sys
results_path, tasks_path = os.environ["RESULTS"], os.environ["TASKS"]
exit_code = int(os.environ["EXIT_CODE"])
ok = True
if exit_code != 0:
    print(f"      FAIL: container exited non-zero ({exit_code})"); ok = False
try:
    tasks = json.load(open(tasks_path))
    res = json.load(open(results_path))
except Exception as e:
    print(f"      FAIL: results.json missing or invalid JSON: {e}"); sys.exit(1)
if not isinstance(res, list):
    print("      FAIL: results is not a JSON array"); ok = False
tids_in = [t["task_id"] for t in tasks]
tids_out = [r.get("task_id") for r in res]
print(f"      valid JSON      : yes ({len(res)} entries)")
print(f"      one per task    : {'yes' if len(res)==len(tasks) else 'NO'}")
print(f"      order preserved : {'yes' if tids_out==tids_in else 'NO'}")
print(f"      schema (id+ans) : {'yes' if all('task_id' in r and 'answer' in r and isinstance(r.get('answer'),str) for r in res) else 'NO'}")
if len(res)!=len(tasks) or tids_out!=tids_in: ok = False
print("\n      --- answers (for accuracy review) ---")
for r in res:
    ans = (r.get("answer") or "").replace(chr(10), " ")
    print(f"      [{r.get('task_id')}] {ans[:90]}")
print("\n      VERDICT:", "PASS ✅" if ok else "FAIL ❌")
sys.exit(0 if ok else 1)
PY
