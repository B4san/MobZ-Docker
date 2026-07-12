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
# Everything needed is created automatically on run (input tasks, temp input/
# output dirs, permissions). The ONLY external requirement is a .env (or exported
# env) with three values: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS.
#
# Usage:
#   scripts/judge_run.sh [IMAGE] [TASKS_JSON]
#     IMAGE       default: b4san/mobz:latest
#     TASKS_JSON  optional; if omitted, a built-in unseen task set is generated.
#
# Credentials/config are read from the environment (or a local .env):
#   FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
# ============================================================================
set -euo pipefail

IMAGE="${1:-b4san/mobz:latest}"
TASKS_JSON="${2:-}"   # optional; if empty, a default task set is generated below
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

# Tasks: use the file given as arg 2, or auto-generate a default unseen set.
if [[ -n "$TASKS_JSON" && -f "$TASKS_JSON" ]]; then
  cp "$TASKS_JSON" "$WORKDIR/input/tasks.json"
  TASKS_SRC="$TASKS_JSON"
else
  cat > "$WORKDIR/input/tasks.json" <<'TASKS'
[
  { "task_id": "j1", "prompt": "Is the sentiment of this tweet positive or negative? 'Absolutely thrilled with the new update, everything runs so much smoother now!'" },
  { "task_id": "j2", "prompt": "Compute 128 divided by 4, then multiply the result by 7. Give only the final number." },
  { "task_id": "j3", "prompt": "Write a Python function called count_vowels(s) that returns how many vowels are in the string s." },
  { "task_id": "j4", "prompt": "Return a JSON object with keys \"language\" and \"year\" for this fact: Python was created by Guido van Rossum and first released in 1991." },
  { "task_id": "j5", "prompt": "In one sentence, explain why the sky appears blue during the day." },
  { "task_id": "j6", "prompt": "Who painted the Mona Lisa?" },
  { "task_id": "j7", "prompt": "A shirt costs $40 and is discounted by 25%. What is the final price? Show the calculation briefly." },
  { "task_id": "j8", "prompt": "List three prime numbers between 10 and 20." },
  { "task_id": "j9", "prompt": "Translate into Spanish, one sentence only: I will call you tomorrow after the meeting." },
  { "task_id": "j10", "prompt": "Extract the person and the city as JSON with keys person and city: Ada Lovelace lived in London." }
]
TASKS
  TASKS_SRC="auto-generated (built-in default set)"
fi

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
echo "   tasks  : $TASKS_SRC ($NTASKS tasks)"
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
# Decode common non-zero exits for a clearer diagnosis.
EXIT_NOTE=""
if [[ "$EXIT_CODE" -eq 124 ]]; then
  EXIT_NOTE=" (TIMEOUT: exceeded ${BUDGET_SECONDS}s budget)"
elif [[ "$EXIT_CODE" -gt 128 ]]; then
  SIG=$(( EXIT_CODE - 128 ))
  case "$SIG" in
    4)  EXIT_NOTE=" (SIGILL: a native lib used an unsupported CPU instruction — likely gptqmodel/torchao kernels loading the GPTQ compressor on CPU; the process is killed hard so the heuristic fallback cannot catch it)";;
    9)  EXIT_NOTE=" (SIGKILL: killed — usually OOM)";;
    11) EXIT_NOTE=" (SIGSEGV: native segfault)";;
    6)  EXIT_NOTE=" (SIGABRT: native abort)";;
    *)  EXIT_NOTE=" (killed by signal $SIG)";;
  esac
fi
echo "      exit code: $EXIT_CODE${EXIT_NOTE} | elapsed: ${ELAPSED}s"

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
