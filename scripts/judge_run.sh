#!/usr/bin/env bash
# ============================================================================
# judge_run.sh — Track 1 self-check harness (AMD Hackathon: Token-Efficient
# Routing Agent).
#
# Reproduces how the judges evaluate a PUBLISHED image, using the Track 1 PUBLIC
# validation examples (retired scoring cases) as the input tasks:
#   1. Pull the image from the public registry.
#   2. Report its COMPRESSED size (the ≤10GB gate).
#   3. Run the container exactly like the harness (mount /input ro + /output,
#      inject FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS), 10-min budget.
#   4. Persist /output/results.json to the LOCAL directory (./judge_run/output/
#      and a copy at ./results.json) so the answers are inspectable.
#   5. Self-check like the judge FAQ: valid JSON, one result per task, task IDs
#      preserved exactly, schema, no skipped/empty answers, runtime under limit,
#      plus a per-task correctness rubric derived from the published Expected
#      criteria (automated approximation — final judging uses an LLM judge).
#
# Everything is created automatically. The ONLY external requirement is a .env
# (or exported env) with three values: FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
# ALLOWED_MODELS.
#
# Usage:  scripts/judge_run.sh [IMAGE]
#           IMAGE  default: b4san/mobz:latest
# ============================================================================
set -euo pipefail

IMAGE="${1:-b4san/mobz:latest}"
BUDGET_SECONDS="${MOBZ_MAX_RUNTIME_SECONDS:-600}"
HERE="$(pwd)"
RUN_DIR="$HERE/judge_run"
IN_DIR="$RUN_DIR/input"
OUT_DIR="$RUN_DIR/output"
LOG="$RUN_DIR/container.log"
RESULTS="$OUT_DIR/results.json"
LOCAL_COPY="$HERE/results.json"

# --- Config from .env (harness injects these itself in real judging) --------
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (env or .env)}"
: "${FIREWORKS_BASE_URL:?set FIREWORKS_BASE_URL (env or .env)}"
: "${ALLOWED_MODELS:?set ALLOWED_MODELS (env or .env)}"

# --- Fresh, PERSISTENT local work dirs (NOT cleaned up) ---------------------
rm -rf "$RUN_DIR"
mkdir -p "$IN_DIR" "$OUT_DIR"

# --- Track 1 PUBLIC validation tasks (verbatim prompts, exact task IDs) -----
cat > "$IN_DIR/tasks.json" <<'TASKS'
[
  { "task_id": "T01",  "prompt": "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB." },
  { "task_id": "T01b", "prompt": "What is the difference between machine learning and deep learning? Briefly explain how each works." },
  { "task_id": "T01c", "prompt": "Explain the difference between RAM and ROM in a computer. What is each type used for?" },
  { "task_id": "T02",  "prompt": "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?" },
  { "task_id": "T02b", "prompt": "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?" },
  { "task_id": "T03",  "prompt": "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'" },
  { "task_id": "T03b", "prompt": "Classify the sentiment of this tweet as Positive, Negative, or Neutral and give a one-sentence reason: 'Just got my order. Box was dented and the manual was missing, but honestly the device itself is flawless and set up in under 5 minutes.'" },
  { "task_id": "T04",  "prompt": "Summarize the following passage in exactly two sentences:\n\n'Machine learning is increasingly deployed in healthcare for diagnosis, treatment planning, and patient monitoring. These systems analyse medical images, predict patient deterioration, and spot patterns in electronic health records that might be missed by human clinicians. However, concerns remain about model interpretability, data privacy, liability when errors occur, and the potential for algorithmic bias to worsen existing healthcare disparities. Regulatory frameworks are still catching up with the pace of deployment, creating uncertainty for healthcare providers and technology developers alike.'" },
  { "task_id": "T04b", "prompt": "Summarize the following passage in exactly three bullet points, each no longer than 15 words:\n\n'Remote work has transformed how companies operate globally. Employees gain flexibility and reduced commute times, leading to reported improvements in work-life balance. However, challenges persist around collaboration, company culture, and the blurring of personal and professional boundaries. Organisations are responding by investing in digital collaboration tools and rethinking office space as a hub for social and creative work rather than daily attendance.'" },
  { "task_id": "T05",  "prompt": "Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE:\n\n'On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety.'" }
]
TASKS

# Container runs as a non-root user: input readable, output writable.
chmod 755 "$RUN_DIR"; chmod -R a+rX "$IN_DIR"; chmod 777 "$OUT_DIR"
NTASKS="$(python3 -c "import json;print(len(json.load(open('$IN_DIR/tasks.json'))))")"

echo "==================================================================="
echo " MobZ — Track 1 judge self-check"
echo "   image  : $IMAGE"
echo "   tasks  : Track 1 public validation set ($NTASKS tasks)"
echo "   output : $RESULTS  (+ copy at $LOCAL_COPY)"
echo "==================================================================="

# --- 1. Pull ----------------------------------------------------------------
echo "[1/5] Pulling image ..."
docker pull "$IMAGE"

# --- 2. Compressed size gate (≤10GB) ----------------------------------------
echo "[2/5] Checking compressed image size (10GB limit) ..."
CB="$(docker manifest inspect "$IMAGE" 2>/dev/null \
  | python3 -c "import json,sys; m=json.load(sys.stdin); print(sum(l['size'] for l in m.get('layers',[])))" 2>/dev/null || echo 0)"
[[ "$CB" -gt 0 ]] && python3 - "$CB" <<'PY'
import sys; gb=int(sys.argv[1])/1e9
print(f"      compressed size: {gb:.2f} GB  -> {'PASS ✅' if gb<=10 else 'FAIL ❌ (>10GB)'}")
PY

# --- 3. Run like the harness ------------------------------------------------
echo "[3/5] Running container (budget ${BUDGET_SECONDS}s) ..."
START="$(date +%s)"
set +e
timeout "$BUDGET_SECONDS" docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$IN_DIR:/input:ro" \
  -v "$OUT_DIR:/output" \
  "$IMAGE" >"$LOG" 2>&1
EXIT_CODE=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

# Decode common non-zero exits.
NOTE=""
if [[ "$EXIT_CODE" -eq 124 ]]; then NOTE=" (TIMEOUT > ${BUDGET_SECONDS}s)"
elif [[ "$EXIT_CODE" -gt 128 ]]; then
  SIG=$(( EXIT_CODE - 128 ))
  case "$SIG" in
    4) NOTE=" (SIGILL: native lib used an unsupported CPU instruction — e.g. gptqmodel/torchao loading the GPTQ compressor on CPU; kills the process so the fallback can't catch it)";;
    9) NOTE=" (SIGKILL: usually OOM)";; 11) NOTE=" (SIGSEGV)";; 6) NOTE=" (SIGABRT)";;
    *) NOTE=" (killed by signal $SIG)";;
  esac
fi
echo "      exit code: $EXIT_CODE${NOTE} | elapsed: ${ELAPSED}s (limit ${BUDGET_SECONDS}s)"
echo "      --- last container log lines ---"
tail -n 6 "$LOG" | sed 's/^/      | /'

# Surface routing + token metrics (Track 1: efficiency after correctness).
echo "      --- routing & tokens ---"
grep -oE "Task [A-Za-z0-9]+ routed to [^ ]+" "$LOG" | sed 's/^/      | /' || true
grep -oE "in=[0-9]+ out=[0-9]+ total=[0-9]+" "$LOG" | tail -1 | sed 's/^/      | tokens (billed by proxy): /' || true

# Persist a copy in the current directory.
[[ -f "$RESULTS" ]] && cp "$RESULTS" "$LOCAL_COPY"

# --- 4/5. Self-check + accuracy rubric --------------------------------------
echo "[4/5] Structural self-check (judge FAQ) ..."
RESULTS="$RESULTS" TASKS="$IN_DIR/tasks.json" EXIT_CODE="$EXIT_CODE" \
ELAPSED="$ELAPSED" BUDGET="$BUDGET_SECONDS" python3 <<'PY'
import json, os, re, sys

results_path, tasks_path = os.environ["RESULTS"], os.environ["TASKS"]
exit_code = int(os.environ["EXIT_CODE"]); elapsed=int(os.environ["ELAPSED"]); budget=int(os.environ["BUDGET"])
tasks = json.load(open(tasks_path))
tids = [t["task_id"] for t in tasks]

def fail(msg): print(f"      ❌ {msg}")
def ok(msg):   print(f"      ✅ {msg}")

structural_ok = True
if exit_code != 0:
    fail(f"RUNTIME: container exited non-zero ({exit_code})"); structural_ok = False
else:
    ok("container exited 0 (clean run)")
if elapsed <= budget: ok(f"runtime under limit ({elapsed}s ≤ {budget}s)")
else: fail(f"TIMEOUT ({elapsed}s > {budget}s)"); structural_ok = False

try:
    res = json.load(open(results_path))
except Exception as e:
    fail(f"OUTPUT_MISSING / INVALID_RESULTS_SCHEMA: {e}")
    print("\n      VERDICT: FAIL ❌ (no valid output)"); sys.exit(1)

ok("results.json is valid JSON")
if not isinstance(res, list): fail("results is not a JSON array"); structural_ok=False
out_ids = [r.get("task_id") for r in res]
if len(res)==len(tasks): ok(f"one result per task ({len(res)}/{len(tasks)})")
else: fail(f"MISSING_TASKS: got {len(res)} of {len(tasks)}"); structural_ok=False
if out_ids==tids: ok("task IDs preserved exactly and in order")
else: fail(f"task IDs mismatch: {out_ids} vs {tids}"); structural_ok=False
schema_ok = all(isinstance(r,dict) and isinstance(r.get("task_id"),str) and isinstance(r.get("answer"),str) for r in res)
ok("schema: each item has string task_id + answer") if schema_ok else fail("INVALID_RESULTS_SCHEMA: task_id/answer types")
if not schema_ok: structural_ok=False
ans = {r.get("task_id"): (r.get("answer") or "") for r in res if isinstance(r,dict)}
empties = [t for t in tids if not ans.get(t,"").strip() or ans.get(t,"").strip().upper() in ("N/A","NA","NONE")]
if empties: fail(f"empty/placeholder answers (accuracy gate risk): {empties}")
else: ok("no empty/placeholder answers")

# ---------- Accuracy rubric (approx of the published Expected criteria) ----------
def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
def bullets(t):
    return [l for l in t.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+", l)]
def has(t, *ws):
    tl=_norm(t); return all(w in tl for w in ws)
def any_(t, *ws):
    tl=_norm(t); return any(w in tl for w in ws)
def _norm(t):
    # normalise unicode hyphens (‑ – —) to '-' so matches are robust
    return t.lower().replace("\u2011","-").replace("\u2013","-").replace("\u2014","-")

def r_T01(a):  return has(a,"red","green","blue") and any_(a,"additive","emit","light")
def r_T01b(a): return any_(a,"subset","sub-field","subfield","sub field","branch","type of") and "neural" in _norm(a) and any_(a,"feature","representation")
def r_T01c(a): return "volatile" in a.lower() and any_(a,"firmware","bios") and "ram" in a.lower() and "rom" in a.lower()
def r_T02(a):  return ("1672" in a) or ("1,672" in a)
def r_T02b(a): return any_(a,"1.875","1.87","1.88") and any_(a,"4.50","4.5 ","$4.50","4,50")
def _mixedlabel(a):
    tl=a.lower()
    # must not be classified purely Negative
    label_pos = any_(a,"positive","neutral","mixed")
    only_neg = ("negative" in tl) and not label_pos
    return label_pos and not only_neg
def r_T03(a):  return _mixedlabel(a) and any_(a,"late","damag") and any_(a,"work","support","resolv","hour")
def r_T03b(a): return _mixedlabel(a) and any_(a,"dent","missing","manual") and any_(a,"flawless","set up","setup","minute")
def r_T04(a):
    s=sentences(a); opp=any_(a,"image","predict","pattern","diagnos","monitor","clinical")
    ch=any_(a,"interpretab","privacy","bias","liabilit","regulat")
    return len(s)==2 and opp and ch
def r_T04b(a):
    b=bullets(a)
    if len(b)!=3: return False
    if any(len(re.sub(r"^\s*([-*•]|\d+[.)])\s+","",l).split())>15 for l in b): return False
    return any_(a,"flexib","work-life","balance") and any_(a,"collaborat","culture","boundar") and any_(a,"tool","office","digital")
def r_T05(a):
    tl=a.lower()
    ents = all(e in tl for e in ["sundar pichai","google","zurich","eth zurich"]) and bool(re.search(r"march\s*15,?\s*2023", tl))
    labels = all(l in a.upper() for l in ["PERSON","ORGANIZATION","LOCATION","DATE"])
    return ents and labels

RUBRIC={"T01":r_T01,"T01b":r_T01b,"T01c":r_T01c,"T02":r_T02,"T02b":r_T02b,
        "T03":r_T03,"T03b":r_T03b,"T04":r_T04,"T04b":r_T04b,"T05":r_T05}

print("\n[5/5] Accuracy rubric (automated approximation of Expected criteria):")
passed=0
for t in tids:
    a=ans.get(t,"")
    try: good=RUBRIC[t](a)
    except Exception: good=False
    passed+=good
    print(f"      [{t:4}] {'PASS' if good else 'CHECK'}  {a.strip()[:110].replace(chr(10),' ')}")
print(f"\n      rubric: {passed}/{len(tids)} likely-correct (final = LLM judge)")
verdict = "PASS ✅" if (structural_ok and not empties) else "FAIL ❌"
print(f"      structure/reliability: {verdict}")
sys.exit(0)
PY

echo
echo "Output persisted at:"
echo "  $RESULTS"
echo "  $LOCAL_COPY   (copy in current directory)"
echo "Full container log: $LOG"
