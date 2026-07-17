#!/usr/bin/env bash
# Deferred same-inference SAE labeling course runner (user Option B) - HARDENED.
# Runs from isolated /tmp/course_run (extracted from commit 54221ed0, repo-correct
# layout so REPO_ROOT/PROMPTS_DIR/dataset paths resolve). Non-destructive to sibling tree.
#
# HARDENED vs prior versions:
#  - PRE-FLIGHT checks all deps before doing anything irreversible.
#  - DOES NOT release the slot on a launch/setup error (only on genuine completion).
#  - WATCHDOG: hard-stops the course python at SLOT_END minus margin, so it can NEVER
#    overrun the reservation and collide with the next holder (Claude Code).
#  - Records start time; if it cannot finish within the window, it stops and leaves
#    the slot for the next reservation.
set +e
RESV=2e9ea8a0
PY=/home/darkstar/.venv-gemma4/bin/python
WORK=/tmp/course_run/experiments/v8_nla_local/labeled_outputs
SCRIPT=$WORK/sae_labeled_course.py
OUTDIR=$WORK/runs
LOG=/tmp/run_sae_course_deferred.log
# Slot end (from gpusched reserve). Hard-stop MARGIN_MIN before it.
SLOT_END="2026-07-16T19:30:00-07:00"
MARGIN_MIN=20

echo "[wait] blocking until reservation $RESV starts ..." | tee -a "$LOG"
gpusched wait "$RESV" 2>&1 | tee -a "$LOG"

# --- PRE-FLIGHT ---
if [ ! -f "$SCRIPT" ]; then echo "[preflight] FAIL: $SCRIPT missing" | tee -a "$LOG"; exit 1; fi
if [ ! -f "$WORK/labeler_service.py" ] || [ ! -f "$WORK/labeler_config.yaml" ]; then
  echo "[preflight] FAIL: labeler_service.py or labeler_config.yaml missing" | tee -a "$LOG"; exit 1; fi
if [ ! -d "/tmp/course_run/experiments/v8_nla_local/prompts" ]; then
  echo "[preflight] FAIL: prompts dir missing" | tee -a "$LOG"; exit 1; fi
DS=$(/home/darkstar/.venv-gemma4/bin/python -c "import yaml;c=yaml.safe_load(open('$WORK/labeler_config.yaml'));print(c['run']['datasets'][0]['path'])")
if [ ! -f "$WORK/$DS" ]; then echo "[preflight] FAIL: dataset $WORK/$DS missing" | tee -a "$LOG"; exit 1; fi
for pf in $($PY -c "import yaml;c=yaml.safe_load(open('$WORK/labeler_config.yaml'));s=set();[s.update({v for k,v in pset.items() if k in ('labeler','labeler_a','labeler_b','auditor','instruction')}) for pset in c['prompt_sets'].values()];print(' '.join(sorted(s)))"); do
  if [ ! -f "/tmp/course_run/experiments/v8_nla_local/prompts/$pf" ]; then
    echo "[preflight] FAIL: prompt $pf missing" | tee -a "$LOG"; exit 1; fi
done
echo "[preflight] OK" | tee -a "$LOG"

# --- RAM gate ---
for i in $(seq 1 40); do
  FREE=$(free -g | awk '/Mem/{print $7}')
  if [ "$FREE" -ge 150 ]; then echo "[gate] attempt $i: RAM ${FREE}GB free -> proceed" | tee -a "$LOG"; break; fi
  echo "[gate] attempt $i: RAM ${FREE}GB (<150) -> wait 120s" | tee -a "$LOG"; sleep 120
done

OUT=$OUTDIR/sae_course_deferred_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
echo "OUT=$OUT" | tee -a "$LOG"

cd "$WORK"
# Hard OS-level timeout: kill the python at SLOT_END minus margin, no matter what.
# This is the ultimate backstop against the 07-15 collision (waiter watched wrong PID).
TIMEOUT_S=$(( ( $(date -d "$SLOT_END - ${MARGIN_MIN} minutes" +%s) - $(date +%s) ) ))
[ "$TIMEOUT_S" -lt 60 ] && TIMEOUT_S=60
HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null) \
  timeout -s TERM "$TIMEOUT_S" $PY "$SCRIPT" \
    --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
    --layers 0,16,32,48,63 \
    --model Qwen/Qwen3.5-27B \
    --dtype float16 \
    --hf-cache /tmp/hf_cache \
    --device cuda:0 \
    --max-new-tokens 2200 \
    --max-rows 5 \
    --out "$OUT/sae_course.jsonl" > "$OUT/run.log" 2>&1 &
COURSE_PID=$!
echo "[run] course pid=$COURSE_PID started $(date)" | tee -a "$LOG"

# --- WATCHDOG: hard-stop at SLOT_END - MARGIN_MIN so we NEVER overrun ---
# Bulletproof: (1) timeout kills the python by its REAL pid (no pipe, so $! is correct);
# (2) name-based pkill fallback in case the pid check misses (the bug that caused the 07-15 collision).
STOP_EPOCH=$(date -d "$SLOT_END - ${MARGIN_MIN} minutes" +%s 2>/dev/null)
echo "[watchdog] hard-stop epoch=$STOP_EPOCH (slot end $SLOT_END minus ${MARGIN_MIN}m)" | tee -a "$LOG"
while kill -0 $COURSE_PID 2>/dev/null; do
  NOW=$(date +%s)
  if [ -n "$STOP_EPOCH" ] && [ "$NOW" -ge "$STOP_EPOCH" ]; then
    echo "[watchdog] SLOT END NEAR - hard-stopping course (pid=$COURSE_PID + pkill)" | tee -a "$LOG"
    kill -TERM $COURSE_PID 2>/dev/null; sleep 5
    pkill -TERM -f sae_labeled_course 2>/dev/null; sleep 5
    kill -9 $COURSE_PID 2>/dev/null; pkill -9 -f sae_labeled_course 2>/dev/null
    break
  fi
  sleep 30
done
wait $COURSE_PID; RC=$?
echo "=== course exit $RC ===" | tee -a "$LOG"

if [ "$RC" -eq 0 ]; then
  echo "=== releasing reservation $RESV (genuine completion; not greedy) ===" | tee -a "$LOG"
  gpusched release "$RESV" 2>&1 || true
  echo "ALL_DONE" | tee -a "$LOG"
else
  echo "[WARN] course exited $RC (or watchdog-stopped) -- NOT releasing slot $RESV (leave held for next holder)" | tee -a "$LOG"
fi
