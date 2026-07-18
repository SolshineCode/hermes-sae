#!/bin/bash
# D_pilot: agent-trace SAE instrument run — additive wrapper around sae_labeled_course.py
# Booked slot: gpusched 3da0c561 (gpu=both, 2026-07-18 08:35 -> 12:35, -07:00)
# HARD-STOP: watchdog kills the python at 12:25 (10 min before slot end).
#
# Run location: $WORK (the working copy that has prompts/ + patched sae_labeled_course.py),
# so labeler_service.LS resolves prompts correctly (PROMPTS_DIR = $WORK/../prompts).
set -u
WORK=/tmp/course_run/experiments/v8_nla_local/labeled_outputs
cd "$WORK"
export HF_HOME=/tmp/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p runs
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$WORK/runs/d_pilot_agent_trace_$TS"
mkdir -p "$OUTDIR"
echo "[d_pilot] out=$OUTDIR config=d_pilot_config.yaml"
# hard-stop watchdog: kill the instrument at 12:25 (slot ends 12:35) — MUST kill, not release-after
( sleep $(( $(date -d '2026-07-18T12:25:00-07:00' +%s) - $(date +%s) )) ; \
  echo "[d_pilot][watchdog] HARD STOP at $(date) — killing instrument"; \
  pkill -f sae_labeled_course.py; exit 0 ) &
WATCHDOG_PID=$!
/home/darkstar/.venv-gemma4/bin/python sae_labeled_course.py \
  --config d_pilot_config.yaml \
  --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
  --layers 0,16,32,48,63 --dtype float16 \
  --out "$OUTDIR/sae_course.jsonl" \
  --max-new-tokens 1500 --max-rows 60 \
  --hf-cache /tmp/hf_cache --device cuda:0
RC=$?
kill $WATCHDOG_PID 2>/dev/null
echo "[d_pilot] instrument exited rc=$RC at $(date)"
echo "D_PILOT_DONE"
