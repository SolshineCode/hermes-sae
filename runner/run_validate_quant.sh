#!/usr/bin/env bash
# B: quant-vs-unquant sameness validation runner. Fires at booking 56a2d647 (gpu=1, 21:35).
# Compares 4-bit Qwen3.5-27B SAE features vs the fp16 row-0 already on disk.
set +e
RESV=56a2d647
PY=/home/darkstar/.venv-gemma4/bin/python
WORK=/tmp/hermes-sae
REF=/tmp/course_run/experiments/v8_nla_local/labeled_outputs/runs/sae_course_deferred_20260716_033020/sae_course.jsonl
LOG=/tmp/validate_quant.log
SLOT_END="2026-07-17T01:35:00-07:00"
MARGIN_MIN=15

echo "[wait] blocking until reservation $RESV (gpu=1) starts ..." | tee -a "$LOG"
gpusched wait "$RESV" 2>&1 | tee -a "$LOG"

cd "$WORK"
# Hard OS timeout backstop at slot end minus margin
TIMEOUT_S=$(( ( $(date -d "$SLOT_END - ${MARGIN_MIN} minutes" +%s) - $(date +%s) ) ))
[ "$TIMEOUT_S" -lt 60 ] && TIMEOUT_S=60
echo "[run] timeout=${TIMEOUT_S}s, loading int4 Qwen3.5-27B on gpu=1" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=1 timeout -s TERM "$TIMEOUT_S" $PY validate_quant_sameness.py \
  --ref-jsonl "$REF" --row-idx 0 \
  --model Qwen/Qwen3.5-27B --dtype float16 --quant int4 \
  --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
  --layers 0,16,32,48,63 --hf-cache /tmp/hf_cache --device cuda:0 \
  --max-new-tokens 2200 > "$WORK/validate_run.log" 2>&1
RC=$?
echo "=== quant validation exit $RC ===" | tee -a "$LOG"
# release only on clean completion (not on overrun/timeout)
if [ "$RC" -eq 0 ]; then
  gpusched release "$RESV" 2>&1 || true
  echo "[done] report -> $WORK/quant_sameness_report.json" | tee -a "$LOG"
else
  echo "[warn] non-zero exit $RC; NOT releasing (investigate)" | tee -a "$LOG"
fi
