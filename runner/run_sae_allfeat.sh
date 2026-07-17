#!/usr/bin/env bash
# ALL-features SAE populate (Goodfire parity). Booking 3e5a9c36 (gpu=both, 07-17 01:40).
# Runs the PATCHED sae_labeled_course.py (captures full 81920-width allf_a/b/aud).
# --max-rows 1: one all-features row is enough to demo the dashboard; more if time.
set +e
RESV=3e5a9c36
PY=/home/darkstar/.venv-gemma4/bin/python
WORK=/tmp/course_run/experiments/v8_nla_local/labeled_outputs
SCRIPT=$WORK/sae_labeled_course.py          # patched: all_features() present
LOG=/tmp/run_sae_allfeat.log
SLOT_END="2026-07-17T18:40:00-07:00"
MARGIN_MIN=20

echo "[wait] blocking until reservation $RESV (gpu=both) starts ..." | tee -a "$LOG"
gpusched wait "$RESV" 2>&1 | tee -a "$LOG"

# pre-flight: RAM + free GPUs
FREE=$(free -g | awk 'NR==2{print $7}')
echo "[preflight] free RAM ${FREE}GB" | tee -a "$LOG"
if [ "${FREE:-0}" -lt 150 ]; then echo "[preflight] FAIL ram"; exit 3; fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
echo "[preflight] GPU used MiB: $USED" | tee -a "$LOG"
if [ "${USED:-0}" -gt 2000 ]; then echo "[preflight] FAIL gpu in use"; exit 4; fi

cd "$WORK"
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR=$WORK/runs/sae_course_allfeat_$TS
mkdir -p "$OUTDIR"
echo "[run] out=$OUTDIR" | tee -a "$LOG"

# Hard OS timeout backstop at slot end minus margin (can't overrun)
TIMEOUT_S=$(( ( $(date -d "$SLOT_END - ${MARGIN_MIN} minutes" +%s) - $(date +%s) ) ))
[ "$TIMEOUT_S" -lt 60 ] && TIMEOUT_S=60

HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null) \
  timeout -s TERM "$TIMEOUT_S" $PY "$SCRIPT" \
    --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
    --layers 0,16,32,48,63 --model Qwen/Qwen3.5-27B \
    --dtype float16 --hf-cache /tmp/hf_cache \
    --device cuda:0 --max-new-tokens 2200 --max-rows 1 \
    --out "$OUTDIR/sae_course.jsonl" > "$OUTDIR/run.log" 2>&1
RC=$?
echo "=== all-features populate exit $RC ===" | tee -a "$LOG"

# verify the all-features capture landed
if [ -s "$OUTDIR/sae_course.jsonl" ]; then
  /home/darkstar/.venv-gemma4/bin/python -c "
import json
r=json.loads(open('$OUTDIR/sae_course.jsonl').readline())
af=r.get('allf_a',{})
L=list(af.keys())[0]
print('row0 ok=',r['meta']['ok'],'| all_features_viewable=',r['meta'].get('all_features_viewable'),'| d_sae=',af[L]['d_sae'],'| profile_len=',len(af[L]['max_profile']),'| sparse=',len(af[L]['sparse']))
print('DASHBOARD: open dashboard.html, load', '$OUTDIR/sae_course.jsonl')
"
fi

if [ "$RC" -eq 0 ]; then
  gpusched release "$RESV" 2>&1 || true
  echo "[done] all-features row produced" | tee -a "$LOG"
else
  echo "[warn] non-zero exit $RC; NOT releasing (investigate)" | tee -a "$LOG"
fi
