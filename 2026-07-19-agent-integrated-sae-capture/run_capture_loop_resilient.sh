#!/bin/bash
# run_capture_loop_resilient.sh -- continuous Hermes-Agent-turn SAE capture (GPU grant)
# Resilient variant: checks nla_server health before each turn; restarts it
# if unreachable; only hits Hermes when the server is actually up.
# This avoids the ~4h APIConnectionError tail seen in the first run.
set -u
cd /home/caleb/nla_run
LOG=logs/capture_loop_resilient.log
PROMPTS=(
  "What is 2+2? One sentence."
  "Write a haiku about a GPU rendering a fractal."
  "List three benefits of sparse autoencoders for interpretability."
  "Explain recursion in one sentence."
  "What is the capital of France? One sentence."
  "Summarize mechanistic interpretability in two sentences."
  "If I have 3 apples and you give me 2 more, how many? One sentence."
  "Name a color that is not red, in one word."
  "What is the opposite of hot? One word."
  "Compose one short line of Python that prints hello."
  "What is 7 times 8? One sentence."
  "Give a one-sentence definition of a sparse autoencoder."
)
SERVER_PID=""
ensure_server() {
  # (re)start nla_server.py if not already running and healthy
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    if curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
      return 0   # server healthy, nothing to do
    fi
    echo "[$(date +%H:%M:%S)] server unhealthy (pid $SERVER_PID) — killing and restarting" >> "$LOG"
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
  fi
  nohup .venv/bin/python nla_server.py --model-dir weights/gemma-4-E2B --port 8000 --max-tokens-cap 16 --log-dir logs/nla_server >/dev/null 2>&1 &
  SERVER_PID=$!
  echo "[$(date +%H:%M:%S)] started nla_server pid=$SERVER_PID" >> "$LOG"
  # wait for health (up to ~60s while model loads)
  for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] server healthy after ${i}s" >> "$LOG"
      return 0
    fi
    sleep 2
  done
  echo "[$(date +%H:%M:%S)] WARNING: server did not become healthy within 60s" >> "$LOG"
  return 1
}
END=$(($(date +%s) + 32400))   # ~9h
i=0
echo "[$(date +%H:%M:%S)] resilient capture loop start; will run until $(date -d @$END +%H:%M:%S)" >> "$LOG"
while [ "$(date +%s)" -lt "$END" ]; do
  p="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
  if ! ensure_server; then
    echo "[$(date +%H:%M:%S)] skipping turn $i (server not healthy)" >> "$LOG"
    sleep 15
    continue
  fi
  echo "[$(date +%H:%M:%S)] turn $i :: $p" >> "$LOG"
  timeout 600 hermes -p nla-local chat -q "$p" >> "$LOG" 2>&1
  ec=$?
  echo "[$(date +%H:%M:%S)] turn $i exit=$ec" >> "$LOG"
  i=$((i+1))
  sleep 3
done
echo "[$(date +%H:%M:%S)] capture loop complete after $i turns (server_pid=$SERVER_PID)" >> "$LOG"
