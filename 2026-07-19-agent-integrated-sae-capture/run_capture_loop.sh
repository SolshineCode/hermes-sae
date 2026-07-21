#!/bin/bash
# run_capture_loop.sh -- continuous Hermes-Agent-turn SAE capture (GPU grant)
# Drives the local Gemma-4-E2B (served by nla_server.py) through varied
# Hermes Agent turns so the SAME-INFERENCE layer-23 activations accumulate
# in logs/nla_server/{activations.jsonl, run_*.npz}.
# Server caps max_tokens=16, so each turn ~2-4 min on the 4GB GPU.
set -u
cd /home/caleb/nla_run
LOG=logs/capture_loop.log
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
END=$(($(date +%s) + 32400))   # ~9h (leave buffer before the 10h wind-down timer)
i=0
echo "[$(date +%H:%M:%S)] capture loop start; will run until $(date -d @$END +%H:%M:%S)" >> "$LOG"
while [ "$(date +%s)" -lt "$END" ]; do
  p="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
  echo "[$(date +%H:%M:%S)] turn $i :: $p" >> "$LOG"
  timeout 1200 hermes -p nla-local chat -q "$p" >> "$LOG" 2>&1
  ec=$?
  echo "[$(date +%H:%M:%S)] turn $i exit=$ec" >> "$LOG"
  i=$((i+1))
  sleep 3
done
echo "[$(date +%H:%M:%S)] capture loop complete after $i turns" >> "$LOG"
