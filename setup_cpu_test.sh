#!/usr/bin/env bash
# setup_cpu_test.sh — one-command replication of the PROPEL-SAE CPU-only pipeline.
#
# Downloads the professionally released SAE weights for a preset into the on-disk layout
# cpu_dpilot_test.py expects, then runs the test on CPU (no GPU). The model itself is
# auto-downloaded by transformers.AutoModelForCausalLM during the run.
#
# Usage:
#   ./setup_cpu_test.sh qwen25b      # MODERN Qwen2.5-0.5B + released residual SAE (validated)
#   ./setup_cpu_test.sh gpt2         # gpt2-small + Joseph Bloom released GPT2-Small SAE (validated)
#   ./setup_cpu_test.sh qwen15b      # DeepSeek-R1-Distill-Qwen-1.5B + EleutherAI 65k SAE
#                                    #   NOTE: that model hangs on from_pretrained on this box
#                                    #   (transformers 5.13 / torch 2.4.1). SAE download works;
#                                    #   use qwen25b instead for a clean modern run.
#
# Full pipeline deps (pip):  torch transformers safetensors
# No sae_lens / bitsandbytes needed for the qwen25b / gpt2 presets.
set -euo pipefail

PRESET="${1:-qwen25b}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="/home/darkstar/hermes_cache/cpu_sae"
source "${HERE%/}/../.." 2>/dev/null || true   # no-op, keeps shellcheck quiet

echo "=== PROPEL-SAE CPU test setup: preset=$PRESET ==="

# --- preset -> (repo_id, layer list, local dir, file layout) ---
case "$PRESET" in
  gpt2)
    REPO="jbloom/GPT2-Small-SAEs"
    LAYERS="7 11"
    SAE_DIR="$CACHE_ROOT/gpt2-small-sae"
    LAYOUT="pt"   # remote layer_{L}/final_sae/sae.safetensors -> local layer{L}.sae.pt
    ;;
  qwen25b)
    REPO="HuggingAnalist/sae-qwen2.5-0.5B-res"
    LAYERS="16 18"
    SAE_DIR="$CACHE_ROOT/qwen25b-res"
    LAYOUT="flat" # remote layer_{L}/final_sae/{cfg.json,sae_weights.safetensors} -> local layer{L}.sae.safetensors
    ;;
  qwen15b)
    REPO="EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
    LAYERS="0 6"
    SAE_DIR="$CACHE_ROOT/qwen15b-eleuther"
    LAYOUT="mlp"  # remote layers.{L}.mlp/{cfg.json,sae.safetensors} -> local layers.{L}.mlp/sae.safetensors
    ;;
  *)
    echo "unknown preset: $PRESET (use gpt2|qwen25b|qwen15b)"; exit 2 ;;
esac

mkdir -p "$SAE_DIR"
echo "=== downloading released SAE ($REPO) into $SAE_DIR ==="
for L in $LAYERS; do
  if [ "$LAYOUT" = "pt" ]; then
    # Bloom GPT2-Small SAEs: remote layer_{L}/final_sae/sae.safetensors -> local layer{L}.sae.pt
    curl -sSL -o "$SAE_DIR/layer${L}.sae.pt" \
      "https://huggingface.co/$REPO/resolve/main/layer_${L}/final_sae/sae.safetensors"
    echo "  layer $L -> layer${L}.sae.pt ($(du -h "$SAE_DIR/layer${L}.sae.pt" | cut -f1))"
  elif [ "$LAYOUT" = "flat" ]; then
    mkdir -p "$SAE_DIR/layer${L}"
    curl -sSL -o "$SAE_DIR/layer${L}/cfg.json" \
      "https://huggingface.co/$REPO/resolve/main/layer_${L}/final_sae/cfg.json"
    curl -sSL -o "$SAE_DIR/layer${L}/sae_weights.safetensors" \
      "https://huggingface.co/$REPO/resolve/main/layer_${L}/final_sae/sae_weights.safetensors"
    cp "$SAE_DIR/layer${L}/sae_weights.safetensors" "$SAE_DIR/layer${L}.sae.safetensors"
    echo "  layer $L -> layer${L}.sae.safetensors ($(du -h "$SAE_DIR/layer${L}.sae.safetensors" | cut -f1))"
  else  # mlp
    mkdir -p "$SAE_DIR/layers.${L}.mlp"
    curl -sSL -o "$SAE_DIR/layers.${L}.mlp/cfg.json" \
      "https://huggingface.co/$REPO/resolve/main/layers.${L}.mlp/cfg.json"
    curl -sSL -o "$SAE_DIR/layers.${L}.mlp/sae.safetensors" \
      "https://huggingface.co/$REPO/resolve/main/layers.${L}.mlp/sae.safetensors"
    echo "  layer $L -> layers.${L}.mlp/sae.safetensors ($(du -h "$SAE_DIR/layers.${L}.mlp/sae.safetensors" | cut -f1))"
  fi
done

echo "=== running CPU test (model auto-downloads; ~110s load + ~20s/gen on CPU) ==="
cd "$HERE"
PY="${PYTHON:-python}"
timeout 900 "$PY" cpu_dpilot_test.py \
  --preset "$PRESET" --n-per 4 --max-tokens 24 \
  --out "$SAE_DIR/${PRESET}_cpu_capture.jsonl"
echo "=== DONE. captures -> $SAE_DIR/${PRESET}_cpu_capture.jsonl ; separability -> $SAE_DIR/${PRESET}_cpu_separability.json ==="
