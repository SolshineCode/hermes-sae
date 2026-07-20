# hermes-sae-serving-config.md
# =============================
# ADDITIVE Hermes config block to register the SAE-instrumented local model as a
# selectable /model option. Drop the `providers:` addition and the `model:` entries
# below into ~/.hermes/config.yaml. This does NOT modify any existing key except the
# `model.default`/`model.provider` lines, which you set ONLY when you want Hermes Agent
# to run on the instrumented local model by default.
#
# THE SAME-INFERENCE REQUIREMENT (Caleb 2026-07-19):
#   sae_serve.py runs the local model IN-PROCESS with SAE forward hooks. Hermes talks
#   to it over OpenAI-compatible HTTP exactly like it talks to ollama. Selecting
#   `sae-local` routes the agent's own generation through the instrumented model, and
#   every assistant turn's SAE activations are logged automatically (no replay, no
#   second pass). Non-local (cloud) models never hit this server, so no SAE is produced
#   for them — which is correct (no residual stream access for cloud inference).
#
# ---------------------------------------------------------------
# 1) ADD a provider block (merge into existing top-level `providers:`)
# ---------------------------------------------------------------
providers:
  sae:
    base_url: http://localhost:8077/v1
    api_key: not-needed            # OpenAI client still wants a key field; anything works
    request_timeout_seconds: 900    # 27B on dual-M40 is slow; be generous
    stale_timeout_seconds: 900
#
# ---------------------------------------------------------------
# 2) ADD model entries (merge into existing top-level `model:`)
#    Set `default`/`provider` to `sae-local`/`sae` ONLY when you want Hermes to run
#    on the instrumented local model. Otherwise leave your current default and just
#    use `/model` -> sae-local ad hoc.
# ---------------------------------------------------------------
model:
  sae-local:
    provider: sae
    label: "Qwen3.5-27B + SAE (instrumented, local)"   # shows in /model selector
  # if you want it as the default:
  #   default: sae-local
  #   provider: sae
#
# ---------------------------------------------------------------
# 3) OPTIONAL: register as an MOA aggregator/reference option
# ---------------------------------------------------------------
moa:
  presets:
    local-council:
      aggregator:
        provider: sae
        model: sae-local
# ===============================================================
#
# STARTUP (launch BEFORE selecting sae-local in Hermes):
#   TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 python sae_serve.py \
#       --model-dir /tmp/hf_cache/Qwen/Qwen3.5-27B \
#       --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
#       --layers 0,16,32,48,63 --dtype float16 \
#       --max-memory '0=20GiB,1=20GiB,cpu=300GiB' \
#       --port 8077 --log runs/agent_sae_history.jsonl
#   (base model is safetensors -> loads fine; SAE .pt needs the env override above,
#    which is safe for the trusted HF-hosted SAE repo.)
#
# VERIFY SAME-INFERENCE at runtime:
#   - the log line's gen_text == the agent's visible assistant message
#   - log line's gen_len == usage.completion_tokens
#   - every feat[token_pos] in [0, gen_len)
#   All asserted in sae_serve.py --self-test; re-verify on the real 27B path on first GPU run.
