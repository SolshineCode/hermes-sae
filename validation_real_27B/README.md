# Real-hardware validation: live SAE serving on Qwen3.5-27B

**Date:** 2026-07-19 · **GPU slot:** 133141e2 (both M40) · **Model:** Qwen3.5-27B fp16 (~52GB, device_map=auto over 2×M40+CPU) · **SAE:** 5-layer 80K PoC (`SAE-Res-Qwen3.5-27B-W80K-L0_50`)

## What this proves (user's absolute top-level requirement)
> "The EXACT SAME model instance that is running Hermes Agent (local LLM) is the one we
> gather the SAE activation records from, while it is doing the EXACT inference that runs
> Hermes Agent."

`sae_serve.py` runs the model **in-process** with forward hooks on decoder layers 0/16/32/48/63.
One `generate()` per request produces BOTH the text returned to the client AND the SAE
feature activations. No replay, no second pass. Verified on the real 27B model:

- `validation_real_27B/real_27B_sae_history.jsonl` — log line where `gen_text` ==
  HTTP response content (byte-identical) and `gen_len` (40) == `usage.completion_tokens` (40).
  `same_inference: true`. Every `feat[token_pos]` in `[0, gen_len)`. 887–1097 distinct
  SAE features fired per layer from a 40-token generation.
- `validation_real_27B/real_27B_serve_response.json` — the actual OpenAI-compatible response.
- `validation_real_27B/real_allfeat_row.jsonl` + `real_27B_allf_response.json` — a FULL-
  dictionary (all-81920-width) capture enable via `--allf`, feeding B2/D_Pilot.

So selecting `sae-local` in Hermes and running the agent normally will automatically record
the SAE feature history of the agent's own activity, turn by turn. Non-local models never
hit this server (no residual access for cloud inference — correct).

## B2: distinct-feature analysis on REAL data (audit fix)
`sae_realfeat_analysis.py` (corrected: DISTINCT features only, cross-cosine excludes
self-pairs, dead-feature stats) run on `real_allfeat_row.jsonl`:

| Layer | distinct feats | occurrences | max act | topk cross-cos (NO self-pairs) | dead frac |
|---|---|---|---|---|---|
| 0  | 33   | 48    | 3.57  | -0.0000 | 0.0000 |
| 16 | 731  | 1124  | 23.17 | 0.0119  | 0.0000 |
| 32 | 2591 | 3960  | 33.69 | 0.0154  | 0.0000 |
| 48 | 5125 | 7915  | 93.78 | 0.0185  | 0.0000 |
| 63 | 7575 | 12692 | 258.33| 0.0065  | 0.0000 |

**The prior "L32 clustering peak cos=0.33" was a duplicate-self-pair artifact** (Fable 5
audit 1c). With distinct features and no self-pairs, cross-cosine is ~0 across all layers —
distinct SAE features are near-orthogonal, as expected for a well-trained SAE. The corrected
result is descriptive only; it does NOT support any "semantic clustering" interpretation.

## Reproduction
```
TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 CUDA_VISIBLE_DEVICES=0,1 python sae_serve.py \
  --model-dir /tmp/hf_cache/Qwen/Qwen3.5-27B \
  --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
  --layers 0,16,32,48,63 --dtype float16 \
  --max-memory '0=20GiB,1=20GiB,cpu=300GiB' --port 8077 \
  --log runs/agent_sae_history.jsonl [--allf]
# then: curl the /v1/chat/completions endpoint; SAE features logged per turn.
```
