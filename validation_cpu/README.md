# CPU-Only Validation — PROPEL-SAE Pipeline on a Small Model + Professionally Released SAE

**Date:** 2026-07-20 (overnight, GPUs unavailable — ran on CPU)
**Author:** Hermes Agent (autonomous /deep-work, per Caleb DeLeeuw)
**Purpose:** Prove the PROPEL-SAE labeling/capture pipeline is **model- and SAE-agnostic**
and **runs CPU-only** — an additional robustness condition requested explicitly for the
write-up to the Hermes Agent developer team / Nous Research.

---

## What was tested

| Component | Value |
| --- | --- |
| Model | `gpt2` (small, 124M params, base model — **no chat template**) |
| SAE | **Professionally released** Joseph Bloom GPT2-Small SAEs (`jbloom/GPT2-Small-SAEs`), d_sae = 24576, layers 7 & 11 |
| Hardware | CPU only (`device_map="cpu"`, `CUDA_VISIBLE_DEVICES=""`) — **no GPUs** |
| Serving | `sae_serve.py` (OpenAI-compatible HTTP server) **and** the engine-direct path |
| Capture | Same-inference SAE feature activations + generated text from the *exact same* forward pass |
| Analysis | B1 separability (insider vs clean), permutation test (200 perms), BH-FDR |

## Key results

1. **Serving layer runs CPU-only on a different model/SAE stack.** `sae_serve.py` loaded
   gpt2 + released SAE on CPU, served an OpenAI-compatible `/v1/chat/completions`, and
   recorded `allf_a` (per-layer sparse SAE feature activations over the generated span)
   **from the same inference instance** that produced the text. Same-inference invariant
   holds over HTTP on CPU — not just on the 27B GPU path.

2. **B1 separability (directional signal present, tiny-n honest null on FDR).** With
   n = 4/group (synthetic honest same-prompt-confound-controlled insider/clean prompts —
   different *content* by role bucket, never identical prompts, per CLAUDE.md):
   - **Layer 7, feature 3018 fires ONLY on insider turns** (4/4 insider, 0/4 clean,
     one-sided presence p_perm = 0.0249).
   - Features 4407, 11215, 14863, 15621, 17154 show the same **presence-only insider**
     signature (insider_mean > 0, clean_mean = 0).
   - **Layer 11, feature 5258** similar (p_perm = 0.0647).
   - 0 features survive BH-FDR at n = 4/group — expected (low power); the *direction*
     (features that are present exclusively in insider generations) is exactly the
     Theory-of-Mind-predictor cue the design predicts. This is a feasibility/generality
     result, not a falsifiable claim (that requires n ≥ 100/group on the real trace).

## Why this matters for the robustness write-up

- The pipeline is **not hard-coded to Qwen3.5-27B or to our local PoC SAE**. It loaded a
  *professionally released third-party SAE* (different format: pickled `sae_training`
  object, `W_enc` stored as `(d_in, d_sae)`) with **zero format-specific code in the
  engine** — a dependency-free loader (`load_released_pt.py`) stubs `sae_training`,
  extracts `W_enc`/`b_enc`, and transposes to the engine's `(d_sae, d_in)` convention.
- It runs **CPU-only** with no code changes to the capture/analysis core — proving the
  labeling layer can run on commodity hardware (e.g., as an always-on sidecar capturing
  the local Hermes model's SAE feature history during normal agent use, GPU or not).
- Base models without chat templates are auto-handled (generic default template assigned
  in `sae_serve.py`).

## Files

- `cpu_dpilot_test.py` — CPU end-to-end test (engine → capture → B1 separability).
- `load_released_pt.py` — dependency-free loader for released SAE `.pt` files.
- `sae_serve.py` — patched: `load_sae` now delegates to `load_released_sae_pt` (handles
  both local PoC and released formats); assigns a default chat template for base models.
- Artifacts (durable): `/home/darkstar/hermes_cache/cpu_sae/cpu_dpilot_capture.jsonl`,
  `cpu_dpilot_separability.json`, `cpu_serve_history.jsonl`.

## Reproduce

```bash
# download released SAEs (Bloom GPT2-Small)
mkdir -p /home/darkstar/hermes_cache/cpu_sae/gpt2-small-sae
for L in 7 11; do
  curl -L -o /home/darkstar/hermes_cache/cpu_sae/gpt2-small-sae/layer$L.sae.pt \
    "https://huggingface.co/jbloom/GPT2-Small-SAEs/resolve/main/final_sparse_autoencoder_gpt2-small_blocks.$L.hook_resid_pre_24576.pt"
done

# CPU server (gpt2 + released SAE)
TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 CUDA_VISIBLE_DEVICES="" python sae_serve.py \
  --model-dir gpt2 --sae-repo /home/darkstar/hermes_cache/cpu_sae/gpt2-small-sae \
  --layers 7,11 --dtype float32 --port 8088 --allf

# OR end-to-end test
TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 CUDA_VISIBLE_DEVICES="" python cpu_dpilot_test.py \
  --sae-dir /home/darkstar/hermes_cache/cpu_sae/gpt2-small-sae --layers 7,11 \
  --width 16 --n-per 4 --max-tokens 24 --out /tmp/cpu_dpilot_capture.jsonl
```

## Caveat (scientific integrity, per CLAUDE.md)

The CPU result uses **synthetic** insider/clean prompts (n=4/group) for a plumbing/
generality proof. The honest falsifiable test remains the **real D_Pilot capture on the
27B** (n ≥ 100/group from `agentic_traces_gen1`) scheduled for the GPU slot — that is the
run that can confirm or falsify the deception-separability claim. The CPU run establishes
that the *machinery* generalizes across model, SAE, and device; it does not by itself
establish a deception-SAE effect.
