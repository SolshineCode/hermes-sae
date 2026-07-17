# hermes-sae — same-inference SAE + label instrument

Faithful mechanistic readout for any model-generated text: the **exact same
`model.generate()` call that produces the label text also yields the SAE
feature activations over the residual stream.** No replay, no second inference,
no cached-activations round-trip.

## Why "same inference" matters (Caleb's correctness rule)

If you label with one model instance and then re-run the text through a hooked
model to get SAE features, you have **two inferences** — and activations can
diverge (sampling, KV state, device placement). That is *unfaithful*.

Here, a single `model.generate()` runs with forward hooks registered on every
decoder layer. After it returns:
- `out_ids[0, P:]` → decode → **label text**
- captured residuals sliced to the same generated positions → **SAE features**

Both derive from one forward pass / one KV cache, written into one JSON record.
`meta.same_inference == True` is asserted per row.

## Multi-purpose scope

The instrument is model-agnostic. Same-inference readout supports:
1. Deception / honesty
2. NLA capability classification (role_playing, domain, factual_claim)
3. Safety / refusal
4. Agent-trace cognition (planning vs executing)
5. Sycophancy / faithfulness

Label *schema* is config-driven (`labeler_config.yaml` + prompt set), so each
use-case selects its own labeling head while reusing the SAE capture path.

## Layout

- `sae_labeled_course.py` — hooked same-inference course (labeler_a/b/auditor).
- `labeler_service.py` — prompt assembly + `extract_label` (incl. truncated-JSON recovery).
- `validate_quant_sameness.py` — B: prove quant (int4) vs fp16 SAE-feature sameness.
- `runner/run_sae_course_deferred.sh` — scheduler-safe waiter (triple backstop:
  no-pipe-before-& PID, OS `timeout`, `pkill` fallback) so it can never overrun a slot.

## Correctness invariants

- `device_map=auto`; `inp.to(model.device)` (NOT hardcoded cuda:0 — would yield empty text).
- Hooks capture `out[0]` residual per layer; clone to avoid autograd aliasing.
- `--max-new-tokens 2200` so the label schema isn't truncated mid-object (recovery as backstop).
- SAEs trained on fp16 activations → quantization sameness MUST be re-validated (see validate_quant_sameness.py).

## Status

- ✅ Same-inference capture proven (row 0: ok=True, agree=1.0, same_inf=True).
- ⏳ Quant sameness validation (booking 56a2d647, 2026-07-16 21:35, gpu=1).
- ⏳ Throughput: fp16 CPU-offload ~13.5 h/row → 4-bit quant on single GPU is the scale unlock.
