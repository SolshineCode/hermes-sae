# Nous Research Report — Draft (local SAE × Hermes Agent)

## TL;DR for Nous
We can now run a local open-weight model **as the Hermes Agent model** and capture
the **same-inference** SAE / residual-stream activations of the exact inference
that powers the agent's actions — on a 4 GB consumer GPU. This is the faithful,
local, cheap path to *"watch the internals of the inferences that drive the agent,"*
which is the basis for understanding model behavior and building reliability signals.

## What is proven (this session)
- `nla_server.py` loads Gemma-4-E2B (4-bit NF4) ONCE and serves it
  OpenAI-compatibly. A forward hook on layer 23 captures the full
  residual-stream activation at every position of each generation.
- Hermes Agent (profile `nla-local`) is pointed at that server. Every agent
  turn is serviced by the *same* instance that carries the hook.
- End-to-end proof: a real Hermes turn produced a capture whose
  `prompt_preview` is the Hermes system prompt, with **96 layer-23 vectors**
  stored in one `.npz`.
- **Same-inference invariant holds**: no replay, no second model — the SAE
  features and the agent's output text come from one `model.generate()`.

## Key behavioral finding (read straight from the activations)
The 2B model **looped in text** ("4." ×96) on an arithmetic turn, BUT its
**layer-23 residual stream did NOT collapse**: consecutive-token cosine
**mean 0.569, max 0.796, loop-fraction(cos>0.9) = 0.0** across 95 pairs.
So *surface degeneration ≠ representational fixed-point at this layer.*
Implication: an output-text monitor would flag this turn "unreliable"; an
activation monitor at L23 would **not**. SAE/activation tracking gives a
*different, here less alarmist*, read of "degenerate" behavior — exactly the
nuance a reliability report needs to state honestly.

## Honest limitations (so Nous isn't misled)
- We capture **RAW layer-23 residuals** (what a trained SAE decomposes into
  features). No trained SAE for Gemma-4-E2B L23 exists in this setup yet →
  feature-level readout is the immediate next step.
- 2B is a **weak agent** (looping is model behavior, surfaced not caused by capture).
- 4 GB → slow decode (~0.3 tok/s); we cap tokens for tractability.
- So far **single layer** (23). Other layers untested for the collapse question.

## Why this serves the stated goal
- **Faithful same-inference readout** → causal claims about "what produced this
  agent action" are valid (the core hermes-sae invariant, REPORT.md §4).
- **Local + cheap** (no API, no second model) → can run continuously (we are,
  for the 10h grant) to build a behavioral corpus.
- **Norm-stable baseline** (~55–59 at L23) → a sudden L23 norm excursion is a
  clean, cheap anomaly / reliability signal.
- **SAE-ready capture format** (`acts [N,1536]`, `norms`, `is_prefill`,
  `token_ids`, `gen_ids`) → drop in a trained SAE later, no re-capture.

## Reproducibility
Everything (server, Hermes profile snippet, capture loop, analysis script,
framing + findings, sample `.npz`) is in the hermes-sae repo under
`2026-07-19-agent-integrated-sae-capture/`.

## Recommended next steps for the Nous deliverable
1. Train/obtain an SAE for Gemma-4-E2B L23; re-run analysis at feature level
   (does a feature collapse where raw cosine doesn't?).
2. Multi-layer capture (early/late) to localize where loop dynamics live.
3. Propagate `X-Hermes-Session-Id` for per-agent-activity attribution.
4. Scale to a stronger local model if one fits 4 GB.
5. Build the reliability signal: L23 norm-excursion + feature-sparsity drift
   as a live "agent confidence / degeneration" monitor.
