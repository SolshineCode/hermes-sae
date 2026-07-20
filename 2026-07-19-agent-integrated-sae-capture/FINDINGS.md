# Findings (mechanistic, from same-inference captures)

First batch analyzed (2026-07-20, 3 captures: 2 curl, 1 Hermes-Agent turn).
Metrics from `analyze_captures.py` on the layer-23 residual stream.

## 1. Text-looping ≠ representational collapse (layer 23)

- The Hermes-Agent turn (`0bed84ea`, 96 tokens) produced **repetitive surface text**
  ("4." repeated) — a classic small-model degeneration.
- But its **layer-23 residual stream did NOT collapse**: consecutive-generated-token
  cosine similarity **mean 0.569**, **max 0.796**, loop-fraction(cos>0.9) = **0.0**
  across all 95 consecutive pairs.
- Interpretation: the looping is a **decoding/degeneration phenomenon**, not a
  representational fixed-point at this layer. The model's internal state at layer 23
  keeps moving even while its output text stalls.
- **Why this matters for the Nous report:** an output-text monitor would flag this turn
  as "degenerate / unreliable." An activation-space monitor at layer 23 would **not**
  (per-token vectors still diverge). SAE/activation tracking gives a *different, here
  less alarmist*, read of "degenerate" behavior than the text alone — exactly the
  kind of nuance the report should surface.
- **Caveats:** (a) a *trained SAE* might still show feature-level collapse even
  where raw cosine is moderate — need the SAE to confirm. (b) layer 23 may not be
  where loop dynamics live; comparing other layers is future work.

## 2. Residual stream is norm-stable across generation

- Prefill norm ≈ 57; generation norms ≈ 55–59 across all 3 captures and across
  prompts.
- The layer-23 residual stream does not drift/explode during generation — a stable,
  information-rich readout suitable for monitoring.
- Implication for reliability: a sudden norm excursion at layer 23 would be a clean
  anomaly signal; baseline is ~55–59.

## 3. Same-inference invariant holds in the agentic setting

- The Hermes turn's `prompt_preview` is the Hermes system prompt, and its
  activations are from the *exact* `generate()` that produced the agent's reply — no
  replay, no second model. Faithful to `REPORT.md` §4.

## Open questions for the next captures

- Does a *trained* SAE on Gemma-4-E2B layer 23 show feature-level loop
  signals the raw cosine misses?
- Do other layers (early vs late) show the collapse the text implies?
- Across many varied prompts, what is the distribution of loop-fraction? Is the 2B
  model systematically degenerate, or only on arithmetic/short-answer prompts?
