# Local Model *inside* Hermes Agent — the 4 GB branch of `hermes-sae`

This folder is the **local-model-in-Hermes-Agent** capability of the `hermes-sae`
project: run a small open-weight model **as the Hermes Agent model**, and capture
the **same-inference** SAE / residual-stream activations of the *exact* inference
that powers the agent's actions.

It is the direct, small-hardware complement to the existing `REPORT.md` pipeline
(Qwen3.5-27B fp16 on 2× Tesla M40). Same correctness invariant, opposite
scale.

## Why this branch matters (the Nous Research story)

The project's defining rule (REPORT.md §4) is the **same-inference invariant**:
SAE feature activations and the model's output label must come from the *exact same*
`model.generate()` call — no replay, no second model. Our `nla_server.py`
preserves this while letting the **Hermes Agent harness** be the thing that drives
the inference:

```
Hermes Agent (profile nla-local)
   └─ OpenAI-compatible HTTP ─► nla_server.py
                                   ├─ loads Gemma-4-E2B (NF4) ONCE
                                   ├─ L23 forward hook on model.generate()
                                   └─ captures ALL residual activations
                                      of the SAME call that produced the
                                      agent's output text
```

So the activations we save are provably the ones that *produced* the agent's
behavior — not a post-hoc re-run. That is the property the Nous Research report
needs to credibly claim: *"we can run SAEs on local models inside Hermes Agent
and watch the internals of the exact inferences that drive the agent."*

## Hardware reality (4 GB GTX 1650 Ti)

- Gemma-4-E2B in 4-bit NF4 fits (~3.9 GB), leaving ~0.1 GB.
- Decode is slow (~0.3 tok/s); a 16-token turn is ~2–4 min. We cap
  `max_tokens=16` for tractable sessions and raise it for real use.
- The 2B model is a **weak agent** (it loops — see the analysis docs); it is the
  right *vehicle* to prove the capture, not for agent quality.

## How the captures are produced (repro)

```bash
# 1) server (loads model once, then idles ready)
cd /home/caleb/nla_run
.venv/bin/python nla_server.py --model-dir weights/gemma-4-E2B --port 8000 --max-tokens-cap 16

# 2) Hermes profile nla-local (one-time)
hermes profile create nla-local --clone
hermes -p nla-local config set model.provider custom
hermes -p nla-local config set model.base_url http://127.0.0.1:8000/v1
hermes -p nla-local config set model.default gemma-4-e2b
hermes -p nla-local config set model.api_key sk-local
hermes -p nla-local config set model.context_length 65536   # Hermes min is 64K

# 3) real agent turn -> produces a capture whose prompt_preview is the Hermes system prompt
hermes -p nla-local chat -q 'What is 2+2? One sentence.'

# 4) continuous capture (GPU grant): varied turns back-to-back
bash run_capture_loop.sh

# 5) analyze
.venv/bin/python analyze_captures.py        # -> logs/nla_server/ANALYSIS.md
```

## What we capture

Per request, `nla_server.py` writes:

| Artifact | Contents |
|-----------|----------|
| `activations.jsonl` | one line/request: `request_id`, `session_id`, `timestamp`, `model`, `layer`, `d_model`, `n_records`, `n_gen_tokens`, `prompt_preview`, `npz_path`, `gen_text_preview` |
| `run_<reqid>.npz` | `acts [N,1536]` float32 (1 prefill + N-1 generated), `norms`, `is_prefill`, `token_ids`, `gen_ids` |
| `server.log` | lifecycle + per-request log (Hermes requests show `stream=True`) |

These are **raw layer-23 residual activations** — exactly what a *trained SAE*
would decompose into interpretable features. The capture format is SAE-ready; wiring
a trained SAE for Gemma-4-E2B layer 23 is the natural next step (the repo's
existing SAEs target Qwen3.5-27B layers).

## Current status (this branch)

- ✅ Proven end-to-end: a real Hermes Agent turn was serviced by the local
  Gemma-4-E2B instance and **all 96 layer-23 residual activations** of that
  exact inference were captured (see `evidence/activations.jsonl` line 2).
- ✅ Analysis script (`analyze_captures.py`) characterizes behavior from the
  activations — e.g. it detects the degenerate **loop attractor** (sustained
  consecutive-cosine > 0.9) that corresponds to the model repeating "4.".
- 🔜 Next: train/obtain an SAE for Gemma-4-E2B layer 23 and map the raw
  vectors to named features; propagate `X-Hermes-Session-Id` for per-agent
  attribution; scale to a stronger local model if one becomes available.

## Honest caveats

1. **4 GB is the bottleneck** — slow decode; cap tokens for tractability.
2. **2B is a weak agent** — looping is model behavior, surfaced (not caused) by the capture.
3. **Raw residuals, not yet feature-decomposed** — no trained SAE for this model/layer yet.
4. **Session correlation** is timestamp + prompt-preview based (`session_id=null` today);
   the server already supports `X-Hermes-Session-Id` for tighter binding.
