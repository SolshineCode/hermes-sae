# Agent-Integrated SAE Capture — Gemma-4-E2B as the Hermes Agent Model (2026-07-19)

Full granular write-up with all logs: **`REPORT.md`**.

## What this is

`nla_server.py` is a model-agnostic, OpenAI-compatible model server that loads
**Gemma-4-E2B (NF4)** exactly once and, on every generation, captures the
**complete layer-23 residual-stream activation at each position** ("all the SAE
activations"). Hermes Agent is pointed at this server via the `nla-local`
profile, so every agent turn is serviced by the *same* model instance that
carries the forward hook — activations are captured automatically.

## Headline result

A real Hermes Agent turn was serviced by the local Gemma-4-E2B (NF4) instance,
and **all 96 layer-23 residual activations** of that exact inference were
captured into the npz below — see `evidence/activations.jsonl` line 2, whose
`prompt_preview` is the Hermes system prompt
(`"system: You are Hermes Agent, an intelligent AI assistant created by Nous Research…"`).

## Contents

| File | Purpose |
|------|---------|
| `REPORT.md` | Detailed report: objective, architecture, environment, full `nla_server.py` source, Hermes wiring, evidence/logs, caveats, next steps. |
| `nla_server.py` | The hooked OpenAI-compatible server (loads Gemma-4-E2B NF4 once; L23 forward hook captures ALL residual activations per generation; logs npz + jsonl). |
| `nla-local.profile.model.yaml` | The Hermes `nla-local` profile model block that points Hermes at the server. |
| `evidence/activations.jsonl` | 3 capture records (line 2 = the Hermes Agent turn). |
| `evidence/server.log` | Server lifecycle + request log (`[req d1d5c0ef] … stream=True` is the Hermes request). |
| `evidence/run_0bed84ea58e5485fa827fdcf2a10552e.npz` | The actual Hermes-turn activation array: `acts [96, 1536]` float32, `norms`, `is_prefill`, `token_ids`, `gen_ids`. |

## Quick reproduce

```bash
# 1) server (loads model once, then idles)
cd /home/caleb/nla_run
.venv/bin/python nla_server.py --model-dir weights/gemma-4-E2B --port 8000 --max-tokens-cap 16

# 2) Hermes profile (one-time)
hermes profile create nla-local --clone
hermes -p nla-local config set model.provider custom
hermes -p nla-local config set model.base_url http://127.0.0.1:8000/v1
hermes -p nla-local config set model.default gemma-4-e2b
hermes -p nla-local config set model.api_key sk-local
hermes -p nla-local config set model.context_length 65536

# 3) real agent turn -> produces the Hermes-turn capture
hermes -p nla-local chat -q 'What is 2+2? One sentence.'

# 4) inspect
cat logs/nla_server/activations.jsonl
python -c "import numpy as np; d=np.load('logs/nla_server/run_0bed84ea58e5485fa827fdcf2a10552e.npz'); print(d['acts'].shape)"
```
