# Nous Research materials

Everything related to introducing this work to Nous Research lives here, so the
Nous-facing thread has one home instead of being spread across loose local notes.

## Contents

- `NOUS_WRITEUP_DRAFT.md` — the technical memo Tristan asked for (2026-07-15),
  covering hooking SAE probes into an agent harness for a live readout of internal
  model state while the agent runs. Drafted 2026-07-17, human-writing-check clean.
  **Not yet sent** as of 2026-07-20, pending review.
- `NOUS_WRITEUP_PLAN_2026-07-15.md` — the D1/D2/D3 plan behind the memo.
  D1 (writeup) and D2 (probe demo) are live; D3 (PRs to Nous's GitHub) is on hold.
- `figures/` — figures referenced by the memo.
- `../probe-demo/` — the D2 demo: a standalone SAE probe runner (SAELens
  safetensors loader, per-token top-k hooks), smoke-tested on SmolLM2-135M.
- `../2026-07-19-agent-integrated-sae-capture/NOUS_REPORT_DRAFT.md` — the later
  agent-integrated capture report from the same thread.

## Naming caution

This repo's own agent harness is a personal tool that shares a name with Nous's
product. In anything Nous-facing, do not use that name for the local tool: say
"the local agent harness" instead, so the two are never confused.

## Provenance

The memo originates from a private DM exchange. Keep the correspondence itself out
of this repo. Only the technical content belongs here, and this repo is currently
PRIVATE.
