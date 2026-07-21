# Nous Research / Hermes Agent materials

Materials prepared for the Nous Research community around integrating this
project with [Hermes Agent](https://github.com/NousResearch/hermes-agent).

- [`WRITEUP.md`](WRITEUP.md) — the memo: what runs today, the evidence tiers,
  the `sae_trace` observability plugin, how this relates to Nous's own
  neuron-steering work, and honest limits. Also available as
  [`Hermes-SAE-Writeup.pdf`](Hermes-SAE-Writeup.pdf) (same content, repo links absolute).
- [`figures/`](figures/) — figures referenced by the memo.
- [`../probe-demo/`](../probe-demo/) — minimal standalone probe demo.
- [`../2026-07-19-agent-integrated-sae-capture/`](../2026-07-19-agent-integrated-sae-capture/)
  — the agent-integrated capture: a local model serving as the Hermes Agent
  model with same-inference residual-stream capture.

Naming note: "Hermes Agent" in these documents always refers to Nous
Research's product. The multi-agent scenario harness that generated the 27B
trace dataset is a separate local research tool and is referred to only as
"the local agent harness" or "scenario engine."
