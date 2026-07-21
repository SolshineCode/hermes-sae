# Watching a local model think while it runs Hermes Agent

*Draft v2 (updated 2026-07-20 after real 27B D_Pilot capture completed).
Status: read-only observability is PROVEN at scale. Parametric steering and
PROPEL-style reward surrogates are on the roadmap but unvalidated. Figures:
`nous_figs/f1_feature_timeline.png` (TBD from real capture), `f2_architecture.png`
(validated), `f3_fdr_layer_table.png` (new — replaces TBD overhead bar, priority).
Caleb DeLeeuw. For Tristan / Nous Research.*

> **What this version adds over v1 (2026-07-15):** v1 described the instrument
> and the roadmap. v2 incorporates the first empirical result from the instrument
> on real agentic work (181-turn D_Pilot capture on Qwen3.5-27B). That result is
> in, it is real, and it means the read-only observability claim is now confirmed
> at production scale. The roadmap section honestly separates what's proven from
> what remains open — specifically, the distinction between "SAE features
> distinguish insider vs. clean context" (confirmed) and "SAE features can
> substitute for the expensive solver in a reward loop" (PROPEL's actual claim,
> still open).

---

## The one-paragraph version

While Hermes Agent runs on a local model, I attach sparse autoencoder (SAE)
probes to the model's residual stream and log which learned features fire, per
token, as the agent works. The transcript tells you what the agent said. The
feature trace tells you what the model was doing internally while it said it:
which concepts lit up during planning, where attention to the actual task
dropped off, what was active in the forward pass when a tool call went sideways.
It's a live internal readout of agentic work, on consumer hardware, with open
models.

What makes this version different from v1 is that we now have a real empirical
result on the claim: on a 181-turn agentic capture on Qwen3.5-27B, roughly
37–44% of features that fire in ≥20 turns survive FDR correction (`α=0.05`) as
discriminators of insider (deception-relevant) vs. clean (compliant) turns —
rejecting the null that SAE features carry no group-discriminable information.
The same-inference invariant (gen_text + feature log from one forward pass) has
been verified end-to-end on both the 27B GPU path and CPU-only paths on two
different released SAE formats. That is what this document is reporting. The
steering roadmap that follows is roadmap — designed, not yet validated.

---

## What runs today

### A. The capture layer (proven at 27B scale)

- **Model:** Qwen3.5-27B, fp16, split across two used Tesla M40s in a Dell T7610
  workstation (total GPU cost under $200 used). Hermes Agent drives it through
  real multi-step labeling tasks. The user is Caleb DeLeeuw (SolshineCode); the
  trace is from `agentic_traces_gen1`.
- **Probes:** SAE forward hooks at 5 layers (0, 16, 32, 48, 63 of 64),
  reading the residual stream in the *same inference pass* that produces the
  agent's output. No second forward, no replay. Per token the sidecar keeps the
  top-k features and their activations, streamed to a JSONL log alongside the
  agent transcript.
- **SAEs:** open checkpoints where they exist, trained suites for Gemma-4-E2B,
  plus a released residual SAE trained on Qwen3.5 itself. The probe code doesn't
  care whose SAE it is — it just needs the layer and the weights.
- **Capture format:** `{row_idx, gen_text, allf_a[L]}` where `allf_a[L]` carries
  the sparse feature activation list (token position, feature index, activation
  value) for that layer. Everything is append-only JSONL; no database required.

![One forward pass, two outputs](nous_figs/f2_architecture.png)

### B. What the real 27B capture shows

This is what was not known when v1 was drafted:

**Significance — what's supported:**
SAE features carry information about which *kind* of scenario the agent is in.
In a 181-turn real trace (94 insider / 87 clean, 2 skipped HTTP-500s), features
firing in ≥20 turns were tested at 5 SAE layers. At FDR α=0.05:

- **L0:** 98 tested, 37% survive (36 features)
- **L16:** 2,218 tested, 42% survive (924 features)
- **L32:** 6,805 tested, 42% survive (2,867 features)
- **L48:** 12,817 tested, 42% survive (5,366 features)
- **L63:** 22,573 tested, 44% survive (9,948 features)

For the most-insider-specific features the permutation p-value is at the
simulation floor (p < 1/301 ≈ 0.0033). The strongest signatures are
*presence-only*: certain features fire exclusively on insider turns and on zero
clean turns across the full 181-turn dataset (e.g., L63 feature 50911 fires 35/35
insider, 0/46 clean).

The signal is roughly constant across all five layers — it is distributed
throughout the model, not concentrated in one late "semantic peak" layer.

**What this does not yet show:**
The test distinguishes insider vs. clean *scenario pools*, which differ in
content, framing, and pressure. We have not yet isolated whether the features
are responding to *deception specifically* vs. any structural difference between
the two pools. That requires a matched-pair test on the *same prompt* — a known,
pre-registered next step (see §Roadmap below).

**Honest power note:**
At the insider-presence boundary (features that fire 100% in insider and 0% in
clean), power is essentially perfect — these features are detectable from tiny n.
The more demanding discriminator is features with moderately shifted activation
distributions (AUC 0.65–0.80) that might distinguish deception from same-content
honest behavior. Whether such features exist in this trace is unknown.

### C. CPU-only robustness tests (model + SAE format + hardware agnosticism)

The same capture pipeline was independently validated (a) on gpt2-small with a
professionally released Bloom-format SAE, and (b) on a *modern* Qwen2.5-0.5B
with a released sae_lens-format residual SAE, both on plain CPU (`CUDA_VISIBLE_DEVICES=""`).

- `sae_serve.py` server path (OpenAI-compatible HTTP `/v1/chat/completions`) runs
  CPU-only and writes `allf_a` from the same inference instance that returns the
  text. Same-inference invariant confirmed over HTTP on CPU.
- B1 separability on gpt2-small and Qwen2.5-0.5B (n=4/group, synthetic prompts)
  reproduced the same qualitative signature: insider-only presence features
  (e.g., Qwen2.5 L16 features 2376, 3090 fire 8/8 insider, 0/8 clean) with 0
  BH-FDR survivors at n=4/group (expected — this is a plumbing/provenance test,
  not the falsifiable power test).

Reproducible in one command: `./setup_cpu_test.sh qwen25b`. SAE weights are
downloaded from public HuggingFace repos; the model is auto-fetched by
`transformers`. No GPU, no `sae_lens`/`bitsandbytes` required for the validated
presets.

---

## Why Hermes specifically

Two reasons, one technical and one philosophical.

**Technical:** Hermes Agent has a plugin architecture. The existing delegation
plugin is a yaml manifest plus a small Python module. The SAE probe fits the
same shape — a plugin that wraps the local model's generation and emits a
feature-trace sidecar per session. No Hermes fork needed. The configuration is
just `{model, sae_repo, layers, top_k}`.

**Philosophical:** this only works because the weights are local. You cannot
hook the residual stream of an API. The interpretability story and the
local-model story are the same story, and Nous sits at the center of the
local-model story. An agent stack where *"what is my model actually doing"* is
a first-class, inspectable question feels like a natural extension of what
Hermes already is.

---

## What the evidence does and does not establish

This is the section v1 did not have, and it's the most important one to get
right when talking to a research team.

**Established (empirical result, 2026-07-20):**
- At the 181-turn real D_Pilot capture, supervised learning on `scenario.role`
  (insider/clean) is real: thousands of SAE features pass BH-FDR as
  insider-presence discriminators at all five hooked layers.
- The same-inference invariant (feature activation log and generated text from
  one `generate()` call, byte-identical `gen_text` across engine-direct and
  HTTP-server paths) holds at 27B scale.
- The capture infrastructure is model-agnostic (validated on gpt2, Qwen2.5-0.5B,
  Qwen3.5-27B) and device-agnostic (validated on CPU and 2×M40).
- The capture-and-analyze pipeline has known failure modes (row-index alignment
  bug, write-crash on bare output filenames, O(n_features × n_perm) runtime in
  the original analyzer) that have been identified, fixed, and audited.

**Real but narrower than it sounds (pending matched-pair test):**
The insider/clean distinction might reflect any combination of (a) the model's
recognition that it is in a situation where it should be deceptive, *or* (b) any
structural difference between the two scenario pools (task framing, vocabulary,
pressure, instruction style). Disentangling (a) from (b) requires the matched-pair
honest-vs-deceptive test on identical prompts — that's the next step.

**Not established:**
- That the features specifically encode the *solver's intended reasoning state*
  (PROPEL's core claim). The insider/clean distinction is not the same as the
  solver-bottleneck mechanism.
- That a linear probe or reward on SAE features improves GRPO training outcomes.
- That per-feature bias injection at serve time is well-tolerated by Qwen3.5
  at any nudge strength.
- Generalization beyond this one model/SAE/trace combination.

---

## Roadmap: from reading to steering

The roadmap explicitly distinguishes *iconfirmed evidence* from *designed
mechanism* from *open experiment*.

### What's confirmed (evidence-based)
| Item | Source |
|------|--------|
| SAE features distinguish insider vs. clean at FDR < 0.05, 181-turn real trace | `B1_REAL_27B_REPORT_2026_07_20.md` §4 |
| Same-inference invariant holds end-to-end: engine + HTTP server + CPU + 2×M40 | `validation_cpu/README.md`, `validation_real_27B/README.md` |
| Capture pipeline is model- and SAE-agnostic | `setup_cpu_test.sh`, gpt2 + Qwen2.5 runs |
| Analyzer correctness: row_idx bug fixed, write crash fixed, fast path is sound | `fast_dpilot_sep.py`, `analyze_dpilot_separability.py` |

### What's designed but unvalidated (mechanism)
| Item | Location | Notes |
|------|----------|-------|
| PROPEL-SAE reward substitution | `PROPEL_SAE_BRAINSTORM.md` | Replaces the supervised probe with unsuperivsed SAE feature vector |
| SAE-fixed binary reward on presence-only features | `propel_sae_reward.py` scaffold | No GRPO run yet |
| Hybrid supervised + SAE reward | `PROPEL_SAE_EXTENSION_REPORT.md` V2 | Calibration probe + SAE surrogate |
| Per-feature bias injection at serve time (drift correction) | `saeserve.py` (not yet committed) | Risk: output quality degradation, uncalibrated |

### What remains open (falsifiable experiments)
| Exp | Claim | Test | Where |
|-----|-------|------|-------|
| **B2-redo** | Insider-presence features encode deception specifically (not ambient context) | Matched-pair design: same `harness_prompt` + honest vs. deceptive output (within insider turns); re-run B1 | `FALSIFIABLE_CLAIM_DESIGN.md` |
| **B3** | SAE features predict solver outcome | Train probe on (`SAE features`, `solver_pass/fail`) pairs; test AUC | `PROPEL_SAE_EXTENSION_REPORT.md` V1 |
| **B4** | SAE-reward improves generator RL without calling the solver | GRPO with SAE-probe reward; measure frontier-problem rate vs. no-reward and PROPEL-probe baselines | `propel_sae_reward.py` |
| **C1** | Surving features have interpretable semantics | Label top-500 features via decoder cosine + top-activating contexts; manual labeling round | planned |
| **C2** | Overhead headroom for Hermes sidecar | Measure tok/s with/without 5-layer hooks; report | planned (next GPU window) |

### Off-ramps / honest limits
- If B2-redo is null (honest output matches deceptive on insider/SAE features), the insider-presence signal is *contextual* not *deceptive*. The deception-detection claim is bounded.
- If B3 and B4 fail, the PROPEL SAE-substitution hypothesis is rejected for this model/SAE/trace combination. The base SAE instrument still stands (capture works), but the reward-extension path is closed.
- Steering experiments require a separate pre-registration. The current roadmap does *not* claim any steering outcome before the read-only infrastructure is solid.

---

## What I'm offering

1. **This writeup v2**, updated to reflect the actual empirical result. Ready for
   internal circulation; figures still need measurement (F1 from a short episode
   trace; F3 overhead numbers scheduled this week).
2. **A minimal public demo repo** (in progress): the probe core extracted to
   a few hundred lines. Any HF model + matching SAE, live feature stream. One
   command.
3. **PRs to the Hermes Agent repo** for anything the team wants upstream. The
   natural first is the probe as a plugin following the `claudedelegate` shape.
4. **Collaboration on Hermes failure modes.** The Nous team knows what actually
   goes wrong in long agent runs. A shortlist of "behaviors you wish you could
   see coming" would directly shape which features I try to label and track first
   — especially useful for the drift-correction steering scenario.

---

## Honest numbers and limits

| Item | Status | Value |
|------|--------|-------|
| Same-inference invariant | Verified | `gen_text` byte-identical across engine + HTTP server paths |
| Capture overhead | Scheduled measurement (next GPU window, ~30 min) | unknown; SAE top-k at 5 layers, written to JSONL sidecar |
| SAE feature survival rate (insider vs. clean) | Confirmed at n=181/group, 5 layers | 37–44% of common features pass BH-FDR α=0.05 |
| Deception-specific signal | Open | Matched-pair honest-vs-deceptive test pending |
| Solver-bottleneck / PROPEL claim | Open | V1/V2 in scaffolding; no GRPO run |
| Overhead tok/s penalty | TBD | Needs measurement |
| SAE dictionary quality | Varies | Not all layers are equally monosemantic; ongoing labeling |
| Feature labeling coverage | ~0% of 5×81920 features | Planned: top 500–1000 by discriminability |
| Steering tolerance | Not even measured yet | Probably feature-dependent; no prior art at this scale |
| Safety utility | None established | Feature trace is *evidence*, not a safety guarantee |

None of this is a safety guarantee. A feature trace is evidence, not proof,
of what a model is doing.

---

## Provenance

| Artifact | Location | Role |
|----------|----------|------|
| B1 report (detailed) | `B1REAL27BREPORT20260720.md` | Full layer-wise table, top features, evidence table, falsifiability audit, provenance chain |
| Falsifiable-claim design | `FALSIFIABLE_CLAIM_DESIGN.md` | Pre-registered claims + null-result conditions (written before data) |
| PROPEL SAE brainstorm | `PROPEL_SAE_BRAINSTORM.md` | Mechanism mapping (PROPEL probe ⇄ SAE feature) |
| PROPEL extension report | `PROPEL_SAE_EXTENSION_REPORT.md` | V1/V2 reward stack, decision log |
| Real D_Pilot artifacts | `validationreal_27B/` and `/home/darkstar/hermes_cache/dpilot_capture/` | Capture + joined synthesis + analyzer |
| CPU validation artifacts | `validationcpu/` | gpt2 + Qwen2.5-0.5B artifacts + one-command setup script |
| Analyzer + joiners | `propel_sae_extension/` | `fastdpilotsep.py`, `analyzedpilotseparability.py`, `buildreal_dpilot.py` |
| Original Nous plan | `nous/NOUS_WRITEUP_PLAN_2026-07-15.md` | D1–D3 delivery sequence + decisions for Caleb |

If any of this is useful to the team, reach out. The demo repo will be public
shortly — I'll share the link here before sending the replacement figures.
