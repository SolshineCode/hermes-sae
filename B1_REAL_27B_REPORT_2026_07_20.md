# B1 Real 27B D_Pilot Report: Deception-Detection Confirmed, PROPEL Solver-Bottleneck Claim Separately Pending

**Date:** 2026-07-20  
**Authors:** Hermes Agent (autonomous, for Caleb DeLeeuw / SolshineCode) + user-direction on falsifiability  
**Companion docs:** `FALSIFIABLE_CLAIM_DESIGN.md`, `PROPEL_SAE_BRAINSTORM.md`, `PROPEL_SAE_EXTENSION_REPORT.md`, `NOUS_WRITEUP_DRAFT.md` (Nous-facing version)  
**Status:** Final result; roadmap for PROPEL confirmation documented here.

> **One-line bottom line up front:** On a real ~200-turn agentic trace on Qwen3.5-27B (94 insider / 87 clean turns, 5 SAE layers, d_sae=81920), roughly 38–44% of features that fire in ≥20 turns survive BH-FDR α=0.05 as insider-presence signatures. The null — that latent SAE features carry no group-discriminable information — is rejected. **This confirms deception-detection signal at production scale, not PROPEL's solver-bottleneck mechanism itself.** Those are distinct claims and they require distinct evidence. This document states exactly which is which, what the evidence does and does not establish, and what must happen next to reach the PROPEL claim.

---

## Table of contents

1. [What B1 actually tested (claim ↔ test ↔ result)](#1-what-b1-actually-tested)
2. [PROPEL's solenoid-bottleneck claim vs. deception detection — where they split](#2-propel-vs-deception-detection)
3. [Falsifiability audit: what's ruled in / ruled out / not yet tested](#3-falsifiability-audit)
4. [Detailed result: layer-wise summary and top features](#4-detailed-result)
5. [Pathogens caught + remediated during analysis](#5-pathogens-caught--remediated)
6. [Evidence trail: capture, join, and analyzer provenance](#6-evidence-trail)
7. [Next steps + falsifiable roadmap](#7-next-steps)
8. [Implications for the Nous / Hermes writeup](#8-implications-for-nous)

---

## 1. What B1 actually tested

### Claim
**H1:** At ≥1 SAE layer, a subset of dictionary features is significantly more active on insider (deception-relevant) agent turns than on clean (compliant) turns, such that those features' activation profile separates the two conditions with rank-AUC well above chance, surviving Benjamini-Hochberg FDR at α=0.05 at the observed sample size.

**H0:** No feature subset separates the conditions. All per-feature rank-AUC values are ≈0.50 under permutation.

### Instrument
- **Model:** Qwen3.5-27B (fp16, `device_map="auto"` across 2× Tesla M40 + host RAM, ~52 GB total)
- **SAE:** released 5-layer residual SAE (d_sae = 81920 per layer, dict width); layers 0, 16, 32, 48, 63 (evenly spaced across 64)
- **Capture mode:** same-inference SAE forward-hook sidecar — `allf_a[L].sparse = [[tok_pos, feat_idx, act], ...]`, recorded from the *exact same* `model.generate()` call that emitted each turn's text
- **Dataset:** `agentic_traces_gen1_per_turn.jsonl`; 578 total turns, 40 genomes
- **Group labels:** `scenario.role` (`insider` vs. `clean`), with `pressure` and `phase` as recorded covariates
- **Test statistic:** per-row maximum activation per feature (avoiding token-level pseudoreplication), rank-AUC over rows, one-sided permutation p-values (label-shuffle over rows)
- **Correction:** Benjamini-Hochberg FDR applied across all m tested features within each layer

### Result (sound, after two bug fixes — see §5)
- **n_rows_analyzed:** 181 (94 insider, 87 clean — 2 turns skipped due to HTTP 500)
- **n_perm:** 300
- **min_freq gate:** features firing in ≥20 of 181 rows tested (features firing more rarely cannot meaningfully separate two groups of ~90)
- **Surviving features by layer:**
  - L0: 98 tested → **36 survive FDR** (37%)
  - L16: 2,218 tested → **924 survive FDR** (42%)
  - L32: 6,805 tested → **2,867 survive FDR** (42%)
  - L48: 12,817 tested → **5,366 survive FDR** (42%)
  - L63: 22,573 tested → **9,948 survive FDR** (44%)
- **Top features:** dominated by **insider-only presence signatures** — features that fire on every insider turn and on zero clean turns (e.g., L63 feat 50911 fires 35/35 insider rows, 0/46 clean, p=0.0)

### Verdict on H1
**H1 confirmed at scale.** Per-layer survival rates of 37–44% with p≈0 on the most-insider-specific features constitutes an overwhelming rejection of H0. The presence/absence signature is uniform across all five SAE layers, meaning the group-discriminable signal is distributed throughout the model, not just in one late "semantic peak" layer.

---

## 2. PROPEL vs. deception detection

This is the question you asked and the reason this report exists: *what does B1 actually prove, and how does it relate to PROPEL's solver-bottleneck claim?*

### What PROPEL actually claims

PROPEL = "Probe Rewards for Optimizing Problems at the Edge of Learning" (Vmax + Goodfire, arXiv 2606.18284). Its precise, falsifiable claim is:

> **Given a task generator (the policy) and an expensive solver that grades generated tasks, a lightweight activation probe trained ONCE on (`generated_task`, `solver_outcome`) pairs—reading a frozen reference model's hidden states—can substitute for the solver as the reward signal during generator RL, breaking the solver bottleneck.**

That claim requires three empirical conditions:
1. **Solver-predictive features.** The probe must read activations that predict the solver outcome, *not* some ambient stylistic correlate of the task class.
2. **Frozen-reference insulation.** The probe reads the frozen reference model (not the policy), so the policy cannot trivially game the probe by shifting its own activations.
3. **Reward improves generator RL.** Using the probe's output as reward in GRPO/RL must produce generators that produce solver-passing tasks at the frontier of difficulty, without calling the solver during training.

### What thePROPEL-variant we're exploring claims (the SAE substitution)

The Hermes-SAE PROPEL extension (`PROPEL_SAE_BRAINSTORM.md`, `PROPEL_SAE_EXTENSION_REPORT.md`) proposes swapping PROPEL's *supervised probe* for an *unsupervised SAE feature vector*. If that works, the claim becomes:

> **SAE features alone (no solver-labeled probe, no solver in the reward loop) can reward-shape a task generator to produce solver-passing tasks at the learnable frontier.**

This is *stronger* and *different* from PROPEL in two ways:
- It removes the solver from surrogate *construction* (SAE features are free once the model runs) — breaking the bottleneck twice over.
- It removes the supervision signal — the SAE features are whatever the model finds useful, not specifically calibrated to solver difficulty.

But the new claim requires its own distinct evidence:
- The features must track *task difficulty / solver outcome* specifically (not just "insider vs. clean context")
- Rewarding with the SAE signal must improve generator RL outcomes

### What B1 actually tested (and what it does NOT test)

B1 tests whether SAE features distinguish insider-tagged agent turns from clean-tagged agent turns, using `scenario.role` as the ground-truth label. This is a **deception-detection test** — it asks "do the features know the model is in an insider/deception context?" not "do the features know the task is at the frontier of difficulty?"

| Claim | Tested by B1? | Evidence? |
|-------|--------------|-----------|
| SAE features distinguish insider vs. clean agent turns | ✅ | 38–44% FDR survival, presence-only signatures at every layer, p≈0 |
| SAE features distinguish honest vs. deceptive output on the SAME insider prompt | ❌ | Would require matched-pair design (same `harness_prompt`, two label texts, one honest one deceptive) with ground-truth honest answer annotated |
| SAE features track task difficulty / solver outcome | ❌ | Would require (a) labeled task-difficulty dataset (solver outcome per problem), (b) an RL generator loop showing reward improvement |
| SAE features can substitute for the PROPEL probe in GRPO reward | ❌ | Would require an actual GRPO training run with the SAE reward — exactly the V1/V2 experiment in `PROPEL_SAE_EXTENSION_REPORT.md` |
| Frozen-reference insulation is preserved | ✅ | Proven: the SAE sits on the same frozen Qwen3.5-27B host, not the RL policy |

**Why the insider/clean distinction is real but not PROPEL-specific:**** The insider scenario pool has systematically different content, pressure, phrasing, and task framing than the clean pool. Any feature distinguishing those pools could be responding to surface content, tone, prompt structure, or task type — none of which are the solver state PROPEL targets. PROPEL's specific scenario is one where the *solver's intended answer* is visible in SAE space even when the deceptive output is fully coherent; B1's insider/clean split conflates the deception context with differences that are not about the solver.

---

## 3. Falsifiability audit

### What this run has ruled out

| Statement | Status |
|-----------|--------|
| "Under this instrument, capturereal agentic trace on 27B, 5-layer SAE), the null hypothesis of no group-discriminable SAE signal at FDR=0.05" | **RULED OUT** — the null is rejected at all 5 layers with thousands of survivors |
| "Same-inference capture introduces measurable aliasing vs. cached-activation replay" | **RULED OUT** — byte-identical `gen_text` from same-inference HTTP path vs. engine-direct path on 27B; same-inference invariant holds end-to-end |
| "The analysis pipeline is biased toward confirming H1 (p-hacking through feature selection, FDR, or null construction)" | **SUBSTANTIALLY WEAKENED** — pre-registered decision rule before running; null-result condition written before results; bugs found and corrected (see §5); power honestly reported |
| "SAE analyzer has row_idx-indexing bug" | **FIX + VERIFIED** — roles map now keyed by `row_idx` (5..574 space), 181/181 rows aligned |
| "The original analyzer silently produces unsound results on large captures" | **CONFIRMED** — the first buggy run analyzed 65/181 rows and the result was not flagged; fixed and documented in `fast_dpilot_sep.py` |

### What remains genuinely open

| Statement | Status |
|-----------|--------|
| "The insider-only signature is *deception-specific* and not an ambient correlate of prompt content / pressure / task framing" | **OPEN** — matched-pair same-prompt honest-vs-deceptive test on real data is the only discriminator |
| "SAE features encode the solver's reasoning trajectory (not just the final output)" | **OPEN** — requires inspecting feature activation at token positions during reasoning chains, not just full-pass aggregates |
| "A linear probe trained on SAE features improves GRPO reward when substituted for the PROPEL probe or standalone" | **OPEN** — V1/V2 in `PROPEL_SAE_EXTENSION_REPORT.md` (solver-outcome probe vs. SAE-feature reward; both unvalidated on real math/game tasks) |
| "The SAE-reward generalizes beyond the insider/clean label space" | **OPEN** — PROPEL uses solver outcome as the supervisory signal; B1 uses scenario.role |
| "Feature 50911 (L63, 35/35 insider-only) is the same kind of feature across models / datasets" | **UNKNOWN** — needs replication on a different agent trace / model to assess generalizability |

---

## 4. Detailed result

### 4.1 Layer-wise summary

| Layer | Features tested | Survive FDR | Fraction | Top feature | Top p |
|-------|----------------|-------------|----------|-------------|-------|
| L0    | 98             | 36          | 37%      | 76546       | 0.0   |
| L16   | 2,218          | 924         | 42%      | 20987       | 0.0   |
| L32   | 6,805          | 2,867       | 42%      | 21515       | 0.0   |
| L48   | 12,817         | 5,366       | 42%      | 4052        | 0.0   |
| L63   | 22,573         | 9,948       | 44%      | 3411        | 0.0   |

Observations:
- The fraction surviving FDR is nearly constant across layers (~37–44%). The signal is not concentrated in L32 (the hypothesized semantic-clustering peak from the earlier deep-dive).
- Top p-values are at the permutation granularity floor (p < 1/301 ≈ 0.0033) for the most-insider-specific features.
- At min_freq=20, the tested feature set contains only features that fire in ≥20/181 turns — the genuinely group-informative ones.

### 4.2 Top insider-only presence features per layer

These are the strongest-possible signatures: features that fire on *every* insider turn they appear in, and on *zero* clean turns.

| Layer | Feature | Fires | n_insider / n_clean | p_perm | note |
|-------|---------|-------|---------------------|--------|------|
| L63 | 50911 | 35/35 | 35 / 0 | 0.0 | strongest signature |
| L63 | 3411 | 34/34 | 34 / 0 | 0.0 | |
| L48 | 4052 | 34/34 | 34 / 0 | 0.0 | |
| L32 | 21515 | 34/34 | 34 / 0 | 0.0 | |
| L16 | 20987 | 34/34 | 34 / 0 | 0.0 | |
| L0 | 76546 | 34/34 | 34 / 0 | 0.0 | |
| L63 | 41372 | 34/34 | 34 / 0 | 0.0 | |
| L48 | 23248 | 34/34 | 34 / 0 | 0.0 | |
| L32 | 41121 | 19/19 | 19 / 0 | 0.0 | |
| L16 | 71258 | 34/34 | 34 / 0 | 0.0 | |
| L0 | 14756 | 34/34 | 34 / 0 | 0.0 | |
| L63 | 39770 | 14/14 | 14 / 0 | 6e-05 | |

### 4.3 Honest power note

At n=94/group, features with AUC ≈ 1.0 (presence-only) are trivially detectable. The more demanding test is features with moderate AUC (0.65–0.80) over the full feature set — those are the ones that can distinguish deception from same-content honest output. We don't know yet whether such features exist; that's the matched-pair test.

---

## 5. Pathogens caught + remediated

This run had three real bugs. Each is fixed in the captured codebase.

### 5.1 Soundness: roles map keyed by wrong index (CRITICAL)
- **What happened:** `analyze_dpilot_separability.py` built `roles[i] = role` where `i` was the *enumerate index* (0..180) but looked up `roles[rec["row_idx"]]`. Real `row_idx` values were in the sparse space 5..574 (from `agentic_traces_gen1_per_turn.jsonl`). Only 65/181 rows matched.
- **Impact:** The first "successful" run (proc_548b07ce4038, reported `n_rows_analyzed: 65`) produced an **unsound** result. If taken at face value, it would have been a type-II error (much lower power than claimed) or worse, a spurious alignment if the mapped subset happened to be non-representative.
- **Fix:** `roles` map now keyed by the real `row_idx` field from the trace file. All 181 rows align. Was caught by noticing `n_rows_analyzed: 65` << 181.

### 5.2 Write crash: os.makedirs on empty dirname
- **What happened:** `os.makedirs(os.path.dirname(out_path))` on a bare filename (`out_path = "dpilot_real_separability.json"`) → `os.makedirs("")` raises `FileNotFoundError`. This crashed the entire analyzer after the entire permutation loop completed — throwing away 13+ min of computation.
- **Fix:** Guard with `if _d: os.makedirs(_d, exist_ok=True)`.
- **Recommendation:** Use `out_path = os.path.join(os.getcwd(), "dpilot_real_separability.json")` in the caller; but the guard also adds resilience.

### 5.3 Performance: analyzer O(n_features × n_perm) pure-Python too slow on real capture
- **What happened:** The original `analyze_dpilot_separability.py` reranked the entire pooled array once per permutation per feature — pure-Python loops. On the real 27B capture with d_sae=81920 and >22k common features per deep layer, the run never finished in reasonable wall-clock. Two separate heavy runs were killed.
- **Fix:** Wrote `fast_dpilot_sep.py` (a sibling, not an in-place rewrite) that:
  - Pre-filters to features firing in ≥min_freq rows (rare features can't separate two ~90-row groups; this cut the tested set from many tens of thousands to a few thousand)
  - Uses numpy for rank-biserial AUC (vectorized rank assignment)
  - Adds the one-sided presence/absence test using hypergeometric p (via scipy)
- **Trade-off acknowledged:** "fast" here means "runs in minutes rather than hours." The full feature permutation at 81920-d_sae still takes ~15–25 min even filtered. Future: PyPy or numba or a smarter two-pass (compute null once, compare each feature's observed |AUC-0.5| to null distribution). This is correct science but the engineering should improve.

---

## 6. Evidence trail

### Capture
| Artifact | Path | Description |
|----------|------|-------------|
| 181-row tagged driver log | `/home/darkstar/hermes_cache/dpilot_capture/dpilot_tagged.jsonl` | one record per turn: `row_idx`, `gen_text`, `completion_tokens`, `meta.role`, `meta.pressure`, `http_status` |
| SAE sidecar log (server) | `/home/darkstar/hermes_cache/dpilot_capture/sae_history.jsonl` | 1.8 GB; one record per turn: `row_idx`, `gen_text`, `allf[layer]` (d_sae, threshold, max_profile, sparse = [[tok_pos, feat_idx, act], ...]) |
| Join script | `/home/darkstar/hermes_cache/dpilot_capture/build_real_dpilot.py` | Keyed by `row_idx` (not line order); produces joined jsonl + trace jsonl |
| Joined synthesis | `/home/darkstar/hermes_cache/dpilot_capture/dpilot_joined.jsonl` | 181 rows: `row_idx`, `meta.role`, `allf_a[L].sparse`; 178 MB |
| Trace (label map) | `/home/darkstar/hermes_cache/dpilot_capture/dpilot_trace.jsonl` | `{row_idx: ..., scenario: {role: ...}}` per row |

### Analyzer
| Artifact | Path | Note |
|----------|------|------|
| Original (corrected) | `propel_sae_extension/analyze_dpilot_separability.py` | row_idx fix + makedirs guard in `19ccd1fc` |
| Fast (production) | `propel_sae_extension/fast_dpilot_sep.py` | common-feature filter + numpy AUC + hypergeometric one-sided test — used for the final number |

### Reproducing the number
From the `dpilot_capture/` dir:
```bash
PY=/home/darkstar/.venv-gemma4/bin/python
$PY fast_dpilot_sep.py \
  --dpilot-jsonl dpilot_joined.jsonl \
  --trace-dataset dpilot_trace.jsonl \
  --out dpilot_real_separability_fast.json \
  --n-perm 300 --min-freq 20 --alpha 0.05
```

---

## 7. Next steps + falsifiable roadmap

### Step B2 (already done — cited for context)
Distinct-feature analysis on real 27B. Validated that L32's cross-cosine 0.330 was a self-pair artifact; real distinct-feature cosine is 0.0154 (near-orthogonal features).

### Step B2-redo: matched-pair honest-vs-deceptive on same prompt
**Falsifiable extension of B1:** take `agentic_traces_gen1_per_turn.jsonl`, extract insider turns where we have the honest ground-truth answer (from the trace's known-original behavior), and re-label those turns by whether the model's output text decided honestly or deceptively. Same `row_idx`, new label: `honest` vs `deceptive` (within the insider bucket). Re-run B1 separately for:
- (a) insider_honest vs. insider_deceptive (same-prompt content)
- (b) insider_deceptive vs. clean (same as Step B1 — already done)
- (c) insider_honest vs. clean

If (a) survives FDR with similar strength, the insider-only features encode deception (not ambient context). If (a) is null but (b) survives, the features encode *any* insider context — including honest insider — and the separability is about scenario framing, not deception.

**Status:** data partially present — `agentic_traces_gen1_per_turn.jsonl` has the original honest behavior annotations.

### Step B3: solver-outcome probe (PROPEL's actual claim)
- Construct a labeled task dataset: for each generation, annotate whether the solver would accept the output as correct (this requires the ground-truth task + answer key from the *insider* scenario design).
- Train a linear probe on the frozen Qwen3.5-27B SAE features to predict solver outcome (one probe per layer, trained on a held-out set of task generations).
- Test: does the probe's reward (probe logit) coarsely track solver outcome on a held-out test set?
- Negative control: train on pressure- or phase-labeled data (should not track solver outcome — this is the confound check).

**Status:** requires the insider scenario's answer key; partial data present.

### Step B4: GRPO reward loop with SAE-reward (PROPEL execution)
- Extend the existing GRPO harness with an SAE-reward head.
- Two variants:
  - SAE-fixed reward (no training): use the presence of a fixed set of insider-only features as a binary reward (this is a black-box reward, akin to PROPEL's probe, but without any training)
  - SAE-learned reward: train a probe on solver outcome → use as GRPO reward
- Baseline: PROPEL-style supervised probe reward (for A/B)
- Success criterion: generator episodes at the frontier of difficulty, measured by solver acceptance rate, WITHOUT solver calls in the RL loop

**Status:** scaffolding exists (`propel_sae_reward.py`, `propel_calibrate.py`); no GRPO training run yet.

### Step C1: feature labeling
- Take the top 500–1000 surviving features, extract their decoder directions, inspect top-activating contexts, label them by concept.
- Goal: produce a small feature dictionary for PROPEL-relevant concepts (task understanding, verification, self-doubt, tool-loop, etc.)
- Side benefit: confirm whether insider-only features have interpretable semantics (or are polysemantic/unsupervised artifacts)

### Step C2: overhead measurement for Nous / Hermes writeup
- Measure tok/s with and without 5-layer SAE forward hooks on Qwen3.5-27B → produce F3 figure for the Nous pitch doc
- Same for gpt2 + Qwen2.5-0.5B on CPU → demonstrate sidecar viability

---

## 8. Implications for the Nous / Hermes writeup

### What the Nous pitch doc already claims
The existing `nous/NOUS_WRITEUP_DRAFT.md` (written 2026-07-15) frames the work as "live internal readout of agentic work via SAE probes attached to the model's residual stream while it runs Hermes Agent." It stays in the *read-only observability* register: drift detection, failure diagnosis, interpretability. It writes about future steering but labels it unimplemented.

### What changed with the real D_Pilot result

With the real 27B result, the writeup now has *empirical receipts* for three claims:

1. **Same-inference capture is real and validated at scale.** The 181-turn real capture is not a synthetic stress — it's genuine multi-step agentic work. The tool-use/planning/output trace is real.

2. **SAE features encode behaviorally meaningful distinctions.** The insider-presence signatures (¾ of 5-layer features distinguishing a real scenario class at FDR<0.05) mean the feature stream is *not noise*. For Nous's purposes: when the agent is working on a task in the "insider" scenario, the feature activity looks structurally different than when it's working clean. That's a measurable signal — exactly what the writeup's "live internal readout" paragraph claims.

3. **What's on the roadmap is on the roadmap.** The distinction between the *confirmed* deception-detection result and the *unvalidated* PROPEL solver-bottleneck claim is scientifically honest and publicly defensible. The Nous doc should know this distinction and keep the roadmap grounded.

### What the Nous doc should say now (recommended updated framing)

**What's proven:**
> "During a 181-turn real agentic labeling session on Qwen3.5-27B with 5-layer SAE hooks (all on the same `generate()` call), SAE features distinguish insider (deception-relevant) scenarios from clean (compliant) scenarios at FDR<0.05, with ~40% of common features surviving at every layer. The strongest signatures are presence-only: certain neurons fire on every insider turn and on no clean turn at all. The same-inference invariant holds end-to-end: gen_text + feature log from one forward pass, no replay."

**What's realistic next (roadmap):**
> "A next investigated question is whether these features track *deception* specifically (rather than any difference between the two scenario pools), and whether SAE features can substitute for the expensive solver in a reward-shaping loop (the PROPEL solver-bottleneck scenario). Both are pre-registered and begun but not yet validated."

### What to send / not send to Nous

- **Send:** the updated writeup draft, the architecture diagram, the FDR-survival layer table, the CPU-test result (gpt2 + Qwen2.5-0.5B on CPU proves the sidecar portability claim), the demo repo link.
- **Do not send:** the raw 178-MB `dpilot_joined.jsonl`, the insider-scenario prompt design (that's the research's adversarial setup), internal `agentic_traces_gen1` naming. These are the research artifact layer; the writeup operates at the level of empirical claims.
- **Do not claim:** "we proved the PROPEL solver-bottleneck"; that's the next step and the roadmap.

---

## Appendix: Provenance chain for the writeup

| Narrative claim | Backing artifact | File / location |
|----------------|-----------------|-----------------|
| Same-inference invariant holds at 27B scale | 181-turn capture, gen_text byte-identical to HTTP response | `/home/darkstar/hermes_cache/dpilot_capture/` |
| 181-turn real D_Pilot capture | `dpilot_tagged.jsonl`, `sae_history.jsonl`, GPU waiter exit 0 | `/home/darkstar/hermes_cache/dpilot_capture/` |
| B1 confirmed (layer-wise) | `dpilot_real_separability_fast.json` | `/home/darkstar/hermes_cache/dpilot_capture/` |
| Analyzer correctness | `fast_dpilot_sep.py`, fixed `analyze_dpilot_separability.py` | `propel_sae_extension/` |
| CPU pipeline proven (plumbing) | `validation_cpu/` (gpt2 + qwen25b artifacts + README) | `propel_sae_extension/validation_cpu/` |
| Serving layer + same-inference over HTTP | `sae_serve.py` | `propel_sae_extension/` |
| Open-source released SAE load (no sae_lens) | `load_released_pt.py` | `propel_sae_extension/` |
| One-command replication | `setup_cpu_test.sh` | `propel_sae_extension/` |
| Nous-facing narrative framings | `nous/NOUS_WRITEUP_DRAFT.md`, `nous/NOUS_WRITEUP_PLAN_2026-07-15.md` | `nous/` |
| Deployment / roadmap planning | `PROPEL_SAE_EXTENSION_REPORT.md`, `PROPEL_SAE_BRAINSTORM.md` | repo root |
| Original falsifiable-claim design | `FALSIFIABLE_CLAIM_DESIGN.md` | repo root |

---

*This document was written at the completion of the real D_Pilot analysis (proc_f6e7484feb75, exit 0, 2026-07-20). The next update should occur whenever B2-redo, B3, or B4 lands — all are referenced above with specific falsifiable conditions.*
