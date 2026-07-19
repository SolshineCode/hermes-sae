# PROPEL-SAE: Running Vmax's PROPEL Solver-Bottleneck Method with SAE Features
# Status: BRAINSTORM (Hermes Agent + Caleb) · 2026-07-18
# Extends: experiments/v8_nla_local/notes/PROPEL_VMAX_and_AR_AV_mapping.md
#   (that doc covers PROBE-as-reward; this one covers SAE-feature-as-reward —
#    the probe replacement/extension the prior doc deferred as "idea #5 / next escalation")

## Recap: what PROPEL actually does (solver-bottleneck method)
PROPEL = "Probe Rewards for Optimizing Problems at the Edge of Learning"
(Vmax + Goodfire, arXiv:2606.18284). The bottleneck it breaks:
- Naively training a *task generator* needs the **solver** to grade each generated
  problem → solver calls are the expensive inner loop → generator RL is slow/$$.
- PROPEL's fix: train a lightweight **activation probe ONCE** on
  (generated-task, solver-outcome) pairs, using a FROZEN reference model's hidden
  states. Then generator RL uses the probe's logit as reward → **zero solver calls
  in the loop**. The probe reads the frozen reference (NOT the policy) so the policy
  can't shift activations to game it. Min-over-ensemble (WCO) prevents collapse.

Core transferable idea: **replace the expensive solver with a cheap, frozen,
activation-derived reward surrogate.**

## The SAE substitution (the novel part you asked for)
We already have, WORKING and validated:
- The same-inference SAE hook (sae_labeled_course.py) on Qwen3.5-27B.
- 5 SAE layers, 81920 dict features each, capturing the EXACT activation profile
  that produced a given output (faithful — no replay).
- A real-data deep-dive showing L32 is a semantic-clustering peak and activation
  mass grows 1400× L0→L63.

PROPEL uses a **supervised probe** (trained on solver outcomes) as the surrogate.
We can substitute/extend with **SAE features**, which are *unsupervised* but
*monosemantic-ish* and already span the full latent space:

### Mapping table (PROPEL probe  ⇄  Hermes-SAE feature)
| PROPEL concept | PROPEL impl | Hermes-SAE equivalent |
|---|---|---|
| Frozen reference model | frozen base LM | Qwen3.5-27B fp16 (our SAE host) |
| Activation probe (reward surrogate) | linear head trained on solver outcomes | **SAE top-K feature vector** (unsupervised; no solver needed) |
| Probe reads frozen ref → policy can't game | yes | SAE is on the *reference* host, not the policy → same game-proof property ✓ |
| Reward = probe logit | scalar | reward = f(SAE feature activations) — see variants |
| WCO min-over-ensemble | 2+ probes | min/max over feature *subsets* or over layers |
| Learnable-frontier band (curriculum) | sample tasks at probe-decodability edge | sample generations at SAE-activation frontier (high-but-not-saturated mass) |

### Why SAEs could be BETTER than probes here
1. **No solver needed to BUILD the surrogate** — PROPEL needs (task, solver-outcome)
   pairs to train the probe. SAE features are free once the model runs. The solver
   bottleneck is broken *twice over*: not just removed from the RL loop, but removed
   from surrogate *construction* too.
2. **Game-proof by architecture**: SAE sits on the frozen reference; a trained policy
   can't shift reference activations (PROPEL's own safety property, inherited).
3. **Interpretable reward**: a probe is a black box; an SAE-feature reward is
   inspectable ("we rewarded feature 20857 firing, which we independently found
   corresponds to X"). That's a credibility win for the worst-case-across-seeds
   reporting doctrine.

### Why probes might still be KEPT (hybrid)
- SAE features are *unsupervised* — they encode whatever the model finds useful,
  not necessarily "task difficulty / solvability." PROPEL's probe is *trained* to
  predict solver outcome, so it's directly difficulty-aligned.
- Hybrid: use SAE features for the *cheap, always-on* reward signal (no solver),
  and keep a (rarely-called) solver-labeled probe for *calibration* (does the
  SAE-feature reward track real solver outcomes? periodic audit, not inner loop).

## Concrete PROPEL-SAE reward variants (for the Hermes harness)
Let `F(x) = allf_a[L].sparse` be the SAE feature activation vector for generation x.

- **V1 — Frontier-mass reward (closest to PROPEL curriculum):** reward = total
  activation mass in a target band (e.g. layer L32, act ∈ [5, 50]). Encourages
  generations at the "learnable frontier" (rich-but-not-saturated). Directly
  maps PROPEL's "train generator at the edge of learning."
- **V2 — Separating-feature reward (builds on D_Pilot):** once D_Pilot finds the
  insider/clean-separating feature set S*, reward = Σ_{f∈S*} act_f. This is the
  SAE analog of PROPEL's probe-logit reward, but the "probe" is *discovered*
  unsupervised from the falsifiable-claim experiment. (Probe replacement achieved.)
- **V3 — WCO min-over-layers:** reward = min_L( frontier_mass(L) ). PROPEL's
  anti-collapse min-operator, applied across SAE layers instead of across probes.
- **V4 — Validity-gated:** invalid/empty generation → r_bad (PROPEL's hard gate),
  applied before any SAE reward is scored.

## What "overcoming the solver bottleneck in the Hermes harness" means concretely
Today the NLA GRPO lever uses tf-idf content checker (cheap) or AR-as-reward
(broken at V8 scale, per PROPEL mapping doc §"caveat"). PROPEL-SAE offers a
THIRD verifier — SAE-feature-derived — that is:
- cheaper than a solver (zero solver calls),
- harder/more-gameable-resistant than tf-idf (captures paraphrase/semantics via
  features, not surface tokens),
- inspectable (unlike AR-which-is-content-blind or a black-box probe),
- already built + validated in this very repo (sae_labeled_course.py, health-checked).

## Open questions to resolve WITH Caleb (brainstorm, not decisions)
1. **Reward-target semantics:** do we want the SAE reward to track *task
   difficulty* (PROPEL's goal) or *behavioral intent* (D_Pilot's finding)? These
   may need different layers (difficulty ≈ mid, intent ≈ deep/L32).
2. **Supervise or not:** pure-unsupervised SAE reward (V1/V2) vs hybrid with a
   solver-calibrated probe (keeps difficulty-alignment).
3. **Policy vs reference:** PROPEL keeps probe on frozen reference. In our
   actor-critic (AV/AR), would the SAE reward come from the *policy's* own
   activations or a frozen reference's? Game-proofing argues for frozen reference.
4. **Scale:** 81920 features × 5 layers is a big reward space. Do we pre-select
   a sparse feature subset (from D_Pilot separability) or use full-vector
   compression (e.g..decoder-norm weighting)?
5. **Faithfulness under RL:** PROPEL's safety is "policy can't shift frozen-ref
   activations." If we instead read the *policy's* activations during generation,
   the SAE sees policy-shifted states → potential reward-hacking. Must use a
   frozen reference host (as we do now) or risk collapse.

## Next-step proposal (if Caleb green-lights)
- Step 1 (this week, cheap): D_Pilot delivers separability → gives us a real
  SAE feature subset (V2 reward candidate).
- Step 2: add a `--sae-reward V1|V2|V3` flag to a GRPO harness variant that
  reads the frozen-reference SAE vector as reward; first test on the NLA frontier
  curriculum (replaces tf-idf arm).
- Step 3: hybrid calibration — periodically score SAE-reward vs a real solver on
  a held-out set; if correlation < threshold, fall back to probe.
