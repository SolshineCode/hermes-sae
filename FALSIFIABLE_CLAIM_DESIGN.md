# Falsifiable-Claim Experiment Design: SAE-Feature Separability of Insider vs Clean Agent Behavior
# Status: DRAFT (ready for D_Pilot data) · Author: Hermes Agent for Caleb DeLeeuw
# Date: 2026-07-18 · Companion to REPORT.md §11–§15 and D_PILOT_PLAN.md

## 0. Why this document exists
Fable 5's independent audit ("PARTIALLY INACCURATE") flagged that the deception claim
needs (a) an explicit behavioral ground truth and (b) a contrastive (honest vs deceptive)
split. This design pre-registers the falsifiable claim BEFORE D_Pilot runs, so the
result is a genuine test, not a post-hoc story. The agent-trace dataset already carries
the required split (`scenario.role ∈ {insider, clean}` + `pressure`), so no new field
is needed — the audit's objection is resolved by data that already exists.

## 1. The claim (stated before seeing results)
**H1 (specific):** At ≥1 SAE layer, a subset of dictionary features is significantly
more active on `insider` (deception-relevant) agent turns than on `clean`
(compliant) turns, such that those features' activation profile separates the two
conditions with rank-AUC ≥ 0.70 (controlling for `phase`/`pressure`).

**H0 (null):** No feature subset separates the conditions (all per-feature rank-AUC ≈ 0.50).
The latent SAE features carry no behavioral signal distinguishing insider from clean turns.

A *positive* result supports "SAE features track agent behavioral intent."
A *null* result does NOT disprove deception detection — it bounds the claim:
under this instrument + dataset, latent SAE features are not a cheap stand-alone
deception signal. That is still a publishable, falsification-first outcome.

## 2. Design (falsification-first)
- **Unit of analysis:** one agent turn (`agentic_traces_gen1_per_turn.jsonl`), N≈578 turns
  across 40 genomes. Labelable text = `harness_prompt` (the action point).
- **Instrument:** same-inference SAE hooking (sae_labeled_course.py) on Qwen3.5-27B,
  5 layers (0,16,32,48,63), captures full 81920-width activation profile per layer
  (`allf_a[L].sparse = [token_idx, feat_id, activation]`).
- **Ground truth:** `scenario.role` read per row_idx from the trace dataset
  (insider=1, clean=0). `pressure` (high/low/none) and `phase` recorded as covariates.
- **Test statistic:** per-feature rank-AUC (insider vs clean), computed by
  `analyze_dpilot_separability.py` (self-test validated: known signal → AUC=1.0;
  null-shuffle baseline ≈ 0.055). Threshold 0.70 chosen as a conventional "medium"
  effect before multiple-comparison correction.
- **Multiple comparisons:** 81920 features/layer × 5 layers. Report
  Benjamini-Hochberg FDR per layer. Claim H1 stands only if ≥1 feature survives
  FDR<0.05 at AUC≥0.70.
- **Covariate control:** within-layer, compare insider vs clean *stratified by phase*
  (so a feature that merely tracks "phase 3 vs phase 1" is not mistaken for
  "deceptive vs honest"). Also report AUC conditioning on `pressure`.

## 3. Pre-registered positive controls (must pass or the pipeline is broken)
1. **Synthetic signal control:** inject a feature that fires 2× on insider; the
   pipeline MUST rank it top (self-test already does this — AUC=1.0).
2. **Layer-known structure:** the real SAE deep-dive (sae_realfeat_analysis.json)
   found L32 is the semantic-clustering peak (top-50 decoder cosine 0.33 vs L16 0.0075)
   and activation mass grows ~1400× L0→L63. Prediction: if a behavioral signal exists,
   it is most likely at L32 or deeper — a directional prior, not a post-hoc excuse.
3. **Null-shuffle sanity:** shuffle roles across rows; top per-layer |AUC−0.5| must
   collapse toward ~0.05 (chance). Already observed in self-test.

## 4. Pre-registered null-result condition (explicit, per your falsification-first doctrine)
If H0 holds (no feature ≥0.70 post-FDR at any layer), the conclusion is:
> "Under the Hermes-SAE instrument (Qwen3.5-27B, 5-layer 80K SAE, 578-turn
>  agentic trace), latent dictionary features do not, by themselves, separate
>  insider/deceptive from clean/compliant agent behavior at AUC≥0.70. The SAE
>  instrument remains valid for activation-recording; the *deception claim* is
>  NOT supported by this evidence and is bounded accordingly."
This is written now, before data, so a null cannot be spun into a win.

## 5. What D_Pilot must produce
`sae_course.jsonl` (one row/turn) with `allf_a[L].sparse` for all 5 layers per turn,
joined to `row_idx` → trace `role`. The separability script consumes exactly this.
Hard-stop watchdog ensures the run ends before the 3da0c561 slot (08:35→12:35) expires.

## 6. Decision rule (pre-registered)
| Observation | Conclusion |
|---|---|
| ≥1 feature AUC≥0.70, FDR<0.05, survives phase/pressure stratification | H1 supported → "SAE features track behavioral intent" (bounded claim) |
| Signal only at L32+ (not shallow) | Consistent with deep-layer semantic encoding; report layer-specificity |
| No feature survives FDR | H0 → claim bounded as in §4 |
| Signal vanishes under phase stratification | Artifact (phase-correlated), NOT a behavioral finding |

## 7. Extensions this design enables (next steps, not part of the core test)
- **PROPEL-SAE:** use the separating features (or the max-AUC feature vector) as a
  *reward surrogate* in place of PROPEL's supervised probe — see PROPEL_SAE_BRAINSTORM.md.
- **Multi-turn aggregation:** collapse per-turn AUC to per-episode (genome) for a
  coarser but higher-power test.
- **Contrastive pair test:** within-genome, insider-turn vs its matched clean-turn
  (same genome, swapped scenario) → paired AUC, removes genome-level confounds.
