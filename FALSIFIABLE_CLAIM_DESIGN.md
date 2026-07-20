# Falsifiable-Claim Experiment Design: SAE-Feature Separability of Insider vs Clean Agent Behavior
# Status: REVISED per Fable 5 verify/disverify audit (2026-07-19) · Author: Hermes Agent for Caleb DeLeeuw
# Date: 2026-07-19 · Companion to FABLE5_AUDIT.md and the PROPEL-SAE extension files.

## 0. Why this document exists
Fable 5's independent audit (FABLE5_AUDIT.md) flagged three DISVERIFIED headline claims in
the prior draft. This revision corrects them BEFORE any D_Pilot run:
  (A) "L32 is the semantic-clustering peak (cosine 0.33)" — DISVERIFIED: a duplicate-self-
      pair artifact (top-50 were token OCCURRENCES, 33–57% of cosine pairs were a feature
      compared with itself, cos=1.0; L16's top-50 was just two antipodal features). The
      sae_realfeat_analysis.py script is rewritten to use DISTINCT features + no self-pairs;
      it now reports DESCRIPTIVE stats only and does NOT assert semantic clustering.
  (B) The pre-registered decision rule "AUC>=0.70 AND FDR<0.05 at n=10/group" — DISVERIFIED
      as unsatisfiable: exact p(AUC=0.70 | 10v10) ≈ 0.065 > 0.05, so it cannot survive BH at
      any m; even AUC=1.0 fails BH over 81920 features unless >=9 features are simultaneously
      perfect. FIXED below (honest power + effect-size-driven rule).
  (C) "calibration rho=1.0 validates V1" — DISVERIFIED as circular (both signals strictly
      increasing in the same index). FIXED: the calibration self-test now validates the
      remediation DECISION LOGIC, not a meaningless rho=1.0.

## 1. The claim (stated before seeing results)
**H1 (specific):** At >=1 SAE layer, a subset of dictionary features is significantly more
active on `insider` (deception-relevant) agent turns than on `clean` (compliant) turns,
such that those features' activation profile separates the two conditions with rank-AUC
well above chance, surviving Benjamini-Hochberg FDR < 0.05 at the OBSERVED sample size.

**H0 (null):** No feature subset separates the conditions (all per-feature rank-AUC ~0.50).
The latent SAE features carry no behavioral signal distinguishing insider from clean turns.

A positive result supports "SAE features track agent behavioral intent." A null result does
NOT disprove deception detection — it bounds the claim under this instrument + dataset.

## 2. Design (falsification-first)
- **Unit of analysis:** one agent turn (`agentic_traces_gen1_per_turn.jsonl`), N≈578 turns
  across 40 genomes. Labelable text = `harness_prompt` (the action point).
- **Instrument:** same-inference SAE hooking (sae_labeled_course.py / sae_serve.py) on
  Qwen3.5-27B, 5 layers (0,16,32,48,63), captures full 81920-width activation profile per
  layer (`allf_a[L].sparse = [token_idx, feat_id, activation]`).
- **Ground truth:** `scenario.role` read per row_idx from the trace dataset
  (insider=1, clean=0). `pressure` (high/low/none) and `phase` recorded as covariates.
- **Test statistic (MATCHED, no pseudoreplication — audit fix 1b/1c):** each (feature, row)
  collapses to the MAX activation over that row's token occurrences; rank-AUC is computed
  over ROWS (per-row labels). The null uses the SAME per-row-max aggregation via label
  permutation. Observed and null are on the same statistic.
- **Multiple comparisons + FDR (audit fix 1c/§5):** for each feature, a permutation p-value
  is computed (label-shuffle over rows; per-feature valid null). Benjamini-Hochberg is
  applied across the m tested features (analyze_dpilot_separability.py implements bh_fdr).
  A feature "passes" iff its BH-adjusted p < alpha.
- **One-sided features (audit fix 1f/H2):** a feature firing ONLY on insider rows is the
  strongest possible signature (presence/absence). It is included via a dedicated presence
  statistic + permutation over row-presence (no longer silently dropped).
- **Covariate control:** within-layer, compare insider vs clean stratified by phase, and
  report AUC conditioning on pressure.

## 3. Pre-registered positive controls (must pass or the pipeline is broken)
1. **Synthetic signal control:** inject features firing ~2x on insider; the pipeline MUST
   rank them top and produce finite p-values (self-test does this; one-sided feat detected).
2. **Descriptive layer structure:** sae_realfeat_analysis.py reports DISTINCT-feature counts,
   cross-cosine (no self-pairs), max activation, and dead-feature fraction per layer. These
   are DESCRIPTIVE ONLY — no "clustering peak" claim is made. Any layer-specificity in the
   separability result is reported as an empirical finding, not a pre-justified prior.
3. **Null calibration:** under a no-signal injection, permutation p-values must be ~uniform on
   average (the pipeline produces finite p in [0,1]; a uniformity check belongs in the real run).

## 4. Pre-registered null-result condition (explicit)
If H0 holds (no feature survives BH-FDR at the observed n at any layer), the conclusion is:
> "Under the Hermes-SAE instrument (Qwen3.5-27B, 5-layer 80K SAE, agentic trace), latent
>  dictionary features do not, by themselves, separate insider/deceptive from clean/compliant
>  agent behavior at FDR<0.05. The SAE instrument remains valid for activation-recording; the
>  *deception claim* is NOT supported by this evidence and is bounded accordingly."
This is written now, before data, so a null cannot be spun into a win.

## 5. Power & sample size (HONEST — audit fix §5)
The prior "n>=10/group" gate was mathematically unsatisfiable for the pre-registered AUC
threshold. Corrected statement:
- At n=10/group, only features with AUC ~0.95+ survive BH over ~1000 tested features; an
  AUC~0.70 feature (d≈1.67) does NOT. This is a DATA/FACT of the test, not a bug.
- Recommended PRE-REGISTERED sample size for AUC~0.70 power (~50%): n ≈ 100/group
  (~200 turns, ~2700 GPU-hours at ~13.5 h/turn on dual-M40) OR episode-level aggregation
  (collapse per-turn AUC to per-genome) which raises effective n.
- The decision rule is EFFECT-SIZE-DRIVEN: H1 stands iff >=1 feature survives BH-FDR at the
  observed n. We report the min-p and the effect size (AUC) of any surviving feature, and we
  pre-commit that a result is only "supported" if it survives FDR — not at a fixed absolute
  AUC cutoff that chance can achieve at small n (expected no-signal top AUC ≈ 0.75 at the
  pilot's ~1000-feature scale).
- Until the larger sample is run, a pilot at n=10/group is explicitly a FEASIBILITY/
  PIPELINE-VALIDATION run, NOT a falsifiable test of H1 (the audit's M2 distinction, honored).

## 6. Decision rule (pre-registered)
| Observation | Conclusion |
|---|---|
| >=1 feature survives BH-FDR at the observed n (effect size reported) | H1 supported (bounded claim) |
| Signal only at deep layers | empirical layer-specificity; report |
| No feature survives FDR | H0 -> claim bounded as in §4 |
| Signal vanishes under phase stratification | Artifact (phase-correlated), NOT a behavioral finding |

## 7. Extensions this design enables
- **PROPEL-SAE:** separating features as a reward surrogate (needs full-scale H1 first — M2).
- **Multi-turn aggregation:** per-turn -> per-episode for higher power.
- **Contrastive pair test:** within-genome insider vs matched clean turn -> paired AUC.
