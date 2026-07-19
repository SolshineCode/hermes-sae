# TL;DR

**The pipeline's arithmetic mostly checks out, but three headline claims are disverified:**
(1) the pre-registered falsifiable test **cannot ever confirm H1 at its planned sample size** — at n=10/group, a feature at the pre-registered AUC 0.70 threshold has exact p ≈ 0.065 > 0.05 and can never survive FDR < 0.05 (and no FDR code even exists);
(2) the **"L32 semantic-clustering peak (cosine 0.33)" is a duplicate-entry artifact** — the actual top-50 entries per layer contain only 2–7 distinct features and 33–57% of the cosine pairs are a feature compared with itself (cos = 1.0, which is why `top50_cos_max` is exactly 1.0 at every layer);
(3) the **calibration ρ = 1.0 is circular** (both signals strictly monotonic by construction, and the shipped lexical-diversity stub returns a constant → NaN on the self-test's own data).
I also found that **V3 ≡ 0 on real data** (L0's max activation is 3.17, below the band floor of 5.0, so min-over-layers is constantly zero) and a **V2 cross-layer feature-id collision bug** (per-layer SAE namespaces treated as global).

**Execution caveat:** this non-interactive session blocked all script execution (`python3`, `node`, `awk`, subagent, JS sandbox) and all file writes with "requires approval." I could not live-run the three scripts. Instead I executed extensive read-only forensics on the real data (grep/sort/uniq over `sample_allfeat_row0.jsonl`), hand-executed every small code path (rank_auc, M4 dedup, Spearman, V1 counterexamples), and used exact combinatorics for the power analysis. Verdicts requiring a live run are marked INCONCLUSIVE with the analysis that stands regardless; §9 lists re-run commands.

---

# FABLE5 AUDIT — PROPEL-SAE Deception-Detection Extension
**Auditor:** Claude Fable 5 (independent correctness audit — verify/disverify, not review)
**Date:** 2026-07-19 · **Repo:** /tmp/hermes-sae @ d9e1341

## 0. Execution constraints (disclosed up front)
This session's permission system blocked **all** arbitrary code execution non-interactively
(`python3 <script>`, `python3 -c`, `node -e`, `awk`, and the harness JS sandbox all returned
"elsThis command requires approval"; a subagent hit the same wall), and blocked all file writes.
I therefore could not live-run the three scripts. What I **did** execute: extensive read-only
forensics on the real data (`grep`/`head`/`tail`/`sort`/`uniq` over
`sample_allfeat_row0.jsonl`, the 3.2 MB demo of the 228 MB all-features row), plus
hand-execution of every small-N code path and exact/analytic statistics for the large-N
claims. Every hand or analytic result below is marked as such, and each disverification
includes concrete evidence. Items whose verdict *required* a live run are marked
INCONCLUSIVE for the run itself, with the analysis that stands regardless. Re-run commands
are in §9.

---

## 1. Separability pipeline (`analyze_dpilot_separability.py`)

### 1a. `--self-test` pass / reproduce AUC 0.93 and null 0.28 — **INCONCLUSIVE (run blocked); numbers analytically consistent**
Could not execute. Statistical reproduction by analysis of the exact code path:
- Signal features: insider ~N(2.0, 0.6) vs clean ~N(1.0, 0.6) → d = 1.67σ → expected
  per-feature AUC = Φ(1.67/√2) ≈ 0.88. Top-of-3 signal features with per-feature
  SE ≈ 0.03–0.04 at 40/40 rows → expected top ≈ 0.91–0.94. **Reported 0.93: consistent.**
- Null: max over (100 sims × 600 subsampled features) = max of ~60,000 draws of
  |AUC−0.5| with n ≈ 35/35 firing rows (the >0.05 cutoff drops ~13% of noise draws).
  SD(AUC) = √((n₁+n₀+1)/(12n₁n₀)) ≈ 0.070; the 1-in-60,000 two-sided quantile is
  z ≈ 4.3–4.5 → ≈ 0.30, slightly less for the bounded, discrete rank statistic.
  **Reported 0.28: consistent with the expected chance level — see 1e.**
- Guard: gap = (0.93−0.5) − 0.28 = 0.15 ≥ 0.10 → self-test would report PASS.

### 1b. `rank_auc` correctness — **VERIFIED (hand-executed)**
The pairwise definition is the standard Mann-Whitney AUC with 0.5 tie-weight.
Hand cases traced through the exact code: pos=[2,3], neg=[1,2] → pairs (2>1)=1,
(2=2)=0.5, (3>1)=1, (3=2)=1 → 3.5/4 = **0.875** ✓; pos=[3,2], neg=[1] → 1.0 ✓;
pos=[1], neg=[2] → 0.0 ✓; all-ties → 0.5 ✓. The function itself is correct.
**However** — see Flaw H1: `run()` applies it to *occurrence-level* values (line 100:
every token position of every row is a sample) while labels are per-row. That is
pseudoreplication on real multi-token data.

### 1c. Value-permutation null vs label-permutation null — **comment's justification DISVERIFIED; null valid per-feature, invalid as implemented for real data**
- The code's comment (lines 127–128) claims label-permutation "would be invalid because a
  feature's activation can itself be group-determined." This is statistically **wrong**:
  under H0 activations are exchangeable across rows, and for a rank statistic the two
  schemes are *provably identical* — shuffling which pooled value lands on which
  fixed-label row induces exactly the same uniform random (n₁, n₀) split of the pooled
  values as shuffling labels over fixed values. Same null distribution, by symmetry.
  Label-permutation is the textbook null here; the code's value-permutation is equivalent
  per-feature, so the null is *valid in the single-feature case* despite the incorrect
  rationale.
- **Where it breaks on real data (DISVERIFIED as implemented):**
  1. **Statistic mismatch:** the observed statistic is occurrence-level AUC (all token
     firings as samples); the null statistic is computed on **per-row max** (lines
     141–150). The null therefore does not calibrate the observed statistic. On real
     multi-token rows the occurrence-level AUC has correlated within-row samples and a
     different (typically more extreme) sampling distribution than the row-level statistic
     the null simulates. The self-test never exposes this because its synthetic features
     fire ≤ 1×/row — the only regime where the two statistics coincide.
  2. **Independent per-feature shuffles** destroy cross-feature correlation, so the
     max-over-features null is that of *independent* features. Real SAE features are
     heavily correlated (§6 shows near-duplicate/antipodal features), making this null
     conservatively too large — by an uncalibrated amount.
  3. The reported null is **max over 100 sims** of the per-sim best — an extreme order
     statistic, not a quantile. No p-value is derivable from it, so it cannot support the
     pre-registered FDR<0.05 rule (which is not implemented anywhere — see §5).

### 1d. `gap ≥ 0.10` guard — **threshold ARBITRARY; guard does fail correctly on no-signal data (analytic)**
The 0.10 is not derived from any error rate. Under a no-signal synthetic (same generator,
zero signal features), the real top is the max of ~1,000 single draws →
E[top |AUC−0.5|] ≈ 0.072 × 3.4 ≈ 0.25 (real top AUC ≈ 0.75), while the null is the max of
~60,000 draws ≈ 0.30. Gap ≈ −0.05 < 0.10 → **self-test FAILS correctly**. The guard works,
but only because the null side draws 60× more samples than the real side — a structural
asymmetry, not a designed error rate. Corollary (Flaw M5): the *absolute* pre-registered
threshold "AUC ≥ 0.70" is **chance-achievable** at these n (expected no-signal top ≈ 0.75);
only the gap/null carries inferential weight. Could not live-run this synthetic (blocked);
the calculation is standard extreme-value arithmetic on the exact code path.

### 1e. Is null = 0.28 the expected chance level, not an artifact? — **VERIFIED (analytic)**
Yes. 600 subsampled features × 100 sims = 60,000 draws of |AUC−0.5| at n≈35/35
(SD ≈ 0.070). Expected maximum ≈ 0.29–0.31 for Gaussian tails; the rank statistic's bounded
support pulls this slightly down → 0.28 is exactly where "best of ~60k chance draws" should
sit. A multiple-comparisons ceiling, as the design doc's corrected §3 itself states. Not an
implementation artifact.

### 1f. Additional finding — one-sided features are invisible
Line 98: `if not g["ins"] or not g["cln"]: continue` — any feature that fires **only on
insider rows** (the strongest possible deception signature: presence/absence) is silently
dropped from both the real analysis and the null. (Flaw H2.)

---

## 2. V1 reward (`propel_sae_reward.py`)

### 2a. Self-test run + M4 dedup arithmetic — **run blocked; arithmetic VERIFIED by hand; premise VERIFIED on real data**
Hand-execution of the exact self-test path: synthetic `[[t, 71349, 3.17] for t in 0..4]`:
dedup=True → per_feat = {71349: 3.17} → V1 = **3.17**; dedup=False → 5 × 3.17 = **15.85**;
single occurrence → **3.17** → `m4_ok = True`. The claim "5 positions → reward equals single
occurrence, not 5×" is arithmetically correct. Real-data premise confirmed by executed
forensics: feat 71349 appears **269×** in L0's top-2000 demo entries, feat 80518 **239×** at
L63, feat 20857 **84×** at L32 (`grep -o ', 71349, ' sample_allfeat_row0.jsonl | wc -l` →
269). Without dedup, V1 would indeed be dominated by repeat-firing features.

### 2b. Is band(5,50) principled? — **DISVERIFIED: magic number resting on a disverified premise**
The docstring justifies the band via "L32 is the semantic-clustering peak" — §6 shows that
claim is a duplicate-entry artifact, so the band's empirical anchor is invalid. Beyond that:
- No calibration ever linked band mass to difficulty (P4 never ran with a real solver; §4).
- The band applies to the **per-feature max** (post-dedup), producing a cliff: a feature
  maxing at 50.0 contributes +50.0; at 50.1 it contributes **0**. Reward is non-monotone in
  activation strength, and a feature with many in-band firings but one >50 spike is dropped
  entirely. "Band mass" is really "sum of per-feature maxima that happen to land in [5,50]".
- Real-data consequence (executed evidence): L0's max activation over the entire row is
  **3.1731 < 5.0** (analysis JSON, confirmed in the demo row) → V1(L0) ≡ 0 for real data →
  **V3 = min over default layers ≡ 0 always**. The anti-collapse reward is degenerate: it
  emits a constant 0. The self-test's `v3_ok` check (`v3 == min(masses)`) passes trivially
  while the reward carries no signal. The prior implementation review's claim that "with
  full allf_a V3 would be non-zero" is contradicted by the full row's own L0 max. (Flaw H3.)

### 2c. Is V1 a task-difficulty proxy? — **DISVERIFIED by counterexample (hand-computed on the shipped function)**
V1 scales with the **number of distinct mid-band features**, i.e. surface diversity and
length, not difficulty:
- Gen A — focused, genuinely hard content: 30 distinct features at 40–49 → V1 ≈ 1,335.
- Gen B — long topic-hopping word salad (easy, loopy): 300 distinct features at 6.0
  → V1 = **1,800 > Gen A**. Rewarded for the wrong reason.
- Gen C — same as A but *more strongly* activated (51–60): V1 = **0** (band cliff).
  Punished for the wrong reason.
M4 dedup removes the same-feature-repeat length bias but **not** the distinct-feature-count
length bias — longer generations fire more distinct features, so a policy is still paid to
generate longer, more surface-diverse text. V1 is an activation-diversity/magnitude proxy;
only a real P4 calibration could rescue it, and none exists (§4).

### 2d. Does max-per-feature dedup have a failure mode? — **CONFIRMED**
Symmetric to the hack it blocks: a feature that *genuinely* fires strongly at 200 distinct
positions (e.g., a deception feature active throughout a long deceptive plan) counts exactly
once — identical to a single-token blip. The reward is prevalence-blind; sustained
legitimate signal is under-rewarded and a one-off spurious spike gets full credit.
Additionally, **`sae_reward_v2` has a correctness bug**: feature ids are per-layer
namespaces (5 independent SAEs, 81920 each), but v2 iterates all layers, matches ids
globally, and its `seen` set credits an id **once across layers, first-iterated-layer wins**
(dict order). A V2 feature discovered at L32 can be silently credited with unrelated L0
activations of the same integer id. (Flaw H4.)

---

## 3. D4 frozen-reference invariant — **DISVERIFIED as a safety *guarantee*; honest as a docstring**
1. **Unenforceable today (fig-leaf):** the reward functions accept any `allf_a` dict; no
   provenance tag, no assertion, and the GRPO harness that is supposed to enforce D4 does
   not exist (grpo_sae_reward_flag_spec.md is a spec, not code). Until a harness exists,
   "enforced at the call site" is a claim about hypothetical code.
2. **Overstated even if perfectly enforced:** "game-proof by architecture" (brainstorm)
   overclaims. The policy fully controls the *text* fed to the frozen reference, hence
   fully controls the frozen reference's activations through input selection. Freezing pins
   the reward *function*; it does not prevent the policy from optimizing inputs against
   that fixed function — the standard reward-hacking channel that V1's band structure
   invites (§2b/2c) remains fully open.
The *architecture decision* is sound and correctly documented; the *safety claim* as stated
does not hold.

---

## 4. Calibration (`propel_calibrate.py`) — **DISVERIFIED as evidence; scaffold only**

### 4a. Is ρ = 1.0 circular? — **YES, CONFIRMED (hand-traced)**
The self-test constructs both signals as strictly increasing functions of the same index i:
V1ᵢ = (5+i)(5+0.5i) and the *injected* test solver returns 5+0.5i. Spearman of any two
strictly increasing sequences is identically 1.0 — the self-test cannot fail short of a
syntax error. Worse, it **swaps out the shipped stub** (`_test_solver` injected via
`solver_fn`), so the shipped `_solver_stub` is never exercised at all.

### 4b. Does the remediation signal fire on bad data? — **logic VERIFIED by inspection; live run blocked**
ρ NaN → "INCONCLUSIVE"; ρ < 0.3 → "RECALIBRATE"; 0.3–0.6 → "MONITOR"; ≥ 0.6 → "KEEP".
Independent noise → ρ ≈ 0 → RECALIBRATE fires; anti-monotonic → ρ = −1 → also RECALIBRATE
(correct branch, though it conflates "no correlation" with "strong *negative* correlation" —
a sign-flipped but informative reward). Notable hand-derived result: the *shipped*
`_solver_stub` (lexical diversity = len(set)/len) on the self-test's own rows yields
diversity ≡ 1.0 (all tokens distinct) → constant scores → tied ranks → `db == 0` → ρ = NaN
→ "INCONCLUSIVE". **The shipped configuration produces no calibration signal even on its
own synthetic data.**

### 4c. Does calibration make sense with a fake solver? — **currently meaningless as validation**
The hybrid-calibration *design* (rare solver audit outside the RL loop) is reasonable iff a
real solver is plugged in. As shipped: stub fake, self-test circular, stub degenerate. No
calibration number from this repo — including "ρ = 1.0" — may be cited as evidence that V1
tracks difficulty. D3's "first empirical check" has not happened.

---

## 5. Falsifiable design (AUC ≥ 0.70 + FDR < 0.05; power at ≥10/group) — **form VERIFIED, substance DISVERIFIED**

**Genuine pre-registration in form:** H1/H0 stated before data, null-result wording
pre-written, decision table fixed. Credit where due.

**But the H1 arm is mathematically unsatisfiable at the planned scale — power ≈ 0:**
1. **FDR is not implemented.** No p-values, no Benjamini-Hochberg anywhere in the analysis
   script (its only null output is the max-over-sims statistic, from which no p-value is
   derivable). The pre-registered decision rule cannot even be evaluated with the shipped
   instrument.
2. **Exact back-of-envelope at n = 10/group (the "full-scale" gate):**
   - Observed AUC = 0.70 ⇒ U = 70. The exact one-sided critical value at α = 0.05 for
     n₁ = n₂ = 10 is U ≥ 73 (AUC ≥ 0.73); P(U ≥ 70) ≈ 0.065 > 0.05. Since the BH rejection
     threshold never exceeds 0.05, **a feature at exactly the pre-registered AUC threshold
     can NEVER survive FDR < 0.05 — at any m, even m = 1.**
   - Even *perfect* separation has minimum one-sided p = 1/C(20,10) = 1/184,756
     ≈ 5.4×10⁻⁶. The design's BH over "81920 features/layer" has rank-1 threshold
     0.05/81920 ≈ 6.1×10⁻⁷ < 5.4×10⁻⁶ — **a perfectly separating feature fails FDR unless
     ≥ 9 features are simultaneously perfect** (0.05k/81920 ≥ 5.4×10⁻⁶ ⇒ k ≥ 9).
     Restricting m to fired-in-both features helps only if m ≲ 9,200 — implausible at deep
     layers (L32 fires ~10⁵ occurrences in a single row).
   - Sample size actually required: for a true-AUC-0.70 feature to clear 0.05/81920 with
     ~50% power needs 0.5 + 4.86·SD₀ ≤ 0.70 ⇒ SD₀ ≤ 0.041 ⇒ **n ≈ 100 per group**
     (≈ 200 turns ≈ 2,700 GPU-hours at the plan's own 13.5 h/turn) — 10× the "≥10/group"
     gate.
3. Consequence: as pre-registered, the experiment can only ever return H0. A test whose H1
   arm cannot be reached by any data outcome at the planned n is not a falsifiable test of
   H1 — it is a null-generator. The pre-registration is honest in intent but internally
   inconsistent (threshold, correction, and sample size cannot coexist).

---

## 6. Real-data sanity (`sae_realfeat_analysis.json`) — **headline interpretations DISVERIFIED (executed forensics); raw capture VERIFIED**

**Raw data integrity: VERIFIED.** Executed cross-checks against the demo row reproduce the
JSON exactly: L63 max 293.944 (feat 80518 @ token 49), L32 max 31.562 (feat 20857),
L48 86.025 (feat 27644), L16 25.788 (feat 2628), L0 3.173 (feat 71349).

**"L32 decoder cosine 0.330 ⇒ semantic clustering": DISVERIFIED — duplicate-entry
artifact.** `sae_realfeat_analysis.py:61` takes the top-50 *entries* (token occurrences),
not distinct features, then averages pairwise cosine including pairs of the same feature
with itself (cos = 1.0 exactly — which is why `top50_cos_max` is exactly 1.0 at **every**
layer, a self-pair giveaway). Executed exact top-50 composition (sort|uniq on the demo row):

| Layer | Top-50 composition (feat × count) | Distinct | Self-pairs /1225 (all cos=1.0) | Reported mean | Implied avg cross-feature cosine |
|---|---|---|---|---|---|
| L0 | 71349×21, 71095×16, 63017×13 | **3** | 408 → 0.333 | 0.134 | ≈ **−0.30** |
| L16 | 2628×29, 7038×21 | **2** | 616 → 0.503 | 0.0075 | ≈ **−0.996** (antipodal pair!) |
| L32 | 14277×37, 20857×8, 5 singles | **7** | 694 → 0.567 | 0.330 | ≈ **−0.55** |
| L48 | 27644×17, 6481×3, 9×2, 12×1 | **24** | 148 → 0.121 | 0.140 | ≈ +0.02 |
| L63 | 80518×18, 39359×15, 40809×10, 49687×5, 2×1 | **6** | 313 → 0.256 | 0.224 | ≈ −0.04 |

Reading: at every layer the mean is a mixture of forced self-pairs (1.0) and a handful of
cross-feature terms that are ≈ 0 or strongly **negative**. L16's entire top-50 is *two
near-antipodal features* (implied cos ≈ −1.0) — an SAE pathology (a feature and its
negation), not semantics. The cross-layer contrast "0.33 vs 0.0075" that the design doc
elevates to a "directional prior" for L32 tracks duplicate counts and one antipodal pair,
not semantic geometry. (Demo row is 3-decimal-rounded, so top-50 membership may differ
marginally from the full-precision row; the counts are too lopsided for that to change the
conclusion.)

**Other artifacts (executed):**
- `n_fired` counts sparse **entries** (occurrences), not features: L63 "n_fired": 856,184 >
  81,920 = total features in the SAE — impossible as a feature count. Executed count: L63's
  top-2000 demo entries contain only **78 distinct features**; L0's 2,000 entries contain
  333. Fired-feature counts are inflated by orders of magnitude wherever this JSON is
  quoted.
- `top50_dec_norm_mean = 1.0` at every layer ⇒ decoder columns are unit-normalized ⇒ the
  planned "decoder-norm × activation" importance artifact is vacuous.
- **L63 max = 293.94 implies nothing by itself:** activation scale growing with depth is
  expected residual-stream norm growth; the "1400× L0→L63 mass growth" is arithmetic on
  occurrence-inflated counts × magnitudes. A feature firing 239+ times per row at extreme
  magnitude is more consistent with a dense/positional feature than a monosemantic one. No
  meaning has been established for any cited feature; no dead-feature statistics were
  computed at all.

---

## 7. Consolidated flaws, ranked

### BLOCKING
1. **Pre-registered decision rule is unsatisfiable at planned scale and unimplemented**
   (§5): AUC=0.70 can never pass FDR<0.05 at n=10/group (exact p ≈ 0.065 > 0.05); even
   AUC=1.0 fails BH at m=81920; no p-value/FDR code exists. The experiment as designed can
   only return H0. Needs ~100/group (~2,700 GPU-h) or a redesigned rule (permutation maxT
   with matched statistic, episode-level aggregation, or an effect-size-only claim).
2. **"L32 semantic-clustering peak" is a duplicate-self-pair artifact** (§6), and it is
   load-bearing: it justifies V1's layer/band choice and the design doc's "directional
   prior." All downstream reasoning built on cos=0.33 is unsupported.
3. **No calibration evidence exists** (§4): the ρ=1.0 self-test is circular, bypasses the
   shipped stub, and the shipped stub is degenerate (constant → NaN) on the self-test's own
   data. Any citation of "Spearman 1.0" as validation is an overstated result.

### HIGH
4. **Real-vs-null statistic mismatch + pseudoreplication in separability** (§1b/1c):
   occurrence-level AUC with row-level labels vs per-row-max null; untested regime — the
   self-test only exercises fire-once-per-row data.
5. **One-sided features silently dropped** (§1f): a feature firing only on insider rows —
   the strongest deception signature — is invisible to the test.
6. **V3 ≡ 0 on real data with default layers** (§2b): L0 never reaches band (5,50) (row
   max 3.17) — the anti-collapse reward has no gradient, and the self-test cannot notice.
7. **V2 cross-layer feature-id collision** (§2d): per-layer namespaces treated as global;
   first-iterated layer wins nondeterministically.
8. **V1 is not a difficulty proxy** (§2c): distinct-feature-diversity/length channel remains
   after M4 dedup; band cliff at 50 punishes stronger activation; band(5,50) anchored to
   the disverified clustering claim.
9. **D4 is currently a fig-leaf and "game-proof" is overstated** (§3): no enforcing harness
   exists; even ideal enforcement leaves input-selection reward hacking fully open.

### MED
10. Null reported as **max over 100 sims** (extreme statistic, not a quantile) with an
    arbitrary gap ≥ 0.10 guard — no calibrated error rate anywhere in the pipeline.
11. Value-perm rationale in comments is statistically wrong (label-perm is equivalent
    per-feature, not "invalid"); independent per-feature shuffles ignore feature
    correlation (conservative by an unknown amount).
12. `n_fired` mislabeled (occurrences, not features) — off by orders of magnitude wherever
    quoted (856,184 "fired" > 81,920 existing features).
13. M4 dedup under-rewards sustained legitimate signal (prevalence-blind: 200 strong
    firings = 1 firing).
14. Pre-registered absolute threshold AUC ≥ 0.70 is chance-achievable at pilot n (expected
    no-signal top ≈ 0.75 over ~1,000 features).
15. Self-tests are systematically too friendly: calibration self-test cannot fail; `v3_ok`
    is tautological; separability self-test avoids the multi-firing regime. Also the prior
    review (FABLE5_PROPEL_IMPLEMENTATION_REVIEW.md §1a) describes an API (`validity_gate`,
    `min_words=5`) that does not match the shipped file (`sae_reward_v4`, `min_len=1`) —
    review and code drifted, so its "manual verification" does not attest current code; and
    its claim that V3 would be non-zero on full data is contradicted by the full row's own
    L0 max.

---

## 8. Disverification evidence (exact commands + outputs)

**§6 duplicate-artifact (executed in this session):**
```
$ head -c 2000 sample_allfeat_row0.jsonl | grep -o '\[[0-9]*, [0-9]*, [0-9.]*\]' \
    | head -50 | grep -o ', [0-9]*,' | sort | uniq -c | sort -rn
     37 , 14277,
      8 , 20857,
      1 , 80191,  1 , 59859, ...  # L32 top-50: 7 distinct
$ (same pipeline over L16 region)        # L16 top-50:
     29 , 2628,
     21 , 7038,                        # 2 distinct features!
$ grep -o ', [0-9]*,' sample_allfeat_row0.jsonl | sort -u | wc -l   # L63 top-2000 distinct
78                                       # L63: 78 distinct in top-2000
$ grep -o ', 71349, ' sample_allfeat_row0.jsonl | wc -l
269                                       # single feature, 269 positions
```
Combined with `sae_realfeat_analysis.py:61` (`topk = sorted(sparse,...)[:50]` over entries)
and `top50_cos_max = 1.0` at all 5 layers: self-pairs (cos = 1.0) constitute 33–57% of the
1,225 pairs; solving mean = (self + cross)/1225 gives the implied cross-cosines in the §6
table (L16: (0.0075·1225 − 616)/609 = −0.996).

**§5 power (exact combinatorics, no simulation needed):**
P(AUC ≥ 0.70 | H0, 10v10) = P(U ≥ 70); exact critical value U₀.₀₅ = 73 ⇒ p(70) ≈ 0.065
> 0.05 ⇒ fails any BH threshold. min p = 1/C(20,10) = 5.41×10⁻⁶ > 0.05/81920 = 6.10×10⁻⁷
⇒ perfect separation also fails at the design's own m.

**§2 V1 counterexample (hand-executed on the shipped function):**
band(5,50), dedup: {300 distinct feats @6.0} → 1800 > {30 distinct @40–49} → 1335;
{30 distinct @51–60} → 0. And L0: all activations ≤ 3.1731 < 5 ⇒ V1(L0)=0 ⇒ V3≡0.

**§4 calibration (hand-traced on the shipped functions):**
V1ᵢ = (5+i)(5+0.5i) and solverᵢ = 5+0.5i are both strictly increasing in i ⇒ ρ ≡ 1.0 for
any monotone pair (circular). Shipped `_solver_stub` on the same rows:
`len(set(["w0".."wN"]))/N = 1.0` ∀rows ⇒ constant ⇒ tied ranks ⇒ `db=0` ⇒ ρ=NaN ⇒
"INCONCLUSIVE".

## 9. Re-run commands (for an interactive session; expected results per this audit)
```
python3 analyze_dpilot_separability.py --self-test   # expect PASS, top≈0.93, null≈0.28
python3 propel_sae_reward.py                          # expect PASS; check [V3] == 0.0 (degenerate)
python3 propel_calibrate.py                           # expect PASS with rho=1.0 (circular)
```

## 10. Verdict summary
| Item | Verdict |
|---|---|
| 1a self-test run / 0.93 & 0.28 reproduction | INCONCLUSIVE (run blocked); values analytically consistent |
| 1b rank_auc correctness | VERIFIED (function); pseudoreplication flaw in usage (H1) |
| 1c value-perm null validity | Rationale DISVERIFIED; per-feature null valid; as-implemented on real data DISVERIFIED (statistic mismatch, no p-values) |
| 1d gap ≥ 0.10 guard | Threshold ARBITRARY; guard fails correctly on no-signal (analytic) |
| 1e null 0.28 = chance level | VERIFIED (analytic; multiple-comparisons ceiling, not artifact) |
| 2a M4 dedup claim (3.17 vs 15.85) | VERIFIED (hand + real-data premise) |
| 2b band(5,50) principled | DISVERIFIED (magic number; anchor artifact; band cliff; V3≡0) |
| 2c V1 as difficulty proxy | DISVERIFIED (counterexample) |
| 2d dedup failure mode | CONFIRMED (prevalence-blind; + V2 namespace bug) |
| 3 D4 frozen-ref safety | DISVERIFIED as guarantee (unenforced; input-channel hacking remains) |
| 4 calibration ρ=1.0 | DISVERIFIED as evidence (circular; shipped stub degenerate; remediation logic itself sound) |
| 5 pre-registration + power | Form VERIFIED; power/decision-rule DISVERIFIED (H1 unreachable at n=10/group) |
| 6 real-data meaning (cos 0.33, max 293.94) | Raw capture VERIFIED; interpretations DISVERIFIED (duplicate artifact; occurrence-inflated counts; magnitude uninterpreted) |

FABLE5_AUDIT_DONE
