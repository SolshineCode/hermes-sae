# Fable 5 — PROPEL-SAE Plan Review (Second Pass)
# Role: independent senior ML-systems reviewer — REVIEW ONLY, no implementation
# Date: 2026-07-18 · Primary: PROPEL_SAE_IMPLEMENTATION_PLAN.md
# Prior pass: FABLE5_PROPEL_IMPLEMENTATION_REVIEW.md (overstepped into code; bugs confirmed fixed)

---

## VERDICT: APPROVE-WITH-CHANGES

The plan is coherent and broadly executable. Both bugs from the prior pass are confirmed
fixed in the current `analyze_dpilot_separability.py` (Bug A: 3-element sparse indexing at
lines 85–86; Bug B: value-permutation null at lines 130–167 — both correct). The phase
ordering is sound, D1/D4/D5 are honored structurally, and the scale honesty added to §2
is a genuine improvement. Four issues require mandatory plan edits before execution. None
are in the reward code; all are in documentation and the statistical decision protocol.

---

## Issue 1 — BLOCKING: Null-threshold claim in P1 conflicts with what the code computes

**Plan §3 says:** "Null-shuffle sanity must collapse to ~0.05."
**FALSIFIABLE_CLAIM_DESIGN §3 control 3 says:** "top per-layer |AUC−0.5| must collapse
toward ~0.05 (chance)."

**What the code actually computes:** `null_top_abs_auc_off05` is the MAXIMUM over 100
permutation simulations of the BEST |AUC−0.5| across up to 600 features. Under true H0,
this max-over-many-features-over-many-sims statistic will NOT collapse to ~0.05 — multiple
testing drives it substantially higher. The self-test already reflects this: it accepts
`nl < 0.15`, not `< 0.05`. The prior Fable 5 self-test report cited "~0.055" — that was
from a synthetic dataset with relatively few signal-carrying features; real 81920-feature
data will push this null metric higher even under H0.

**Scientific risk:** if the plan declares "null collapsed to ~0.05" as the sanity gate but
the code computes a metric that won't pass that gate on clean permuted data, the sanity
check is uninterpretable and the H0 boundary is undefined.

**Mandated change M1:** Update plan §3 and FALSIFIABLE_CLAIM_DESIGN §3 control 3 to read:
"Null-shuffle sanity: the value-permutation null's `null_top_abs_auc_off05` (max-over-
sims of best |AUC−0.5| across subsampled features) must be substantially below the
observed real-data top |AUC−0.5|. Under true H0, expect this metric at ~0.10–0.15 with
600 features and 100 sims (not ~0.05); self-test threshold is <0.15. The operative sanity
check is: observed top |AUC−0.5| >> null_top_abs_auc_off05, not that the null reaches a
specific absolute value." Also add one sentence explaining why 0.05 was wrong (multiple
testing over features × simulations).

---

## Issue 2 — HIGH: Pilot H1/H0 verdict must NOT gate V2 feature selection

**Plan §3 says:** "Apply pre-registered decision rule (FALSIFIABLE_CLAIM_DESIGN.md §6)
… ≥1 feature AUC≥0.70 FDR<0.05 … → H1 supported."
**Plan §5 says:** "if P1 = H1, load the discovered feature set S* from separability.json;
sae_reward_v2(allf_a) = Σ_{f∈S*} act_f."

**Problem:** P0 explicitly scopes to a 2–4 turn pilot, which is "underpowered for real
effects" per plan §2. But plan §3 says P1 will render an H1/H0 verdict that directly
gates V2 (plan §5). A 2-turn pilot can produce a spurious H1 by chance (small-n
instability will yield arbitrary top-AUC features that do not replicate). If V2 is built
on a pilot-derived feature set, it will be a phantom reward: semantically uninformative
features that happened to differ by chance between the pilot's 1–2 insider and 1–2 clean
turns.

The plan says "the pre-registered H1/H0 stands for the full run" — correct — but then
wires P5 V2 to P1's result without specifying WHICH P1. There are two P1 runs: pilot and
full. The plan doesn't make that distinction.

**Mandated change M2:** Add an explicit paragraph to §3 and §5 distinguishing:
- **Pilot P1** (from 2–4 turn data): pipeline smoke-test only. Verdict is "pipeline
  intact / broken." No H1/H0 claim. No V2 feature set extracted. Output is
  `pilot_separability.json` (not `separability.json`).
- **Full-scale P1** (from the accumulated multi-session run, once ≥N turns per group):
  this is the pre-registered falsifiable test. Its verdict gates V2.
State the minimum per-group N required before full-scale P1 is run (recommend ≥10 per
group as the minimum for FDR-corrected AUC over 81920 features to have any power; or
defer this to user judgment but require explicit user sign-off before running the full
decision rule).

---

## Issue 3 — HIGH: Pilot session count unquantified

**Plan §2 step 1 says:** "Per 4h slot: ~0.3 turns … a 2–4 turn pilot is NOT one
booking."

**Gap:** ~0.3 turns/slot means 2 turns ≈ 7 sessions, 4 turns ≈ 13 sessions. The plan
says "NOT one booking" but doesn't say what it is. This is a planning horizon gap: the
human cannot calendar or reserve GPU time for a pilot whose session count is unstated.

Additionally, the prior failure (slot 3da0c561 expired unused, no process armed) is
identified as the root cause for D_Pilot not running. Plan §2 step 2 says "Waiter MUST
hard-stop before slot end (existing watchdog)" — but the watchdog's existence is
asserted, not verified. If it didn't run last time, "existing watchdog" may be
aspirational.

**Mandated change M3:** In plan §2 step 1, add: "At ~0.3 turns/4h slot, 2 turns requires
~7 sessions, 4 turns ~13 sessions. Calendar accordingly. Each session must independently
arm the watchdog before the slot starts; do NOT assume the watchdog persists across
sessions." Also add a launch-verification step: before booking each slot, confirm
`run_d_pilot_agent_trace.sh` is present at the expected path and executable.

---

## Issue 4 — HIGH: max-per-feature deduplication unmentioned in P2 test specification

**Plan §4 (P2) says:** "unit-test v1 on the real all-features row (sae_realfeat_analysis.json)
— confirm it returns a finite, monotonic reward across layers."

**Gap:** sae_realfeat_analysis.json reveals that at L0, L16, L48, and L63, the top-5
activation slots are all the SAME feature repeated (71349×5 at L0, 2628×5 at L16,
27644×5 at L48, 80518×4 at L63). Without max-per-feature deduplication, V1 would
sum these repetitions, inflating the band reward 2–5× for the dominant feature — a
direct length-bias reward hacking vector even under the frozen reference (policy can
learn to generate semantically uniform content that fires the dominant feature at many
token positions). The reward code (per prior review) implements `_max_per_feature()`
to handle this, but the plan's P2 unit test doesn't name this as a check, so future
developers won't know it's required.

**Mandated change M4:** In plan §4 unit-test spec, add: "Confirm max-per-feature
deduplication is active: the reward for L0 (where feature 71349 appears 5× in the
top-5 of sae_realfeat_analysis.json) must equal the reward for a synthetic record
containing that feature once — repeated token-position entries for the same feat_id
must not inflate the score."

---

## D1/D2/D3/D4/D5 compliance check

- **D1 (add, not replace):** Compliant. Plan §4/§7 explicitly prohibit in-place edits to
  `sae_labeled_course.py`. All new files are additive. ✓
- **D2 (difficulty target):** Partially compliant. V1's frontier-mass band (5.0–50.0) at
  L32 is selected empirically from the real SAE deep-dive; its connection to task
  DIFFICULTY (vs semantic content in general) is an assumption, not a measurement.
  P4 calibration (Spearman ρ vs solver outcomes) is the correct validator, and it runs
  after P2/P3. This ordering means V1 is used before its difficulty-alignment is
  confirmed. Acceptable as a research choice, but name it explicitly in §9 risks: "V1
  band=(5.0,50.0) assumed to track task difficulty; P4 calibration is the first
  empirical check on this assumption." ✓ (risk named in §9 but not this specifically)
- **D3 (hybrid → unsupervised):** Compliant. P4 calibration harness is the correct
  mechanism for the transition. The "end-state goal" framing is honest. ✓
- **D4 (frozen reference, binding):** Architecturally sound; enforcement is a harness-
  level invariant. The plan correctly identifies this as a harness integration test
  (§9). One addition needed when the harness is built: `assert all(not p.requires_grad
  for p in frozen_ref.parameters())` as a startup check, not a comment. This is not
  a plan-text change (harness doesn't exist yet) but warrants a note in §9.
- **D5 (V1 first, then V2):** Compliant. P2 precedes P3. V2 is explicitly conditional
  on P1 H1. ✓

---

## Separability analysis scientific validity (post-bug-fix)

Both bugs are confirmed fixed in the current file:
- **Bug A (3-element format):** lines 85–86 now index `e[1]` and `e[2]`. Self-test
  generates `[0, feat, act]` at line 203. Correct.
- **Bug B (null):** Lines 130–167 implement a value-permutation null that fixes group
  labels per row and shuffles activation values. This correctly breaks the label-
  activation association under H0. The prior null (shuffling existing AUC ordering)
  was scientifically invalid and is gone.

The H1/AUC≥0.70/FDR decision rule (FALSIFIABLE_CLAIM_DESIGN §6) remains valid as a
falsifiable test WHEN applied to the full-scale run. The FDR (Benjamini-Hochberg) over
81920 features × 5 layers is correctly pre-registered. The null check's expected range
needs the correction noted in M1 above, but the underlying permutation logic is correct.

---

## Remaining execution risks

1. **V1 band saturation collapse** (plan §9 mentions D4; add this): if GRPO discovers
   generating semantically homogeneous content saturates L32 band activation, V1 reward
   converges prematurely. V3 min-over-layers is the guard. P4 calibration detects it on
   a lag. Acceptable with named monitoring.

2. **D4 leakage timing:** the frozen reference and the trained policy share architecture.
   Early in training (few steps), their activation distributions are nearly identical.
   D4 compliance is untestable until the policy has diverged. This should be noted in
   the harness integration test specification — not a plan change, but an implementation
   note to add to §9.

3. **The prior slot failure pattern:** the root cause was "no process, no waiter armed."
   Fixing this requires BOTH (a) arming the watchdog before the GPU slot starts AND
   (b) verifying the launch script exists and is executable at booking time. M3 above
   covers this, but it's worth calling out separately: the plan currently says "existing
   watchdog" without evidence that watchdog setup is part of the automated booking flow.

---

## What the prior pass got WRONG — ignore these

- **Prior pass §6** included Python code for Bug A and Bug B fixes. That code was
  correct in intent but is now superseded by the actual implementation, which is also
  correct. Ignore all code blocks in §6 of the prior review.
- **Prior pass M3** stated "≥20 per group = ≥40 sessions minimum." This was the FULL
  578-turn run math, applied as if it were the pilot target. The plan has correctly
  scoped to a 2–4 turn pilot with the full run as a later multi-session effort. The
  "40 sessions" figure is not wrong for the full run but should not be cited as a pilot
  session count.
- **Prior pass §4** mandated a "harness guard" Python snippet for the empty-feature-set
  edge case. This was implementation code in a review document. The edge case is real
  (H1 verdict but empty feature_set after FDR), but the fix belongs in the harness when
  it is built — not retroactively attributed to the plan text.
- **Prior pass verdict "IMPLEMENTABLE (CONDITIONAL on M1+M2)"** referred to the REWARD
  CODE being implementable, with bugs blocking ONLY the P1 separability analysis. The
  plan currently reads as if P2 is also blocked — it is not. V1 reward code can be
  developed in parallel with pilot data collection; P2 does not depend on P1.

---

## Summary

```
Verdict:                APPROVE-WITH-CHANGES
Bug A (sparse format):  FIXED ✓
Bug B (null shuffle):   FIXED — value-permutation correct ✓
D1:                     Compliant ✓
D2:                     Assumed for V1; P4 calibration validates (name risk explicitly)
D3:                     Compliant ✓
D4:                     Architecturally sound; harness startup assertion still needed
D5:                     Compliant ✓
Falsifiable claim:      Scientifically valid for full-scale run; null threshold needs
                        correction from ~0.05 to ~0.10-0.15 (M1 — BLOCKING)
Pilot gate / V2:        Pilot must not render H1/H0 verdict that gates V2 (M2 — HIGH)
Pilot session count:    Unquantified; state ~7-13 sessions (M3 — HIGH)
max-per-feature test:   Missing from P2 unit spec; add to test list (M4 — HIGH)
P2 parallelism:         V1 reward code is independent of P1; can proceed now
```

Mandated changes before execution, ordered by priority:
- M1 (BLOCKING): Fix null-threshold language in plan §3 and FALSIFIABLE_CLAIM_DESIGN §3.
- M2 (HIGH): Add pilot-vs-full-run distinction to §3 and §5; pilot P1 is smoke-test only.
- M3 (HIGH): Quantify pilot session count (~7-13 × 4h); add launch-verification step to §2.
- M4 (HIGH): Add max-per-feature deduplication as a named unit-test check in plan §4.
```
