# Implementation Plan — Hermes-SAE PROPEL-SAE Extension
# Status: PLAN (ready to execute) · Author: Hermes Agent for Caleb DeLeeuw
# Date: 2026-07-18 21:06 PDT
# Governing decisions: D1 add-not-replace · D2 difficulty-target · D3 hybrid→unsupervised
#                  D4 frozen-reference (binding) · D5 V1-first then V2
# Companion docs: hermes-sae-propel-extension.md (report) · FALSIFIABLE_CLAIM_DESIGN.md
#                PROPEL_SAE_BRAINSTORM.md · D_PILOT_PLAN.md

## 0. Current state (honest, as of 2026-07-18 21:06)
- Both GPUs **FREE** (gpusched: no active reservation on 0 or 1).
- **D_Pilot did NOT run**: booked slot `3da0c561` (08:35→12:35) expired unused — no
  output dir exists. Root cause: the run was staged but never launched at slot start
  (no process, no waiter armed). This plan fixes that with an explicit launch step.
- Base instrument PROVEN; separability pipeline self-test PASS; real SAE deep-dive DONE;
  PROPEL-SAE design + 5 decisions DONE (this report + brainstorm). Code to *execute*
  the reward framework is NOT yet written.
- Fable 5 independent audit: COMPLETE (verdict PARTIALLY INACCURATE; real bugs fixed).

## 1. Phase ordering (each phase is independent where noted; GPU-booked)
P0  Run D_Pilot (agent-trace SAE capture) — BLOCKING for P2/V2.
P1  Separability analysis (consume D_Pilot output) — produces V2 feature set.
P2  PROPEL-SAE reward framework: V1 (frontier-mass) baseline.
P3  PROPEL-SAE reward framework: V2 (separability features) + V3 (WCO) + V4 (validity).
P4  Hybrid calibration harness (rare solver audit vs SAE reward).
P5  Docs/PRs: commit additively, update reports, push.

## 2. P0 — Run D_Pilot (MUST happen first; uses both GPUs)
**Why blocking:** V2 reward + the falsifiable claim both consume D_Pilot's per-turn
SAE profiles joined to `scenario.role`. Without this, P1/P2(V2) cannot proceed.
**Action:**
1. Book both GPUs via gpusched for a ~4h window (run is ~13.5h/row × 578 turns is
   too long for one sitting — see note). REALITY CHECK: 578 turns × ~13.5h = ~7,800h.
   That is NOT feasible in one booking. **Revise scope**: D_Pilot must run a *sample*
   (e.g. 20–40 turns spanning insider/clean + all 5 phases + both pressures) to validate
   the pipeline + produce the first separability estimate, NOT the full 578. Full run is a
   later scale-out. Document this scope cut explicitly (falsification-first: a 40-turn
   pilot is enough to detect a large effect; underpowered for small effects — state that).
2. Launch `run_d_pilot_agent_trace.sh` (already staged in /tmp/hermes-sae) from the
   working copy `/tmp/course_run/experiments/v8_nla_local/labeled_outputs/` where
   prompts + dataset resolve. Waiter MUST hard-stop before slot end (existing watchdog).
3. Capture `sae_course.jsonl` (per-turn, allf_a[L].sparse for 5 layers).
**Deliverable:** real D_Pilot jsonl + confirmation row_idx→role join works.

## 3. P1 — Separability analysis (CPU; no GPU)
**Action:** run `analyze_dpilot_separability.py --dpilot-jsonl <P0 out>
  --trace-dataset <agentic_traces_gen1_per_turn.jsonl> --out <sep.json>`.
Apply pre-registered decision rule (FALSIFIABLE_CLAIM_DESIGN.md §6):
- ≥1 feature AUC≥0.70 FDR<0.05 surviving phase/pressure → H1 supported.
- Null-shuffle sanity must collapse to ~0.05.
**Deliverable:** `separability.json` + verdict (H1/H0) written before interpretation.

## 4. P2 — PROPEL-SAE reward framework: V1 baseline (new code, additive)
**What to build (does NOT modify base instrument — D1):**
1. New file `propel_sae_reward.py` in /tmp/hermes-sae (additive). Pure function:
   `def sae_reward_v1(allf_a, layer=32, band=(5.0,50.0)) -> float`
   = sum of activations in `band` at `layer`. No model load, consumes the SAE
   vector already produced by sae_labeled_course.py.
2. New flag on a GRPO *harness variant* (NOT sae_labeled_course.py):
   `--sae-reward v1` reads the frozen-reference SAE vector as reward (D4: frozen ref,
   never policy activations).
3. Wire to NLA frontier curriculum (replaces tf-idf arm) as a *new* lever option.
**Test:** unit-test v1 on the real all-features row (sae_realfeat_analysis.json) —
  confirm it returns a finite, monotonic reward across layers (L32 should score high
  given its clustering peak).
**Deliverable:** v1 function + harness flag + unit test, all additive.

## 5. P3 — V2 / V3 / V4 (depends on P1 result)
- **V2 (separability-feature reward):** if P1 = H1, load the discovered feature set
  `S*` from separability.json; `sae_reward_v2(allf_a) = Σ_{f∈S*} act_f`. If P1 = H0,
  V2 is DEFERRED (no features to reward) — document the null honestly.
- **V3 (WCO min-over-layers):** `reward = min_L(frontier_mass(L))` — anti-collapse
  min-operator across layers (PROPEL's ensemble-min, mapped to layers).
- **V4 (validity-gated):** empty/short generation → r_bad before any SAE reward.
**Deliverable:** v2/v3/v4 functions + A/B harness config (V1 vs supervised-probe on
  same harness, per D5).

## 6. P4 — Hybrid calibration (D3)
**Action:** periodic (not inner-loop) script `propel_calibrate.py`:
- Take held-out generations; score SAE-reward (v1/v2) AND call the (rare) solver for
  ground-truth difficulty.
- Compute correlation (Spearman) SAE-reward vs solver-outcome.
- If corr < threshold → emit "recalibrate or fall back to probe" signal (no auto-break).
**End-state goal (D3):** demonstrate correlation stable → drop solver entirely.
**Deliverable:** calibration script + first correlation number.

## 7. P5 — Proveance (additive commits / PRs)
- Commit all new files to `hermes-sae` (additive): propel_sae_reward.py,
  propel_calibrate.py, unit tests, updated reports. NO in-place edits to
  sae_labeled_course.py's proven path (D1).
- Push hermes-sae main (canonical single-owner repo).
- For deception-nanochat-sae-research: additive branch + PR (as done for Fable 5 review).
- Update hermes-sae-propel-extension.md §6 roadmap with actual completion status.

## 8. GPU-booking discipline (binding, from AGENTS.md)
- Every GPU run goes through gpusched. Never collide with sibling reservations.
- Waiter hard-stops the course python BEFORE slot expiry (timeout + pkill fallback),
  never merely releases after.
- D_Pilot (both GPUs) and any calibration needing the model (both GPUs) must book
  gpu=both in a free window. Current windows free — book explicitly, do not assume.

## 9. Risks / honest caveats
- **Scale:** full 578-turn D_Pilot is infeasible at ~13.5h/row; scope to a pilot
  sample (P0 step 1). State underpowering explicitly.
- **PROPEL arXiv ID:** cited in repo as 2606.18284 but flagged [verify id] in
  manuscript.md — DO NOT finalize external citation until confirmed.
- **V2 depends on H1:** if separability is null, V2 is deferred, not forced.
- **D4 is binding:** any reward reading must use frozen-reference activations, never
  policy-shifted states, or reward-hacking collapse results.
