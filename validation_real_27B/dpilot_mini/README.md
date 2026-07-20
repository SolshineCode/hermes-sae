# Mini-D_Pilot feasibility: live-serving capture + corrected B1 separability (REAL data)

**Date:** 2026-07-19 · **Model:** Qwen3.5-27B fp16 + 5-layer 80K SAE PoC · **Capture:** `sae_serve.py` `--allf` (LIVE same-inference)

## What this is
A **feasibility/plumbing validation** of the full corrected PROPEL-SAE pipeline on REAL
hardware and REAL agentic-trace turns — NOT the falsifiable test (which needs larger n). It
proves the chain works end-to-end:
1. `mini_dpilot_capture.py` drives `sae_serve.py` with 10 real `agentic_traces_gen1_per_turn.jsonl`
   turns (5 insider / 5 clean, by `scenario.role`), tagged by `row_idx`.
2. `join_minidpilot.py` merges the server `--allf` log (full 81920-width dict) with the
   tagged log into `allf_a` schema.
3. `analyze_dpilot_separability.py` (corrected B1) computes per-feature AUC on the
   MATCHED per-row-max statistic, permutation max-T p-values, real BH-FDR.

## Result (n=5/group, REAL data, 5 layers)
| Layer | feats tested | survive BH-FDR | min p | top feat |
|---|---|---|---|---|
| 0  | 67    | 0 | 0.17166 | 45211 |
| 16 | 1582   | 0 | 0.06986 | 20987 |
| 32 | 4985   | 0 | 0.02595 | 21515 |
| 48 | 9814   | 0 | 0.01796 | 4052  |
| 63 | 16885  | 0 | 0.01996 | 1739  |

**0 features survive FDR at every layer.** This is the EXACT outcome predicted by the
corrected design (B1) + power analysis (Fable 5 audit §5): with n=5/group and ~16k
features tested at L63, the BH-FDR threshold is 0.05×1/16885 ≈ 3e-6, far below the
best observed p≈0.018. The falsifiable claim H1 is **NOT supported at this n**, and the
pipeline correctly reports that rather than inflating significance.

This is a **null result**, reported honestly per the falsification-first mandate. The
corrected `FALSIFIABLE_CLAIM_DESIGN.md` specifies the n≥~100/group (or episode
aggregation) required for the real test. Top rank-ed features (4052, 1739, 21515…)
are candidate signals worth tracking at larger n.

## Files
- `minidpilot_capture.jsonl` — raw `--allf` server log (full dict per turn).
- `minidpilot_tagged.jsonl` — response + ground-truth role/row_idx per turn.
- `minidpilot_joined.jsonl` — joined `allf_a` schema fed to the analyzer.
- `minidpilot_separability.json` — per-layer BH-FDR result above.
