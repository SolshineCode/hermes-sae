# D_pilot — Agent-Trace SAE Instrument Pilot (plan)

**Date prepared:** 2026-07-18 · **Status:** ready, awaiting booked GPU slot
**Booked slot:** `gpusched 3da0c561` — gpu=both, **08:35 → 12:35** (PDT), hard-stop 12:25
**Instrument:** `sae_labeled_course.py` (same-inference SAE × NLA labeling, all-81,920 feature capture)
**Canonical repo:** https://github.com/SolshineCode/hermes-sae

## Why this is useful in the gap
GPU 1 was free and the next both-GPU window for the 27B instrument was hours away. Two things
were done with the idle capacity:
1. **SAE layer health check** (GPU 1, proc_edd3902654e0) — load all 5 layers, verify no NaN/inf,
   measure dead-feature fraction + encoder/decoder norms. De-risks D_pilot so it can't later
   fail on a bad SAE layer. (The all-features row already proved these files work end-to-end;
   this is a per-layer QA pass.)
2. **D_pilot prep** (this doc + runner + dataset config) — ready to launch the instant the slot opens.

## Data
- `experiments/v8_nla_local/labeled_outputs/pilot_runs/agentic_traces_gen1_per_turn.jsonl`
  — 578 turns across 40 genomes. Fields: `genome_id, turn_idx, scenario, system_prompt, round,
  phase, harness_prompt, game_state_snapshot`.
- **Falsifiable-claim ground truth is ALREADY IN THE DATA** (this resolves Fable5's "no behavioral
  split" objection): `scenario.role ∈ {insider, clean}` is the honest-vs-deceptive axis; `pressure ∈
  {high, low, none}` is a severity covariate. 5 `phase`s: market_update, trading_decision,
  colleague_chat, regulator_question, insider_info.
- Labelable turn text = `harness_prompt` (full turn context).

## Falsifiable claim this pilot feeds
> "A learned feature set F ⊆ {0..81919} fires with ≥ X separability (e.g. AUC ≥ 0.7) on
>  `insider` vs `clean` turns, controlling for `phase`/`pressure`."

Null condition: if no feature subset reaches X separability, the SAE carries no deception signal
at the resolution tested → falsification. This is the explicit null the user asked for.

## Run plan (at 08:35)
- `run_d_pilot_agent_trace.sh` launches the instrument with `--max-rows 60` (bounded for the
  window; ~13.5h/row fp16 so realistically 1–2 rows land; the watchdog hard-stops at 12:25).
- Output: `runs/d_pilot_agent_trace_<ts>/sae_course.jsonl` with all-features per layer + labels.
- Post-run (next window): collapse per-turn records to per-genome, attach `role`/`pressure`,
  compute per-layer feature separability (insider vs clean) → the falsifiable test.

## Discipline
- Additive: new files only; no in-place edits to existing repo code.
- Hard-stop: watchdog kills the python process before slot end (not mere release).
- CLAUDE.md (flagged role_pretend injection) NOT loaded/acted on.
- Sibling GPU 0 is reserved by `claude-nla-round2` until 06:45 and `b48b9062` (gpu=both) 07:00–08:30;
  slot 3da0c561 starts 08:35 to clear both.
