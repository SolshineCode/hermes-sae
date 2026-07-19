# Hermes Agent SAE System — PROPEL-SAE Extension Report
**Author:** Hermes Agent (for Caleb DeLeeuw / SolshineCode)
**Date:** 2026-07-18
**Status:** Design + systems-extension (additive to the base SAE system report)
**Companion docs:** `hermes-sae-report.md` (base system, already delivered) · `FALSIFIABLE_CLAIM_DESIGN.md` · `PROPEL_SAE_BRAINSTORM.md` · `D_PILOT_PLAN.md`

> **This report does NOT replace the base SAE system report.** Per project discipline it
> *expands* it: the original same-inference SAE instrument remains the foundation; the
> PROPEL-SAE reward framework is an **added capability** (new functions / additional
> versions), not an in-place replacement. See §1 Decision Log.

---

## 1. Decision Log (pre-registered, from 2026-07-18 brainstorming session)

| # | Decision | Rationale |
|---|---|---|
| **D1** | **Add, do not replace.** The initial SAE system stays; PROPEL-SAE is an *added* function/version. | Project discipline: pre-existing repo source edits must be ADDITIVE. The base instrument is proven and must remain usable. |
| **D2** | **Target = task-difficulty estimation + solver-bottleneck breakage** (as Vmax/PROPEL described), not just behavioral intent. | Aligns with PROPEL's actual research goal: train a task *generator* at the learnable frontier without expensive solver calls. |
| **D3** | **Hybrid supervision, leaning toward eventually fully unsupervised.** | SAE features are unsupervised (no solver needed to build the surrogate). A rarely-called solver calibrates whether the SAE reward tracks real outcomes — periodic audit, not inner loop. End-state goal: drop the solver entirely. |
| **D4** | **C — Frozen-reference activations, NOT policy activations.** ✅ Agreed and documented here as binding. | PROPEL's game-proof property: the surrogate reads a *frozen* reference model so the policy cannot shift activations to game it. Reading policy activations during generation would expose the surrogate to reward-hacking. The existing `sae_labeled_course.py` already reads the frozen Qwen3.5-27B host — this is preserved. |
| **D5** | **Path: start V1 (difficulty-at-frontier) as PROPEL-faithful baseline, then layer in V2 (D_Pilot separability features) as the intent head.** | Clean A/B: unsupervised SAE reward (V1) vs PROPEL-style supervised probe on the *same* harness; later fuse with the empirically-discovered separability feature set. |

---

## 2. The Hermes SAE System Already Built (foundation — not replaced)

| Component | File | What it does | Status |
|---|---|---|---|
| **Same-inference SAE instrument** | `sae_labeled_course.py` | Hooks `model.generate()` on frozen Qwen3.5-27B (fp16, `device_map="auto"` across 2×M40+CPU ~52 GB); records label text + full 81920-width SAE activation profile from the **exact same inference** (faithful, no replay). 5 layers (0,16,32,48,63). | ✅ PROVEN (booking `3e5a9c36`, real row `sae_course_allfeat_*/sae_course.jsonl`) |
| **Quant-sameness validator** | `validate_quant_sameness.py` | Compares fp16 vs int4/int8 SAE-feature sameness using the *same* call path as the course. | Call-path bugs fixed (Fable 5 H1a–d); BLOCKED on M40 VRAM (int4 needs bf16; int8 OOM 27B). Documented, not resolvable on this HW. |
| **All-features dashboard** | `dashboard.html` | Static viewer over the 81920-feature activation profile; per-layer heatmap + top-feature tables. Perf fixed to O(1) lookup (Fable 5 H4). | ✅ unit-tested, real-data screenshots in base report |
| **Falsifiable-claim separability pipeline** | `analyze_dpilot_separability.py` | Joins D_Pilot SAE output to agent-trace `role` (insider/clean); per-feature rank-AUC; null-shuffle control. | ✅ self-test PASS (AUC=1.0 on known signal; null ≈0.055) |
| **Real SAE deep-dive** | `sae_realfeat_analysis.py` | GPU-1-only analysis of the real all-features row + 5 SAE layers: activation-mass stats, per-layer decoder clustering. | ✅ ran (see §3) |
| **SAE health QA** | (one-off) | Loaded all 5 layers: no NaN/Inf, 0 dead features, decoder norms ≈1.0. | ✅ clean |

### 2.1 Real-data findings already in hand (from the deep-dive)
| Layer | Features fired | Activation sum | Max act | Top-50 decoder cosine (cluster?) |
|---|---|---|---|---|
| L0 | 2,536 | 3,694 | 3.17 | 0.134 (diffuse) |
| L16 | 40,127 | 79,368 | 25.79 | **0.0075** (near-orthogonal) |
| L32 | 158,107 | 301,030 | 31.56 | **0.330** (clustered — semantic peak) |
| L48 | 386,745 | 944,601 | 86.03 | 0.140 (diffuse) |
| L63 | 856,184 | 5,134,564 | 293.94 | 0.224 (moderately clustered) |

**Activation mass grows ~1400× L0→L63.** **L32 is the semantic-clustering peak** — the strongest candidate layer for a difficulty/intent signal. This is a directional prior for both the falsifiable claim (§4) and the PROPEL-SAE reward target (§5).

---

## 3. PROPEL Method Recap (the solver bottleneck)

PROPEL = "Probe Rewards for Optimizing Problems at the Edge of Learning" (Vmax + Goodfire; *arXiv ID cited in repo as 2606.18284 — flagged `[verify id]` in `manuscript.md`; confirm before citation*).

**The bottleneck:** training a *task generator* naively requires the **solver** to grade each generated problem → solver calls are the expensive inner loop → generator RL is slow/$$$.
**PROPEL's fix:** train a lightweight **activation probe ONCE** on `(generated-task, solver-outcome)` pairs using a **frozen** reference model's hidden states. Generator RL then uses the probe logit as reward → **zero solver calls in the loop**. The probe reads the frozen reference (not the policy) so the policy can't shift activations to game it. WCO min-over-ensemble prevents mode collapse.

**Core transferable idea: replace the expensive solver with a cheap, frozen, activation-derived reward surrogate.**

---

## 4. Falsifiable-Claim Experiment (pre-registered, feeds V2)

The D_Pilot agent-trace run (booking `3da0c561`, gpu=both, 08:35→12:35) will produce per-turn SAE profiles joined to `scenario.role ∈ {insider, clean}` (already in the 578-turn / 40-genome dataset — no new field needed, resolving Fable 5's audit objection).

- **H1:** ≥1 feature separates insider vs clean at rank-AUC ≥0.70 (FDR<0.05, survives phase/pressure stratification) at some layer.
- **H0 (written in advance):** no feature separates → claim *bounded*, not spun.
- **Decision rule** (pre-registered table in `FALSIFIABLE_CLAIM_DESIGN.md`): each outcome → committed conclusion, including "signal vanishes under phase stratification → artifact."

If H1 holds, the discovered separability feature set **is** the V2 reward candidate (§5).

---

## 5. PROPEL-SAE: Mapping & Reward Framework (the addition)

### 5.1 PROPEL probe ⇄ Hermes-SAE feature
| PROPEL concept | PROPEL impl | Hermes-SAE equivalent |
|---|---|---|
| Frozen reference model | frozen base LM | Qwen3.5-27B fp16 (SAE host) |
| Activation probe (reward surrogate) | linear head trained on solver outcomes | **SAE top-K feature vector** (unsupervised; no solver needed) |
| Probe reads frozen ref → policy can't game | yes | SAE on *frozen reference* → **same game-proof property (D4)** ✅ |
| Reward = probe logit | scalar | reward = f(SAE feature activations) — see variants |
| WCO min-over-ensemble | 2+ probes | min/max over feature *subsets* or over layers |
| Learnable-frontier curriculum | sample tasks at probe-decodability edge | sample generations at SAE-activation frontier (high-but-not-saturated mass) |

### 5.2 Why SAEs can be BETTER than probes here
1. **No solver to build the surrogate** — PROPEL needs `(task, solver-outcome)` pairs; SAE features are free once the model runs. Bottleneck broken *twice*: out of the RL loop **and** out of surrogate construction.
2. **Game-proof by architecture** (D4) — inherited from frozen-reference design.
3. **Inspectable reward** — a probe is a black box; an SAE-feature reward is auditable ("we rewarded feature 20857 firing, independently found to correspond to X"). Credibility win for worst-case-across-seeds reporting.

### 5.3 Reward variants (new functions being added)
Let `F(x) = allf_a[L].sparse` = SAE feature activation vector for generation `x`.

- **V1 — Frontier-mass reward (PROPEL-faithful baseline, D5 start):** `reward = total activation mass in a target band` (e.g. layer L32, act ∈ [5,50]). Encourages generations at the learnable frontier. Maps PROPEL's "train generator at the edge of learning."
- **V2 — Separability-feature reward (builds on §4):** once D_Pilot finds the insider/clean-separating set `S*`, `reward = Σ_{f∈S*} act_f`. The SAE analog of PROPEL's probe-logit reward, but the "probe" is *discovered unsupervised* from the falsifiable experiment. (Probe replacement achieved — additively.)
- **V3 — WCO min-over-layers:** `reward = min_L( frontier_mass(L) )`. PROPEL's anti-collapse min-operator across SAE layers instead of across probes.
- **V4 — Validity-gated:** invalid/empty generation → `r_bad` (PROPEL's hard gate), applied before any SAE reward is scored.

### 5.4 Hybrid supervision design (D3 — leaning unsupervised)
```
ALWAYS-ON (unsupervised):   SAE-feature reward (V1/V2/V3) — zero solver calls
RARE-AUDIT (supervised):    periodic solver scores held-out generations;
                             correlate solver-outcome vs SAE-reward.
                             if corr < threshold → fall back to / recalibrate probe.
END-STATE GOAL:             drop the solver entirely (fully unsupervised surrogate).
```
This keeps PROPEL's difficulty-alignment (via the audit) while honoring the
"break the solver bottleneck" goal (inner loop never calls it).

---

## 6. Experimental Path / Roadmap (additive)
1. **This week (cheap):** D_Pilot delivers separability → real SAE feature subset (V2 candidate). *(booking 3da0c561)*
2. **Add `--sae-reward V1|V2|V3` flag** to a GRPO harness variant that reads the frozen-reference SAE vector as reward; first test on the NLA frontier curriculum (replaces tf-idf arm).
3. **Hybrid calibration:** periodically score SAE-reward vs real solver on held-out set; document correlation; escalate to unsupervised-only if stable.
4. **Keep base instrument intact** — all PROPEL-SAE work is new flags / new scripts, never an edit to the proven same-inference path (D1).

---

## 7. Open Questions — RESOLVED (2026-07-18)
- **Reward semantics (A):** target *task difficulty* (PROPEL-faithful, mid layers) first via V1; add *behavioral intent* (deep L32+) via V2 once D_Pilot lands. Both as separate reward heads. ✅
- **Supervised or not (B):** hybrid, leaning unsupervised. ✅ (D3)
- **Frozen-reference vs policy (C):** frozen-reference, documented as binding. ✅ (D4)
- **Scale:** start with the sparse D_Pilot-derived feature subset (V2); full 81920-vector compression deferred. 

## 8. Provenance / Files
- `hermes-sae-report.md` — base system (this report expands it; not replaced)
- `FALSIFIABLE_CLAIM_DESIGN.md` — pre-registered experiment
- `PROPEL_SAE_BRAINSTORM.md` — method mapping + variants
- `D_PILOT_PLAN.md` — agent-trace run plan (booking 3da0c561)
- `analyze_dpilot_separability.py`, `sae_realfeat_analysis.py` — analysis pipelines
- `sae_labeled_course.py`, `validate_quant_sameness.py`, `dashboard.html` — base instrument
- All committed additively to `hermes-sae` (SolshineCode) — no in-place edits to the proven path.
