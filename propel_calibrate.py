#!/usr/bin/env python3
"""
propel_calibrate.py — Hybrid calibration harness (P4, D3: hybrid -> unsupervised).
Implemented by Hermes Agent (not Fable 5) per user directive.

Purpose: periodically (NOT in the RL inner loop) check whether the SAE reward actually
tracks REAL task difficulty, by comparing SAE-reward scores against rare SOLVER
outcomes on held-out generations. If correlation holds, the solver can be dropped
entirely (end-state of D3).

D4 INVARIANT (binding): SAE-reward inputs MUST be frozen-reference activations. This
script only *consumes* reward scores; the frozen-ref guarantee is enforced at the
harness call site that produced them.

The solver is a PLUGIN (rare, expensive). Replace _solver_stub with the real solver
call. Calibration NEVER calls the solver in the inner loop.
"""
import json
import math
import random
from propel_sae_reward import sae_reward_v1, sae_reward_v2  # additive sibling import


def _spearman(a, b):
    """Pure-stdlib Spearman rho with tie handling."""
    n = len(a)
    if n < 2 or len(b) != n:
        return float("nan")
    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def _solver_stub(generation_text):
    """PLACEHOLDER for the rare, expensive solver. Returns a difficulty/outcome score.
    Replace with the real solver call. Called ONLY in calibration, never in RL loop."""
    # stub: proxy difficulty = length-normalized lexical diversity (clearly marked fake)
    toks = str(generation_text).split()
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def calibrate(rows, feature_set=None, solver_fn=None, seed=0):
    """rows: list of dicts with keys 'allf_a' and 'gen_text'.
    solver_fn: callable(gen_text)->difficulty score. Defaults to _solver_stub (PLACEHOLDER).
    Returns Spearman rho between SAE-reward and solver-outcome, + remediation signal.
    feature_set: if given, use V2 (separability) reward; else V1 (frontier-mass)."""
    if solver_fn is None:
        solver_fn = _solver_stub
    rng = random.Random(seed)
    sae_scores, solver_scores = [], []
    for row in rows:
        af = row["allf_a"]
        gen = row.get("gen_text", "")
        if feature_set:
            s = sae_reward_v2(af, feature_set)
        else:
            s = sae_reward_v1(af, layer=32, band=(5.0, 50.0))
        sae_scores.append(s)
        solver_scores.append(solver_fn(gen))  # RARE solver call (calibration only)
    rho = _spearman(sae_scores, solver_scores)
    # remediation: if correlation weak, signal recalibrate or fall back to probe
    if math.isnan(rho):
        action = "INCONCLUSIVE (degenerate scores)"
    elif rho < 0.3:
        action = "RECALIBRATE or FALL-BACK-TO-PROBE (SAE reward not tracking difficulty)"
    elif rho < 0.6:
        action = "MONITOR (weak correlation; keep solver audit)"
    else:
        action = "KEEP (strong correlation; candidate to drop solver)"
    return {"spearman_rho": round(rho, 4) if not math.isnan(rho) else None,
            "n": len(rows), "action": action}


if __name__ == "__main__":
    import sys, os, glob
    # self-test: synthetic — V1 score correlates POSITIVELY with solver-stub.
    # Make both monotonic in richness: V1 band-mass scales with i; solver returns
    # mean per-token activation (also scales with i). They should correlate strongly.
    synth_rows = []
    for i in range(20):
        n_tok = 5 + i
        toks = [f"w{j}" for j in range(n_tok)]
        gen = " ".join(toks)
        act = 5.0 + i * 0.5
        af = {"32": {"sparse": [[j, 100 + j, act] for j in range(n_tok)]}}
        synth_rows.append({"allf_a": af, "gen_text": gen})
    # solver stub for the test: mean activation (monotonic in i == monotonic in n_tok)
    def _test_solver(gen):
        return 5.0 + (len(gen.split()) - 5) * 0.5
    out = calibrate(synth_rows, solver_fn=_test_solver)
    print(f"[calibrate self-test] {out}")
    ok = out["action"] in ("KEEP (strong correlation; candidate to drop solver)",
                           "MONITOR (weak correlation; keep solver audit)")
    print("[self-test]", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)
