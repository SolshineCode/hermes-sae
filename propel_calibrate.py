#!/usr/bin/env python
"""
propel_calibrate.py — Hybrid calibration harness (P4, D3: hybrid -> unsupervised).
Implemented by Hermes Agent (not Fable 5) per user directive.

Purpose: periodically (NOT inside the tight RL loop) compare the SAE-reward signal
against a RARE solver's independent difficulty outcome. If they agree (Spearman high,
positive) the reward is tracking real difficulty and we may DROP the solver (save cost).
If they disagree, RECALIBRATE (re-derive band / re-run P1) — this is the D3 safety valve.

AUDIT-FIX NOTES (Fable 5, 2026-07-19):
  * The prior self-test was CIRCULAR: it built both V1 and the injected solver as
    strictly-increasing functions of the same index, so rho===1.0 by construction and
    the self-test could not fail. FIXED: the self-test now verifies the REMEDIATION
    DECISION LOGIC on informative cases (anti-correlated -> RECALIBRATE; noise -> KEEP/
    RECALIBRATE) and explicitly does NOT claim rho=1.0 validates V1. A real calibration
    requires a REAL solver (not the lexical-diversity stub, which is degenerate: it
    returns a constant on distinct-token data -> NaN). D3's "first empirical check" has
    not happened; this scaffold is ready but must not be cited as evidence.
  * Even a STRONG negative correlation is informative (sign-flipped reward) and is
    reported distinctly from "no correlation".

Usage:
  python propel_calibrate.py --self-test
  python propel_calibrate.py --rows agent_turns.jsonl --solver-endpoint http://...
"""
import json, math, os, sys, argparse


def _spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float('nan')
    return num / (dx * dy)


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def decision(rho):
    """Map Spearman to a remediation action (audit logic verified)."""
    if rho != rho:  # NaN
        return "INCONCLUSIVE (degenerate scores)"
    if rho < 0.3:
        return "RECALIBRATE (weak/negative correlation)"
    if rho < 0.6:
        return "MONITOR (moderate correlation)"
    return "KEEP (strong positive correlation; candidate to drop solver)"


def calibrate(rows, feature_set=None, solver_fn=None, seed=0):
    """rows: list of dicts with keys 'allf_a' and 'gen_text'. solver_fn: callable
    (gen_text)->difficulty score. Returns Spearman rho + remediation action."""
    from propel_sae_reward import sae_reward_v1
    xs = [sae_reward_v1(r["allf_a"]) for r in rows]
    ys = []
    for r in rows:
        if solver_fn is None:
            ys.append(_shipped_stub(r["gen_text"]))
        else:
            ys.append(solver_fn(r["gen_text"]))
    rho = _spearman(xs, ys)
    return {"spearman_rho": (round(rho, 4) if rho == rho else None),
            "n": len(rows), "action": decision(rho)}


def _shipped_stub(gen_text):
    """Degenerate lexical-diversity stub (documented): returns a constant on distinct
    token data -> NaN. Real calibration MUST supply a real solver_fn."""
    toks = str(gen_text).split()
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def self_test():
    """Verify REMEDIATION DECISION LOGIC on informative cases (not circular rho).
    Uses a trivial local reward (sum of token indices) and solver stubs with known
    relationships; asserts the decision branches fire correctly. Does NOT claim rho=1.0
    validates the real V1 (that requires a real solver — D3 first empirical check)."""
    # local stand-in reward (monotone in row index) so we can control the relationship
    def local_reward(i):
        return float(i)

    # (a) positively correlated solver -> KEEP
    rows_pos = [{"allf_a": {}, "gen_text": f"w{i}"} for i in range(20)]
    rewards_pos = [local_reward(i) for i in range(20)]
    # emulate via monkeypatch of sae_reward_v1 is intrusive; call decision directly
    rho_pos = _spearman(rewards_pos, [float(i) for i in range(20)])
    # (b) anti-correlated solver -> RECALIBRATE
    rho_neg = _spearman(rewards_pos, [float(19 - i) for i in range(20)])
    # (c) noise solver -> weak -> RECALIBRATE/INCONCLUSIVE boundary
    rng = __import__("random").Random(0)
    noise = [rng.random() for _ in range(20)]
    rho_noise = _spearman(rewards_pos, noise)

    print(f"[calibrate self-test] rho_pos={rho_pos:.3f} action={decision(rho_pos)}")
    print(f"                      rho_neg={rho_neg:.3f} action={decision(rho_neg)}")
    print(f"                      rho_noise={rho_noise:.3f} action={decision(rho_noise)}")

    ok = (decision(rho_pos) == "KEEP (strong positive correlation; candidate to drop solver)"
          and decision(rho_neg).startswith("RECALIBRATE")
          and decision(rho_noise).startswith(("RECALIBRATE", "MONITOR")))
    print("[calibrate self-test]", "PASS ✅" if ok else "FAIL ❌")
    print("[calibrate self-test] NOTE: rho=1.0 from a circular self-test is NOT evidence "
          "V1 tracks difficulty; real validation needs a real solver (D3).")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--rows")
    ap.add_argument("--solver-endpoint")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.rows:
        print("need --rows (or --self-test)"); sys.exit(2)
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    res = calibrate(rows)
    print(json.dumps(res, indent=2))
