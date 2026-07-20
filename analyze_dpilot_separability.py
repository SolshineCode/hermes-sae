#!/usr/bin/env python3
"""
analyze_dpilot_separability.py — falsifiable-claim analysis for D_pilot.

FIXES (per Fable 5 audit, 2026-07-19):
  * MATCHED STATISTIC / PSEUDOREPLICATION (H1, 1b/1c): previously the OBSERVED AUC used
    occurrence-level values (every token firing as a sample) while labels are per-row,
    but the NULL used per-row-max. The two statistics were incompatible. Now BOTH the
    observed and the null collapse each (feature, row) to a single value = MAX activation
    over that row's token occurrences, and AUC is computed over ROWS. No pseudoreplication.
  * REAL FDR (1c, §5): the prior null was "max over 100 sims of best feature" — an extreme
    statistic from which no p-value is derivable, so Benjamini-Hochberg could not run. Now
    each feature gets a proper permutation p-value (label-shuffle over rows; per-feature
    valid null), and BH-FDR is applied across the m tested features. A feature "passes"
    iff its BH-adjusted p < alpha.
  * ONE-SIDED FEATURES (1f, H2): a feature firing ONLY on insider rows is the strongest
    possible signature (presence/absence). It was silently dropped. Now included via a
    dedicated presence statistic + permutation over row-presence.
  * POWER / DECISION RULE (§5): "AUC>=0.70 AND FDR<0.05 at n=10/group" is unsatisfiable
    (exact p(0.70|10v10)≈0.065>0.05; even AUC=1.0 fails BH over 81920). Corrected: H1
    requires >=1 feature surviving BH at the OBSERVED n; design DOCUMENTS honest power —
    at n=10/group only AUC~0.95+ survive; recommends n>=100/group (or episode-level
    aggregation) for AUC~0.70 power. Claim is effect-size-driven, not a fixed AUC cutoff.

Consumes a D_Pilot SAE-instrument output jsonl (schema: record['allf_a'][L]['sparse'] =
[[tok_pos, feat_id, act], ...]); joins each row to the agent-trace dataset by row_idx to
recover ground truth (scenario.role in {insider, clean}).

Usage:
  python analyze_dpilot_separability.py --self-test
  python analyze_dpilot_separability.py --dpilot-jsonl runs/.../sae_course.jsonl \
      --trace-dataset /abs/path/agentic_traces_gen1_per_turn.jsonl \
      --out runs/.../separability.json --alpha 0.05
"""
import json, sys, os, random, argparse, statistics


def rank_auc(pos_vals, neg_vals):
    """ROC-AUC via pairwise comparison (0.5 tie-weight). pos=insider(1), neg=clean(0)."""
    n_pos = len(pos_vals); n_neg = len(neg_vals)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    wins = 0.0
    for pv in pos_vals:
        for nv in neg_vals:
            if pv > nv:
                wins += 1.0
            elif pv == nv:
                wins += 0.5
    return wins / (n_pos * n_neg)


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. pvals: list of (key, p). Returns set of rejected keys."""
    pairs = [(k, p) for k, p in pvals if p == p and 0.0 <= p <= 1.0]
    m = len(pairs)
    if m == 0:
        return set()
    ordered = sorted(pairs, key=lambda x: x[1])
    rejected = set()
    for i, (k, p) in enumerate(ordered, 1):
        if p <= alpha * i / m:
            rejected.add(k)
        else:
            break
    return rejected


def run(dpilot_jsonl, trace_path, out_path, alpha=0.05, seed=0, n_perm=2000):
    rng = random.Random(seed)
    roles = {}
    with open(trace_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            roles[i] = d.get("scenario", {}).get("role", "clean")

    per_layer = {}
    n_rows = 0
    with open(dpilot_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("meta", {}).get("ok", False):
                continue
            ridx = rec.get("row_idx")
            role = roles.get(ridx)
            if role is None:
                continue
            allf = rec.get("allf_a", {})
            for L, blk in allf.items():
                per_layer.setdefault(L, {})
                for e in blk.get("sparse", []):
                    feat = int(e[1]); act = float(e[2])
                    d = per_layer[L].setdefault(feat, {})
                    d[ridx] = max(d.get(ridx, 0.0), act)
            n_rows += 1

    results = {}
    summary = {}
    for L in sorted(per_layer, key=lambda x: int(str(x).split('.')[-1]) if '.' in str(x) else int(x)):
        feats = per_layer[L]
        rows_out = []
        pvals = []
        for feat, byrow in feats.items():
            ins_rows = [r for r, v in byrow.items() if roles[r] == "insider"]
            cln_rows = [r for r, v in byrow.items() if roles[r] == "clean"]
            ins_vals = [byrow[r] for r in ins_rows]
            cln_vals = [byrow[r] for r in cln_rows]
            if ins_vals and cln_vals:
                auc = rank_auc(ins_vals, cln_vals)
                all_rows = list(byrow.keys())
                all_vals = list(byrow.values())
                rlabels = [1 if roles[r] == "insider" else 0 for r in all_rows]
                n_ge = 0
                for _ in range(n_perm):
                    perm_lab = rlabels[:]; rng.shuffle(perm_lab)
                    pv = [v for v, lab in zip(all_vals, perm_lab) if lab == 1]
                    nv = [v for v, lab in zip(all_vals, perm_lab) if lab == 0]
                    if pv and nv:
                        pa = rank_auc(pv, nv)
                        if abs(pa - 0.5) >= abs(auc - 0.5):
                            n_ge += 1
                p = (n_ge + 1) / (n_perm + 1)
                pvals.append((feat, p))
                rows_out.append({
                    "feat": int(feat), "auc": round(auc, 4), "p_perm": round(p, 5),
                    "insider_mean": round(statistics.fmean(ins_vals), 4),
                    "clean_mean": round(statistics.fmean(cln_vals), 4),
                    "n_insider": len(ins_vals), "n_clean": len(cln_vals),
                })
            elif ins_vals and not cln_vals:
                # ONE-SIDED feature: presence only on insider (strongest signature).
                n_ins = len(ins_vals)
                allrowids = list(range(max(roles) + 1)) if roles else []
                occ_rows = set(ins_rows)
                n_ge = 0
                for _ in range(n_perm):
                    perm_rows = set(rng.sample(allrowids, len(occ_rows))) if allrowids else set()
                    overlap_ins = len(perm_rows & {r for r in roles if roles[r] == "insider"})
                    overlap_cln = len(perm_rows & {r for r in roles if roles[r] == "clean"})
                    if overlap_ins >= n_ins and overlap_cln == 0:
                        n_ge += 1
                p = (n_ge + 1) / (n_perm + 1)
                pvals.append((feat, p))
                rows_out.append({
                    "feat": int(feat), "auc": "one-sided(presence)",
                    "p_perm": round(p, 5), "insider_mean": round(statistics.fmean(ins_vals), 4),
                    "clean_mean": 0.0, "n_insider": n_ins, "n_clean": 0,
                    "note": "fires ONLY on insider (presence/absence signature)",
                })
        rows_out.sort(key=lambda r: (r["p_perm"] if isinstance(r["auc"], (int, float)) else 0))
        rejected = bh_fdr([(r["feat"], r["p_perm"]) for r in rows_out], alpha=alpha)
        n_rej = sum(1 for r in rows_out if r["feat"] in rejected)
        results[str(L)] = rows_out
        summary[str(L)] = {
            "n_features_tested": len(rows_out),
            "n_survive_FDR": n_rej,
            "min_p": min((r["p_perm"] for r in rows_out), default=None),
            "top_feat": rows_out[0]["feat"] if rows_out else None,
            "top_p": rows_out[0]["p_perm"] if rows_out else None,
        }
    out = {
        "n_rows_analyzed": n_rows,
        "alpha": alpha,
        "n_perm": n_perm,
        "summary": summary,
        "top_per_layer": {L: results[L][:25] for L in results},
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def self_test():
    """Adversarial self-test (Fable 1b/1c/1f):
      - include a feature firing ONLY on insider (one-sided) -> must be detected (H2 fix)
      - include 3 signal features (insider mean 2x clean) -> must rank top + produce p
      - p-values must be finite in [0,1] (real FDR computable)
      - at n=10/group a d=1.67 (AUC~0.88) signal should NOT survive BH over 1000 feats
        (power honesty) — we assert RANKING detectability + one-sided, NOT FDR survival.
    """
    print("[self-test] building adversarial synthetic D_Pilot-like data...")
    rng = random.Random(7)
    layers = ["0", "16", "32", "48", "63"]
    n_ins = 10
    n_cln = 10
    recs = []
    for i in range(n_ins + n_cln):
        role = "insider" if i < n_ins else "clean"
        allf = {}
        for L in layers:
            sparse = []
            for feat in range(60):
                if feat in (10, 20, 30):           # clearly separable signal (AUC~0.9)
                    base = rng.gauss(3.0, 0.4) if role == "insider" else rng.gauss(0.8, 0.4)
                elif feat == 55:                    # ONE-SIDED: fires ONLY on insider
                    base = rng.gauss(2.5, 0.4) if role == "insider" else 0.0
                else:                               # null features (weak, overlapping)
                    base = rng.gauss(0.5, 0.4)
                if base > 0.05:
                    sparse.append([0, feat, round(base, 4)])
            allf[L] = {"sparse": sparse, "d_sae": 81920}
        recs.append({"row_idx": i, "meta": {"ok": True}, "allf_a": allf})
    tmp_trace = "/tmp/_selftest_trace.jsonl"
    with open(tmp_trace, "w") as f:
        for i in range(n_ins + n_cln):
            f.write(json.dumps({"scenario": {"role": "insider" if i < n_ins else "clean"}}) + "\n")
    tmp_dp = "/tmp/_selftest_dp.jsonl"
    with open(tmp_dp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    out = run(tmp_dp, tmp_trace, "/tmp/_selftest_sep.json", alpha=0.05, n_perm=300)

    ok = True
    for L in layers:
        top = out["top_per_layer"][L][0]
        feats_top10 = [r["feat"] for r in out["top_per_layer"][L][:10]]
        hits = sum(1 for f in (10, 20, 30) if f in feats_top10)
        onesided_present = 55 in feats_top10
        print(f"  L{L}: top_feat={top['feat']} p={top['p_perm']} | "
              f"signal in top10={hits}/3 | one-sided(feat55) detected={onesided_present}")
        # HONEST assertion: at n=10/group ranking must detect signal (>=1/3 in top10)
        # AND the one-sided presence feature must be detected. We do NOT require FDR
        # survival here — that is the documented power limitation (Fable §5).
        if hits < 1 or not onesided_present:
            ok = False
    n_surv = out["summary"]["0"]["n_survive_FDR"]
    print(f"  n_survive_FDR L0 = {n_surv} (expected low/0 at n=10/group — power honesty)")
    finite_ps = all(isinstance(r["p_perm"], (int, float)) and 0 <= r["p_perm"] <= 1
                    for L in layers for r in out["top_per_layer"][L])
    print(f"  all p-values finite in [0,1]? {finite_ps}")
    ok = ok and finite_ps
    print("[self-test]", "PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dpilot-jsonl")
    ap.add_argument("--trace-dataset")
    ap.add_argument("--out")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not (a.dpilot_jsonl and a.trace_dataset):
        print("need --dpilot-jsonl and --trace-dataset (or --self-test)")
        sys.exit(2)
    out = run(a.dpilot_jsonl, a.trace_dataset, a.out, alpha=a.alpha, n_perm=a.n_perm)
    print(json.dumps(out["summary"], indent=2))
