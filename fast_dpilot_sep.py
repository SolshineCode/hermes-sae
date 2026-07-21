#!/usr/bin/env python3
"""fast_dpilot_sep.py - fast, sound B1 separability for the REAL 27B D_Pilot capture.

Optimizations over analyze_dpilot_separability.py (which is O(n_features * n_perm)
pure-Python and took >1h on the full feature set):
  * Pre-filter to features firing in >= min_freq rows (rare features cannot separate
    groups; this cuts the tested set from tens-of-thousands to hundreds).
  * Vectorized AUC via numpy rank-biserial (no per-perm Python row loop for AUC).
  * One-sided (presence-only) features handled with exact hypergeometric p (scipy),
    falling back to 1.0 if scipy absent.

Consumes dpilot_joined.jsonl (row_idx, meta.role, allf_a[L].sparse=[[tok,feat,act]...])
and dpilot_trace.jsonl (row_idx -> scenario.role). Writes JSON with per-layer summary
+ top features + BH-FDR survivors.
"""
import json, argparse, statistics, random, os
from collections import defaultdict
import numpy as np


def rank_auc_vec(ins, cln):
    """Vectorized AUC via Mann-Whitney U / rank-biserial (averaged ties)."""
    a = np.asarray(ins, dtype=float)
    b = np.asarray(cln, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.5
    n, m = a.size, b.size
    pooled = np.concatenate([a, b])
    order = pooled.argsort(kind="mergesort")
    ranks = np.empty(pooled.size, dtype=float)
    sortedp = pooled[order]
    i = 0
    while i < sortedp.size:
        j = i
        while j + 1 < sortedp.size and sortedp[j + 1] == sortedp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    R1 = ranks[:n].sum()
    U1 = R1 - n * (n + 1) / 2.0
    return float(U1 / (n * m))


def bh_fdr(pval_pairs, alpha=0.05):
    """Benjamini-Hochberg. pval_pairs: list of (name, p). Returns set of rejected names."""
    pairs = [(n, float(p)) for n, p in pval_pairs if p == p and 0 <= p <= 1]
    m = len(pairs)
    if m == 0:
        return set()
    ranked = sorted(pairs, key=lambda x: x[1])
    rejected = set()
    for i, (name, p) in enumerate(ranked, start=1):
        if p <= alpha * i / m:
            rejected.add(name)
    if rejected:
        max_idx = max(i for i, (n, p) in enumerate(ranked, start=1) if n in rejected)
        for i in range(1, max_idx + 1):
            rejected.add(ranked[i - 1][0])
    return rejected


def run(dpilot_jsonl, trace_path, out_path, alpha=0.05, n_perm=1000,
        min_freq=5, seed=0):
    rng = random.Random(seed)
    roles = {}
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ridx = d.get("row_idx")
            if ridx is None:
                continue
            roles[int(ridx)] = d.get("scenario", {}).get("role", "clean")

    # Pass 1: per-layer feature -> {row_idx: max_act}
    per_layer = defaultdict(dict)
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
            for L, blk in rec.get("allf_a", {}).items():
                for e in blk.get("sparse", []):
                    feat = int(e[1]); act = float(e[2])
                    d = per_layer[L].setdefault(feat, {})
                    d[ridx] = max(d.get(ridx, 0.0), act)
            n_rows += 1

    results = {}
    summary = {}
    for L in sorted(per_layer, key=lambda x: int(str(x))):
        feats = per_layer[L]
        rows_out = []
        pvals = []
        for feat, byrow in feats.items():
            ins_rows = [r for r, v in byrow.items() if roles[r] == "insider"]
            cln_rows = [r for r, v in byrow.items() if roles[r] == "clean"]
            if ins_rows and not cln_rows:
                n_ins = len(ins_rows)
                try:
                    from scipy.stats import hypergeom
                    N = len(roles)
                    K = sum(1 for r in roles.values() if r == "insider")
                    n_fire = n_ins
                    p = float(hypergeom.sf(n_fire - 1, N, K, n_fire))
                except Exception:
                    p = 1.0
                pvals.append((feat, p))
                rows_out.append({
                    "feat": int(feat), "auc": "one-sided(presence)",
                    "p_perm": round(p, 5),
                    "insider_mean": round(statistics.fmean([byrow[r] for r in ins_rows]), 4),
                    "clean_mean": 0.0, "n_insider": n_ins, "n_clean": 0,
                    "note": "fires ONLY on insider (presence/absence signature)"})
                continue
            if not ins_rows or not cln_rows:
                continue
            if len(byrow) < min_freq:
                continue
            ins_vals = np.array([byrow[r] for r in ins_rows], dtype=float)
            cln_vals = np.array([byrow[r] for r in cln_rows], dtype=float)
            auc = rank_auc_vec(ins_vals, cln_vals)
            all_vals = np.concatenate([ins_vals, cln_vals])
            labels = np.array([1] * len(ins_vals) + [0] * len(cln_vals))
            base = abs(auc - 0.5)
            n_ge = 0
            for _ in range(n_perm):
                perm = labels.copy()
                rng.shuffle(perm)
                pv = all_vals[perm == 1]
                nv = all_vals[perm == 0]
                pa = rank_auc_vec(pv, nv)
                if abs(pa - 0.5) >= base:
                    n_ge += 1
            p = (n_ge + 1) / (n_perm + 1)
            pvals.append((feat, p))
            rows_out.append({
                "feat": int(feat), "auc": round(auc, 4), "p_perm": round(p, 5),
                "insider_mean": round(statistics.fmean(ins_vals), 4),
                "clean_mean": round(statistics.fmean(cln_vals), 4),
                "n_insider": len(ins_vals), "n_clean": len(cln_vals)})
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
        "min_freq": min_freq,
        "note": "features firing in < min_freq rows excluded (cannot separate groups)",
        "summary": summary,
        "top_per_layer": {L: results[L][:25] for L in results},
    }
    if out_path:
        _d = os.path.dirname(out_path)
        if _d:
            os.makedirs(_d, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpilot-jsonl", required=True)
    ap.add_argument("--trace-dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--min-freq", type=int, default=5)
    a = ap.parse_args()
    out = run(a.dpilot_jsonl, a.trace_dataset, a.out,
              alpha=a.alpha, n_perm=a.n_perm, min_freq=a.min_freq)
    print(f"[fast-sep] n_rows={out['n_rows_analyzed']} n_perm={out['n_perm']} "
          f"min_freq={out['min_freq']}")
    for L in sorted(out["summary"], key=int):
        s = out["summary"][L]
        print(f"  L{L}: tested={s['n_features_tested']} survive_FDR={s['n_survive_FDR']} "
              f"min_p={s['min_p']} top_feat={s['top_feat']} (p={s['top_p']})")
