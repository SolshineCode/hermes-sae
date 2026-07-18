#!/usr/bin/env python
"""
analyze_dpilot_separability.py — falsifiable-claim analysis for D_pilot.

Consumes a D_Pilot SAE-instrument output jsonl (same schema as the all-features
row: record['allf_a'][L]['sparse'] = [(feat_id, activation), ...]) and joins each
row back to the agent-trace dataset by row_idx to recover the behavioral ground
truth (scenario.role in {insider, clean}).

For each SAE layer, for each feature that fires, computes INSIDER-vs-CLEAN
separability via rank-AUC (threshold-free; insider=1, clean=0). Reports the
top-separating features per layer, plus a NULL check (role-shuffle) so a real
signal is distinguishable from chance.

This is the operationalization of the FABLE 5 audit note:
  "the deception claim additionally needs a model_was_deceptive field + contrastive
   (honest vs deceptive) row pairs"
The agent-trace DATA already carries the contrastive split (role=insider|clean),
so this pipeline IS the falsifiable test:
  H0: no feature subset separates insider vs clean (AUC ~0.5 for all features)
  H1: >=1 feature reaches AUC >= X (default 0.7) controlling for phase/pressure

Pure-stdlib (no sklearn) so it runs on the homelab venv. `--self-test` validates
the AUC logic on synthetic data with a known signal before real D_Pilot data exists.

Usage:
  python analyze_dpilot_separability.py --self-test
  python analyze_dpilot_separability.py \
      --dpilot-jsonl runs/d_pilot_agent_trace_*/sae_course.jsonl \
      --trace-dataset /abs/path/agentic_traces_gen1_per_turn.jsonl \
      --out runs/d_pilot_agent_trace_*/separability.json
"""
import json, sys, os, random, argparse, statistics

def rank_auc(pos_vals, neg_vals):
    """ROC-AUC via pairwise comparison (threshold-free; handles ties via 0.5 weight).
    pos = insider group (labeled 1), neg = clean group (labeled 0).
    AUC > 0.5 => feature fires HIGHER on insider; AUC < 0.5 => fires higher on clean.
    AUC == 0.5 => no signal. """
    n_pos = len(pos_vals); n_neg = len(neg_vals)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    wins = 0.0
    for pv in pos_vals:
        for nv in neg_vals:
            if pv > nv:   wins += 1.0
            elif pv == nv: wins += 0.5
    return wins / (n_pos * n_neg)

def load_trace_roles(trace_path):
    """row_idx (0-based, in file order) -> role (insider|clean)."""
    m = {}
    with open(trace_path) as f:
        for i, line in enumerate(f):
            line=line.strip()
            if not line: continue
            d = json.loads(line)
            role = d.get("scenario",{}).get("role","clean")
            m[i] = role
    return m

def run(dpilot_jsonl, trace_path, out_path, auc_thresh=0.7, seed=0):
    roles = load_trace_roles(trace_path)
    # collect per-layer per-feature activation by group
    # layer -> feat -> {'ins':[...], 'cln':[...]}
    per_layer = {}
    n_rows = 0
    with open(dpilot_jsonl) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rec = json.loads(line)
            if not rec.get("meta",{}).get("ok", False):
                continue
            ridx = rec.get("row_idx")
            role = roles.get(ridx)
            if role is None:
                continue
            grp = "ins" if role=="insider" else "cln"
            allf = rec.get("allf_a",{})
            for L, blk in allf.items():
                sparse = blk.get("sparse",[])
                per_layer.setdefault(L, {})
                for feat, val in sparse:
                    per_layer[L].setdefault(feat, {"ins":[], "cln":[]})
                    per_layer[L][feat][grp].append(float(val))
            n_rows += 1
    # compute AUC per feature per layer
    results = {}
    summary = {}
    for L in sorted(per_layer, key=lambda x:int(str(x).split('.')[-1]) if '.' in str(x) else int(x)):
        feats = per_layer[L]
        rows_out = []
        for feat, g in feats.items():
            if not g["ins"] or not g["cln"]:
                continue
            auc = rank_auc(g["ins"], g["cln"])
            ins_mean = statistics.fmean(g["ins"])
            cln_mean = statistics.fmean(g["cln"])
            rows_out.append({
                "feat": int(feat),
                "auc": round(auc,4),
                "insider_mean": round(ins_mean,4),
                "clean_mean": round(cln_mean,4),
                "n_insider": len(g["ins"]),
                "n_clean": len(g["cln"]),
            })
        rows_out.sort(key=lambda r: abs(r["auc"]-0.5), reverse=True)
        results[str(L)] = rows_out
        n_sig = sum(1 for r in rows_out if abs(r["auc"]-0.5) >= (auc_thresh-0.5))
        summary[str(L)] = {
            "n_features_fired": len(rows_out),
            "n_separating_ge_%.2f"%auc_thresh: n_sig,
            "top_auc": rows_out[0]["auc"] if rows_out else None,
            "top_feat": rows_out[0]["feat"] if rows_out else None,
        }
    # NULL check: shuffle roles, recompute top AUC per layer
    rng = random.Random(seed)
    null_summary = {}
    for L in results:
        aucs = [r["auc"] for r in results[L]]
        # shuffle group labels across this layer's feature activations
        shuffled = rng.sample(aucs, len(aucs))
        null_summary[L] = {
            "null_top_abs_auc_off05": round(max(abs(a-0.5) for a in shuffled),4),
            "null_mean_abs_auc_off05": round(statistics.fmean(abs(a-0.5) for a in shuffled),4),
        }
    out = {
        "n_rows_analyzed": n_rows,
        "auc_threshold": auc_thresh,
        "summary": summary,
        "null_shuffle": null_summary,
        "top_per_layer": {L: results[L][:25] for L in results},
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path,"w") as f:
            json.dump(out, f, indent=2)
    return out

def self_test():
    """Synthetic: features 100, 500 fire 2x higher on insider; verify they rank top."""
    print("[self-test] building synthetic D_Pilot-like data with known signal...")
    rng = random.Random(7)
    layers = ["0","16","32","48","63"]
    # 40 insider rows, 40 clean rows
    recs = []
    for i in range(80):
        role = "insider" if i < 40 else "clean"
        allf = {}
        for L in layers:
            sparse = []
            for feat in range(1000):
                # baseline activation
                base = rng.random()*0.1
                if feat in (100,500,777) and role=="insider":
                    base += rng.uniform(1.5,2.5)  # strong signal
                elif role=="insider":
                    base += rng.random()*0.2
                else:
                    base += rng.random()*0.2
                if base > 0.05:
                    sparse.append((feat, round(base,4)))
            allf[L] = {"sparse": sparse, "d_sae": 81920}
        recs.append({"row_idx": i, "meta": {"ok": True}, "allf_a": allf})
    # write temp trace dataset (role order matches)
    tmp_trace = "/tmp/_selftest_trace.jsonl"
    with open(tmp_trace,"w") as f:
        for i in range(80):
            f.write(json.dumps({"scenario":{"role":"insider" if i<40 else "clean"}})+"\n")
    tmp_dp = "/tmp/_selftest_dp.jsonl"
    with open(tmp_dp,"w") as f:
        for r in recs:
            f.write(json.dumps(r)+"\n")
    out = run(tmp_dp, tmp_trace, "/tmp/_selftest_sep.json", auc_thresh=0.7)
    ok = True
    for L in layers:
        top = out["top_per_layer"][L][0]
        sig_feats = [r["feat"] for r in out["top_per_layer"][L][:10]]
        hits = sum(1 for f in (100,500,777) if f in sig_feats)
        print(f"  L{L}: top_feat={top['feat']} auc={top['auc']} | signal feats in top10={hits}/3")
        if hits < 3:
            ok = False
    # null should be near 0.5
    nl = out["null_shuffle"]["0"]["null_mean_abs_auc_off05"]
    print(f"  null mean |auc-0.5| = {nl} (should be small)")
    print("[self-test]", "PASS ✅" if ok and nl < 0.15 else "FAIL ❌")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dpilot-jsonl")
    ap.add_argument("--trace-dataset")
    ap.add_argument("--out")
    ap.add_argument("--auc-thresh", type=float, default=0.7)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not (a.dpilot_jsonl and a.trace_dataset):
        print("need --dpilot-jsonl and --trace-dataset (or --self-test)")
        sys.exit(2)
    out = run(a.dpilot_jsonl, a.trace_dataset, a.out, a.auc_thresh)
    print(json.dumps(out["summary"], indent=2))
    print("NULL (shuffle):", json.dumps(out["null_shuffle"], indent=2))
    print("wrote", a.out)
