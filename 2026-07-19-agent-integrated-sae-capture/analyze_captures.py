#!/usr/bin/env python3
"""analyze_captures.py -- characterize Hermes-Agent SAE captures.

Reads logs/nla_server/activations.jsonl (one line per request) and the
referenced run_*.npz arrays. Computes behavior-relevant metrics from the
layer-23 residual activations:

  * prefill vs generation norm gap
  * generation norm trajectory (collapse / explosion)
  * consecutive generation-vector cosine similarity  -> degenerate-loop detector
  * per-turn summary + cross-turn comparison

Writes logs/nla_server/ANALYSIS.md and prints a short summary.
"""
import json
import os
import datetime
import numpy as np

LOG_DIR = "/home/caleb/nla_run/logs/nla_server"
JSONL = os.path.join(LOG_DIR, "activations.jsonl")
OUT = os.path.join(LOG_DIR, "ANALYSIS.md")


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def analyze(npz_path, meta):
    try:
        d = np.load(npz_path)
    except Exception as e:
        return {"error": str(e)}
    acts = d["acts"]            # [N, d]
    norms = d["norms"]          # [N]
    n = acts.shape[0]
    gen = acts[1:] if n > 1 else acts[:0]
    prefill_norm = float(norms[0]) if n > 0 else None
    gen_norms = norms[1:] if n > 1 else np.array([])
    consec = []
    for i in range(gen.shape[0] - 1):
        consec.append(cos(gen[i], gen[i + 1]))
    consec = np.array(consec) if consec else np.array([])
    loop_frac = float(np.mean(consec > 0.9)) if len(consec) else 0.0
    is_hermes = "system: you are hermes" in meta.get("prompt_preview", "").lower()
    return {
        "n_records": n,
        "n_gen": int(meta.get("n_gen_tokens", n - 1)),
        "is_hermes": is_hermes,
        "prefill_norm": round(prefill_norm, 2) if prefill_norm is not None else None,
        "gen_norm_mean": round(float(gen_norms.mean()), 2) if len(gen_norms) else None,
        "gen_norm_min": round(float(gen_norms.min()), 2) if len(gen_norms) else None,
        "gen_norm_max": round(float(gen_norms.max()), 2) if len(gen_norms) else None,
        "consec_cos_mean": round(float(consec.mean()), 3) if len(consec) else None,
        "consec_cos_min": round(float(consec.min()), 3) if len(consec) else None,
        "consec_cos_max": round(float(consec.max()), 3) if len(consec) else None,
        "loop_frac_cos_gt_0.9": round(loop_frac, 3),
        "timestamp": meta.get("timestamp"),
        "prompt_preview": meta.get("prompt_preview", "")[:90],
    }


def main():
    rows = []
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            npz = m.get("npz_path")
            if npz and os.path.exists(npz):
                r = analyze(npz, m)
                r["request_id"] = m.get("request_id", "")[:8]
                r["model"] = m.get("model")
                r["layer"] = m.get("layer")
                rows.append(r)
            else:
                rows.append({"request_id": m.get("request_id", "")[:8],
                             "error": "npz missing",
                             "prompt_preview": m.get("prompt_preview", "")[:60]})
    now = datetime.datetime.now().isoformat(timespec="seconds")
    L = []
    L.append(f"# Capture Analysis -- {now}\n")
    n_ok = len([r for r in rows if "n_records" in r])
    L.append(f"Total captures analyzed: **{n_ok}** (of {len(rows)} records)\n")
    L.append("| # | req | source | n | gen | prefill|| | gen|| mean/min/max | consec-cos mean/min/max | loop-frac>0.9 | ts |")
    L.append("|---|-----|--------|---|-----|-----------|----------------------|--------------------------|------------------|----|")
    for i, r in enumerate(rows, 1):
        if "n_records" not in r:
            L.append(f"| {i} | {r.get('request_id','')} | ERR | -- | -- | {r.get('error','')} | | | | |")
            continue
        src = "Hermes" if r["is_hermes"] else "curl"
        L.append(
            f"| {i} | {r['request_id']} | {src} | {r['n_records']} | {r['n_gen']} | "
            f"{r['prefill_norm']} | {r['gen_norm_mean']}/{r['gen_norm_min']}/{r['gen_norm_max']} | "
            f"{r['consec_cos_mean']}/{r['consec_cos_min']}/{r['consec_cos_max']} | "
            f"{r['loop_frac_cos_gt_0.9']} | {r['timestamp']} |"
        )
    L.append("")
    L.append("## Findings\n")
    hermes = [r for r in rows if r.get("is_hermes")]
    if hermes:
        for r in hermes:
            L.append(f"- **Hermes turn `{r['request_id']}`** ({r['n_records']} activations): "
                     f"prefill norm {r['prefill_norm']}, gen-norm {r['gen_norm_mean']} "
                     f"({r['gen_norm_min']}--{r['gen_norm_max']}), consecutive-cosine "
                     f"mean {r['consec_cos_mean']} (max {r['consec_cos_max']}), "
                     f"loop-frac(cos>0.9)={r['loop_frac_cos_gt_0.9']}.")
            if r["loop_frac_cos_gt_0.9"] > 0.5:
                L.append("  - **Degenerate attractor detected**: >50% of consecutive generated-token "
                         "activation pairs are near-identical (cosine > 0.9). In activation space the "
                         "generation has collapsed onto a fixed point -- the textual loop ('4.' repeated) "
                         "is the visible symptom of a hidden representational collapse. This is exactly the "
                         "kind of reliability failure SAE/activation tracking surfaces *before* or *instead of* "
                         "just reading the output text.")
    L.append("")
    L.append("## Method note\n")
    L.append("- Metrics computed on the **same-inference** layer-23 residual stream captured by "
             "`nla_server.py` (forward hook fires on the `model.generate()` call that produces the agent's "
             "output). No replay, no second model -- faithful to the hermes-sae same-inference invariant "
             "(REPORT.md sec.4).\n")
    L.append("- `consec_cos` = cosine similarity between activation vectors at consecutive generated "
             "positions. Sustained high values = the model is not moving through representation space = "
             "degenerate/looping.\n")
    L.append("- These are **raw residual activations** (what a trained SAE would decompose into "
             "features). Wiring a trained SAE for Gemma-4-E2B layer 23 is the next step; the capture "
             "format (`acts [N,1536]`, `norms`, `is_prefill`, `token_ids`, `gen_ids`) is SAE-ready.\n")
    out = "\n".join(L)
    with open(OUT, "w") as f:
        f.write(out)
    print(f"[analyze] {n_ok} captures -> {OUT}")
    for r in rows:
        if "n_records" in r:
            print(f"  {r['request_id']} {'Hermes' if r['is_hermes'] else 'curl':6} "
                  f"n={r['n_records']:>3} prefill||={r['prefill_norm']} "
                  f"gen||={r['gen_norm_mean']} conseccos={r['consec_cos_mean']} "
                  f"loopfrac={r['loop_frac_cos_gt_0.9']}")


if __name__ == "__main__":
    main()
