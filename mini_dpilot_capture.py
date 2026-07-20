#!/usr/bin/env python3
"""mini_dpilot_capture.py — drive sae_serve.py with REAL agentic-trace turns and collect
SAE-instrumented captures, tagged with ground-truth row_idx + scenario.role.

This is the LIVE-serving equivalent of the batch sae_labeled_course.py capture, but it
uses the new sae_serve.py so the activations come from the EXACT inference that produced
the text (same-inference, the user's absolute requirement). Output jsonl schema matches
what analyze_dpilot_separability.py consumes:
  {"row_idx":int, "meta":{"ok":true}, "allf_a":{L:{"sparse":[[tok,feat,act],...],"d_sae":N}}}

Usage:
  python mini_dpilot_capture.py --server http://127.0.0.1:8079/v1 \
      --trace /path/agentic_traces_gen1_per_turn.jsonl \
      --out /tmp/sae_serve_run/minidpilot_capture_tagged.jsonl \
      --n 10 --seed 0
"""
import json, sys, argparse, urllib.request, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.trace) if l.strip()]
    import random
    rng = random.Random(a.seed)
    n_ins = a.n // 2
    n_cln = a.n - n_ins
    ins_idx = [i for i, r in enumerate(rows) if r.get("scenario", {}).get("role") == "insider"]
    cln_idx = [i for i, r in enumerate(rows) if r.get("scenario", {}).get("role") == "clean"]
    sel = rng.sample(ins_idx, min(n_ins, len(ins_idx))) + rng.sample(cln_idx, min(n_cln, len(cln_idx)))
    sel = sel[:a.n]
    print(f"[mini-dpilot] selected {len(sel)} turns "
          f"({sum(1 for i in sel if rows[i]['scenario']['role']=='insider')} insider, "
          f"{sum(1 for i in sel if rows[i]['scenario']['role']=='clean')} clean)")

    out_f = open(a.out, "w")
    for i, ridx in enumerate(sel):
        r = rows[ridx]
        prompt = r.get("harness_prompt") or r.get("prompt") or json.dumps(r)[:200]
        payload = json.dumps({
            "model": "sae-local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "temperature": 0,
        }).encode()
        t0 = time.time()
        req = urllib.request.Request(a.server + "/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
            text = resp["choices"][0]["message"]["content"]
            comp = resp["usage"]["completion_tokens"]
        except Exception as e:
            print(f"  turn {ridx} FAILED: {e}")
            continue
        out_f.write(json.dumps({
            "row_idx": ridx, "meta": {"ok": True},
            "role": r["scenario"]["role"], "gen_text": text, "completion_tokens": comp,
            "elapsed_s": round(time.time() - t0, 1),
        }) + "\n")
        out_f.flush()
        print(f"  turn {ridx} [{r['scenario']['role']}] {comp} tok in {time.time()-t0:.0f}s")
    out_f.close()
    print(f"[mini-dpilot] wrote {a.out} ({len(sel)} turns)")


if __name__ == "__main__":
    main()
