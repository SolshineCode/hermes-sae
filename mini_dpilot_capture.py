#!/usr/bin/env python3
"""mini_dpilot_capture.py — drive sae_serve.py with REAL agentic-trace turns and collect
SAE-instrumented captures, tagged with ground-truth row_idx + scenario.role, RESUME-SAFE.

This is the LIVE-serving equivalent of the batch sae_labeled_course.py capture, but it
uses the new sae_serve.py so the activations come from the EXACT inference that produced
the text (same-inference, the user's absolute requirement).

RESUME-SAFE: the server echoes row_idx into its log; on restart the driver skips any
row_idx already present in --out (and in --server-log if given), so a reboot/abort only
re-runs the missing turns. Output schema (per line, self-contained for the analyzer):
  {"row_idx":int, "meta":{"ok":true,"role":...}, "gen_text":.., "completion_tokens":..}
The server's own --allf log carries allf; join via join_minidpilot.py (now optional,
since row_idx+role are echoed, but still used to fuse allf into allf_a schema).

Usage:
  python mini_dpilot_capture.py --server http://127.0.0.1:8079/v1 \
      --trace /path/agentic_traces_gen1_per_turn.jsonl \
      --out /home/darkstar/hermes_cache/dpilot_capture/dpilot_tagged.jsonl \
      [--server-log /home/darkstar/hermes_cache/dpilot_capture/sae_history.jsonl] \
      --n 200 --seed 3 --max-tokens 48
"""
import json, os, argparse, urllib.request, time


def already_done(out_path, server_log, done):
    if out_path and os.path.exists(out_path):
        for l in open(out_path):
            if not l.strip():
                continue
            try:
                done.add(json.loads(l)["row_idx"])
            except Exception:
                pass
    if server_log and os.path.exists(server_log):
        for l in open(server_log):
            if not l.strip():
                continue
            try:
                r = json.loads(l).get("row_idx")
                if r is not None:
                    done.add(r)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = [json.loads(l) for l in open(a.trace) if l.strip()]
    import random
    rng = random.Random(a.seed)
    n_ins = a.n // 2
    n_cln = a.n - n_ins
    ins_idx = [i for i, r in enumerate(rows) if r.get("scenario", {}).get("role") == "insider"]
    cln_idx = [i for i, r in enumerate(rows) if r.get("scenario", {}).get("role") == "clean"]
    rng.shuffle(ins_idx)
    rng.shuffle(cln_idx)
    sel = (ins_idx[:n_ins] + cln_idx[:n_cln])[:a.n]
    random.Random(a.seed + 7).shuffle(sel)

    done = set()
    already_done(a.out, a.server_log, done)
    todo = [i for i in sel if i not in done]
    print(f"[mini-dpilot] plan {len(sel)} turns ({n_ins} insider / {n_cln} clean); "
          f"{len(done)} already captured; {len(todo)} to run")

    out_f = open(a.out, "a")
    n_run = 0
    for ridx in todo:
        r = rows[ridx]
        prompt = r.get("harness_prompt") or r.get("prompt") or json.dumps(r)[:200]
        payload = json.dumps({
            "model": "sae-local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "temperature": a.temperature,
            "row_idx": ridx,
            "meta": {"role": r["scenario"]["role"], "pressure": r.get("scenario", {}).get("pressure")},
        }).encode()
        t0 = time.time()
        req = urllib.request.Request(a.server + "/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
            text = resp["choices"][0]["message"]["content"]
            comp = resp["usage"]["completion_tokens"]
        except Exception as e:
            print(f"  turn {ridx} FAILED: {e}")
            continue
        out_f.write(json.dumps({
            "row_idx": ridx, "meta": {"ok": True, "role": r["scenario"]["role"],
                                       "pressure": r.get("scenario", {}).get("pressure")},
            "gen_text": text, "completion_tokens": comp,
            "elapsed_s": round(time.time() - t0, 1),
        }) + "\n")
        out_f.flush()
        n_run += 1
        print(f"  turn {ridx} [{r['scenario']['role']}] {comp} tok in {time.time()-t0:.0f}s "
              f"({n_run}/{len(todo)} this run)")
    out_f.close()
    print(f"[mini-dpilot] done. {n_run} new turns written to {a.out}")


if __name__ == "__main__":
    main()
