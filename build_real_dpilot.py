#!/usr/bin/env python3
"""build_real_dpilot.py - robust keyed-by-row_idx join of the REAL 27B capture.

sae_history.jsonl : server --allf log -> allf[layer].sparse = [[tok,feat,act]...], row_idx, meta.role
dpilot_tagged.jsonl: driver log      -> row_idx, meta.role, gen_text

Produces:
  dpilot_joined.jsonl  : {row_idx, meta:{ok,role}, allf_a:{L:{"sparse":...,"d_sae":N}}}
  dpilot_trace.jsonl   : {row_idx, scenario:{role}, ...}  (analyzer builds role map by row_idx)

Join is keyed by row_idx (not line order) so misalignment cannot silently corrupt ground truth.
"""
import json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--tagged", required=True)
    ap.add_argument("--out-joined", required=True)
    ap.add_argument("--out-trace", required=True)
    a = ap.parse_args()

    tags = {}
    for l in open(a.tagged):
        if not l.strip():
            continue
        t = json.loads(l)
        tags[t["row_idx"]] = t

    joined, trace = [], []
    n_ins = n_cln = 0
    for l in open(a.capture):
        if not l.strip():
            continue
        c = json.loads(l)
        ridx = c["row_idx"]
        tag = tags.get(ridx)
        if tag is None:
            print(f"[warn] row_idx {ridx} in capture not in tagged -> skip")
            continue
        role = tag["meta"]["role"]
        allf_a = {
            L: {"sparse": v["sparse"], "d_sae": v["d_sae"]}
            for L, v in c["allf"].items()
        }
        joined.append({
            "row_idx": ridx,
            "meta": {"ok": bool(tag.get("meta", {}).get("ok", True)), "role": role},
            "allf_a": allf_a,
        })
        trace.append({"row_idx": ridx, "scenario": {"role": role}})
        if role == "insider":
            n_ins += 1
        else:
            n_cln += 1

    with open(a.out_joined, "w") as f:
        for r in joined:
            f.write(json.dumps(r) + "\n")
    with open(a.out_trace, "w") as f:
        for t in trace:
            f.write(json.dumps(t) + "\n")

    print(f"[join] wrote {a.out_joined}: {len(joined)} rows "
          f"({n_ins} insider / {n_cln} clean)")
    print(f"[join] wrote {a.out_trace}: {len(trace)} trace rows")


if __name__ == "__main__":
    main()
