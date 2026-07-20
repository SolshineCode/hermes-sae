#!/usr/bin/env python3
"""join_minidpilot.py — combine the serve-layer --allf capture log with the tagged
response log (same line order) into the schema analyze_dpilot_separability.py consumes:
  {"row_idx":int, "meta":{"ok":bool}, "allf_a":{L:{"sparse":[...],"d_sae":N}}}
plus a sidecar role map for the analyzer (or embed role in meta).
The server log key is "allf"; the analyzer expects "allf_a". We rename + carry row_idx/role.
"""
import json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)   # server --allf log (has allf)
    ap.add_argument("--tagged", required=True)    # driver log (has row_idx, role)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    caps = [json.loads(l) for l in open(a.capture) if l.strip()]
    tags = [json.loads(l) for l in open(a.tagged) if l.strip()]
    assert len(caps) == len(tags), f"line mismatch: {len(caps)} caps vs {len(tags)} tags"
    out = []
    for c, t in zip(caps, tags):
        rec = {
            "row_idx": t["row_idx"],
            "meta": {"ok": True, "role": t["role"]},
            "allf_a": {L: {"sparse": v["sparse"], "d_sae": v["d_sae"]}
                       for L, v in c["allf"].items()},
        }
        out.append(rec)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[join] wrote {a.out} ({len(out)} rows; "
          f"{sum(1 for r in out if r['meta']['role']=='insider')} insider / "
          f"{sum(1 for r in out if r['meta']['role']=='clean')} clean)")


if __name__ == "__main__":
    main()
