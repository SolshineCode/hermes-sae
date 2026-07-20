#!/usr/bin/env python
"""
sae_realfeat_analysis.py — GPU SAE deep-dive on the REAL all-features row.

FIXES (per Fable 5 audit, 2026-07-19):
  - B2: top-50 were TOKEN OCCURRENCES, not distinct features, and the pairwise cosine
    included self-pairs (cos=1.0) which inflated top50_cos_mean. Now we:
      * operate over DISTINCT fired features (occurrence counts tracked separately),
      * compute cross-feature cosine over distinct features only (no self-pairs),
      * report dead-feature fraction (W_dec columns near-zero norm) and per-layer
        occurrence-vs-feature counts so nothing is mislabeled by orders of magnitude.
  - The "L32 semantic-clustering peak" interpretation is REMOVED as a conclusion; this
    script now only reports descriptive stats (cross-cosine mean, distinct-feature
    count, max activation) WITHOUT asserting semantic meaning. Downstream claims must
    not rely on cos=0.33 (it was an artifact).

No 27B model needed: SAE enc/dec (~3 GB) fits on one M40; REAL SAE activations already
on disk (all-features row from booking 3e5a9c36).

Usage:
  CUDA_VISIBLE_DEVICES=1 TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 python sae_realfeat_analysis.py \
      --allfeat /tmp/course_run/.../sae_course_allfeat_*/sae_course.jsonl \
      --sae /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
      --out sae_realfeat_analysis.json
"""
import torch, json, os, argparse, statistics


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-9 or nb < 1e-9:
        return float('nan')
    return float(torch.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allfeat", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layers", default="0,16,32,48,63")
    ap.add_argument("--out", default="sae_realfeat_analysis.json")
    ap.add_argument("--topk", type=int, default=50,
                    help="number of DISTINCT features to rank by activation for cosine")
    ap.add_argument("--d-sae", type=int, default=81920)
    ap.add_argument("--dead-norm-thr", type=float, default=1e-3,
                    help="W_dec column norm below this counts as a dead/near-dead feature")
    a = ap.parse_args()
    dev = "cuda:0"
    print("cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    layers = [int(x) for x in a.layers.split(",")]
    Wdec = {}
    for L in layers:
        sd = torch.load(f"{a.sae}/layer{L}.sae.pt", map_location=dev,
                        weights_only=False)
        Wdec[L] = sd["W_dec"].to(dev)
        print(f"  loaded L{L} decoder {tuple(Wdec[L].shape)}")

    rec = json.loads(open(a.allfeat).readline())
    allf = rec["allf_a"]
    out = {}
    for L in layers:
        sparse = allf[str(L)]["sparse"]
        per_feat = {}
        for e in sparse:
            f = int(e[1]); act = float(e[2])
            if f not in per_feat or act > per_feat[f][0]:
                per_feat[f] = (act, 0)
            per_feat[f] = (max(act, per_feat[f][0]), per_feat[f][1] + 1)
        n_occurrences = sum(c for _, c in per_feat.values())
        n_distinct = len(per_feat)
        max_act = max((v[0] for v in per_feat.values()), default=0.0)

        top_feats = sorted(per_feat, key=lambda f: -per_feat[f][0])[:a.topk]
        vecs = [Wdec[L][:, f] for f in top_feats]
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(cos(vecs[i], vecs[j]))
        cross_cos_mean = statistics.fmean(sims) if sims else float('nan')
        cross_cos_max = max(sims) if sims else float('nan')

        norms = Wdec[L].norm(dim=0)
        n_dead = int((norms < a.dead_norm_thr).sum().item())
        dead_frac = n_dead / norms.shape[0]

        out[str(L)] = {
            "n_occurrences": n_occurrences,
            "n_distinct_features": n_distinct,
            "max_activation": round(max_act, 4),
            "topk_cross_cos_mean": round(cross_cos_mean, 4),
            "topk_cross_cos_max": round(cross_cos_max, 4),
            "dead_feature_frac": round(dead_frac, 4),
            "top5_features": [[f, round(per_feat[f][0], 4), per_feat[f][1]]
                              for f in top_feats[:5]],
            "note": ("DESCRIPTIVE ONLY. Cross-cosine over DISTINCT features, no "
                     "self-pairs. Do NOT interpret as 'semantic clustering' without "
                     "independent evidence."),
        }
        print(f"L{L}: distinct={n_distinct} occ={n_occurrences} max_act={max_act:.4f} "
              f"cross_cos_mean={cross_cos_mean:.4f} dead_frac={dead_frac:.4f}")
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
