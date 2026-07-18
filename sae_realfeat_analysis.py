#!/usr/bin/env python
"""
sae_realfeat_analysis.py — GPU-1-only SAE deep-dive on the REAL all-features row.

No 27B model needed: the SAE encoder/decoder (~3 GB fp32) fits on one M40, and
we already have REAL SAE activations on disk (the all-features row from booking
3e5a9c36: Qwen3.5-27B hidden states -> SAE). This script:

  1. Loads all 5 SAE layers on CUDA:1.
  2. Reads the real all-features row (allf_a[L].sparse = [(feat,act)]).
  3. Per layer: activation-mass stats, #features fired, max activation, and the
     decoder-norm of the fired features (do high-norm "specific" features dominate?).
  4. Computes pairwise cosine similarity of decoder vectors for the top-50 most
     active features per layer -> are the model's active features CLUSTERED
     (semantically related), a real interpretability signal.
  5. Saves a per-layer feature-importance artifact (decoder-norm x activation)
     that the D_Pilot separability pipeline can later cross-reference.

Pure analysis of already-captured data + SAE weights: uses the FREE GPU 1 now,
de-risks and enriches the falsifiable-claim work, zero collision with sibling.

Usage:
  CUDA_VISIBLE_DEVICES=1 python sae_realfeat_analysis.py \
      --allfeat /tmp/course_run/.../sae_course_allfeat_*/sae_course.jsonl \
      --sae /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
      --out sae_realfeat_analysis.json
"""
import torch, json, os, argparse, math, statistics

def cos(a,b):
    return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-9))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allfeat", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layers", default="0,16,32,48,63")
    ap.add_argument("--out", default="sae_realfeat_analysis.json")
    a = ap.parse_args()
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 -> physical gpu1 shows as cuda:0
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    # load SAE decoder vectors per layer
    layers = [int(x) for x in a.layers.split(",")]
    Wdec = {}
    for L in layers:
        sd = torch.load(f"{a.sae}/layer{L}.sae.pt", map_location=dev)
        Wdec[L] = sd["W_dec"].to(dev)  # (d_in, d_sae)
        print(f"  loaded L{L} decoder {tuple(Wdec[L].shape)}")

    # read real all-features row
    rec = json.loads(open(a.allfeat).readline())
    allf = rec["allf_a"]
    out = {}
    for L in layers:
        sparse = allf[str(L)]["sparse"]  # entries are [token_idx, feat_id, activation]
        acts = [float(e[2]) for e in sparse]
        feats = [int(e[1]) for e in sparse]
        # decoder norm of fired features
        dec_norms = []
        topk = sorted(sparse, key=lambda x:-float(x[2]))[:50]  # by activation
        top_feats = [int(e[1]) for e in topk]
        vecs = [Wdec[L][:, f] for f in top_feats]  # each (d_in,)
        # pairwise cosine among top-50
        n = len(vecs)
        sims = []
        for i in range(n):
            for j in range(i+1, n):
                sims.append(cos(vecs[i], vecs[j]))
        stats = {
            "n_fired": len(feats),
            "activation_sum": round(sum(acts),3),
            "activation_mean": round(statistics.fmean(acts),5),
            "activation_max": round(max(acts),4),
            "top50_dec_norm_mean": round(statistics.fmean([float(Wdec[L][:,f].norm()) for f in top_feats]),4),
            "top50_cos_mean": round(statistics.fmean(sims),4) if sims else None,
            "top50_cos_max": round(max(sims),4) if sims else None,
            "top5_features": [[int(e[1]), round(float(e[2]),4)] for e in topk[:5]],
        }
        out[str(L)] = stats
        print(f"L{L}: fired={stats['n_fired']} act_sum={stats['activation_sum']} "
              f"top50_cos_mean={stats['top50_cos_mean']} top_feat={stats['top5_features'][0]}")
    json.dump(out, open(a.out,"w"), indent=2)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
