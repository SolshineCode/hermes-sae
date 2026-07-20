"""sae_probe.py — live SAE feature readout for any HF model while it generates.

Attach sparse-autoencoder probes to chosen layers of a local HuggingFace model
and stream the top-k firing features per generated token, in the same forward
pass that produces the output. One file, no framework.

Usage:
  python sae_probe.py --model HuggingFaceTB/SmolLM2-135M \
      --sae-repo <hf-repo-with-sae-state-dicts> --layers 7 11 \
      --prompt "Explain what a sparse autoencoder is." --top-k 5

SAE format: SAELens-compatible — either loose .pt state dicts or per-SAE
subdirectories containing sae_weights.safetensors (+ cfg.json), with W_enc
(d_in, d_sae) and b_enc. Subdir names are matched on "L{layer}" tokens; use
--sae-variant to disambiguate (e.g. "topk", "mixed"). Feature activation is
relu(x @ W_enc + b_enc).

Output: one console line per token (position, token text, per-layer top-k
feature ids + activations) and a full JSONL sidecar next to this script.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch


def load_saes(repo_or_dir, layers, variant=""):
    """Load one SAE per layer from a local dir or HF repo. Supports loose .pt files
    and SAELens-style subdirs (sae_weights.safetensors + cfg.json)."""
    p = Path(repo_or_dir)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(repo_or_dir))
    def matches(name, L):
        n = name.lower()
        return (variant.lower() in n) and any(t in n for t in (f"l{L}_", f"_l{L}_", f"l{L}.", f"layer{L}", f"layer_{L}", f"-l{L}-"))
    saes = {}
    for L in layers:
        cands = sorted([f for f in p.rglob("sae_weights.safetensors") if matches(f.parent.name, L)])              or sorted([f for f in p.rglob("*.pt") if matches(f.name, L)])
        if not cands:
            avail = sorted({f.parent.name for f in p.rglob("sae_weights.safetensors")})[:12]
            raise FileNotFoundError(f"no SAE for layer {L} (variant={variant!r}) under {p}; available: {avail}")
        f = cands[0]
        if f.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(f)
        else:
            sd = torch.load(f, map_location="cpu", weights_only=True)
            sd = sd.get("state_dict", sd)
        W, b = sd["W_enc"].float(), sd["b_enc"].float()
        saes[L] = (W, b, f.parent.name if f.suffix == ".safetensors" else f.name)
        print(f"[sae] layer {L}: {saes[L][2]}  d_in={W.shape[0]} d_sae={W.shape[1]}", file=sys.stderr)
    return saes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sae-repo", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--sae-variant", default="", help="substring filter when a repo holds multiple SAE variants per layer")
    ap.add_argument("--prompt", default="The most important idea in interpretability is")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32 if a.device == "cpu" else torch.float16,
        device_map={"": a.device}, low_cpu_mem_usage=True)
    model.eval()
    saes = load_saes(a.sae_repo, a.layers, a.sae_variant)

    # find the decoder layer list across common architectures
    base = model
    for attr in ("model", "transformer"):
        if hasattr(base, attr):
            base = getattr(base, attr)
    layer_list = getattr(base, "layers", None) or getattr(base, "h", None)
    assert layer_list is not None, "could not locate decoder layers on this architecture"

    trace = {L: [] for L in a.layers}   # per layer: list of (token_idx, [(feat, act), ...])
    step = {"i": -1}                     # -1 during prefill; >=0 per generated token

    def make_hook(L):
        W, b, _ = saes[L]
        W_d, b_d = W.to(a.device), b.to(a.device)
        def hook(_m, _i, out):
            h = (out[0] if isinstance(out, tuple) else out).detach()
            x = h[0, -1].float()                      # last position: the token being generated
            acts = torch.relu(x @ W_d + b_d)
            top = torch.topk(acts, a.top_k)
            trace[L].append((step["i"], list(zip(top.indices.tolist(),
                                                 [round(v, 4) for v in top.values.tolist()]))))
            return out
        return hook

    handles = [layer_list[L].register_forward_hook(make_hook(L)) for L in a.layers]
    ids = tok(a.prompt, return_tensors="pt").input_ids.to(a.device)
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    for h in handles:
        h.remove()

    new_tokens = gen[0][ids.shape[1]:]
    # token i was sampled from forward i (prefill = forward 0), so trace[i] is its readout
    print(f"\n[probe] {len(new_tokens)} tokens in {dt:.1f}s ({len(new_tokens)/dt:.1f} tok/s with probes)\n")
    out_path = Path(a.out or Path(__file__).parent / "feature_trace.jsonl")
    with open(out_path, "w") as f:
        for i, t in enumerate(new_tokens):
            word = tok.decode([t])
            row = {"pos": i, "token": word,
                   "features": {str(L): trace[L][i][1] if i < len(trace[L]) else [] for L in a.layers}}
            f.write(json.dumps(row) + "\n")
            feats_str = " | ".join(f"L{L}: " + " ".join(f"{fid}:{act}" for fid, act in row["features"][str(L)][:3])
                                   for L in a.layers)
            print(f"{i:4d} {word!r:>14}  {feats_str}")
    print(f"\n[probe] full trace -> {out_path}")


if __name__ == "__main__":
    main()
