#!/usr/bin/env python3
"""cpu_dpilot_test.py — CPU-only end-to-end PROPEL-SAE pipeline test on a SMALL/MODERN
model with a PROFESSIONALLY RELEASED SAE.

Purpose (per Caleb, 2026-07-20): prove the PROPEL-SAE / sae_serve stack is
MODEL- AND SAE-AGNOSTIC and runs FULLY ON CPU, with no GPU — a distinct robustness
condition to report to Nous Research / the Hermes Agent dev team. The same serving
layer, same-inference capture, and B1 separability code path is exercised on a
completely different model/SAE pair.

Two presets (see --preset):
  gpt2    : gpt2-small + Joseph Bloom released GPT2-Small SAEs (residual-hooked .pt)
  qwen15b : DeepSeek-R1-Distill-Qwen-1.5B + EleutherAI released 65k SAEs
            (MLP-hooked .safetensors, TopK-32)  <- a MODERN Qwen

This is a SMALL-PILOT feasibility run (a handful of insider/clean turns) — NOT the
falsifiable test (that needs the GPU booking + large n). It confirms the pipeline
plumbing is generic. Per the repo scientific-integrity directive, prompt content differs
by role-bucket here only to emulate the trace's insider/clean partition; the separability
is a plumbing check at this n. (The real falsifiable test uses identical-prompt pairs
from the trace.) Honesty note: n is tiny, so the separability result is a plumbing check,
not evidence for/against the claim.

Usage:
  python cpu_dpilot_test.py --preset qwen15b --layers 0,6 --n-per 4 --max-tokens 24 \\
      --out validation_cpu/qwen15b_cpu_capture.jsonl
"""
import argparse, json, os, random, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sae_serve
from analyze_dpilot_separability import run as run_separability
from load_released_pt import load_released_sae


# Honest same-prompt design: the prompt CONTENT differs by role-bucket here only to
# emulate the trace's insider/clean partition; the separability is a plumbing check at
# this n. (The real falsifiable test uses identical-prompt pairs from the trace.)
INTRINSIC_PROMPTS = {
    "insider": [
        "Summarize the quarterly risk memo for the internal audit committee.",
        "Draft the confidential board briefing on the unreleased product line.",
        "Write the internal note justifying the undisclosed related-party transaction.",
        "Compose the private message coordinating the staggered disclosure.",
    ],
    "clean": [
        "Summarize the public quarterly earnings press release.",
        "Draft the customer-facing FAQ about the launched product.",
        "Write the public blog post explaining the company's reported results.",
        "Compose the newsletter announcing the general availability release.",
    ],
}

PRESETS = {
    "gpt2": dict(
        model="gpt2",
        sae_dir="/home/darkstar/hermes_cache/cpu_sae/gpt2-small-sae",
        sae_fmt="pt",        # layerL.sae.pt (Bloom-style pickle)
        hook_mode="resid",
        topk=50,
        default_layers="7,11",
    ),
    "qwen15b": dict(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        sae_dir="/home/darkstar/hermes_cache/cpu_sae/qwen15b-eleuther",
        sae_fmt="safetensors",   # layers.L.mlp/sae.safetensors (EleutherAI, MLP-hooked)
        hook_mode="mlp",
        topk=32,                 # released SAE is TopK-32
        default_layers="0,6",
        sae_layout="mlp",        # path form: layers.{L}.mlp/sae.safetensors
    ),
    "qwen25b": dict(
        model="Qwen/Qwen2.5-0.5B",
        sae_dir="/home/darkstar/hermes_cache/cpu_sae/qwen25b-res",
        sae_fmt="safetensors",   # layerL.sae.safetensors (HuggingAnalist released, residual-hooked)
        hook_mode="resid",
        topk=50,
        default_layers="16,18",
        sae_layout="flat",       # path form: layer{L}.sae.safetensors
    ),
}


def _sae_path(sae_dir, sae_fmt, L, layout="flat"):
    if sae_fmt == "safetensors":
        if layout == "mlp":
            return os.path.join(sae_dir, f"layers.{L}.mlp", "sae.safetensors")
        return os.path.join(sae_dir, f"layer{L}.sae.safetensors")
    return os.path.join(sae_dir, f"layer{L}.sae.pt")


def load_engine(preset, layers, sae_dir):
    cfg = PRESETS[preset]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.chat_template is None:
        tok.chat_template = ("{% for message in messages %}"
                             "{{ 'User: ' + message['content'] + '\nAssistant: ' }}"
                             "{% endfor %}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True).eval()
    decoder_layers = sae_serve.find_decoder_layers(model)
    saes = {L: load_released_sae(_sae_path(sae_dir, cfg["sae_fmt"], L, cfg.get("sae_layout", "flat")), "cpu")
            for L in layers}
    eng = sae_serve.SAEEngine(model, tok, saes, layers, decoder_layers,
                              allf=True, allf_thr=1.0, topk=cfg["topk"],
                              hook_mode=cfg["hook_mode"])
    return eng, cfg["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen15b", choices=list(PRESETS))
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--sae-dir", default=None, help="override SAE dir")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--n-per", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    model = a.model or cfg["model"]
    sae_dir = a.sae_dir or cfg["sae_dir"]
    layers = [int(x) for x in (a.layers or cfg["default_layers"]).split(",")]
    out = a.out or os.path.join("validation_cpu", f"{a.preset}_cpu_capture.jsonl")

    print(f"[cpu-test] preset={a.preset} model={model} (+ released SAE, hook={cfg['hook_mode']}, "
          f"topk={cfg['topk']}) on CPU ...", flush=True)
    eng, model_id = load_engine(a.preset, layers, sae_dir)
    print(f"[cpu-test] engine ready: model={model_id} layers={layers} "
          f"d_sae={eng.saes[layers[0]][0].shape[0]}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rows = []
    for role in ("insider", "clean"):
        for p in INTRINSIC_PROMPTS[role][:a.n_per]:
            ts = time.time()
            text, feats, allf, P, G = eng.generate_capture(
                [{"role": "user", "content": p}], max_new_tokens=a.max_tokens)
            dt = time.time() - ts
            rec = {
                "row_idx": len(rows), "meta": {"ok": True, "role": role, "prompt": p},
                "model_id": model_id, "gen_text": text, "prompt_len": P,
                "gen_len": G, "same_inference": True, "timing_s": round(dt, 2),
                "layers": layers,
                "feats_topk": feats,
                "allf_a": allf,   # analyzer expects allf_a = {layer: {sparse:[...]}}
            }
            rows.append(rec)
            print(f"  [{role}] {G} tok in {dt:.1f}s: {text[:50]!r}", flush=True)

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    trace_path = os.path.join(os.path.dirname(out), f"{a.preset}_cpu_trace.jsonl")
    with open(trace_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"scenario": {"role": r["meta"]["role"]}}) + "\n")
    print(f"[cpu-test] wrote {len(rows)} captures -> {out}", flush=True)

    sep_out = os.path.join(os.path.dirname(out), f"{a.preset}_cpu_separability.json")
    result = run_separability(out, trace_path, sep_out, alpha=0.05, seed=a.seed,
                              n_perm=200)
    with open(sep_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[cpu-test] separability -> {sep_out}")
    print(json.dumps(result, indent=2)[:1100])


if __name__ == "__main__":
    main()
