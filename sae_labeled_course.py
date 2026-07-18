#!/usr/bin/env python3
"""
sae_labeled_course.py — ADDITIONAL labeling course with SAE feature activations
recorded from the EXACT SAME inference instance as the label output.

CRITICAL CORRECTNESS CONSTRAINT (Caleb, 2026-07-11):
  "for SAE feature activations of a model output to be faithful and correct,
   the feature activations AND the model output MUST be from the exact same
   inference instance."
=> We CANNOT use ollama's labels + replay through HF (two different inferences).
=> We MUST generate the label text INSIDE the HF model that carries the SAE hook.
   One model.generate() call produces BOTH the label tokens AND the residual-stream
   features for those exact tokens. They share one computation. Faithful by construction.

HOW SAME-INFERENCE IS GUARANTEED:
  - Hooks registered on model.model.layers[L] for each SAE layer L.
  - model.generate() runs (autoregressive). With KV-cache, the PREFILL step fires
    the hook with shape (1, P, D) for all prompt tokens; each DECODE step fires with
    (1, 1, D) for one new token. We APPEND every fire per layer and cat along dim=1
    to reconstruct (1, P+G, D), then slice [P:] to keep only the G generated-token
    residuals. tok_pos 0..G-1 in feats index the generated label tokens.
  - Label = tokenizer.decode(generated_ids)  -> parsed via the SAME extract_label()
    used by labeler_service.py.
  - Features = TopK(50) of (residual @ W_enc.T + b_enc) for every generated
    token position, per layer.
  - Both derive from the single generate() call. No second forward pass.

PROMPT FIDELITY (labels comparable to run7e):
  - Reuses labeler_service.py's exact prompt assembly: load_prompt(labeler_a/b/auditor)
    + schema, build_user(text), FEWSHOT_USER/FEWSHOT_ASST prefix, build_user(text, ja)
    for the auditor pass. So the labeler prompts are IDENTICAL to the ollama course;
    only the inference engine differs (HF q8 vs ollama q4) — documented as a
    quantization-shift caveat (SAE trained on base bf16; q8 closest).

USAGE:
  python sae_labeled_course.py --config labeler_config.yaml \
         --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
         --layers 0,16,32,48,63 --dtype float16 \
         --out runs/sae_course_poc.jsonl --max-new-tokens 5000

OUTPUT (one JSON line per row):
  {"row_idx", "dataset", "input_text",
   "labeler_a": <parsed JSON>, "labeler_b": <parsed>, "auditor_a": <parsed>,
   "agreement_a_b": <float>,
   "feats": { "<layer>": [[tok_pos, feat_idx, value], ... for top-50 per pos] },
   "meta": {...}}

REQUIRES (staged separately, needs GPU free from ollama):
  - base model Qwen/Qwen3.5-27B (q8 ~27GB) downloaded to --hf-cache
  - SAE layers (--sae-repo) downloaded
"""
import argparse, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labeler_service as LS   # reuse EXACT prompt assembly + extract_label + valid_label + agree

SAE_D_MODEL = 5120
SAE_D_SAE   = 81920
SAE_TOPK    = 50

def load_sae(layer, repo_dir, device):
    path = os.path.join(repo_dir, f"layer{layer}.sae.pt")
    # weights_only=False: trusted HF-hosted SAE repo (Qwen official); not user-supplied binary.
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd["W_enc"].to(device), sd["b_enc"].to(device)

@torch.no_grad()
def topk_features(residual, W_enc, b_enc, topk=SAE_TOPK):
    pre = residual @ W_enc.T + b_enc          # (seq, 81920) or (1,seq,81920)
    pre = pre.squeeze(0)  # drop leading batch dim if present -> (seq, 81920)
    vals, idx = pre.topk(topk, dim=-1)      # (seq, 50)
    out = []
    for t in range(vals.shape[0]):
        for k in range(topk):
            v = float(vals[t, k])
            if v > 0:
                out.append([t, int(idx[t, k]), v])
    return out

@torch.no_grad()
def all_features(residual, W_enc, b_enc, thr=1.0):
    """FULL dictionary activations (Goodfire-dashboard parity), not just top-k.
    residual: (G, D) generated-token residuals. Returns a compact but complete
    representation of all 81920 features:
      - max_profile: (81920,) max activation over tokens (for histogram/heatmap)
      - sparse: list of [tok_pos, feat_id, act] for features that ever fired (>thr)
    """
    pre = (residual @ W_enc.T + b_enc).squeeze(0)   # (G, 81920)
    pre = pre.float()
    max_profile = pre.max(dim=0).values            # (81920,)
    fired = pre > thr
    coords = torch.nonzero(fired, as_tuple=False)  # (n, 2): [tok_pos, feat_id]
    sparse = [[int(c[0]), int(c[1]), float(pre[c[0], c[1]])] for c in coords]
    return {"d_sae": pre.shape[-1], "threshold": thr,
            "max_profile": max_profile.tolist(), "sparse": sparse}

def _feat_hist(allf, topn=50):
    """Goodfire-style top-feature ranking across all 81920 dictionary features,
    PER LAYER (FIX Fable5 H3: was layer-0-only and misleadingly named).
    Returns {layer_str: [(feat_id, max_act), ...] sorted desc}."""
    out = {}
    for L in allf:
        prof = allf[L]["max_profile"]
        order = sorted(range(len(prof)), key=lambda i: prof[i], reverse=True)[:topn]
        out[str(L)] = [[i, round(prof[i], 4)] for i in order]
    return out

def generate_with_hooks(model, tok, messages, max_new, layers, saes, device, decoder_layers):
    """Run one chat() through HF with SAE hooks; return (label_text, feats_by_layer).

    Prefill+decode residual reconstruction:
      During model.generate() with KV-cache the hook fires once per autoregressive step.
      The PREFILL step fires with shape (1, P, D) covering all prompt tokens; each
      DECODE step fires with (1, 1, D) for one new token. Overwriting on each fire
      (the prior bug) left only the last decode step's single-token residual. Instead
      we APPEND every fire to a per-layer list and cat along dim=1, yielding (1, P+G, D).
      We then slice [P:] so tok_pos 0..G-1 in feats index the generated label tokens,
      not the prompt prefix. No second forward pass — same-inference guarantee preserved.
    """
    inp = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt")
    if not isinstance(inp, torch.Tensor):
        inp = inp["input_ids"]
    inp = inp.to(model.device)  # MUST match the embedding's device under device_map, not hardcoded cuda:0
    P = inp.shape[1]  # prompt length
    # Each layer accumulates: [prefill (1,P,D), decode_0 (1,1,D), ..., decode_G-1 (1,1,D)]
    captured = {L: [] for L in layers}
    hooks = []
    for L in layers:
        def make_hook(L):
            def _h(module, inp_h, out):
                hid = out[0] if isinstance(out, tuple) else out
                captured[L].append(hid.detach().clone())
            return _h
        hooks.append(decoder_layers[L].register_forward_hook(make_hook(L)))
    out_ids = model.generate(inp, max_new_tokens=max_new, do_sample=False)
    for h in hooks:
        h.remove()
    gen = out_ids[0, P:]               # only newly generated ids
    label_text = tok.decode(gen, skip_special_tokens=True)
    # cat [prefill (1,P,D)] + [G x (1,1,D)] => (1, P+G, D); slice to generated tokens only.
    feats = {}
    allf = {}   # FULL dictionary view (Goodfire parity): per-layer (max_profile, sparse)
    for L in layers:
        full_resid = torch.cat(captured[L], dim=1).float()   # (1, P+G, D)
        gen_resid = full_resid[0, P:, :].cpu()               # (G, D) — tok_pos 0..G-1 = label tokens
        feats[str(L)] = topk_features(gen_resid, saes[L][0], saes[L][1])
        allf[str(L)] = all_features(gen_resid, saes[L][0], saes[L][1])
    return label_text, feats, allf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "labeler_config.yaml"))
    ap.add_argument("--sae-repo", required=True,
                    help="dir containing layer{N}.sae.pt (staged SAE)")
    ap.add_argument("--layers", default="0,16,32,48,63")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--hf-cache", default="/tmp/hf_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Cap number of dataset rows processed (0 = all). For bounded deferred runs.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[args.dtype]
    layers = [int(x) for x in args.layers.split(",")]

    # --- reuse labeler_service config + prompt assembly (IDENTICAL to ollama course) ---
    cfg = LS.yaml.safe_load(open(args.config))
    prof = dict(cfg["defaults"])
    if "model_profiles" in cfg:
        prof.update(cfg["model_profiles"][cfg["run"]["model"]])
    ps = cfg["run"]["prompt_set"]
    pset = cfg["prompt_sets"][ps]
    if pset.get("schema") != "json":
        raise SystemExit("only json schema wired")
    LABELER_A_SYS = LS.load_prompt(pset["labeler_a"]) + "\n\nSchema:\n" + LS.load_prompt(pset["instruction"])
    LABELER_B_SYS = LS.load_prompt(pset["labeler_b"]) + "\n\nSchema:\n" + LS.load_prompt(pset["instruction"])
    AUDITOR_SYS    = LS.load_prompt(pset["auditor"])  + "\n\nSchema:\n" + LS.load_prompt(pset["instruction"])

    print(f"[sae-course] loading base model {args.model} ({args.dtype}) device_map=auto (local dir) ...", flush=True)
    # Files were staged flat (not in HF snapshots/ layout), so load directly from the
    # local dir. Passing cache_dir would make transformers negotiate with the hub mirror
    # and hang. trust_remote_code needed for the custom Qwen3_5 arch.
    LOCAL_MODEL_DIR = os.path.join(args.hf_cache, "Qwen", "Qwen3.5-27B")
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR, dtype=DTYPE, device_map="auto",
        trust_remote_code=True,
        max_memory={0: "20GiB", 1: "20GiB", "cpu": "300GiB"}).eval()
    # decoder-layer container can be model.model.layers or model.model.model.layers
    # (nested ForConditionalGeneration). Resolve once for hooks.
    def _find_layers(m):
        for path in ("model.layers", "model.model.layers", "language_model.model.layers",
                     "model.language_model.layers"):
            obj = m; ok = True
            for a in path.split("."):
                if hasattr(obj, a): obj = getattr(obj, a)
                else: ok = False; break
            if ok and hasattr(obj, "__len__"):
                print(f"[sae-course] decoder layers at model.{path} (n={len(obj)})", flush=True)
                return obj
        raise SystemExit("could not locate decoder layers for hooks")
    DECODER_LAYERS = _find_layers(model)
    # embedding device = where hooks' residual lands; features computed on CPU anyway
    args.device = str(next(model.parameters()).device)
    print(f"[sae-course] loading SAE layers {layers} (on CPU for feature calc) ...", flush=True)
    saes = {L: load_sae(L, args.sae_repo, "cpu") for L in layers}

    # load dataset rows (same loader logic as labeler_service)
    rows = []
    for ds in cfg["run"]["datasets"]:
        p = ds["path"] if os.path.isabs(ds["path"]) else \
            os.path.join(LS.REPO_ROOT, "experiments/v8_nla_local/labeled_outputs", ds["path"])
        with open(p) as f:
            for gi, line in enumerate(f):
                line = line.strip()
                if not line: continue
                r = json.loads(line); r["row_idx"] = gi; r["_dataset"] = ds["name"]
                r["_text_field"] = ds.get("text_field", "prompt")
                rows.append(r)

    if args.max_rows and args.max_rows > 0:
        rows = rows[:args.max_rows]
        print(f"[sae-course] --max-rows {args.max_rows}: processing {len(rows)} rows", flush=True)

    out_f = open(args.out, "w")
    for r in rows:
        text = r.get(r["_text_field"], "")
        if not text: continue
        # --- pass A (SAME inference: generate + capture features) ---
        msgs_a = [{"role":"system","content":LABELER_A_SYS}]
        if prof and prof.get("few_shot"):
            msgs_a += [{"role":"user","content":LS.FEWSHOT_USER},
                        {"role":"assistant","content":LS.FEWSHOT_ASST}]
        msgs_a.append({"role":"user","content":LS.build_user(text)})
        raw_a, feats_a, allf_a = generate_with_hooks(model, tok, msgs_a, args.max_new_tokens,
                                             layers, saes, args.device, DECODER_LAYERS)
        ja = LS.extract_label(raw_a, "fenced_json_or_last_balanced")
        # --- pass B (audit sees A's prior, same as labeler_service) ---
        msgs_b = [{"role":"system","content":LABELER_B_SYS}]
        if prof and prof.get("few_shot"):
            msgs_b += [{"role":"user","content":LS.FEWSHOT_USER},
                        {"role":"assistant","content":LS.FEWSHOT_ASST}]
        msgs_b.append({"role":"user","content":LS.build_user(text)})
        raw_b, feats_b, allf_b = generate_with_hooks(model, tok, msgs_b, args.max_new_tokens,
                                             layers, saes, args.device, DECODER_LAYERS)
        jb = LS.extract_label(raw_b, "fenced_json_or_last_balanced")
        jaud, raw_aud, feats_aud, allf_aud = None, "", None, None
        if ja:
            msgs_aud = [{"role":"system","content":AUDITOR_SYS}]
            if prof and prof.get("few_shot"):
                msgs_aud += [{"role":"user","content":LS.FEWSHOT_USER},
                             {"role":"assistant","content":LS.FEWSHOT_ASST}]
            msgs_aud.append({"role":"user","content":LS.build_user(text, ja)})
            raw_aud, feats_aud, allf_aud = generate_with_hooks(model, tok, msgs_aud, args.max_new_tokens,
                                                     layers, saes, args.device, DECODER_LAYERS)
            jaud = LS.extract_label(raw_aud, "fenced_json_or_last_balanced")

        ok = bool(ja and jb and LS.valid_label(ja) and LS.valid_label(jb))
        ag = LS.agree(ja, jb)
        rec = {"row_idx": r["row_idx"], "dataset": r.get("_dataset"),
                "input_text": text[:2000],
                "labeler_a": ja, "labeler_b": jb, "auditor_a": jaud,
                "agreement_a_b": ag,
                "raw_a": raw_a, "raw_b": raw_b, "raw_aud": raw_aud,
                "feats_a": feats_a, "feats_b": feats_b,
                "feats_aud": feats_aud,
                "allf_a": allf_a, "allf_b": allf_b, "allf_aud": allf_aud,
                "feat_hist": _feat_hist(allf_a),
                "meta": {"raw_a_len": len(raw_a), "raw_b_len": len(raw_b),
                          "ok": ok, "model": args.model, "dtype": args.dtype,
                          "layers": layers, "same_inference": True,
                          "all_features_viewable": True,
                          "d_sae": allf_a[str(layers[0])]["d_sae"]}}
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        print(f"[sae-course] row {r['row_idx']} ok={ok} agree={ag} "
              f"feats_a_layers={list(feats_a.keys())}", flush=True)

    print(f"[sae-course] DONE -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
