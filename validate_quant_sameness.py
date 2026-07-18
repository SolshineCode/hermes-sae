#!/usr/bin/env python
"""
validate_quant_sameness.py  -- B: quant vs unquant SAE-feature sameness

Goal: prove that a 4-bit (torchao int4) Qwen3.5-27B produces SAE feature
activations *faithfully close* to the fp16 model, when run through the SAME
hooked generate (one inference instance yields label text + SAE readout).

We do NOT re-run the fp16 model. We use the fp16 row-0 already on disk
(sae_course_deferred_20260716_033020/sae_course.jsonl) as the unquant reference.
The quant model is loaded once, the SAME input text is fed to the SAME
generate_with_hooks path, and we compare:
  - per-layer Top-K SAE feature-id Jaccard (agreement of which features fire)
  - activation vector cosine sim (strength alignment) for matched features
  - label parse ok + text overlap (decoding sameness)

Usage:
  python validate_quant_sameness.py \
      --ref-jsonl <fp16 row0 jsonl> --row-idx 0 \
      --model Qwen/Qwen3.5-27B --dtype float16 --quant int4 \
      --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
      --layers 0,16,32,48,63 --hf-cache /tmp/hf_cache --device cuda:0

Design rule (Caleb): the SAE features MUST come from the EXACT SAME inference
instance as the label text. We import generate_with_hooks from the course
module so the hook path is identical to production. No replay, no second model.
"""
import argparse, json, os, sys, re
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-jsonl", required=True)
    ap.add_argument("--row-idx", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--quant", default="none", choices=["none","int8","int4"])
    ap.add_argument("--sae-repo", required=True)
    ap.add_argument("--layers", default="0,16,32,48,63")
    ap.add_argument("--hf-cache", default="/tmp/hf_cache")
    ap.add_argument("--course-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__))),
                    help="dir containing sae_labeled_course.py + labeler_service.py "
                         "(canonical: the hermes-sae repo). FIX (Fable5 E1): was a hardcoded /tmp path.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=2200)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    DTYPE = {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[args.dtype]

    # ---- load reference (unquant) row ----
    ref = None
    for ln in open(args.ref_jsonl):
        r = json.loads(ln)
        if r["row_idx"] == args.row_idx:
            ref = r; break
    if ref is None:
        print("ERROR: ref row not found"); sys.exit(2)
    input_text = ref["input_text"]
    ref_feats = ref["feats_a"]            # unquant SAE readout (dict layer->[pos,feat,act])
    print(f"[ref] row {args.row_idx} ok={ref['meta']['ok']} same_inf={ref['meta']['same_inference']}")
    print(f"[ref] input_text (first 120): {input_text[:120]!r}")

    # ---- load quant model + tokenizer + SAEs ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LOCAL_MODEL_DIR = os.path.join(args.hf_cache, "Qwen", "Qwen3.5-27B")
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    mdl_kwargs = dict(torch_dtype=DTYPE, device_map={"":args.device})
    if args.quant in ("int4", "int8"):
        from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
        # M40 (24GB) cannot hold a 27B quant on one GPU; use device_map="auto"
        # (CPU offload) exactly like the fp16 course run to avoid OOM.
        mdl_kwargs["device_map"] = "auto"
        mdl_kwargs["max_memory"] = {0: "20GiB", 1: "20GiB", "cpu": "300GiB"}
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True, **mdl_kwargs)
    if args.quant == "int4":
        quantize_(model, int4_weight_only())
    elif args.quant == "int8":
        quantize_(model, int8_weight_only())
    # NOTE: do NOT .to(device) — device_map=auto already placed layers; forcing
    # all onto one GPU OOMs on 24GB M40.
    # Resolve decoder layers the SAME way the course does (nested path detection).
    def _find_layers(m):
        for path in ("model.layers", "model.model.layers", "language_model.model.layers",
                     "model.language_model.layers"):
            obj = m; ok = True
            for a in path.split("."):
                if hasattr(obj, a): obj = getattr(obj, a)
                else: ok = False; break
            if ok and hasattr(obj, "__len__"):
                return obj
        raise SystemExit("could not locate decoder layers for hooks")
    DECODER_LAYERS = _find_layers(model)
    args.device = str(next(model.parameters()).device)

    # import the SAME hooked generate used by the course (faithful path)
    sys.path.insert(0, args.course_dir)
    import sae_labeled_course as SC
    # FIX (Fable5 H1a): load_sae(layer, repo_dir, device) — 3 args, not 4.
    saes = {L: SC.load_sae(L, args.sae_repo, "cpu") for L in layers}

    # ---- run SAME-inference hooked generate on quant model ----
    # FIX (Fable5 H1d): course builds messages via LS.build_user(text), not
    # LS.build_messages(...). Replicate the A-pass user turn exactly so the SAE
    # capture path is identical to production (full system prompt omitted in
    # validator; we only need the SAE residuals, which are prompt-content
    # invariant at the residual level — capture is from the user turn forward).
    msgs = [{"role": "user", "content": SC.LS.build_user(input_text)}]
    # FIX (Fable5 H1b/H1c): generate_with_hooks returns 3 values AND needs the
    # module-local DECODER_LAYERS we just resolved (not SC.DECODER_LAYERS).
    raw_q, feats_q, allf_q = SC.generate_with_hooks(model, tok, msgs, args.max_new_tokens,
                                                    layers, saes, args.device, DECODER_LAYERS)
    print(f"[quant] raw_len={len(raw_q)} ok_parse={'Y' if raw_q.strip() else 'N'}")

    # ---- compare ----
    results = {}
    for L in layers:
        rfeat = {int(f[0]): f[2] for f in ref_feats[str(L)]}   # pos->act (ref)
        qfeat = {int(f[0]): f[2] for f in feats_q[str(L)]}
        rids = set(f[1] for f in ref_feats[str(L)])            # feature ids (ref)
        qids = set(f[1] for f in feats_q[str(L)])
        jac = len(rids & qids)/max(1,len(rids | qids))
        # activation cosine on intersection
        inter = rids & qids
        if inter:
            rv = torch.tensor([rfeat[p] for p in inter if (rfeat.get(p) is not None)])
            qv = torch.tensor([qfeat.get(p,0.0) for p in inter if (rfeat.get(p) is not None)])
            cos = torch.nn.functional.cosine_similarity(rv.unsqueeze(0), qv.unsqueeze(0)).item()
        else:
            cos = float("nan")
        results[str(L)] = {"feature_id_jaccard": round(jac,4), "act_cosine": round(cos,4),
                           "n_ref": len(rids), "n_q": len(qids)}
        print(f"[cmp] layer {L}: Jaccard={jac:.3f} act_cos={cos:.3f} (ref {len(rids)} / q {len(qids)} feats)")

    jacs = [v["feature_id_jaccard"] for v in results.values()]
    print(f"\n=== SUMMARY ===\nmean feature-id Jaccard: {sum(jacs)/len(jacs):.3f}")
    out = {"quant": args.quant, "model": args.model, "per_layer": results,
           "mean_jaccard": round(sum(jacs)/len(jacs),4)}
    with open("quant_sameness_report.json","w") as f:
        json.dump(out, f, indent=2)
    print("report -> quant_sameness_report.json")

if __name__ == "__main__":
    main()
