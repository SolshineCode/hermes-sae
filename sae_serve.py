#!/usr/bin/env python3
"""
sae_serve.py — LIVE SAE-instrumented OpenAI-compatible inference server.

TOP-LEVEL REQUIREMENT (Caleb, 2026-07-19):
  "The EXACT SAME model instance that is running Hermes Agent (local LLMs only, not
   when it's a nonlocal LLM) is the one we gather the SAE activation records from,
   while it is doing the EXACT inference that runs Hermes Agent."

HOW THIS SATISFIES IT (same-inference by construction — identical guarantee to
sae_labeled_course.generate_with_hooks, now on the SERVING path):
  - One in-process HF model. Forward hooks on decoder layers L.
  - Every /v1/chat/completions request runs ONE model.generate(). The hook fires on
    prefill (1,P,D) and each decode step (1,1,D); we append + cat -> (1,P+G,D), slice
    [P:] to keep only generated-token residuals, SAE-encode them. The text returned to
    Hermes AND the SAE features are produced by that SINGLE generate() call. No replay,
    no second forward pass. Faithful by construction.
  - Because Hermes talks to this server over the OpenAI API exactly like it talks to
    ollama, running Hermes with model=sae-local automatically records the SAE feature
    history of the agent's own activity, turn by turn, with zero extra steps.

WHAT IS **NOT** CAPTURED (by design, per requirement):
  - Non-local models (cloud). Hermes only routes through this server when the selected
    model is `sae-local`. Cloud turns never hit this process, so no SAE is produced for
    them (and none is possible — no residual stream access). This is correct behavior.

INTEGRATION (no Hermes core change):
  Add an OpenAI-compatible provider pointing at this server, and a model entry, e.g.
    providers:
      sae:
        base_url: http://localhost:8077/v1
    model:
      default: sae-local
      provider: sae
  Then `/model` -> sae-local routes all agent generation through the instrumented model.

FEATURE-HISTORY LOG (JSONL, one line per request):
  {"ts","request_id","model","messages_sha","prompt_len","gen_len","gen_text",
   "layers","feats_topk": {layer: [[tok,feat,act],...]},
   "allf": {layer: {"d_sae","threshold","max_profile","sparse"}},  # optional (--allf)
   "same_inference": true, "timing": {...}}

USAGE:
  TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 python sae_serve.py \
     --model-dir /tmp/hf_cache/Qwen/Qwen3.5-27B \
     --sae-repo /tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 \
     --layers 0,16,32,48,63 --dtype float16 --port 8077 \
     --log runs/agent_sae_history.jsonl [--allf]

  # Plumbing smoke-test with a tiny in-memory model (no GPU, no big download):
  python sae_serve.py --self-test
"""
import argparse, json, os, sys, time, hashlib, threading, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch


def load_sae(layer, repo_dir, device):
    # Dependency-free loader: handles BOTH the local Qwen PoC .pt format AND professionally
    # released SAEs (Bloom .pt pickles AND EleutherAI-style .safetensors). Dispatches on
    # extension via load_released_sae (stubs sae_training for .pt, reads safetensors for .safetensors).
    try:
        from load_released_pt import load_released_sae
        return load_released_sae(os.path.join(repo_dir, f"layer{layer}.sae.pt"), device)
    except FileNotFoundError:
        # try .safetensors variant (released SAEs may be named layerL.sae.safetensors)
        from load_released_pt import load_released_sae as _lrs
        return _lrs(os.path.join(repo_dir, f"layer{layer}.sae.safetensors"), device)
    except Exception as e:
        raise RuntimeError(f"failed to load SAE layer {layer} from {repo_dir}: {e}")


def load_released_sae(layer, release, device, width=None, hook="hook_resid_pre"):
    """Load a PROFESSIONALLY RELEASED SAE (sae_lens / Neuronpedia) for a small model
    (e.g. gpt2-small) and adapt to the (W_enc, b_enc) the engine expects. No local .pt
    required — proves the pipeline is model/SAE-agnostic and runs CPU-only.

    sae_lens stores state as W_enc / W_dec / b_enc / b_dec / threshold; we only need
    W_enc + b_enc for the probe-topk and all-feature encodings, so we extract those.
    """
    try:
        from sae_lens import SparseAutoencoder
    except Exception as e:
        raise RuntimeError(
            "sae_lens not installed (need: pip install sae_lens) to use --sae-source saelens") from e
    sae_id = f"blocks.{layer}.{hook}"
    if width:
        sae_id = f"{sae_id}_{width}"
    sae = SparseAutoencoder.load_from_hf(release, sae_id)
    # sae_lens exposes torch tensors (sometimes wrapped in a structure); pull (W_enc, b_enc)
    W_enc = sae.W_enc  # (d_sae, d_in)
    b_enc = sae.b_enc  # (d_sae,)
    if not isinstance(W_enc, torch.Tensor):
        W_enc = W_enc()
    if not isinstance(b_enc, torch.Tensor):
        b_enc = b_enc()
    return W_enc.detach().float().to(device), b_enc.detach().float().to(device)


@torch.no_grad()
def topk_features(residual, W_enc, b_enc, topk=50):
    pre = residual @ W_enc.T + b_enc
    pre = pre.squeeze(0)
    if pre.dim() == 1:
        pre = pre.unsqueeze(0)
    k = min(topk, pre.shape[-1])
    vals, idx = pre.topk(k, dim=-1)
    out = []
    for t in range(vals.shape[0]):
        for j in range(k):
            v = float(vals[t, j])
            if v > 0:
                out.append([t, int(idx[t, j]), v])
    return out


@torch.no_grad()
def all_features(residual, W_enc, b_enc, thr=1.0):
    pre = (residual @ W_enc.T + b_enc)
    if pre.dim() == 3:
        pre = pre.squeeze(0)
    pre = pre.float()
    max_profile = pre.max(dim=0).values
    fired = pre > thr
    coords = torch.nonzero(fired, as_tuple=False)
    sparse = [[int(c[0]), int(c[1]), float(pre[c[0], c[1]])] for c in coords]
    return {"d_sae": pre.shape[-1], "threshold": thr,
            "max_profile": max_profile.tolist(), "sparse": sparse}


def find_decoder_layers(m):
    for path in ("model.layers", "model.model.layers", "language_model.model.layers",
                 "model.language_model.layers", "transformer.h"):
        obj = m
        ok = True
        for a in path.split("."):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            return obj
    raise RuntimeError("could not locate decoder layers for hooks")


class SAEEngine:
    """Holds ONE model instance + SAE weights. generate_capture() runs a single
    generate() and returns (text, feats_topk, allf) from that SAME inference."""

    def __init__(self, model, tok, saes, layers, decoder_layers, allf=False,
                 allf_thr=1.0, topk=50, hook_mode="resid"):
        self.model = model
        self.tok = tok
        self.saes = saes
        self.layers = layers
        self.decoder_layers = decoder_layers
        self.allf = allf
        self.allf_thr = allf_thr
        self.topk = topk
        self.hook_mode = hook_mode
        # Resolve which module to hook per layer. resid -> decoder layer residual_pre;
        # mlp -> the MLP submodule output (for MLP-hooked released SAEs, e.g. EleutherAI).
        self.hook_modules = {}
        for L in layers:
            if hook_mode == "mlp":
                try:
                    self.hook_modules[L] = self.model.model.layers[L].mlp
                except AttributeError:
                    raise RuntimeError(
                        f"hook_mode='mlp' but model has no model.layers[{L}].mlp")
            else:
                self.hook_modules[L] = self.decoder_layers[L]
        self._lock = threading.Lock()

    @torch.no_grad()
    def generate_capture(self, messages, max_new_tokens=512, temperature=0.0):
        with self._lock:
            tok = self.tok
            inp = tok.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=True, return_tensors="pt")
            if not isinstance(inp, torch.Tensor):
                inp = inp["input_ids"]
            inp = inp.to(self.model.device)
            P = inp.shape[1]
            captured = {L: [] for L in self.layers}
            hooks = []
            for L in self.layers:
                def make_hook(L):
                    def _h(module, inp_h, out):
                        hid = out[0] if isinstance(out, tuple) else out
                        captured[L].append(hid.detach().clone())
                    return _h
                hooks.append(self.hook_modules[L].register_forward_hook(make_hook(L)))
            gen_kwargs = dict(max_new_tokens=max_new_tokens)
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs.update(do_sample=False)
            try:
                out_ids = self.model.generate(inp, **gen_kwargs)
            finally:
                for h in hooks:
                    h.remove()
            gen = out_ids[0, P:]
            text = tok.decode(gen, skip_special_tokens=True)
            feats, allf = {}, {}
            for L in self.layers:
                full = torch.cat(captured[L], dim=1).float()
                gen_resid = full[0, P:, :].cpu()
                W_enc, b_enc = self.saes[L]
                feats[str(L)] = topk_features(gen_resid, W_enc, b_enc, self.topk)
                if self.allf:
                    allf[str(L)] = all_features(gen_resid, W_enc, b_enc, self.allf_thr)
            return text, feats, allf, P, int(gen.shape[0])


class Handler(BaseHTTPRequestHandler):
    engine = None
    model_name = "sae-local"
    log_path = None
    log_lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": self.model_name, "object": "model", "owned_by": "sae-serve"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._send(200, {"status": "ok", "model": self.model_name})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "only /v1/chat/completions supported"})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        messages = req.get("messages", [])
        # OPTIONAL: caller may pass row_idx / meta for join-safe, resume-safe capture.
        # (sae_serve echoes them into the log record so the capture file is self-contained.)
        row_idx = req.get("row_idx", None)
        meta_in = req.get("meta", {}) or {}
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 512)
        temperature = float(req.get("temperature", 0.0) or 0.0)
        t0 = time.time()
        try:
            text, feats, allf, P, G = self.engine.generate_capture(
                messages, max_new_tokens=max_new, temperature=temperature)
        except Exception as e:
            self._send(500, {"error": {"message": f"generate failed: {e}",
                                       "type": "sae_serve_error"}})
            return
        dt = time.time() - t0
        rid = "chatcmpl-" + hashlib.sha1(f"{t0}{text[:32]}".encode()).hexdigest()[:24]
        if self.log_path:
            rec = {
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "request_id": rid, "model": self.model_name,
                "messages_sha": hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16],
                "prompt_len": P, "gen_len": G, "gen_text": text,
                "layers": self.engine.layers, "feats_topk": feats,
                "same_inference": True, "timing": {"seconds": round(dt, 3)},
                "row_idx": row_idx, "meta": meta_in,
            }
            if allf:
                rec["allf"] = allf
            with self.log_lock:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        self._send(200, {
            "id": rid, "object": "chat.completion", "created": int(t0),
            "model": self.model_name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": P, "completion_tokens": G,
                      "total_tokens": P + G},
        })


def build_engine_from_args(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    layers = [int(x) for x in args.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.chat_template is None:
        # Base models (e.g. gpt2) ship without a chat template. Provide a minimal generic
        # one so generate_capture's apply_chat_template works unmodified.
        tok.chat_template = "{% for message in messages %}{{ 'User: ' + message['content'] + '\nAssistant: ' }}{% endfor %}"
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for kv in args.max_memory.split(","):
            k, v = kv.split("=")
            max_memory[int(k) if k.isdigit() else k] = v
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=DTYPE, device_map="auto",
        trust_remote_code=True, max_memory=max_memory).eval()
    decoder_layers = find_decoder_layers(model)
    if getattr(args, "sae_source", "local") == "saelens":
        # Professionally released SAE (sae_lens / Neuronpedia) — model/SAE-agnostic CPU path.
        saes = {L: load_released_sae(L, args.sae_release, "cpu",
                                     width=getattr(args, "sae_width", None))
                for L in layers}
    else:
        saes = {L: load_sae(L, args.sae_repo, "cpu") for L in layers}
    hook_mode = getattr(args, "hook_mode", "resid")
    return SAEEngine(model, tok, saes, layers, decoder_layers,
                     allf=args.allf, allf_thr=args.allf_thr, topk=args.topk,
                     hook_mode=hook_mode)


def serve(engine, host, port, model_name, log_path):
    Handler.engine = engine
    Handler.model_name = model_name
    Handler.log_path = log_path
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[sae-serve] listening on http://{host}:{port}/v1  model={model_name}  "
          f"layers={engine.layers}  log={log_path}", flush=True)
    httpd.serve_forever()


def _self_test():
    """Plumbing test with a tiny in-memory GPT2 + random SAE (no disk load, no GPU).
    Verifies: (1) one generate() yields BOTH text and per-layer features,
    (2) feature token positions align to generated tokens (G rows),
    (3) same-inference invariant holds (feats derived from the generated residuals),
    (4) OpenAI-compatible request/response + log line round-trips.
    Builds GPT2 from a config (no torch.load) to isolate SAE-hook plumbing from
    checkpoint format. Real serving needs TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 for .pt SAEs
    (trusted HF repo) and works with safetensors base models."""
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    import urllib.request
    D = 64
    d_sae = 128
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=D, vocab_size=50257)
    model = GPT2LMHeadModel(cfg).eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            "assistant: ")
    decoder_layers = find_decoder_layers(model)
    torch.manual_seed(0)
    saes = {L: (torch.randn(d_sae, D) * 0.1, torch.zeros(d_sae)) for L in (0, 1)}
    eng = SAEEngine(model, tok, saes, [0, 1], decoder_layers, allf=True, allf_thr=0.0,
                    topk=10)
    text, feats, allf, P, G = eng.generate_capture(
        [{"role": "user", "content": "hello there"}], max_new_tokens=8)
    assert isinstance(text, str), "no text returned"
    assert G > 0, "no tokens generated"
    for L in (0, 1):
        for tokpos, fid, act in feats[str(L)]:
            assert 0 <= tokpos < G, f"feat tokpos {tokpos} out of generated range {G}"
        assert allf[str(L)]["d_sae"] == d_sae
        assert len(allf[str(L)]["max_profile"]) == d_sae
    print(f"[self-test]   generate(): text={text!r:.40} P={P} G={G} "
          f"feat_layers={list(feats)} L0_feats={len(feats['0'])}")

    Handler.engine = eng
    Handler.model_name = "sae-local-test"
    logf = "/tmp/_sae_serve_selftest.jsonl"
    if os.path.exists(logf):
        os.remove(logf)
    Handler.log_path = logf
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10)
        models = json.loads(r.read())
        assert models["data"][0]["id"] == "sae-local-test", "model list wrong"
        payload = json.dumps({"model": "sae-local-test", "max_tokens": 6,
                              "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        assert resp["object"] == "chat.completion", "bad response object"
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert "content" in resp["choices"][0]["message"]
        assert resp["usage"]["completion_tokens"] > 0
    finally:
        httpd.shutdown()
    with open(logf) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) == 1, f"expected 1 log line, got {len(lines)}"
    rec = lines[0]
    assert rec["same_inference"] is True
    assert rec["gen_len"] == resp["usage"]["completion_tokens"], \
        "log gen_len != response completion_tokens (same-inference mismatch!)"
    assert rec["gen_text"] == resp["choices"][0]["message"]["content"], \
        "log text != returned text (NOT same inference!)"
    assert set(rec["feats_topk"]) == {"0", "1"}
    print(f"[self-test]   HTTP round-trip OK; log gen_len={rec['gen_len']} "
          f"matches response; text identical (same-inference verified)")
    print("[self-test] PASS ✅")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--sae-repo")
    ap.add_argument("--sae-source", default="local", choices=["local", "saelens"],
                    help="local=.pt loader (Qwen PoC); saelens=professionally released SAE (gpt2-small etc.)")
    ap.add_argument("--sae-release", default="",
                    help="sae_lens release name, e.g. 'gpt2-small-res-jb' (with --sae-source saelens)")
    ap.add_argument("--sae-width", default="",
                    help="optional SAE width suffix for the id, e.g. '16' -> blocks.L.hook_resid_pre_16")
    ap.add_argument("--layers", default="0,16,32,48,63")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max-memory", default="",
                    help="e.g. '0=20GiB,1=20GiB,cpu=300GiB' (device_map=auto budget)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--model-name", default="sae-local")
    ap.add_argument("--log", default="/home/darkstar/hermes_cache/dpilot_capture/sae_history.jsonl")
    ap.add_argument("--allf", action="store_true",
                    help="also log FULL-dictionary activations (large)")
    ap.add_argument("--allf-thr", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--hook-mode", default="resid", choices=["resid", "mlp"],
                    help="resid=hook decoder-layer residual (Qwen PoC / Bloom gpt2 SAEs); "
                         "mlp=hook MLP output (EleutherAI released Qwen SAEs)")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.model_dir:
        raise SystemExit("--model-dir required (or use --self-test)")
    if args.sae_source == "local" and not args.sae_repo:
        raise SystemExit("--sae-repo required with --sae-source local (or use --self-test)")
    if args.sae_source == "saelens" and not args.sae_release:
        raise SystemExit("--sae-release required with --sae-source saelens")
    # normalize sae_width to int or None
    if args.sae_width:
        try:
            args.sae_width = int(args.sae_width)
        except ValueError:
            raise SystemExit("--sae-width must be an integer (e.g. 16)")
    else:
        args.sae_width = None
    print(f"[sae-serve] loading {args.model_dir} ({args.dtype}) device_map=auto "
          f"[sae_source={args.sae_source}] ...", flush=True)
    eng = build_engine_from_args(args)
    serve(eng, args.host, args.port, args.model_name, args.log)


if __name__ == "__main__":
    main()
