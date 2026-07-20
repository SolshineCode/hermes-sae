#!/usr/bin/env python3
"""
nla_server.py -- OpenAI-compatible model server with NLA/SAE activation capture.

Loads ONE model instance persistently (default: Gemma-4-E2B NF4). Serves an
OpenAI-compatible API (POST /v1/chat/completions, GET /v1/models, GET /healthz).
On every generation, a forward hook on a configured transformer layer records the
residual-stream activation at EACH position ("all SAE activations"). Records are
written per request to --log-dir:
    activations.jsonl      one line per request (metadata + npz path)
    run_<request_id>.npz   acts [N, d_model] float32, norms, is_prefill,
                           token_ids, gen_ids

INTEGRATION POINT FOR AGENT-INTEGRATED SAE CAPTURE:
Point Hermes Agent's custom model provider at this server (provider=custom,
base_url=http://127.0.0.1:PORT/v1, model=gemma-4-e2b). Then every agent turn
hits /v1/chat/completions -> the hook fires on the SAME model instance that runs
the agent -> activations are captured automatically, tied to that request.

Usage:
    python nla_server.py --model-dir weights/gemma-4-E2B --port 8000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import datetime
import uuid
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------- globals (populated in load()) ----------
TOK = None
MODEL = None
LAYER_MODULE = None
LAYER = 23
D_MODEL = 1536
MODEL_ID = "gemma-4-e2b"
LOG_DIR = "logs/nla_server"
MAX_TOKENS_CAP = 1024
LOCK = threading.Lock()          # serialize generation (single GPU)
active_collector = None          # set during a generation under LOCK
LOG_FILE = None                  # file mirror of log() (set in load())


def log(*a):
    msg = f"[{datetime.datetime.now():%H:%M:%S}] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOG_FILE:
        try:
            LOG_FILE.write(msg + "\n")
            LOG_FILE.flush()
        except Exception:
            pass


def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


class ActCollector:
    """Forward-hook collector.

    Records the last-position residual activation at EVERY forward pass during
    generation:
        records[0]            = prefill step  -> last PROMPT token activation
        records[1 .. N]       = decode steps  -> activation at generated token k
    This captures ALL per-position SAE activations of the inference instance.
    """

    def __init__(self):
        self.records = []

    def hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        vec = h[0, -1].detach().cpu().float().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec)) + 1e-9
        is_prefill = (len(self.records) == 0)
        self.records.append({"vec": vec, "norm": norm, "is_prefill": is_prefill})


def load(args):
    global TOK, MODEL, LAYER_MODULE, LAYER, D_MODEL, MODEL_ID, LOG_DIR, MAX_TOKENS_CAP
    LAYER = args.layer
    D_MODEL = args.d_model
    MODEL_ID = args.model_id
    LOG_DIR = args.log_dir
    MAX_TOKENS_CAP = args.max_tokens_cap
    os.makedirs(LOG_DIR, exist_ok=True)
    global LOG_FILE
    LOG_FILE = open(os.path.join(LOG_DIR, "server.log"), "a", buffering=1)
    log(f"Loading base {args.model} (NF4) from {args.model_dir} ...")
    TOK = AutoTokenizer.from_pretrained(args.model_dir)
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=make_bnb_config(),
        device_map={"": torch.cuda.current_device()},
    )
    MODEL.eval()
    LAYER_MODULE = MODEL.model.language_model.layers[LAYER]
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    log(f"base loaded. GPU free {free:.2f} GB | layer={LAYER} d_model={D_MODEL} model_id={MODEL_ID}")


def format_messages(messages):
    """Render chat messages into a single prompt via the model's chat template."""
    try:
        return TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"


def write_records(request_id, session_id, prompt, records, gen_ids, text):
    ts = datetime.datetime.now().isoformat()
    n = len(records)
    acts = np.stack([r["vec"] for r in records]).astype(np.float32)        # [n, d_model]
    norms = np.array([r["norm"] for r in records], dtype=np.float32)
    is_prefill = np.array([r["is_prefill"] for r in records], dtype=bool)
    token_ids = np.array([r.get("token_id", -1) for r in records], dtype=np.int64)
    npz_path = os.path.join(LOG_DIR, f"run_{request_id}.npz")
    np.savez(npz_path, acts=acts, norms=norms, is_prefill=is_prefill,
             token_ids=token_ids, gen_ids=np.array(gen_ids, dtype=np.int64))
    meta = {
        "request_id": request_id,
        "session_id": session_id,
        "timestamp": ts,
        "model": MODEL_ID,
        "layer": LAYER,
        "d_model": D_MODEL,
        "n_records": n,
        "n_gen_tokens": len(gen_ids),
        "prompt_preview": prompt[:200],
        "npz_path": npz_path,
        "gen_text_preview": text[:200],
    }
    with open(os.path.join(LOG_DIR, "activations.jsonl"), "a") as f:
        f.write(json.dumps(meta) + "\n")
    log(f"  captured {n} activations (1 prefill + {n - 1} generated) -> {os.path.basename(npz_path)}")
    return meta


def generate_capture(prompt, max_tokens, temperature, request_id, session_id):
    global active_collector
    collector = ActCollector()
    ids = TOK.encode(prompt, return_tensors="pt").to(MODEL.device)
    prompt_len = ids.shape[1]
    do_sample = temperature > 0
    gen_kwargs = dict(
        input_ids=ids,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        pad_token_id=TOK.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 1.0
    with LOCK:
        active_collector = collector
        handle = LAYER_MODULE.register_forward_hook(collector.hook)
        try:
            with torch.no_grad():
                out = MODEL.generate(**gen_kwargs)
        finally:
            handle.remove()
            active_collector = None
    gen_ids = out[0][prompt_len:].tolist()
    text = TOK.decode(gen_ids, skip_special_tokens=True)
    # Align generated token ids/text onto decode records (records[1..N]).
    for i in range(1, len(collector.records)):
        if i - 1 < len(gen_ids):
            collector.records[i]["token_id"] = gen_ids[i - 1]
            collector.records[i]["text"] = TOK.decode([gen_ids[i - 1]], skip_special_tokens=False)
    write_records(request_id, session_id, prompt, collector.records, gen_ids, text)
    return text, gen_ids, prompt_len


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        for c in chunks:
            self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = self.path.rstrip("/")
        if p.endswith("/models"):
            self._send({"object": "list",
                        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "nla"}]})
        elif p.endswith("/healthz"):
            self._send({"status": "ok", "model": MODEL_ID, "layer": LAYER, "d_model": D_MODEL})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        p = self.path.rstrip("/")
        if p.endswith("/chat/completions") or p.endswith("/completions"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                req = json.loads(raw or b"{}")
            except Exception as e:
                self._send({"error": f"bad request: {e}"}, 400)
                return
            self.handle_chat(req)
        else:
            self._send({"error": "not found"}, 404)

    def handle_chat(self, req):
        messages = req.get("messages")
        if not messages:
            # plain /completions: treat prompt as a single user turn
            messages = [{"role": "user", "content": req.get("prompt", "")}]
        max_tokens = min(int(req.get("max_tokens", req.get("max_new_tokens", 200))), MAX_TOKENS_CAP)
        temperature = float(req.get("temperature", 0.7))
        stream = bool(req.get("stream", False))
        request_id = (req.get("request_id")
                      or self.headers.get("X-Request-Id")
                      or uuid.uuid4().hex)
        session_id = self.headers.get("X-Hermes-Session-Id") or req.get("session_id")
        prompt = format_messages(messages)
        log(f"[req {request_id[:8]}] session={session_id} msgs={len(messages)} max_tokens={max_tokens} stream={stream}")
        t0 = time.time()
        text, gen_ids, prompt_len = generate_capture(prompt, max_tokens, temperature, request_id, session_id)
        elapsed = time.time() - t0
        log(f"[req {request_id[:8]}] done in {elapsed:.1f}s, {len(gen_ids)} tokens")
        created = int(time.time())
        resp_model = req.get("model", MODEL_ID)   # echo requested name back
        resp = {
            "id": "chatcmpl-" + request_id[:12],
            "object": "chat.completion",
            "created": created,
            "model": resp_model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(prompt_len),
                      "completion_tokens": len(gen_ids),
                      "total_tokens": int(prompt_len) + len(gen_ids)},
        }
        if stream:
            chunk0 = {"id": resp["id"], "object": "chat.completion.chunk", "created": created,
                      "model": resp_model,
                      "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                   "finish_reason": None}]}
            chunk_done = {"id": resp["id"], "object": "chat.completion.chunk", "created": created,
                          "model": resp_model,
                          "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self._send_sse([chunk0, chunk_done])
        else:
            self._send(resp)

    def log_message(self, fmt, *args):
        pass  # silence default per-request logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/caleb/nla_run/weights/gemma-4-E2B")
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--model-id", default="gemma-4-e2b")
    ap.add_argument("--log-dir", default="/home/caleb/nla_run/logs/nla_server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens-cap", type=int, default=1024)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log("ERROR: CUDA unavailable"); sys.exit(1)
    load(args)
    log(f"Serving OpenAI-compatible API at http://{args.host}:{args.port}/v1  (log-dir={LOG_DIR})")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
