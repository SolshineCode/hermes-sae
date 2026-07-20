# hermes-sae-probe

Watch which SAE features fire, per token, while a local HuggingFace model
generates. One file, one command. Built for (but not limited to) local models
running agentic workloads under Hermes Agent: the probe reads the residual
stream in the same forward pass that produces the output, so you get a live
internal trace alongside the transcript.

```
pip install torch transformers huggingface_hub
python sae_probe.py --model HuggingFaceTB/SmolLM2-135M \
    --sae-repo Solshine/deception-saes-smollm2-135m --sae-variant topk --layers 12 \
    --prompt "Explain what a sparse autoencoder is." --top-k 5
```

Prints a per-token feature readout and writes a `feature_trace.jsonl` sidecar.

SAE format: one `.pt` state dict per layer with `W_enc` / `b_enc`
(SAELens-compatible convention). Feature activation is `relu(x @ W_enc + b_enc)`.

Notes:
- Works on CPU for small models; use a GPU for 7B+.
- The hook reads the last position each forward, so with KV-cache generation
  you get exactly one readout per generated token (plus one for prefill).
- This is the read-only half. Steering (feature-level nudges at serve time)
  is a separate, experimental branch of this work.
