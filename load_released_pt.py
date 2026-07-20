#!/usr/bin/env python3
"""load_released_pt.py — dependency-free loader for released SAE .pt files (e.g. Joseph
Bloom's GPT2-Small SAEs in jbloom/GPT2-Small-SAEs).

The released files are `torch.save(full_SparseAutoencoder_object)`, which pickles a
reference to the `sae_training.SparseAutoencoder` class. We avoid installing that heavy
dep by stubbing `sys.modules['sae_training']` with a lightweight class, then letting
torch.load unpickle the real object and reading its `state_dict` (W_enc / b_enc — the
only tensors the PROPEL-SAE probe needs).

Returns (W_enc [d_sae, d_in], b_enc [d_sae]).
"""
import sys
import types
import torch


class _StubSparseAutoencoder:
    """Stand-in for sae_training.* classes during unpickling.

    torch.load rebuilds the object; any missing attributes are tolerated. After load we
    reach the real tensors via the object's state_dict (a plain dict of tensors).
    """
    def __init__(self, *a, **k):
        pass

    def __setattr__(self, k, v):
        super().__setattr__(k, v)

    def state_dict(self, *a, **k):
        return {k: v for k, v in vars(self).items() if isinstance(v, torch.Tensor)}


class _StubModule(types.ModuleType):
    """Synthetic sae_training.* module: any attribute access yields a generic stub class."""
    def __getattr__(self, name):
        # return a callable that builds a _StubSparseAutoencoder (handles both classes
        # and config objects referenced by the pickle)
        return _StubSparseAutoencoder


def _install_stub():
    # Make `sae_training` and any submodule (sae_training.config, ...) resolvable.
    for name in ("sae_training", "sae_training.config", "sae_training.training",
                 "sae_training.model", "sae_training.utils"):
        sys.modules[name] = _StubModule(name)


def _cleanup_stub():
    # Remove stubs so they don't pollute the live import system (torch.library introspection
    # otherwise walks into them and crashes on later imports).
    for name in list(sys.modules):
        if name == "sae_training" or name.startswith("sae_training."):
            del sys.modules[name]


def load_released_sae_pt(path, device="cpu"):
    _install_stub()
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        _cleanup_stub()
    # Released Bloom SAE .pt files are torch.save({cfg, state_dict}); the tensors live in
    # the nested 'state_dict' OrderedDict.
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    else:
        sd = vars(obj)
    W_enc = sd["W_enc"]
    b_enc = sd["b_enc"]
    if not isinstance(W_enc, torch.Tensor):
        W_enc = W_enc.data if hasattr(W_enc, "data") else torch.as_tensor(W_enc)
    if not isinstance(b_enc, torch.Tensor):
        b_enc = b_enc.data if hasattr(b_enc, "data") else torch.as_tensor(b_enc)
    # Released Bloom SAE stores W_enc as (d_in, d_sae); the PROPEL-SAE engine expects
    # (d_sae, d_in) so the activation is residual @ W_enc.T. Transpose to match convention.
    if W_enc.shape[0] < W_enc.shape[1]:
        W_enc = W_enc.T.contiguous()
    return W_enc.detach().float().to(device), b_enc.detach().float().to(device)


if __name__ == "__main__":
    import os
    p = sys.argv[1] if len(sys.argv) > 1 else "/home/darkstar/hermes_cache/cpu_sae/gpt2-small-sae/layer7.sae.pt"
    W, b = load_released_sae_pt(p)
    print("W_enc", tuple(W.shape), "b_enc", tuple(b.shape))
