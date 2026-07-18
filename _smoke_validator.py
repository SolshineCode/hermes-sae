import sys, types, os
# Smoke-test the validator call-path WITHOUT a GPU (Fable5 H1a/H1b/H1c/H1d).
class FakeTok:
    @staticmethod
    def from_pretrained(*a, **k): return object()
class FakeModel:
    def __init__(self,*a,**k): pass
    def generate(self,*a,**k): return None
    def eval(self): return self
    def parameters(self): yield types.SimpleNamespace(device="cpu")
class FakeAM:
    from_pretrained = staticmethod(lambda *a, **k: FakeModel())
import torch as _t
tf = types.ModuleType("transformers")
tf.AutoTokenizer = FakeTok; tf.AutoModelForCausalLM = FakeAM
sys.modules["transformers"] = tf
qa = types.ModuleType("torchao.quantization")
qa.quantize_ = lambda *a, **k: None
qa.int4_weight_only = lambda: None
qa.int8_weight_only = lambda: None
torchao_pkg = sys.modules.setdefault("torchao", types.ModuleType("torchao"))
torchao_pkg.quantization = qa
sys.modules.setdefault("torchao.quantization", qa)

CALLS = {}
SC = types.ModuleType("sae_labeled_course")
def load_sae(layer, repo_dir, device):
    CALLS["load_sae_args"] = (layer, repo_dir, device); return (None, None)
def generate_with_hooks(model, tok, messages, max_new, layers, saes, device, decoder_layers):
    CALLS["gen_args"] = (max_new, layers, device, decoder_layers)
    return "label", {str(L): [[0,1,2.0]] for L in layers}, \
           {str(L): {"max_profile":[0,0],"threshold":1.0,"sparse":[]} for L in layers}
SC.load_sae = load_sae; SC.generate_with_hooks = generate_with_hooks
LAB = types.ModuleType("labeler_service")
LAB.build_user = lambda t: "USER:" + t
SC.LS = LAB
sys.modules["sae_labeled_course"] = SC
sys.path.insert(0, "/tmp/hermes-sae")

import importlib.util
spec = importlib.util.spec_from_file_location("vqs", "/tmp/hermes-sae/validate_quant_sameness.py")
vqs = importlib.util.module_from_spec(spec)
sys.argv = ["vqs", "--ref-jsonl", "/tmp/course_run/experiments/v8_nla_local/labeled_outputs/runs/sae_course_deferred_20260716_033020/sae_course.jsonl", "--row-idx", "0",
            "--model", "Qwen/Qwen3.5-27B", "--dtype", "float16", "--quant", "int8",
            "--sae-repo", "/tmp/hf_cache/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
            "--layers", "0,16,32,48,63", "--hf-cache", "/tmp/hf_cache", "--device", "cpu",
            "--max-new-tokens", "50", "--course-dir", "/tmp/hermes-sae"]
spec.loader.exec_module(vqs)
vqs.main()
print("\n=== CALL-PATH VERIFICATION (Fable5 H1a/b/c/d) ===")
la = CALLS.get("load_sae_args")
print("load_sae called with %r -> 3 args %s" % (la, "OK" if (la and len(la) == 3) else "WRONG"))
print("generate_with_hooks 3-return unpack %s" % ("OK (no ValueError)" if "gen_args" in CALLS else "FAILED"))
print("DECODER_LAYERS resolved locally not via SC.DECODER_LAYERS: OK")
print("build_user used (not build_messages): OK")
