#!/usr/bin/env python3
"""
propel_sae_reward.py — PROPEL-SAE reward functions (ADDITIVE; D1: does NOT modify
sae_labeled_course.py). Implemented by Hermes Agent (not Fable 5) per user directive
"Fable reviews, I run it."

Consumes allf_a[L]["sparse"] = [[tok_pos, feat_id, act], ...] from
sae_labeled_course.py:all_features(). Layer keys are STRINGS ("0","16","32","48","63").

REWARD VARIANTS (per PROPEL_SAE_BRAINSTORM.md, honoring D1-D5):
  V1  frontier_mass   — sum of activations in a band at one layer (task-difficulty proxy)
  V2  separability     — sum of a discovered feature set's activations (needs full P1 H1)
  V3  wco_min_layers   — min over layers of frontier_mass (anti-collapse, PROPEL WCO)
  V4  validity_gated   — wraps any reward; r_bad if generation empty/short

D4 INVARIANT (binding, per user decision D4): these are PASSIVE reward computers.
They accept PRE-CAPTURED activation dicts and CANNOT tell whether those came from the
frozen reference or a live policy. The D4 guarantee MUST be enforced at the GRPO
harness call site (assert frozen-ref activations only). Passing policy activations
here yields a hackable reward.

M4 GUARD (length-bias hack): _max_per_feature dedups repeated (feat_id, tok_pos) so a
single dominant feature firing at N token positions does NOT inflate the score Nx.
sae_realfeat_analysis.json shows L0 top-5 are all feat 71349 repeated — without this
guard V1 is immediately length-biased.
"""
import math


# ── helpers ──────────────────────────────────────────────────────────────────
def _blk(allf_a, layer):
    """Return allf_a block for layer; accepts int or str key. None if missing."""
    k = str(layer)
    if k in allf_a:
        return allf_a[k]
    try:
        ki = int(layer)
        if ki in allf_a:
            return allf_a[ki]
    except (ValueError, TypeError):
        pass
    return None


def _layer_acts(allf_a, layer, dedup=True):
    """Return per-feature activation from [tok, feat, act] sparse.
    dedup=True  -> dict feat_id -> MAX activation (M4 guard: a feature firing at N
                   token positions counts ONCE, not Nx).
    dedup=False -> dict feat_id -> SUM of activations (length-biased; only for testing
                   the guard / ablation)."""
    blk = _blk(allf_a, layer)
    if not blk:
        return {}
    sparse = blk.get("sparse", [])
    per_feat = {}
    for e in sparse:
        feat = int(e[1]); act = float(e[2])
        if dedup:
            if feat not in per_feat or act > per_feat[feat]:
                per_feat[feat] = act
        else:
            per_feat[feat] = per_feat.get(feat, 0.0) + act
    return per_feat


def _in_band(act, band):
    lo, hi = band
    return lo <= act <= hi


# ── V1: frontier-mass (task-difficulty proxy) ─────────────────────────────────
def sae_reward_v1(allf_a, layer=32, band=(5.0, 50.0), dedup=True):
    """Sum of activations in `band` at `layer`, deduped per-feature (M4).
    Band (5,50) chosen empirically from sae_realfeat_analysis.json: L32 is the
    semantic-clustering peak; mid-activation band tracks 'rich but not saturated'
    generations (task-difficulty proxy). D2: difficulty target. P4 calibration is the
    first empirical check that band-mass correlates with real solver difficulty."""
    per_feat = _layer_acts(allf_a, layer, dedup=dedup)
    return float(sum(a for a in per_feat.values() if _in_band(a, band)))


# ── V2: separability-feature reward (needs full-scale P1 H1) ───────────────────
def sae_reward_v2(allf_a, feature_set, dedup=True):
    """Sum of activations for a DISCOVERED feature set S* (from full-scale P1 H1 only).
    Pilot P1 output MUST NOT be used here (M2). feature_set = iterable of feat_ids."""
    total = 0.0
    seen = set()
    for L in allf_a:
        per_feat = _layer_acts(allf_a, L, dedup=dedup)
        for f in feature_set:
            if int(f) in per_feat and int(f) not in seen:
                total += per_feat[int(f)]
                seen.add(int(f))
    return float(total)


# ── V3: WCO min-over-layers (anti-collapse) ───────────────────────────────────
def sae_reward_v3(allf_a, layers=(0, 16, 32, 48, 63), band=(5.0, 50.0)):
    """PROPEL worst-case-optimization: reward = min over layers of frontier_mass.
    Anti-collapse guard — if the policy saturates one layer, the min still penalizes."""
    if not layers:
        return 0.0
    masses = [sae_reward_v1(allf_a, layer=L, band=band) for L in layers]
    return float(min(masses))


# ── V4: validity-gated wrapper ─────────────────────────────────────────────────
def sae_reward_v4(inner_reward, gen_text, min_len=1, r_bad=-1.0):
    """Wrap any reward; r_bad if generation empty/too short (validity gate).
    D4/robustness: never reward malformed generations."""
    if gen_text is None or len(str(gen_text).strip()) < min_len:
        return float(r_bad)
    return float(inner_reward)


# ── batch helper ──────────────────────────────────────────────────────────────
def sae_reward(allf_a, variant="v1", **kw):
    """Dispatch. variant in {v1,v2,v3,v4}. kw: layer,band (v1/v3), feature_set (v2),
    gen_text (v4)."""
    if variant == "v1":
        return sae_reward_v1(allf_a, kw.get("layer", 32), kw.get("band", (5.0, 50.0)))
    if variant == "v2":
        return sae_reward_v2(allf_a, kw["feature_set"])
    if variant == "v3":
        return sae_reward_v3(allf_a, kw.get("layers", (0, 16, 32, 48, 63)),
                             kw.get("band", (5.0, 50.0)))
    if variant == "v4":
        return sae_reward_v4(kw["inner_reward"], kw.get("gen_text", ""))
    raise ValueError(f"unknown variant {variant}")


if __name__ == "__main__":
    import json, sys, os
    # Self-test: run against the REAL all-features row (no GPU needed).
    ALLFEAT = os.environ.get("ALLFEAT_ROW", "")
    if not ALLFEAT:
        # default to the most recent all-features run
        import glob
        cand = sorted(glob.glob(
            "/tmp/course_run/experiments/v8_nla_local/labeled_outputs/runs/"
            "sae_course_allfeat_*/sae_course.jsonl"))
        ALLFEAT = cand[-1] if cand else ""
    if not ALLFEAT or not os.path.exists(ALLFEAT):
        print("NO all-features row found; pass ALLFEAT_ROW=..."); sys.exit(2)
    row = json.loads(open(ALLFEAT).readline())
    af = row["allf_a"]
    print(f"[self-test] row={row.get('row_idx')} layers={list(af.keys())}")

    # M4 check: isolate the repeated-feature behavior. Build a synthetic layer with
    # feat 71349 repeated 5x (same act) and NOTHING else, then confirm:
    #   dedup   == synth(once)   (single occurrence)
    #   no-dedup == 5 * synth(once)  (5x inflation WITHOUT the guard)
    rep_act = 3.17
    synth_rep = {"0": {"sparse": [[t, 71349, rep_act] for t in range(5)]}}
    synth_once = {"0": {"sparse": [[0, 71349, rep_act]]}}
    v1_dedup = sae_reward_v1(synth_rep, layer=0, band=(0.0, 1e9))
    v1_nodedup = sae_reward_v1(synth_rep, layer=0, band=(0.0, 1e9), dedup=False)
    v1_synth = sae_reward_v1(synth_once, layer=0, band=(0.0, 1e9))
    print(f"[M4] repeated-feat: dedup={v1_dedup:.4f}  no-dedup={v1_nodedup:.4f}  once={v1_synth:.4f}")
    m4_ok = abs(v1_dedup - v1_synth) < 1e-6 and abs(v1_nodedup - 5 * v1_synth) < 1e-6
    print(f"[M4] dedup collapses 5x->1x (guard active)? {m4_ok}")
    if not m4_ok:
        print(f"[M4] NOTE: real L0 has 2536 distinct features; the guard correctly "
              f"dedups 71349's 5x but other features remain. Isolation test above is the guard proof.")

    # monotonic / finite across layers
    masses = {L: sae_reward_v1(af, layer=L) for L in af}
    finite = all(math.isfinite(m) for m in masses.values())
    print(f"[V1] per-layer band(5,50) mass: { {k: round(v,1) for k,v in masses.items()} }")
    print(f"[V1] finite={finite}  L32 (clustering peak) mass={masses.get('32',0):.1f}")

    # V3 = min over layers
    v3 = sae_reward_v3(af)
    print(f"[V3] min-over-layers = {v3:.1f}")
    v3_ok = abs(v3 - min(masses.values())) < 1e-6

    # V4 validity gate
    v4_bad = sae_reward_v4(1.0, "")
    v4_ok = sae_reward_v4(1.0, "valid text") == 1.0 and v4_bad == -1.0
    print(f"[V4] empty-> {v4_bad}  valid-> 1.0  ok={v4_ok}")

    ok = m4_ok and finite and v3_ok and v4_ok
    print("[self-test]", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)
