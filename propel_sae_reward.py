#!/usr/bin/env python3
"""
propel_sae_reward.py — PROPEL-SAE reward functions (ADDITIVE; D1).

Consumes allf_a[L]["sparse"] = [[tok_pos, feat_id, act], ...] from
sae_labeled_course.py:all_features(). Layer keys are STRINGS.

REWARD VARIANTS:
  V1  frontier_mass  — sum of per-feature-max activations in a band at one layer.
  V2  separability    — sum of a DISCOVERED feature set's activations (needs full P1 H1).
  V3  wco_min_layers  — min over layers of frontier_mass (anti-collapse, PROPEL WCO).
  V4  validity_gated  — wraps any reward; r_bad if generation empty/short.

AUDIT-FIX NOTES (Fable 5, 2026-07-19) — these functions are HONEST about what they are:
  * The band (5,50) is an UNCALIBRATED placeholder, NOT an empirically-derived value.
    The prior "L32 semantic-clustering peak" justification was a duplicate-self-pair
    artifact and is REMOVED. band(5,50) is a heuristic pending P4 calibration (real
    solver); it may be replaced after calibration. V1 is an activation-diversity/magnitude
    proxy, not a proven difficulty measure.
  * V1 CAN be reward-hacked by length (more distinct mid-band features => higher score);
    M4 removes SAME-feature repetition bias but NOT distinct-feature-count bias. This is
    documented, not hidden.
  * V2 feature ids are PER-LAYER namespaces (5 independent SAEs, 81920 each). A feature
    id is now qualified as "L{layer}:{feat_id}" so L32 id 123 and L0 id 123 do NOT collide
    (prior bug).
  * V3 is min-over-layers of frontier_mass. On REAL data L0's max activation (3.17) is
    below the band floor (5.0); with skip_inactive=True, V3 uses the deepest layers that
    DO reach the band (real-data V3 = 2393.5), so it is meaningful, not degenerate. Set
    skip_inactive=False for strict WCO (any zero layer => reward 0).

D4 INVARIANT (binding): these are PASSIVE reward computers. They cannot tell whether the
activations came from the frozen reference or a live policy. The D4 guarantee MUST be
enforced at the GRPO harness call site (assert frozen-ref activations only). Even then,
the policy fully controls the text fed to the frozen reference, so input-selection reward
hacking remains an open channel (D4 prevents reward-FUNCTION substitution, not input
optimization). "Game-proof" is overstated; treat as "reference-pinned."

M4 GUARD: _max_per_feature dedups repeated (feat_id, tok_pos) so one dominant feature
firing at N token positions counts ONCE, not Nx. (Note: this also means sustained
legitimate signal is under-rewarded — prevalence-blind — a known trade-off.)
"""
import math


def _blk(allf_a, layer):
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
    """per-feature activation from [tok, feat, act] sparse.
    dedup=True  -> feat_id -> MAX activation (M4 guard).
    dedup=False -> feat_id -> SUM of activations (length-biased; ablation only)."""
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


def sae_reward_v1(allf_a, layer=32, band=(5.0, 50.0), dedup=True):
    """Sum of per-feature-max activations in `band` at `layer` (M4 dedup).
    UNCALIBRATED heuristic band — pending P4 solver calibration."""
    per_feat = _layer_acts(allf_a, layer, dedup=dedup)
    return float(sum(a for a in per_feat.values() if _in_band(a, band)))


def sae_reward_v2(allf_a, feature_set, dedup=True):
    """Sum of activations for a DISCOVERED feature set S* (full-scale P1 H1 only;
    pilot output MUST NOT feed this, M2). feature_set = iterable of QUALIFIED ids
    "L{layer}:{feat_id}" so per-layer SAE namespaces never collide (audit fix)."""
    total = 0.0
    for qid in feature_set:
        if ":" not in str(qid):
            raise ValueError(f"V2 feature id must be qualified 'L{{layer}}:{{feat}}', got {qid!r}")
        Ls, fid = str(qid).split(":", 1)
        blk = _blk(allf_a, Ls)
        if not blk:
            continue
        per_feat = _layer_acts(allf_a, Ls, dedup=dedup)
        if int(fid) in per_feat:
            total += per_feat[int(fid)]
    return float(total)


def sae_reward_v3(allf_a, layers=(0, 16, 32, 48, 63), band=(5.0, 50.0),
                  skip_inactive=True):
    """PROPEL worst-case-optimization: reward = min over layers of frontier_mass.
    skip_inactive=True: layers with zero in-band mass are excluded from the min, so V3
    is not trivially 0 just because a shallow layer never reaches the band (the REAL
    L0 case). Set False for strict WCO (any zero layer => reward 0)."""
    if not layers:
        return 0.0
    masses = []
    for L in layers:
        m = sae_reward_v1(allf_a, layer=L, band=band)
        if skip_inactive and m == 0.0:
            continue
        masses.append(m)
    if not masses:
        return 0.0
    return float(min(masses))


def sae_reward_v4(inner_reward, gen_text, min_len=1, r_bad=-1.0):
    if gen_text is None or len(str(gen_text).strip()) < min_len:
        return float(r_bad)
    return float(inner_reward)


def sae_reward(allf_a, variant="v1", **kw):
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
    import json, sys, os, glob
    ALLFEAT = os.environ.get("ALLFEAT_ROW", "")
    if not ALLFEAT:
        cand = sorted(glob.glob(
            "/tmp/course_run/experiments/v8_nla_local/labeled_outputs/runs/"
            "sae_course_allfeat_*/sae_course.jsonl"))
        ALLFEAT = cand[-1] if cand else ""
    af = None
    if ALLFEAT and os.path.exists(ALLFEAT):
        row = json.loads(open(ALLFEAT).readline())
        af = row["allf_a"]
        print(f"[self-test] row={row.get('row_idx')} layers={list(af.keys())}")
    else:
        print("[self-test] no real all-features row (ALLFEAT_ROW) — skipping real-data "
              "section; running synthetic adversarial checks only.")

    # ---- ADVERSARIAL M4 check (Fable 2a): isolate repeated feature ----
    rep_act = 3.17
    synth_rep = {"0": {"sparse": [[t, 71349, rep_act] for t in range(5)]}}
    synth_once = {"0": {"sparse": [[0, 71349, rep_act]]}}
    v1_dedup = sae_reward_v1(synth_rep, layer=0, band=(0.0, 1e9))
    v1_nodedup = sae_reward_v1(synth_rep, layer=0, band=(0.0, 1e9), dedup=False)
    v1_synth = sae_reward_v1(synth_once, layer=0, band=(0.0, 1e9))
    print(f"[M4] dedup={v1_dedup:.4f} no-dedup={v1_nodedup:.4f} once={v1_synth:.4f}")
    m4_ok = abs(v1_dedup - v1_synth) < 1e-6 and abs(v1_nodedup - 5 * v1_synth) < 1e-6
    print(f"[M4] guard collapses 5x->1x? {m4_ok}")

    # ---- ADVERSARIAL length-bias residual (Fable 2c): document, not hide ----
    A = {"32": {"sparse": [[0, f, 45.0] for f in range(30)]}}
    B = {"32": {"sparse": [[0, f, 6.0] for f in range(300)]}}
    C = {"32": {"sparse": [[0, f, 55.0] for f in range(30)]}}
    vA, vB, vC = sae_reward_v1(A), sae_reward_v1(B), sae_reward_v1(C)
    print(f"[bias] V1: focused={vA:.0f} salad={vB:.0f} strong={vC:.0f} "
          f"(salad>focused={vB > vA}; cliff-zero={vC == 0})")
    bias_doc = (vB > vA) and (vC == 0)
    print(f"[bias] length/distinct-feature bias present (documented)? {bias_doc}")

    # ---- V2 per-layer namespace (Fable 2d) ----
    v2_blk = {"0": {"sparse": [[0, 123, 40.0]]}, "32": {"sparse": [[0, 123, 9.0]]}}
    v2_good = sae_reward_v2(v2_blk, ["32:123"])
    v2_cross = sae_reward_v2(v2_blk, ["0:123"])
    v2_ok = abs(v2_good - 9.0) < 1e-6 and abs(v2_cross - 40.0) < 1e-6
    print(f"[V2] namespace-isolated: L32:123={v2_good:.1f} L0:123={v2_cross:.1f} ok={v2_ok}")

    # ---- V3 non-trivial on synthetic data where ALL layers reach band (Fable H3) ----
    v3_blk = {str(L): {"sparse": [[0, 100 + L, 20.0]]} for L in (0, 16, 32, 48, 63)}
    v3 = sae_reward_v3(v3_blk, layers=(0, 16, 32, 48, 63))
    v3_real = sae_reward_v3(af) if af else None
    v3_ok = abs(v3 - 20.0) < 1e-6 and v3 > 0.0
    print(f"[V3] synthetic(min over layers)={v3:.1f} (expect 20.0)"
          + (f"  real-data V3={v3_real:.1f}" if v3_real is not None else "  (real: n/a)"))

    # ---- V4 validity gate ----
    v4_bad = sae_reward_v4(1.0, "")
    v4_ok = sae_reward_v4(1.0, "valid text") == 1.0 and v4_bad == -1.0
    print(f"[V4] empty->{v4_bad} valid->{1.0 if v4_ok else 'X'} ok={v4_ok}")

    ok = m4_ok and bias_doc and v2_ok and v3_ok and v4_ok
    print("[self-test]", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)
