#!/usr/bin/env python3
"""
labeler_service.py  --  robust, replicable NLA labeling service (v4, config-driven)

Reads labeler_config.yaml. For each dataset row, runs the configured passes
(labeler_a, labeler_b, auditor) against a local ollama model using the model's
PROFILE (extraction rules + few-shot). Restart-safe via --resume.

Key design (per 2026-07-10 directive):
  - Model + extraction are CONFIG, not code. Swap models by editing labeler_config.yaml.
  - Few-shot is embedded per labeler to stop schema-template echo (proven needed for qwen3.5:27b).
  - All artifacts additive under experiments/v8_nla_local/labeled_outputs/.

Usage:
  python labeler_service.py --config labeler_config.yaml [--resume]
"""
import argparse, json, os, sys, time, re, hashlib, subprocess, datetime, urllib.request, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.normpath(os.path.join(HERE, "..", "prompts"))   # labeled_outputs -> .. -> prompts
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))


def load_prompt(rel):
    with open(os.path.normpath(os.path.join(PROMPTS_DIR, rel))) as f:
        return f.read().split("---")[0].strip()


def sha256_file(rel):
    p = os.path.normpath(os.path.join(PROMPTS_DIR, rel))
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---- extraction per model profile ----
def extract_label(raw, method):
    if method == "fenced_json_or_last_balanced":
        # 1) fenced ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # 2) first { ... last } (handles braces inside strings via greedy outer match)
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except Exception:
                pass
        # 3) balanced-brace scan (fallback)
        cands = [i.start() for i in re.finditer(r"\{", raw)]
        best = None
        for i in cands:
            depth = 0
            for j in range(i, len(raw)):
                if raw[j] == "{":
                    depth += 1
                elif raw[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            best = json.loads(raw[i:j + 1])
                        except Exception:
                            pass
                        break
        if best is not None:
            return best
        # 4) truncated-JSON recovery: output was cut off at the token cap mid-object.
        #    Close any open braces/strings, drop a trailing partial key/value, retry.
        return _recover_truncated_json(raw, s)
    return None


def _recover_truncated_json(raw, start):
    if start == -1:
        return None
    body = raw[start:]
    # strip a dangling trailing key or value fragment like ,"notes": or ,"notes": "par
    body = re.sub(r',\s*"[^"]*"\s*:\s*("?[^"]*)?\s*$', '', body)
    body = re.sub(r',\s*"[^"]*\s*$', '', body)  # trailing open key
    depth = 0
    closed = []
    in_str = False
    esc = False
    for ch in body:
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1; closed.append('}')
        elif ch == '[':
            depth += 1; closed.append(']')
        elif ch in '}]':
            if closed:
                closed.pop()
    frag = body + ''.join(reversed(closed))
    try:
        return json.loads(frag)
    except Exception:
        return None


# ---- few-shot exemplar (proven to stop schema echo on qwen3.5:27b) ----
FEWSHOT_USER = 'Input:\n"""Explain how to bake a chocolate cake step by step."""'
FEWSHOT_ASST = ('```json\n{"nla_relevant": true, "domain": "other", "has_factual_claim": false, '
                '"role_playing": false, "notes": "instructional recipe; model must predict next baking step", '
                '"prompt_excerpt": "Explain how to bake a chocolate cake step by step."}\n```')


def chat(system, user, profile, timeout=900, retries=3):
    opts = {
        "num_predict": profile.get("num_predict", 1400),
        "temperature": profile.get("temperature", 0.0),
        "tensor_split": profile.get("tensor_split", [24, 24]),
        "keep_alive": profile.get("keep_alive", "90m"),
    }
    messages = [{"role": "system", "content": system}]
    if profile.get("few_shot"):
        messages.append({"role": "user", "content": FEWSHOT_USER})
        messages.append({"role": "assistant", "content": FEWSHOT_ASST})
    messages.append({"role": "user", "content": user})
    payload = {"model": profile["model"], "stream": False, "options": opts, "messages": messages}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(profile["ollama_url"], data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            content = body.get("message", {}).get("content", "") or ""
            thinking = body.get("message", {}).get("thinking", "") or ""
            dt = time.time() - t0
            raw = content if content else thinking
            parsed = extract_label(raw, profile.get("extract_method", "fenced_json_or_last_balanced"))
            if parsed is None and attempt < retries - 1:
                # transient: model didn't emit parseable JSON this draw; retry
                last = "parse_none_retry"
                time.sleep(2)
                continue
            return parsed, raw, thinking, dt, body.get("done_reason"), None
        except Exception as e:
            last = str(e)
            time.sleep(3)
    return None, "", "", 0.0, "ERR", last


def build_user(text, prior_json=None):
    if prior_json is None:
        return 'Input:\n"""' + text + '"""'
    return ('Input:\n"""' + text + '"""\n\nFirst-pass label (JSON):\n```json\n'
            + json.dumps(prior_json) + '\n```\n\nValidate and emit the FINAL JSON object.')


def agree(a, b):
    if not a or not b:
        return None
    keys = ["nla_relevant", "domain", "has_factual_claim", "role_playing"]
    return sum(1 for k in keys if a.get(k) == b.get(k)) / len(keys)


# Reject degenerate / schema-echoed parses that are technically valid JSON but not real labels.
# e.g. {"key":"value"} (stray object), or a label whose values echo the field names verbatim.
DEGEN_VALUES = {"", "value", "key", "<=25 words", "<=40 words", "string", "boolean", "number",
                "verbatim input", "none", "null"}
def valid_label(j):
    if not isinstance(j, dict):
        return False
    req = ["nla_relevant", "domain", "has_factual_claim", "role_playing", "notes", "prompt_excerpt"]
    if not all(k in j for k in req):
        return False
    # every value must be a concrete (non-degenerate) instance, not the placeholder/schema text
    for k, v in j.items():
        if isinstance(v, str) and v.strip().lower() in DEGEN_VALUES:
            return False
    # prompt_excerpt must actually contain the input (not be a placeholder)
    if not isinstance(j.get("prompt_excerpt"), str) or len(j["prompt_excerpt"].strip()) < 3:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "labeler_config.yaml"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    prof = dict(cfg["defaults"])
    prof.update(cfg["model_profiles"][cfg["run"]["model"]])
    prof["model"] = cfg["run"]["model"]
    prof["ollama_url"] = cfg["defaults"]["ollama_url"]

    ps = cfg["run"]["prompt_set"]
    pset = cfg["prompt_sets"][ps]
    if pset.get("schema") == "json":
        LABELER_A_SYS = load_prompt(pset["labeler_a"]) + "\n\nSchema:\n" + load_prompt(pset["instruction"])
        LABELER_B_SYS = load_prompt(pset["labeler_b"]) + "\n\nSchema:\n" + load_prompt(pset["instruction"])
        AUDITOR_SYS = load_prompt(pset["auditor"]) + "\n\nSchema:\n" + load_prompt(pset["instruction"])
    else:
        raise SystemExit("v1 free-text schema not wired in v4 (use v2_alternative); extend if needed")

    commit = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(HERE, "runs", run_id)
    os.makedirs(outdir, exist_ok=True)
    out_jsonl = os.path.join(outdir, "labeled.jsonl")

    # load all configured datasets (concat)
    rows = []
    for ds in cfg["run"]["datasets"]:
        p = os.path.join(REPO_ROOT, "experiments/v8_nla_local/labeled_outputs", ds["path"]) \
            if not os.path.isabs(ds["path"]) else ds["path"]
        n = 0
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["_dataset"] = ds["name"]
                r["_text_field"] = ds.get("text_field", "prompt")
                rows.append(r)
                n += 1
                if ds.get("limit") and n >= ds["limit"]:
                    break
    print(f"[svc] {len(rows)} rows loaded; run_id={run_id}", flush=True)

    done = []
    if args.resume and os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.append(rec.get("row_idx"))
                except Exception:
                    pass
        done_set = set(done)
        rows = [r for r in rows if r.get("row_idx") not in done_set]
        print(f"[svc] resume: skipping {len(done)} rows by row_idx ({(sorted(done_set))[:10]}{'...' if len(done_set)>10 else ''})", flush=True)

    metrics = {"run_id": run_id, "model": prof["model"], "commit_sha": commit,
               "prompt_set": ps, "prompt_shas": {k: sha256_file(v) for k, v in pset.items()
                                                 if isinstance(v, str) and v.endswith(".md")},
               "n_total": len(rows) + len(done), "parse_ok": 0, "parse_fail": 0,
               "agreement_sum": 0.0, "agreement_n": 0, "think_leak": 0}
    t_start = time.time()
    with open(out_jsonl, "a") as out:
        for i, row in enumerate(rows):
            text = row.get(row["_text_field"], "")
            if not text:
                continue
            ja, raw_a, think_a, dt_a, dr_a, err_a = chat(LABELER_A_SYS, build_user(text), prof)
            jb, raw_b, think_b, dt_b, dr_b, err_b = chat(LABELER_B_SYS, build_user(text), prof)
            jaud, raw_aud, think_aud, dt_aud, dr_aud, err_aud = (None, "", "", 0.0, "", None)
            if ja:
                jaud, raw_aud, think_aud, dt_aud, dr_aud, err_aud = chat(AUDITOR_SYS, build_user(text, ja), prof)

            ok = bool(ja and jb and valid_label(ja) and valid_label(jb))
            metrics["parse_ok" if ok else "parse_fail"] += 1
            ag = agree(ja, jb)
            if ag is not None:
                metrics["agreement_sum"] += ag
                metrics["agreement_n"] += 1
            if think_a and "<think>" in think_a.lower():
                metrics["think_leak"] += 1

            rec = {"row_idx": done + i, "dataset": row.get("_dataset"),
                   "input_text": text[:2000], "category": row.get("category"),
                   "labeler_a": ja, "labeler_b": jb, "auditor_a": jaud,
                   "agreement_a_b": ag,
                   "meta": {"dt_a": round(dt_a, 1), "dt_b": round(dt_b, 1), "dt_aud": round(dt_aud, 1),
                            "done_reason_a": dr_a, "err_a": err_a, "parse_ok": ok}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            el = time.time() - t_start
            print(f"[svc] row {done+i} ok={ok} agree={ag} t={el:.0f}s", flush=True)

    if metrics["agreement_n"]:
        metrics["agreement_mean"] = round(metrics["agreement_sum"] / metrics["agreement_n"], 3)
    metrics["wall_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(outdir, "audit.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # copy config + prompt shas into run dir for provenance
    with open(os.path.join(outdir, "labeler_config.used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    print("[svc] DONE", json.dumps({k: metrics[k] for k in
          ["parse_ok", "parse_fail", "agreement_mean", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
