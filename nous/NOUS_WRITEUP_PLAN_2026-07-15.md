# Plan: Nous Research writeup — "Live SAE readout (and soon steering) of local models under Hermes Agent"

Context: Tristan (Nous community engagement) asked for a WRITEUP he can pass internally,
after Caleb's pitch: SAE probes hooked into local models while they run Hermes Agent =
live internal readout of agentic work; next step is live feature steering (Goodfire
Silico-adjacent, but local + open). PRs to the Hermes Agent GitHub welcomed.
This file stays OUT of the public research repo (originates from a DM).

## Deliverable stack (in order)

### D1 — The writeup (primary ask; target: 1,200–1,800 words + 3 figures)
A single markdown doc, tone matching Caleb's blog voice (first-person, homelab-real,
honest caveats), structured for a technical-but-mixed internal audience:

1. **What this is in one paragraph.** SAE probes attached to the local model's residual
   stream *while it serves Hermes Agent* — a live log of which learned features fire per
   token as the agent plans, uses tools, drifts, recovers. Output ≠ understanding; this
   is the difference.
2. **What already runs today (receipts, not vision).**
   - The same-inference SAE labeling course: Qwen3.5-27B doing real agentic labeling
     under Hermes with 5 SAE layers read per token, on 2× Tesla M40s
     (`sae_labeled_course.py`, merged PR #231 in the research repo).
   - Trained open SAE suites this plugs into: Gemma-4-E2B suite, SmolLM2-135M BatchTopK
     sweep (6 layers, §F163/164), Qwen3.5-27B residual SAEs; GLM-5.2 dictionary
     extraction in progress on CPU (440GB RAM run).
   - One concrete debugging win to narrate: catching drift/failure modes in agentic runs
     from feature traces rather than transcripts (pick the crispest real example from the
     course pilot logs; verify the example against the actual JSONL before writing).
3. **Why Hermes specifically.** Hermes Agent's plugin system (yaml + python module, cf.
   claude_delegate) makes the probe a *plugin*, not a fork: same interface as any tool.
   Local-first means no API-side interpretability blackout — the weights are right there.
4. **The steering roadmap (labeled clearly as next, not done).** Read → flag → nudge:
   per-feature bias injection at serve time; guardrail features (off-task drift,
   sycophancy-adjacent, tool-loop detection) as the first candidates. Silico comparison:
   same read+steer space, but open models, local hardware, agent-in-the-loop.
5. **What we're offering Nous.** (a) the writeup itself, (b) a runnable demo, (c) PRs
   (below), (d) collaboration on which features/behaviors matter most for Hermes UX.
6. **Honest limits.** Overhead numbers (tok/s with/without hooks — measure, don't guess);
   fp16 + device_map constraints; SAE quality varies by layer; steering is unvalidated
   as of writing.

Figures (reuse/adapt existing where possible, all real data):
- F1: feature-firing timeline strip for one agentic episode (from course JSONL).
- F2: architecture sketch — Hermes Agent → local model → SAE hook layers → live trace
  (simple diagram; draw with matplotlib/graphviz, keep the validated palette).
- F3: overhead bar — tok/s with vs without probes (needs one short GPU measurement).

### D2 — Minimal public demo repo (supports D1, ~1 day)
`hermes-sae-probe` (new small repo or dir in an existing public one): the course's hook
core extracted to <300 lines — load any HF model + matching SAE, wrap generation, stream
top-k firing features per token to console/JSONL. One command, one README GIF. No
research-repo internals, no §F references (public-artifact language rule).

### D3 — PR candidates for NousResearch Hermes Agent repo (staged, smallest first)
1. **Docs PR**: "Observability plugins" section + link to the demo (low-risk opener).
2. **Plugin PR**: `sae_probe` plugin following the claude_delegate structure — config
   for model/SAE/layers/top-k, emits a feature-trace sidecar file per agent session.
3. (Later, after validation) **Steering hook PR**: per-feature bias injection behind an
   explicit opt-in flag. Do NOT ship in round 1.

## Sequence + effort
1. Tonight (no GPU needed): draft D1 sections 1, 3, 4, 5, 6 + F2 diagram. [~2h]
2. Next GPU window (piggyback 15 min on tonight's 21:45 reservation tail, or the one
   after): capture F1 trace data from a short course episode + F3 overhead numbers.
3. Draft complete → `/human-writing-check` pass → Caleb reviews → Caleb sends to Tristan
   (Claude never contacts Nous directly).
4. D2 demo extraction after Caleb approves D1's framing. D3.1 docs PR alongside D2.
5. Steering experiments get their own pre-registered plan + gpusched windows; nothing
   steering-related is claimed in D1 beyond "roadmap."

## Decisions Caleb owns
- Where D2 lives (new GitHub repo under SolshineCode vs subdir of an existing one).
- Whether the writeup names the deception-research context or stays generic
  ("agentic labeling tasks") — recommend generic for round 1.
- Timing of send (recommend: after D1 + F1/F3 land, don't wait for D2).
