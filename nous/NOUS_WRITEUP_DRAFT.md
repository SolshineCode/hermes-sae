# Watching a local model think while it runs Hermes Agent

*Draft v1 for Tristan / Nous Research, 2026-07-15. Figures: f1_feature_timeline.png,
f2_architecture.png (attached). One overhead figure pending a scheduled measurement
run; numbers below marked TBD will be filled before send. Caleb DeLeeuw.*

## The one-paragraph version

While Hermes Agent runs on a local model, I attach sparse autoencoder (SAE) probes
to the model's residual stream and log which learned features fire, per token, as
the agent works. The transcript tells you what the agent said. The feature trace
tells you what the model was doing internally while it said it: which concepts
lit up during planning, where attention to the actual task dropped off, what was
active in the forward pass when a tool call went sideways. It's a live internal
readout of agentic work, on consumer hardware, with open models. Everything below
runs today except the last section, which is the roadmap.

## What runs today

This isn't a proposal. The current setup:

- **Model:** Qwen3.5-27B, fp16, split across two Tesla M40s (a 2013 Dell
  workstation; total GPU cost under $200 used). Hermes Agent drives it through
  real multi-step labeling tasks.
- **Probes:** SAE forward hooks at 5 layers (0, 16, 32, 48, 63 of 64), reading
  the residual stream in the *same inference pass* that produces the agent's
  output. No second forward, no replay. Per token I keep the top-k features and
  their activations, streamed to a JSONL sidecar alongside the agent transcript.
- **SAEs:** open checkpoints where they exist, plus suites I've trained myself
  (Gemma-4-E2B across layers, a 6-layer SmolLM2-135M sweep, and a dictionary
  extraction for a much larger model currently grinding away on CPU). The probe
  code doesn't care whose SAE it is, it just needs the layer and the weights.

![One forward pass, two outputs](nous_figs/f2_architecture.png)

Here's what a real episode looks like. This is layer 32 during one labeling task,
top features only:

![Live feature firing during one Hermes labeling episode](nous_figs/f1_feature_timeline.png)

Some features fire in a burst early (task comprehension), some persist across the
whole response (topic and format), some spike at specific token positions. When
a run goes wrong, this is where it shows first.

A concrete example of the kind of signal this gives you, from a real failed
run last week: a labeling course produced empty output text for every row. In
the trace sidecars from those same runs, the feature activity looks normal,
features firing across all five hooked layers while the visible output was
blank. That pattern says the model is fine and the bug is downstream of it, and
that's where the bug was: an input tensor forced to the wrong device under
device_map, mangling the decode. (Credit where due: the agent running that
course found the bug through ordinary debugging. The point is the trace had
been saying "not the model" the whole time, and I want that signal on screen
during runs, not discovered in the sidecar afterward.)

## Why Hermes specifically

Two reasons, one technical and one philosophical.

Technical: Hermes Agent has a plugin system. My existing plugins (a delegation
tool that hands hard subtasks to another agent) are a yaml manifest plus a small
Python module. The SAE probe fits the same shape: a plugin that wraps the local
model's generation and emits a feature-trace sidecar per session. No fork of
Hermes needed. Config is just model, SAE repo, layers, top-k.

Philosophical: this only works because the weights are local. You cannot hook
the residual stream of an API. The interpretability story and the local-model
story are the same story, and Nous sits at the center of the local-model story.
An agent stack where "what is my model actually doing" is a first-class,
inspectable question feels like a natural extension of what Hermes already is.

## The roadmap: from reading to steering

Everything above is read-only. The next step is nudging: adding a small bias
along chosen SAE feature directions at serve time, while the agent runs.

The use case I care about most is drift correction. Long autonomous runs drift.
Anyone who has run a 10-hour agent session has watched it happen: the model
slides from the task into a groove that's adjacent to the task. Today you catch
that by reading transcripts after the fact. With feature-level steering you
could catch the drift signature live (task-relevant features decaying, some
attractor feature ramping) and apply a corrective nudge without restarting the
run or rewriting the prompt.

This is the same read-and-steer space Goodfire's Silico works in, with the
obvious differences: open models, local hardware, and the agent loop in the
picture from the start. I want to be clear about status: steering is designed,
not validated. I don't yet know how much nudge a 27B model tolerates before
output quality degrades, or whether drift signatures are consistent enough to
trigger on. Those are the first two experiments, and they're pre-registered
before they run (lab policy, learned the hard way).

## What I'm offering

1. **This writeup**, to circulate however is useful.
2. **A minimal public demo repo** (in progress): the probe core extracted to a
   few hundred lines. Point it at any HF model plus a matching SAE, get a live
   feature stream. One command.
3. **PRs to the Hermes Agent repo** for anything the team wants upstream. The
   natural first one is the probe as a proper plugin; I'll hold off until
   someone on the team confirms that's welcome and points me at conventions.
4. **Collaboration on what to watch for.** I can trace features all day, but
   the Nous team knows Hermes's real failure modes better than I do. A shortlist
   of "behaviors you wish you could see coming" would directly shape which
   features I try to identify and steer first.

## Honest numbers and limits

- Probe overhead: TBD, measurement scheduled this week (top-k readout at 5
  layers; expectation is small relative to generation cost on these GPUs, but
  I'll publish the actual tok/s with and without hooks rather than guess).
- fp16 with device_map across two cards is the tested configuration. Ollama /
  llama.cpp serving paths don't expose the residual stream, so probing currently
  means the transformers serving path, which is slower. That's a real tradeoff
  today, not a footnote.
- SAE quality varies a lot by layer and by training budget. A probe is only as
  interpretable as its dictionary. Feature labeling (which feature means what)
  is its own ongoing work.
- Nothing here is a safety guarantee. A feature trace is evidence, not proof,
  of what a model is doing.

If any of this is useful to the team, I'm easy to reach, and the demo repo will
be public shortly either way.
