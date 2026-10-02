# Launch plan: hermes-sae-trace

Release checklist, catalog submission text, and announcement drafts.
Everything here is a draft for the author to edit in their own voice.

## Why the catalog matters for visibility

The plugin catalog is in-product discovery, not only a Discord mention:

- `hermes plugins search sae` / `hermes plugins catalog` list it in the CLI.
- The Desktop app's catalog shows it, with an install button.
- Every entry gets a docs page at
  `https://hermes-agent.nousresearch.com/docs/plugins/sae_trace`, rendering
  `sae_trace/README.md` at the pinned SHA, plus a maintainer page at
  `/docs/plugins/by/SolshineCode`.
- `hermes plugins install sae_trace` then works by name.

So the order is: standalone repo, then catalog PR, then announcements that
can say "`hermes plugins install sae_trace`".

## 1. Create the standalone repo (PowerShell)

This directory was staged inside `SolshineCode/hermes-sae`.
`git subtree split` extracts it as a clean repo root. The split starts at
the v0.3.0 commit; earlier plugin history (when it lived under
`hermes-plugin/sae_trace/`) stays in `hermes-sae`, and `CHANGELOG.md`
summarizes it.

```powershell
# From your hermes-sae clone, on the branch that contains hermes-sae-trace/
cd hermes-sae
git fetch origin
git checkout ccr-1dee302b-jbcgwn

# Make a branch whose root is the hermes-sae-trace/ directory
git subtree split --prefix=hermes-sae-trace -b sae-trace-standalone
# Expected: prints a single 40-char commit SHA

# Create an EMPTY public repo on GitHub named hermes-sae-trace (no README,
# no license, no .gitignore), then push the split branch as its main:
git push https://github.com/SolshineCode/hermes-sae-trace.git sae-trace-standalone:main
```

Success: `https://github.com/SolshineCode/hermes-sae-trace` shows
`README.md`, `sae_trace/`, `tests/`, and the Actions tab starts the `ci`
workflow.

Troubleshooting:

- `git subtree` not found: Git for Windows includes it; update Git
  (`winget upgrade Git.Git`).
- Push rejected (`fetch first`): the GitHub repo was created with a README.
  Delete and recreate it empty, or push with `--force` once to the new repo.

## 2. Confirm CI is green, then tag

The `hermes-validate` job installs Hermes `main` and runs the same
`hermes plugins validate` gate the catalog CI uses. It was verified locally
(see `CHANGELOG.md`) but its first run on GitHub Actions has not happened
yet. Fix anything it reports before tagging.

```powershell
git clone https://github.com/SolshineCode/hermes-sae-trace
cd hermes-sae-trace
git tag v0.3.0
git push origin v0.3.0
git rev-parse v0.3.0   # the 40-char SHA for the catalog entry
```

## 3. Catalog PR to NousResearch/hermes-agent

1. Fork / update your fork of `NousResearch/hermes-agent` from `main`.
2. Copy `catalog/sae_trace.yaml` to `plugin-catalog/sae_trace.yaml`.
   Replace `sha:` with the `git rev-parse v0.3.0` output. Delete the
   comment header.
3. Validate at that commit:

   ```powershell
   hermes plugins validate C:\path\to\hermes-sae-trace\sae_trace --install-deps
   # Expected last line: Validation passed.
   ```

4. Open the PR with **only** that one file changed.

### PR title

```
plugin-catalog: add sae_trace (SAE feature traces for local models)
```

### PR description draft

```markdown
Adds `plugin-catalog/sae_trace.yaml` for
https://github.com/SolshineCode/hermes-sae-trace (I am the repo owner).

**What it does:** correlates each agent turn with the same-inference SAE
(sparse autoencoder) feature activations that an SAE-hooked local
OpenAI-compatible server logs to a sidecar JSONL, and writes one
correlated line per turn per session. `/sae status|last|dashboard`, plus a
zero-install HTML dashboard. This is the standalone version of the
in-tree observability PR, moved out per the plugins policy.

**Hermes surfaces used:** `register_hook` on `pre_api_request` and
`post_api_request` (read-only, fail-open), `register_command("sae")`,
`ctx.state.data_dir` for the default output directory. No tools, no
middleware, no capabilities, no internal imports, no core overrides.

**Disclosures (rule 13):**
- Reads one user-configured file (`HERMES_SAE_TRACE_FILE`) read-only,
  outside the plugin's own data.
- Appends JSONL to its plugin data directory (or `HERMES_SAE_TRACE_OUT_DIR`).
  Records include a ≤200-char preview of generated text per record.
- No network calls, no telemetry, no shell commands, no background
  processes, no stored credentials. Python stdlib only; no dependencies.

**Validation:** `hermes plugins validate --install-deps` passes at the pinned
SHA (security scan: safe; no core override: pass). `hermes plugins doctor
--ci` passes. 43 tests in the plugin repo.
```

## 4. Announcements

Post after the catalog PR merges if possible, so the install line is one
command. Before then, use the pip line.

### Nous Research Discord, `#plugins-skills-and-skins`

```
sae_trace: watch your local model think while it runs Hermes

If you run Hermes on a local model, this plugin lines up every agent turn
with the SAE (sparse autoencoder) features your model activated *in the
same forward pass* that produced the reply. The transcript shows what the
agent said; the trace shows which internal features drove it.

- read-only observer hooks, stdlib-only, no network
- /sae status | last | dashboard
- zero-install HTML dashboard, live-follows a session
- CPU-only replication of the server side in one command (gpt2-small /
  Qwen2.5-0.5B with public SAEs)

Install: hermes plugins install sae_trace
Repo: https://github.com/SolshineCode/hermes-sae-trace
Research + 27B results: https://github.com/SolshineCode/hermes-sae

Feedback very welcome, especially from anyone serving Gemma or Qwen
locally with released SAEs.
```

### Blog post outline

Working title: *"One forward pass, two outputs: SAE traces for a real agent"*

1. **Hook:** agent transcripts tell you what happened, not why. Most
   interpretability work runs on prompts, not agents doing multi-step work.
2. **The same-inference invariant:** why re-running text through a hooked
   model is a second inference that can diverge, and how hooking the
   serving `generate()` call avoids it (use `nous/figures/f2_architecture`).
3. **What it showed:** the 27B separability result with its caveats
   (`B1_REAL_27B_REPORT_2026_07_20.md`), and the degenerate-loop attractor
   visible in activation geometry before it is obvious in the transcript.
4. **Try it in 5 minutes:** CPU-only replication, then the plugin and
   dashboard. Screenshot or GIF of `dashboard.html` following a live session.
5. **Limitations, stated plainly:** time-window correlation tier, single
   sidecar per process, SAE quality caveats.
6. **Asks:** SAEs for more local models, feature labels, people to try to
   falsify the separability claim (link `FALSIFIABLE_CLAIM_DESIGN.md`).

Where to publish: your own blog or Substack as the canonical post, then
cross-post to LessWrong / the Alignment Forum (interpretability audience)
and a Hugging Face community blog post (local-model audience).

### Mech interp and local-model communities

- **Open Source Mechanistic Interpretability Slack** and the
  **EleutherAI Discord** interpretability channels: short post that leads
  with the same-inference invariant and the CPU replication command.
- **SAELens / Neuronpedia users:** if the served SAE is a released one
  (Gemma Scope, public Qwen SAEs), feature ids in the trace can be looked up
  on Neuronpedia. Worth a sentence in every post, and a possible future
  feature (feature-id links in the dashboard).
- **r/LocalLLaMA:** the "watch your local model think" angle with a
  dashboard GIF. This is the largest audience that already runs local models.
- **X / Bluesky thread:** 4-5 posts: problem, invariant, figure, GIF, links.
  Tag @NousResearch.
