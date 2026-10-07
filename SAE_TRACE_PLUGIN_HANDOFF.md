# Handoff: publish `sae_trace` as a standalone Hermes plugin

Status as of 2026-10-07. Written for the next session, which can create
GitHub repositories. This one could only push to existing branches.

## Background

- The in-tree PR (NousResearch/hermes-agent #68543,
  `plugins/observability/sae_trace`) was declined. Nous asked for a
  standalone plugin repo, promoted in the Nous Discord
  `#plugins-skills-and-skins`.
- Upstream policy (`plugins/AGENTS.md`, June 2026): observability and
  third-party integrations ship out-of-tree. They call it "a coupling decision,
  not a quality judgment".
- The plugin catalog (`plugin-catalog/README.md` upstream) is the only
  discovery system for out-of-tree plugins. Catalog entries are merged by a
  human reviewer, pinned to an exact 40-character SHA, must not self-update,
  must declare capabilities that match what the plugin registers, and must
  not override core. Catalog CI runs `hermes plugins validate` on the pinned
  commit.
- Being listed is how users find it: `hermes plugins search`, the Desktop
  catalog, a docs page at
  `https://hermes-agent.nousresearch.com/docs/plugins/sae_trace`, and
  `hermes plugins install sae_trace`.

## What is already done (in this repo, merged to `main`)

Everything lives in **`hermes-sae-trace/`**, laid out as a repo root:

| Path | Purpose |
| --- | --- |
| `sae_trace/` | The plugin (`plugin.yaml`, `__init__.py`, `dashboard.html`, `README.md`). Works as a directory/catalog install and as the pip package |
| `pyproject.toml` | pip packaging; entry point `hermes_agent.plugins: sae_trace = "sae_trace:register"`; runtime has no dependencies (stdlib only) |
| `tests/test_sae_trace_plugin.py` | 43 tests; run without Hermes (1 live discovery test skips unless Hermes is installed) |
| `.github/workflows/ci.yml` | pytest 3.11-3.13, plus `hermes plugins validate` / `doctor` against Hermes `main` |
| `catalog/sae_trace.yaml` | Draft catalog entry; `sha:` is a placeholder of 40 zeros |
| `CHANGELOG.md` | v0.3.0 changes and verification record |
| `docs/LAUNCH.md` | Detailed release checklist, catalog PR text, announcement drafts |

v0.3.0 fixes made for catalog admission:
`hooks:` -> `provides_hooks:` (the old manifest failed validate),
`requires_hermes: ">=0.21"`, rich `requires_env`, removed Hermes internal
imports, default output dir moved to `ctx.state.data_dir/traces`.

Verified 2026-10-02 against Hermes `main` @ `0a374d1`:
`hermes plugins validate --install-deps` passed every check (security scan
safe, no core override); `hermes plugins doctor --ci` passed; a pip install
loaded via the entry point with both hooks and `/sae` registered; 43/43 tests
passed.

**Not yet verified:** the GitHub Actions workflow has never run (GitHub
only runs workflows from `.github/` at the repo root, so it will first
run in the new repo).

## Follow-up tasks (in order)

### 1. Create the repo and push the subtree

```bash
# Create an EMPTY public repo SolshineCode/hermes-sae-trace
# (no README/license/.gitignore). Description:
#   "Hermes Agent plugin: same-inference SAE feature traces for local models"
# Topics: hermes-agent, sparse-autoencoder, mechanistic-interpretability, interpretability

cd hermes-sae
git checkout main && git pull
git subtree split --prefix=hermes-sae-trace -b sae-trace-standalone
git push https://github.com/SolshineCode/hermes-sae-trace.git sae-trace-standalone:main
git branch -D sae-trace-standalone
```

Expected: the new repo root contains `README.md`, `sae_trace/`, `tests/`,
`.github/`, and similar files. The history starts at the v0.3.0 commit. The
older plugin history stays in hermes-sae under `hermes-plugin/`.

### 2. Get CI green in the new repo

- Watch the `ci` workflow. If `hermes-validate` fails, the likeliest cause is
  installing Hermes in CI (`uv sync --frozen` on a fresh clone of Hermes
  `main`), not the plugin. Fix the workflow. Do not weaken the
  validate/doctor steps.
- Locally the equivalent was: install Hermes `main` in a venv, then
  `HERMES_HOME=$(mktemp -d) hermes plugins validate ./sae_trace --install-deps`
  and `hermes plugins doctor ./sae_trace --ci`.
- Re-run validate against current Hermes `main`. The plugin contract may
  have changed since 2026-10-02.

### 3. Tag the release

```bash
git tag v0.3.0 && git push origin v0.3.0
git rev-parse v0.3.0   # SHA for the catalog entry
```

Optionally create a GitHub Release for v0.3.0 using the CHANGELOG text.

### 4. Catalog PR to NousResearch/hermes-agent

- Work from a fork synced to upstream `main`. The existing fork
  `SolshineCode/hermes-agent` is far behind and has no `plugin-catalog/`,
  so sync it first.
- Add exactly one file: `plugin-catalog/sae_trace.yaml`, copied from
  `catalog/sae_trace.yaml`. Set `sha:` to the v0.3.0 SHA and delete the
  DRAFT comment header.
- Check `requires_hermes: ">=0.21"` is still a valid SemVer floor no newer
  than the current Hermes release (admission rule 14).
- The PR title and description (with the rule-13 disclosures) are in
  `docs/LAUNCH.md` section 3.
- Only the repo owner (SolshineCode) may submit it (admission rule 5). Open
  it from their account.
- Ask the user before opening the PR. It is outward-facing.

### 5. Housekeeping in hermes-sae

- Replace `hermes-sae-trace/` in hermes-sae with a short pointer
  (README row + link) to the new repo. That avoids two copies drifting.
  Alternatively keep it and state that the new repo is the source of truth.
  Ask the user which.
- Update the links in `README.md` and `nous/WRITEUP.md` to
  `https://github.com/SolshineCode/hermes-sae-trace`.
- `nous/Hermes-SAE-Writeup.pdf` was not rebuilt after the WRITEUP note was
  added. Rebuild with `nous/make_pdf.py` if wanted.

### 6. Announcements (drafts only; the user posts)

Drafts are in `docs/LAUNCH.md` section 4: Nous Discord
`#plugins-skills-and-skins` post, blog outline, and outreach targets (Open
Source Mechanistic Interpretability Slack, EleutherAI Discord,
SAELens/Neuronpedia users, r/LocalLLaMA, LessWrong/Alignment Forum, HF
community blog). Post after the catalog PR merges, so the install line is
`hermes plugins install sae_trace`. Before then, use the pip line. A GIF of
`dashboard.html` following a live session would help every post.

## Possible later features (not started)

- Neuronpedia links for feature ids in the dashboard when the SAE is a
  released one (Gemma Scope, public Qwen SAEs).
- Per-profile state in multi-profile processes (currently one sidecar and
  shared counters per process; documented in `sae_trace/README.md`).

## Constraints to keep

- No Hermes internal imports; only `ctx` / documented APIs (a test enforces
  this at source level).
- No self-updating code, no network, no new runtime dependencies without
  updating the README disclosures and catalog description.
- Every catalog SHA bump is a new PR. Bump `version` in `plugin.yaml`,
  `pyproject.toml` (a test keeps them in sync) and the catalog entry together.
