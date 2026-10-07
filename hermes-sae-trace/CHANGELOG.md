# Changelog

## 0.3.0 (2026-10-02): standalone out-of-tree plugin

Re-packaged as its own repository after the in-tree PR to
`NousResearch/hermes-agent` was redirected to the standalone-plugin route
(Hermes policy: observability integrations ship as separate plugin repos).

Changes required to meet the plugin catalog admission rules:

- **Manifest:** `hooks:` renamed to `provides_hooks:` and `provides_tools: []`
  added. With the old key, `hermes plugins validate` failed the
  "declared hooks" check (rule 6: declared capabilities must match what the
  plugin registers).
- **Manifest:** added `requires_hermes: ">=0.21"` (SemVer floor, rule 14) and
  the rich `requires_env` form, so `hermes plugins install` prompts for
  `HERMES_SAE_TRACE_FILE` with a description.
- **No Hermes internal imports.** Removed the `plugins.plugin_utils` import
  (the local `SingletonSlot` is now the only implementation) and the
  `hermes_constants` import. Internal import paths are not plugin API, and
  since the Sep 2026 module split an import of an old path fails at load time.
- **Default output directory** is now `traces/` inside the plugin's
  profile-scoped data directory (`ctx.state.data_dir`, under
  `$HERMES_HOME/plugin-data/`), the location Hermes reserves for
  plugin-written data. It was `$HERMES_HOME/sae_trace/` before. Set
  `HERMES_SAE_TRACE_OUT_DIR` to keep the old location. Without `ctx.state`
  (older Hermes) it falls back to `$HERMES_HOME/plugin-data/sae_trace/traces`.
- **pip distribution:** `pyproject.toml` with a `hermes_agent.plugins` entry
  point (`sae_trace = "sae_trace:register"`); `plugin.yaml` and
  `dashboard.html` ship as package data.
- `/sae` args hint now lists `dashboard`.

Tests: the 35-case suite from the in-tree PR is ported to run without a
Hermes checkout, plus 8 new cases (pip entry point, manifest/pyproject
version sync, stdlib-only dependencies, no internal imports at the source
level, data-directory resolution and its fallbacks).

Verified against Hermes Agent `main` @ `0a374d1` (2026-10-02):
`hermes plugins validate --install-deps` passes all checks (security scan
`safe`, `no core override`), `hermes plugins doctor --ci` passes, and a pip
install loads through the entry point with both hooks and `/sae` registered.

## 0.2.0

Zero-install session dashboard (`dashboard.html`) and `/sae dashboard`.

## 0.1.0

Initial plugin: sidecar tailer, three correlation tiers
(`request_id` > `session_id` > `time_window`), per-session JSONL output,
`/sae status` and `/sae last`.
