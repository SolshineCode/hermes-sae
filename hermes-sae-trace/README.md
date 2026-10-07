# hermes-sae-trace

**Watch a local model think while it runs an agent.** A
[Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that
correlates every agent turn with the **same-inference** SAE (sparse
autoencoder) feature activations your local model produced for it.

The transcript tells you what the agent said; the feature trace tells you
what the model was doing internally while it said it.

- **Read-only observer** on Hermes' `pre_api_request` / `post_api_request`
  hooks: never modifies requests, responses, or the sidecar.
- **Stdlib-only, no network, no GPU.** Pure file correlation. Your model
  server does the SAE work; this plugin joins it to the agent's turns.
- **`/sae` slash command** (`status`, `last`, `dashboard`) and a
  **zero-install session dashboard** (`dashboard.html`, one self-contained
  page, no server).
- **Out-of-tree by design:** public plugin surfaces only (no core patches,
  no internal imports), passes `hermes plugins validate` including the
  security scan and `no core override` checks.

## Quickstart

```bash
pip install "git+https://github.com/SolshineCode/hermes-sae-trace"
hermes plugins enable sae_trace
echo 'HERMES_SAE_TRACE_FILE=/path/to/sae_history.jsonl' >> ~/.hermes/.env
```

Then point Hermes at an SAE-hooked OpenAI-compatible server (reference
server, CPU-only replication, and 27B results:
[SolshineCode/hermes-sae](https://github.com/SolshineCode/hermes-sae)) and
run `/sae status` in a session.

Full documentation (install routes, configuration, the sidecar contract,
output format, correlation tiers and limitations, security disclosures):
**[`sae_trace/README.md`](sae_trace/README.md)**.

## Sidecar contract

Any server works if it appends one JSON object per request to a JSONL file,
in the same inference pass that produced the completion. See the
[schema table](sae_trace/README.md#sidecar-jsonl-schema-the-interface-contract).

## Repo layout

| Path | What it is |
| --- | --- |
| `sae_trace/` | The plugin: `plugin.yaml` + `__init__.py` (directory / catalog install), also the pip package |
| `sae_trace/dashboard.html` | Zero-install session dashboard |
| `tests/` | 43 tests; run without a Hermes checkout, plus a live discovery test when Hermes is installed |
| `catalog/sae_trace.yaml` | Draft entry for the Hermes plugin catalog (SHA filled in at release) |
| `docs/LAUNCH.md` | Release checklist, catalog PR text, and community announcement drafts |

## Development

```bash
pip install -e ".[test]"
python -m pytest -q
# With Hermes Agent installed, run the catalog admission gate:
hermes plugins validate ./sae_trace --install-deps
hermes plugins doctor ./sae_trace --ci
```

## History

The plugin was first proposed in-tree to `NousResearch/hermes-agent` as
`plugins/observability/sae_trace`. Hermes' policy (June 2026) is that
observability integrations ship as standalone plugin repos rather than in
core, a coupling decision rather than a quality judgment, so it lives here
now. See [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT, see [`LICENSE`](LICENSE).
