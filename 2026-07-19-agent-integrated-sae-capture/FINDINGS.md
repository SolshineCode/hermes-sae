# Capture Reliability: Hermes CLI cold-start latency (2026-07-20, ~19:xx)

## Observed failure mode

After the 15th capture (02:01:35), the capture loop (`run_capture_loop.sh`)
started timing out almost every turn. On re-test, a **single manual Hermes
Agent turn** (`hermes -p nla-local chat -q 'What is 2+2? One sentence.'`)
also failed to complete within 600s (10 min), even though the server
received the request.

This is NOT a server bug. Analysis of the failure surface:

| Step | Where time goes | Typical time |
|------|-----------------|--------------|
| (a) `nla_server.py` loads model on boot | one-time, ~13–20s | 13–20s |
| (b) Hermes CLI agent init (imports, module init, TUI framing, env scan) | CLI side | 30–120s+ |
| (c) Server request latency (16 tok @ ~0.3–1 tok/s on GTX 1650 Ti) | server side | ~30-50s generation |
| (d) Hermes response handling + TUI teardown | CLI side | seconds |
| (e) `timeout 600` wrapper fires | driver side | **kills turn ~10 min after launch** |

The original `run_capture_loop.sh` appeared successful on turns 0–5 only
because Hermes had **pre-warmed state** from earlier manual test invocations.
Cold Hermes CLI init >> 10 min; warm Hermes CLI init is 5–8 minutes. The
600s collar was tight enough that a cold start always lost.

## Fixes applied

1. **Resilient loop** (`run_capture_loop_resilient.sh`): checks /healthz before
   each turn, kills+restarts the server if unhealthy.
2. **Timeout bump**: `timeout 600` → `timeout 1200` (20 min) in both scripts,
   so tight collar no longer kills a cold turn.
3. **Recommended (better) fix**: pre-warm Hermes before the loop: run one
   short Hermes turn to completion (or even let it timeout once) so the
   framework's heavy init happens before the main loop starts. Then the
   actual productive 5–8 min turns start immediately.

## Confirmed: the capture path itself works end-to-end

When Hermes does reach the server and the generation completes, the
capture **always** lands: 15/15 turns that reached `write_records` produced
a valid `.npz` with `acts [N, 1536]` float32. The hook, the server, and the
file format are all solid. This is a Hermes-CLI-side inefficiency, not a
capture-bug.
