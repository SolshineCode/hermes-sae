# Capture Analysis -- 2026-07-20T07:42:36

Total captures analyzed: **15** (of 15 records)

| # | req | source | n | gen | prefill|| | gen|| mean/min/max | consec-cos mean/min/max | loop-frac>0.9 | ts |
|---|-----|--------|---|-----|-----------|----------------------|--------------------------|------------------|----|
| 1 | f7005221 | curl | 20 | 20 | 56.71 | 57.77/47.12/68.0 | 0.513/0.335/0.701 | 0.0 | 2026-07-19T21:52:36.597030 |
| 2 | 0bed84ea | Hermes | 96 | 96 | 56.98 | 55.53/48.1/69.07 | 0.569/0.257/0.796 | 0.0 | 2026-07-19T22:13:50.461547 |
| 3 | 9f4da58f | curl | 8 | 8 | 58.16 | 58.97/47.18/68.2 | 0.455/0.337/0.651 | 0.0 | 2026-07-19T22:42:52.149801 |
| 4 | c0664d17 | Hermes | 16 | 16 | 57.08 | 57.2/49.02/67.43 | 0.537/0.294/0.73 | 0.0 | 2026-07-20T01:27:29.757362 |
| 5 | e0327338 | curl | 16 | 16 | 53.21 | 53.73/48.68/59.48 | 0.577/0.264/0.791 | 0.0 | 2026-07-20T01:27:49.840525 |
| 6 | d616fb4a | Hermes | 16 | 16 | 54.98 | 58.31/48.92/67.8 | 0.475/0.163/0.748 | 0.0 | 2026-07-20T01:34:20.399356 |
| 7 | 871fdb59 | curl | 16 | 16 | 52.74 | 55.56/46.79/65.45 | 0.494/0.289/0.868 | 0.0 | 2026-07-20T01:34:40.224955 |
| 8 | 7491ea7c | Hermes | 16 | 16 | 59.46 | 58.3/53.56/63.7 | 0.624/0.367/0.876 | 0.0 | 2026-07-20T01:41:10.311134 |
| 9 | 9075e111 | curl | 16 | 16 | 53.95 | 52.48/45.82/57.41 | 0.642/0.387/0.802 | 0.0 | 2026-07-20T01:41:30.369912 |
| 10 | c9a74bdf | Hermes | 16 | 16 | 54.31 | 61.25/48.3/66.4 | 0.557/0.334/0.836 | 0.0 | 2026-07-20T01:48:13.576280 |
| 11 | 126cbdb3 | curl | 16 | 16 | 52.55 | 56.62/49.92/60.5 | 0.64/0.434/0.785 | 0.0 | 2026-07-20T01:48:33.606974 |
| 12 | a703ffcb | Hermes | 16 | 16 | 52.64 | 59.55/51.47/65.47 | 0.595/0.35/0.833 | 0.0 | 2026-07-20T01:55:17.417655 |
| 13 | 99ae5488 | curl | 16 | 16 | 54.03 | 56.57/49.79/62.78 | 0.664/0.46/0.805 | 0.0 | 2026-07-20T01:55:37.113193 |
| 14 | cda2c8ef | Hermes | 16 | 16 | 55.31 | 60.2/51.15/64.61 | 0.661/0.413/0.93 | 0.071 | 2026-07-20T02:01:15.585280 |
| 15 | dce4250e | curl | 16 | 16 | 53.96 | 57.52/54.01/60.13 | 0.674/0.514/0.861 | 0.0 | 2026-07-20T02:01:35.855440 |

## Findings

- **Hermes turn `0bed84ea`** (96 activations): prefill norm 56.98, gen-norm 55.53 (48.1--69.07), consecutive-cosine mean 0.569 (max 0.796), loop-frac(cos>0.9)=0.0.
- **Hermes turn `c0664d17`** (16 activations): prefill norm 57.08, gen-norm 57.2 (49.02--67.43), consecutive-cosine mean 0.537 (max 0.73), loop-frac(cos>0.9)=0.0.
- **Hermes turn `d616fb4a`** (16 activations): prefill norm 54.98, gen-norm 58.31 (48.92--67.8), consecutive-cosine mean 0.475 (max 0.748), loop-frac(cos>0.9)=0.0.
- **Hermes turn `7491ea7c`** (16 activations): prefill norm 59.46, gen-norm 58.3 (53.56--63.7), consecutive-cosine mean 0.624 (max 0.876), loop-frac(cos>0.9)=0.0.
- **Hermes turn `c9a74bdf`** (16 activations): prefill norm 54.31, gen-norm 61.25 (48.3--66.4), consecutive-cosine mean 0.557 (max 0.836), loop-frac(cos>0.9)=0.0.
- **Hermes turn `a703ffcb`** (16 activations): prefill norm 52.64, gen-norm 59.55 (51.47--65.47), consecutive-cosine mean 0.595 (max 0.833), loop-frac(cos>0.9)=0.0.
- **Hermes turn `cda2c8ef`** (16 activations): prefill norm 55.31, gen-norm 60.2 (51.15--64.61), consecutive-cosine mean 0.661 (max 0.93), loop-frac(cos>0.9)=0.071.

## Method note

- Metrics computed on the **same-inference** layer-23 residual stream captured by `nla_server.py` (forward hook fires on the `model.generate()` call that produces the agent's output). No replay, no second model -- faithful to the hermes-sae same-inference invariant (REPORT.md sec.4).

- `consec_cos` = cosine similarity between activation vectors at consecutive generated positions. Sustained high values = the model is not moving through representation space = degenerate/looping.

- These are **raw residual activations** (what a trained SAE would decompose into features). Wiring a trained SAE for Gemma-4-E2B layer 23 is the next step; the capture format (`acts [N,1536]`, `norms`, `is_prefill`, `token_ids`, `gen_ids`) is SAE-ready.
