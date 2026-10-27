# Experiment log

All times UTC. The checkpoint ledger is in `checkpoints/MANIFEST.md`.

| When | What |
|---|---|
| 2026-10-03 13:34 | Benchmark frozen (SHA-256 in `benchmark/benchmark.sha256`); preregistration written (`PREREGISTRATION.md`) before any model output. |
| 2026-10-03 13:43 | First model output (Qwen3-4B, core tier). Speed calibration selected 120 cases per family. |
| 2026-10-03 | Session 1 completed Qwen3-4B core and part of Phi-4-mini. Infrastructure problems were fixed without changing the benchmark, prompts, scoring, or seeds: Phi-4-mini loaded its own remote code, which was incompatible with the installed Transformers (now loaded with the built-in implementation); the checkpoint writer copied its own folder recursively (now writes ZIPs directly); resume from saved rows was added. |
| 2026-10-03 18:59 | Amendment 1 (adaptive and sampled tiers use D2, not D1) and Amendment 2 (two small models, core only) recorded, after partial core results were inspected and before any output of those tiers existed. |
| 2026-10-04 08:00 | Amendment 3 (defense-aware tier) recorded, before any defense-aware output existed. |
| 2026-10-04 11:51 onward | Best-of-8 tier, then defense-aware (first output 12:08) and sampled (first output 12:23) tiers. |
| 2026-10-04 14:38 and 15:11 | First outputs of the two added models (core tier only). |
| 2026-10-04 15:43 | Last output. 29,280 unique outputs across 11 checkpoint ZIPs and several sessions and accounts. |

Timestamps of amendments precede the first output of the tier they concern: first best-of-8/sampled output 2026-10-04 11:51 and 12:23 (after Amendment 1), first defense-aware output 2026-10-04 12:08 (after Amendment 3), first outputs of the added models 14:38 and 15:11 (after Amendment 2). Some rows appear in several checkpoints because sessions resumed from older checkpoints; see `docs/VALIDATION.md`.

## Changes after the first session (all infrastructure)
1. Built-in model implementation for Phi-4-mini instead of remote code.
2. Checkpoint writer fixed; a failed checkpoint can no longer stop a run.
3. Resume from saved rows; skip finished models.
4. Defense-aware tier, model filter, and optional extra models added (Amendments 2 and 3).

## Earlier work
`archive/` holds the superseded 36-case pilot script and early partial results from the first session. Those results used a different, smaller design and are not pooled with this study.
