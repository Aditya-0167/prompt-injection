# Validation performed

| Check | Where | Result |
|---|---|---|
| Parsing and scoring unit tests (fenced JSON, think blocks, defanged IOCs, abstention, invalid output, case-insensitive severity, ATT&CK ID extraction, escalation threshold) | `tests/test_scoring.py` | pass |
| Benchmark integrity: hash, size, balance, every strategy-placement pair three times per family, valid targets, documentation-only IPs and `.example` domains | `tests/test_benchmark.py` | pass |
| Clean twin differs from the injected report in exactly one line, for each of the 360 cases | `tests/test_benchmark.py` | pass (360) |
| Checkpoints: all 11 ZIPs present; their union equals the released raw outputs (29,280 keys); expected rows per model and tier; merge tool deduplicates | `tests/test_checkpoints.py` | pass |
| Headline numbers recomputed from raw outputs (attack effect, three defense effects, accuracy cost, naive-scoring inflation, best-of-8 rates, unsafe-action effects) | `tests/test_statistics.py` | pass |
| Whole experiment script end to end with a mock model (benchmark, every tier, statistics, figures, ZIPs) | `tests/run_mock_pipeline.sh` | pass |
| One-command reproduction of tables, figures and paper numbers from the raw outputs | `reproduce.sh` | rebuilds the paper source identically |

Total: 394 automated tests.

## Cross-session reproducibility
Some prompts were regenerated in later Colab sessions that resumed from older checkpoints. Across all checkpoint ZIPs, 3,472 outputs were generated more than once from identical prompts. In 3,468 the raw text was byte-identical; in 4 (all from Phi-4-mini) it differed, and no scored outcome (`hit`, `integrity_ok`) differed in any of the 3,472. We attribute the text differences to batch composition under 16-bit arithmetic and did not investigate further. The merged data keeps the earliest copy of each row.

## Parsing
99.8% of outputs are valid JSON; the lowest model-by-condition cell is 98.1% (`statistics/quality.csv`).

## What was not validated
No real incident reports were used; there is no human audit of the rule-based scorer.
