# Checkpoint ledger

Every checkpoint ZIP produced during the Colab runs. Each ZIP holds `raw/rows.jsonl` (all outputs generated so far in that session lineage; byte-identical to the original), the frozen benchmark, the preregistration files, logs and the environment record. Checkpoints are cumulative within a lineage, so later files contain earlier rows, and sessions that resumed from an older checkpoint regenerated some rows (see `docs/VALIDATION.md`). Project-specific identifiers in log and text files were renamed for anonymity; no raw output was changed.

| File | Rows | First output (UTC) | Last output (UTC) | Size (KB) | SHA-256 (first 16) |
|---|---:|---|---|---:|---|
| `cti_checkpoint_01_0900.zip` | 9,984 | 2026-10-03T13:43:50 | 2026-10-04T09:00:02 | 816 | `db5c1a9866dd5b74` |
| `cti_checkpoint_01_1148.zip` | 13,408 | 2026-10-03T13:43:50 | 2026-10-04T11:48:15 | 1,055 | `d0b583945cfb9aa3` |
| `cti_checkpoint_01_1435.zip` | 2,496 | 2026-10-03T13:43:50 | 2026-10-03T14:35:18 | 282 | `7cf669cfa71dcba6` |
| `cti_checkpoint_01_1453_dup.zip` | 24,112 | 2026-10-03T13:43:50 | 2026-10-04T14:53:19 | 1,726 | `fd645192b52aecdf` |
| `cti_checkpoint_01_1959.zip` | 5,488 | 2026-10-03T13:43:50 | 2026-10-03T19:59:40 | 495 | `272cceb51be4d37f` |
| `cti_checkpoint_01_1959_dup.zip` | 5,488 | 2026-10-03T13:43:50 | 2026-10-03T19:59:40 | 495 | `272cceb51be4d37f` |
| `cti_checkpoint_02_1248.zip` | 16,008 | 2026-10-03T13:43:50 | 2026-10-04T12:48:19 | 1,208 | `8703686153b88e2f` |
| `cti_checkpoint_02_1535.zip` | 5,968 | 2026-10-03T13:43:50 | 2026-10-03T15:35:20 | 527 | `d0d76d7365c1ae37` |
| `cti_checkpoint_02_1543.zip` | 29,280 | 2026-10-03T13:43:50 | 2026-10-04T15:43:10 | 2,114 | `4ea8ab05ab8f6e1d` |
| `cti_checkpoint_03_1348.zip` | 20,408 | 2026-10-03T13:43:50 | 2026-10-04T13:48:28 | 1,478 | `75745d1c34a045ea` |
| `cti_final_complete.zip` | 29,280 | 2026-10-03T13:43:50 | 2026-10-04T15:43:10 | 4,122 | `c95280d15e57bc8e` |
