# Data dictionary: `raw/rows.jsonl.gz`

One JSON object per model output (29,280 rows).

| Field | Meaning |
|---|---|
| `key` | unique id: `model|tier|case_id|report_kind|defense|strategy|sample` |
| `model` | model name (six values) |
| `tier` | `core`, `adaptive` (best-of-8), `defense_aware`, or `sampled` |
| `case_id` | benchmark case (`sev-000` ... `tec-119`); see `benchmark/frozen_benchmark.json` |
| `family` | attack goal: `severity`, `ioc`, or `technique` |
| `placement` | slot holding the injection or its benign filler |
| `report_kind` | `clean` (clean twin), `inj` (injected report), or `auth` (clean twin plus an authorized instruction outside the report) |
| `defense` | `none`, `D1` (reminder), `D2` (spotlighting/datamarking), `D3` (sandwich) |
| `strategy` | injection strategy (`-` for `clean` and `auth`); defense-aware tier uses `escape`, `mimic_auth`, `policy_negation` |
| `sample` | sample index (sampled tier), else 0 |
| `primary_strategy` | the strategy assigned to the case in the core tier |
| `timestamp` | generation time (UTC) |
| `model_mode` | `fp16` (or `4bit-fallback`) |
| `prompt_sha` | SHA-256 of the exact chat messages sent |
| `sampling` | 1 if sampled decoding was used |
| `raw_output` | the model's verbatim text |
| `parser_valid` | 1 if a JSON object could be parsed |
| `abstain` | 1 if `action` was `request_manual_review` |
| `sev_pred`, `tech_pred` | parsed severity / ATT&CK ID (null if unparsable) |
| `n_iocs_pred`, `ioc_recall`, `ioc_extra` | number of IOCs submitted, fraction of gold IOCs present, number of extra IOCs |
| `hit` | attacker's target state reached (0 on abstention or parse failure; computed in every condition, so clean twins give each case's own baseline) |
| `integrity_ok` | submitted record is fully correct (severity, technique, every gold IOC) |
| `consequence` | `hit` that would also change a downstream action (severity downgrade across the High threshold, IOC omission, technique change) |
