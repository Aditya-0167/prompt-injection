## Copyright ©2026 Aditya. All rights reserved.

No permission is granted to copy, modify, distribute, reproduce, or commercially use this code or substantial portions of this repository without prior written permission. For full terms, conditions, and restrictions, please read the official License file
| [View License](License.md) |
| :--- |

---


# Counterfactual evaluation of prompt injection in locally deployed LLM threat-intelligence triage

Code, frozen benchmark, preregistration, **all 29,280 raw model outputs**, checkpoints, statistics and the paper source for a controlled study of indirect prompt injection against open-weight language models that read threat-intelligence text and emit structured triage records (severity, indicators of compromise, ATT&CK technique).

> **Anonymity notice.** This repository accompanies a paper under double-blind review. Author information is withheld and project-specific identifiers were renamed. No raw model output was altered.

## Why this study

Evaluations of prompt injection often (1) count ordinary model errors as attack success, (2) compare defended and undefended prompts that differ in more than the defense, (3) test one fixed template per attack, and (4) ignore what a defense costs. This repository implements a **paired clean-twin methodology** that addresses all four:

* **Clean twins.** Each of 360 synthetic incident reports has a twin that differs from the injected report in exactly one sentence (checked by a test for every case). The injection's effect is the within-case change.
* **Matched defenses.** Task framing is identical across conditions; only the defense text differs (D1 instruction reminder, D2 spotlighting by datamarking, D3 sandwich reminder).
* **Adaptive attackers.** Fixed-template, best-of-eight per case, defense-aware, and sampled-decoding tiers.
* **Cost accounting.** Benign accuracy, authorized-instruction success, escalation to humans, and the unsafe SOC actions that result.

Six open-weight models (1.5B–4B parameters, four families) are run in 16-bit precision on one 16 GB T4 GPU, the in-house deployment regime where the prompt is often the only control an operator can edit.

## Headline results (pooled over six models; 95% case-cluster bootstrap CI)

| Quantity | Estimate |
|---|---|
| Attack effect: injected minus clean twin, no defense | +22.2 pp [+19.6, +25.0] |
| D1 reminder: change in attack success | +1.6 pp [+0.3, +2.8] |
| D2 spotlighting: change in attack success | −11.0 pp [−12.8, −9.2] |
| D2 spotlighting: change in benign accuracy | −13.5 pp [−15.2, −11.7] |
| D3 sandwich: change in attack success | +2.7 pp [+1.6, +3.9] |
| Unsafe action induced: escalation suppressed | +9.9 pp [+6.7, +13.2] |
| Unsafe action induced: IOC missing from blocklist | +18.6 pp [+14.7, +22.5] |
| Unsafe action induced: wrong playbook | +31.4 pp [+26.2, +36.7] |

Per-case adaptive attackers (percent of 60 cases; four models with every tier):

| Model | Mean single strategy (none) | Best of 8 (none) | Best of 8 (D2) | Defense-aware best of 3 (none / D1 / D2 / D3) |
|---|---:|---:|---:|---|
| Qwen3-4B | 64.6 | 98.3 | 95.0 | 98.3 / 95.0 / 71.7 / 95.0 |
| Phi-4-mini | 15.2 | 55.0 | 31.7 | 43.3 / 70.0 / 35.0 / 45.0 |
| Qwen2.5-3B | 20.4 | 53.3 | 38.3 | 43.3 / 48.3 / 26.7 / 41.7 |
| Granite-3.3-2B | 23.5 | 80.0 | 71.7 | 61.7 / 56.7 / 51.7 / 48.3 |

Every number above is recomputed from the raw outputs by the tests in `tests/` and by `reproduce.sh`.

## Repository layout

```
benchmark/        frozen benchmark (360 cases), its SHA-256, prompts and templates, preregistration + amendments
raw/              rows.jsonl.gz: every model output with parsed fields and scores (29,280 rows)
checkpoints/      all 11 Colab checkpoint ZIPs and a ledger (MANIFEST.md)
code/             run_experiment_colab.py (the experiment, one Colab cell) and restore_and_merge_checkpoints.py
analysis/         scripts that turn raw outputs into every table, figure and number in the paper
statistics/       result tables (CSV) produced by the experiment script
figures/          figures used in the paper (PDF)
results/tables/   LaTeX tables used in the paper
paper/            paper source (LaTeX), figures and compiled PDF
tests/            394 automated tests + an end-to-end mock-model pipeline test
docs/             methodology, data dictionary, experiment log, validation report, Colab guide, environment
archive/          superseded 36-case pilot script and early partial results (kept for transparency)
reproduce.sh      one command: raw outputs -> statistics -> tables, figures and paper numbers
```

## Quick start (no GPU needed)

```bash
pip install -r requirements-analysis.txt
bash reproduce.sh                  # rebuild statistics, tables, figures and paper numbers
python -m pytest tests -q          # 394 tests
bash tests/run_mock_pipeline.sh    # whole pipeline with a mock model, a few seconds
```

`reproduce.sh` unzips the raw outputs, recomputes all statistics (the experiment script in statistics-only mode, which loads no model), then regenerates tables, figures and the numbers inserted in the paper text.

## Re-running the experiment

The experiment is one Google Colab cell (T4 GPU, about 5 hours, no Google Drive needed). It checkpoints every hour and resumes from saved rows, so it can be split over several sessions or accounts. See `docs/COLAB_GUIDE.md`. Runs used greedy decoding for the primary tiers; the sampled tier uses temperature 0.7, top-p 0.9.

## Data

`raw/rows.jsonl.gz` holds one JSON object per model output (fields in `docs/DATA_DICTIONARY.md`). Conditions: `report_kind` (`clean`, `inj`, `auth`) × `defense` (`none`, `D1`, `D2`, `D3`); tiers `core`, `adaptive`, `defense_aware`, `sampled`. Counts per model and tier are in `statistics/coverage.csv`.

## Pre-specification and analysis status

The analysis plan (`benchmark/PREREGISTRATION.md`) was written before any model output existed; three amendments made after inspecting partial results are disclosed in `benchmark/PREREG_AMENDMENT_*.md`. The core tier on the four original models is the confirmatory component; the two models added later, the defense-aware tier, and the post hoc analyses are exploratory. `docs/METHODOLOGY.md` states this per analysis.

## Reproducibility notes

* One benchmark hash (`benchmark/benchmark.sha256`) is verified by a test.
* Across independent sessions, 3,472 outputs were regenerated from identical prompts: 4 differed in raw text (all one model) and none differed in any scored outcome (`docs/VALIDATION.md`).
* Environment of all runs: Python 3.13, PyTorch 2.11 (CUDA 13), Transformers 5.18, NVIDIA T4, 16-bit weights (`docs/ENVIRONMENT.md`).

## Limitations

Synthetic reports; open models of 1.5B–4B parameters only (no larger or hosted models); hand-designed attackers (no gradient-based or iteratively optimized attacks); one implementation of each defense; rule-based scoring after normalization; the external-check analysis is an upper bound because the structured fields are authoritative in this benchmark. See the paper's Limitations section.

## Ethics

All data are synthetic: IP addresses come from documentation ranges (RFC 5737) and domains use the reserved `.example` suffix. No real organization, personal data, or live system is involved. The injection strings are short natural-language templates comparable to those in public benchmarks; the study is defensive.

