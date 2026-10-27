# Running the experiment on Google Colab

Requirements: a Colab runtime with a T4 GPU (about 5 hours per session). No Google Drive is used; checkpoints are downloaded to your computer.

1. **Single session from scratch:** open a new notebook, select the T4 runtime, paste the whole of `code/run_experiment_colab.py` into one cell and run it. It installs its dependencies, builds the frozen benchmark, runs all models and tiers, writes a ZIP checkpoint every hour (the browser downloads it) and a final `CTITRIAGE_V3_COMPLETE.zip`.
2. **Resuming or splitting work:** paste `code/restore_and_merge_checkpoints.py` into a first cell, run it, and select every checkpoint ZIP you have. It merges them, refuses to merge if benchmark hashes differ, and keeps the earliest preregistration files. Then run the experiment cell; it skips every output that already exists. To split models across sessions or accounts set `MY_MODELS` in the first cell, for example `"Phi-4-mini,Qwen2.5-3B"`.
3. **Statistics only:** in the first cell set `RUN_STATS_ONLY = True`, or on your own machine run `CTITRIAGE_DRY_RUN=1 CTITRIAGE_GEN_STOP=0 CTITRIAGE_ROOT=<folder> python code/run_experiment_colab.py` (see `reproduce.sh`). No model is loaded.

Environment variables (prefix `CTITRIAGE_`): `ROOT`, `GEN_STOP` (seconds before generation stops), `CKPT_EVERY` (seconds between ZIPs), `ONLY_MODELS`, `N_PER_FAMILY`, `DRY_RUN` (mock model).

The first model call calibrates how many cases per family fit the time budget (120, 80 or 40; the published run used 120) from wall-clock speed only, before any scored output.
