Written: 2026-10-03T18:59:30.799906+00:00
# Amendment 2 (written before any output from the extra models existed)
Two small models (Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct) are run on the same frozen benchmark, core tier only,
and ONLY if generation time remains after all four primary models finished every tier. Purpose: a scale trend inside the
Qwen2.5 family (1.5B vs 3B) and one more model family. They do not change any primary result. A model whose core tier is
incomplete is reported as incomplete and excluded from pooled estimates.
