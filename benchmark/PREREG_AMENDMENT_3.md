Written: 2026-10-04T08:00:01.542160+00:00
# Amendment 3 (written before any defense-aware output existed)
A third attack tier is added: three defense-aware injection strategies (delimiter escape, authorized-channel mimicry,
policy negation) on the 20-per-family adaptive case subset, against all four defense settings. Reason: a defense evaluation
is only meaningful against an attacker who knows the defense; the strategies were designed from the public descriptions of the
defenses, not from per-case results. This was decided after the first core results (Qwen3-4B complete, Phi-4-mini partial)
were inspected. Pre-specified question: does each defense's measured benefit survive these strategies?
Tests: Wilson intervals per model x defense x strategy and best-of-3 per case. No core outcome definition changes.
