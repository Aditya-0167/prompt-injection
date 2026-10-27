Written: 2026-10-03T18:59:30.800284+00:00
# Amendment 1 (protocol change made BEFORE any adaptive or sampled output existed)
The defense set for the adaptive best-of-8 tier and the sampled-decoding tier was changed from {none, D1} to {none, D2}.
Reason: in the first completed core results (Qwen3-4B complete, Phi-4-mini partial), D1 (instruction reminder) had a pooled
effect on injected reports close to zero, while D2 (spotlighting) was the strongest defense; the adaptive question that matters is
whether the strongest defense survives a stronger attacker. This choice was informed by core results only; no adaptive or sampled
row had been generated at this time. The core tier, benchmark, prompts and scoring are unchanged.
