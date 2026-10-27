# Environment

All 11 checkpoint ZIPs record the same environment (`environment/environment.json` inside each ZIP):

| Item | Value |
|---|---|
| GPU | Tesla T4 (16 GB), CUDA available: True |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.18.0 |
| NumPy / pandas | 2.1.3 / 3.0.6 |
| Model precision | 16-bit floating point; 4-bit loading exists only as an automatic fallback if a model failed a smoke test (the `model_mode` column of the raw outputs records the mode used) |
| Decoding | greedy (primary tiers), `max_new_tokens=160`; sampled tier: temperature 0.7, top-p 0.9, three samples |

Analysis scripts and tests need only the packages in `requirements-analysis.txt`.
