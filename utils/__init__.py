"""Shared interpretability utilities for the EFM PPMA notebooks.

Signal-agnostic building blocks reused by `ppma_interp.ipynb` (SHAP→IG baseline) and
`ppma_sae.ipynb` (SAE features). The split:

- `efm_model` — load the fine-tuned EFM, forward helpers (pooled + per-token hidden states).
- `bake`      — pull/iterate the packed tokenized bake + tokenizer runtime.
- `tokens`    — token→transaction/field machinery and raw-value rendering (no model, no plotting).
- `viz`       — rendering: field×signal heatmap, top-transaction token table (length-invariant ranking).

Modules that import `efm_*` packages assume the EFM repo `code/src` is already on `sys.path`
(the notebooks do this in their setup cell before importing `utils`).
"""
