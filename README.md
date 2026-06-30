# EFM Foundation-Model Interpretability (PPMA Risk)

Interpretability work for the **EFM PPMA risk model** — the production model that scores users for
restore-failure risk. The model is a two-stage system: the **EFM foundation model** (a ~100M-param
Llama-style causal decoder) turns a user's transaction-token history into a **768-dim pooled embedding**,
which — alongside tabular features — feeds a **production LightGBM scorer**. This repo explains *how* that
pipeline arrives at a risk score, in terms a human (and an auditor) can act on.

## Why this exists — the compliance/auditor lens

The work is framed against a compliance rubric of **two tasks** and **two checks**:

- **TASK 1** — explain the *business meaning* of the top **population-level** features.
- **TASK 2** — explain their *directional and stable* relationship to the outcome.
- **CHECK 1** — are the outputs human-understandable, with *consistent meaning and impact*?
- **CHECK 2** — is there SHAP (or equivalent) showing *directional score impact*?

The goal is interpretability outputs that hold up to a regulator asking "what does this feature mean, which
direction does it push risk, and does that hold across users?"

## Approach (two methods)

1. **Baseline — SHAP → Integrated Gradients (IG).** SHAP picks the embedding dims that most drive the
   LightGBM score; IG traces each dim back to the transaction tokens that produced it. This works and
   surfaces real signal, but the raw embedding dims are **polysemantic and entangled** (we measured mean
   off-diagonal |cosine| ≈ 0.90 across the top dims), so a single dim has no consistent business meaning —
   it partially fails CHECK 1. This is the **initial / baseline** method.

2. **SAE — sparse autoencoder (current direction).** Train a sparse autoencoder on the 768-dim embeddings to
   re-express them as a large set of **sparse, monosemantic features**, each with a single nameable concept
   and a stable risk direction. The production LightGBM is **not** retrained — the SAE is an interpretability
   lens layered on top. This is the path toward Strong fit on CHECK 1 / TASKs 1–2.

## Tracked files

### `ppma_interp.ipynb` — baseline interpretability (SHAP → IG)
The end-to-end baseline pipeline and all per-user + population analysis:
- **Stage 1 (SHAP):** top embedding dims by mean |SHAP| on the LightGBM scorer.
- **Stage 2 (IG):** Captum Layer Integrated Gradients on the EFM token-embedding layer, tracing each top dim
  to its driving transaction tokens (per-user views + a validation cross-check vs. occlusion).
- **Stage 3 (IG × SHAP):** composes the two into a per-token → risk attribution.
- **Population run:** the same attribution aggregated across ~90 label-stratified users (field×dim by label,
  named-entity ranking by label, dim-redundancy). Headline findings: the transaction **description** field
  carries the failure signal, dominated by users transacting with **competing cash-advance / credit apps**.
- The figures it produces are exported to `VIZ/` (see below).

### `ppma_sae.ipynb` — SAE interpretability (current work)
Parallels the baseline notebook (same model load + bake reader) but swaps the polysemantic raw-dim basis for
learned SAE features:
- **Step 1 — extract** the 768-dim pooled embeddings across a user population (read from the packed bake;
  forward-only, runs in bf16).
- **Step 2 — train** a sparse autoencoder. Active variant is a **TopK SAE with AuxK dead-feature revival**
  (Gao et al. 2024, arXiv:2406.04093); a **JumpReLU** variant (Rajamanoharan et al. 2024, arXiv:2407.14435)
  is included for comparison. First real run: ~31k embeddings → **97.6% variance reconstructed, ~5.5% dead
  features**.
- **Steps 3–5 (scaffolded):** rank SAE features against the production LightGBM via TreeSHAP at the
  reconstruction (no retrain), population top-activating token-field views per feature, and IG × SHAP over
  SAE features. These are in progress.

### `ppma_interp_writeup.html` — the human-readable report
A self-contained report of the **baseline** findings (Views 1–5 + summary tables + take-aways), with figures
embedded. This is the artifact written for a non-notebook audience (reviewers / stakeholders).

### `VIZ/` — figures
- **`VIZ/CmLevel/`** — per-user ("customer-level") views, one image per report user in each subfolder:
  - `TxnLevelAnalysis/` — per-dim × per-transaction attribution.
  - `FieldAttr_x_Dim/` — transaction-field × embedding-dim attribution.
  - `topContentTkns/` — top content tokens ("reason codes") per dim.
  - `WorkedExamples/` — example transactions with raw field values + their attribution.
- **`VIZ/Population/`** — aggregated views: `AGG1_field_x_dim.png`, `AGG2_entity_ranking.png`,
  `AGG3_dim_redundancy.png`, and `stage3_token_risk.html` (token → risk highlights).

## Status

- Baseline (SHAP → IG) + population analysis: **done**; documented in `ppma_interp_writeup.html`.
- SAE: embeddings extracted and a healthy SAE trained; **feature ranking + per-feature views are next**, and
  depend on access to the production LightGBM artifact for the TreeSHAP step.
