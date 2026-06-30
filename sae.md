# SAE Interpretability Pipeline (PPMA) — Reference

Working reference for the **sparse-autoencoder (SAE) interpretability** track on the EFM PPMA risk model.
Captures the goal, the full pipeline (done + planned), the math, the data/access facts, and current status.
Companion to the runnable notebooks: [`ppma_sae.ipynb`](ppma_sae.ipynb) (SageMaker) and
[`databricks/sae_step3_feature_ranking.py`](databricks/sae_step3_feature_ranking.py) (Databricks).

Status legend: ✅ done · 🔧 in progress · ⬜ planned.

---

## 0. Goal & compliance framing

The production PPMA risk model is two stages: the **EFM foundation model** (~100M-param Llama-style causal
decoder) turns a user's transaction-token history into a **768-dim pooled embedding** (`last_token` pooling),
which — with tabular features — feeds a **production LightGBM scorer** that outputs a risk score.

We need interpretability that holds up to a **compliance/auditor** review, judged on:

- **TASK 1** — business meaning of the top **population-level** features.
- **TASK 2** — their **directional & stable** relationship to the outcome.
- **CHECK 1** — outputs are human-understandable, with **consistent meaning + impact**.
- **CHECK 2** — SHAP (or equivalent) shows **directional** score impact.

## 1. Why an SAE (the baseline's gap)

The baseline method (`ppma_interp.ipynb`): SHAP picks the top embedding dims, Integrated Gradients traces each
dim to transaction tokens. It works but **fails CHECK 1**: the raw embedding dims are **polysemantic and
entangled** — measured mean off-diagonal |cosine| ≈ **0.896** across the top dims — so a single dim has no
consistent business meaning.

The SAE fixes the *basis*: it re-expresses the 768 entangled dims as a large set of **sparse, monosemantic
features**, each with (ideally) one nameable concept and a stable risk direction. **The production LightGBM is
never retrained** — the SAE is an interpretability lens layered on the embedding it already consumes.

## 2. Pipeline overview

| Step | What | Where | Status |
|---|---|---|---|
| 1 | Extract 768-d pooled embeddings over a user population | SageMaker (`ppma_sae.ipynb`) | ✅ |
| 2 | Train the SAE (TopK; JumpReLU for comparison) | SageMaker | ✅ |
| 3 | Rank SAE features vs the production scorer (Path B now, Path A verify) | Databricks | 🔧 |
| 4 | Derivable views (unified ranking, token-field views, direction plots) | DBX + SageMaker | 🔧 |
| 5 | IG × SHAP over SAE features (token → risk via a monosemantic feature) | SageMaker | ⬜ |

---

## 3. Data & access

### 3.1 SAE training data ✅
- Population embeddings extracted from the **train bake** (15 shards, ~2k users/shard) via one forward pass
  per user (`pooled_embedding`, bf16, full sequence). Result: **30,866 users × 768** →
  `_cache/sae/pop_emb_train15sh.npz` (852 positive / 30,014 negative).

### 3.2 The inference log (the production attributions) ✅
`main_prod.ml_data.efm_ppma_v1_inference_log` — **91.4M rows**, current through 2026-06-16. Per row:
- `payload.<feature>` — **raw feature VALUES** (incl. all 768 `decoder39k_ppma_lora_emb_0..767` + tabular).
- `response.shap_values.<feature>` — **per-feature SHAP** (production TreeSHAP output).
- `response.shap_values.base_value` = **-4.9679** (stable across 80.5M rows) — the model's expected output.
- `response.score` — the risk score. Additivity: $\text{score} = \text{base\_value} + \sum_{\text{features}} \text{SHAP}$.
- `model_name` / `model_version` columns exist but are **NULL** (the log does not self-identify the binary).
- **Data quirk:** ~28% of rows log SHAP but have **NULL raw embedding values** in `payload`. Path B needs the
  raw values (to encode with the SAE), so the pull filters `payload.decoder39k_ppma_lora_emb_0 IS NOT NULL`.

### 3.3 Access reality
- Databricks **PATs are disabled org-wide** → no token-based access from the SageMaker kernel.
- A **Databricks SQL MCP connector** is available to Claude Code (read-only DBSQL); it can query Unity Catalog
  but **cannot load an mlflow model or run TreeSHAP**.
- Registered **models** are not exposed via `information_schema` / `SHOW MODELS`; the booster binary needs the
  model owner or `mlflow.load_model` inside a DBX notebook.
- **Decision:** run Step 3 **inside Databricks**, where the user is auto-authenticated. Per-user governed data
  (embeddings + SHAP) **stays in DBX**; only the SAE weights (~19 MB) go in and a feature-level aggregate
  ranking comes out. **Do not export per-user rows / user_ids.**

---

## 4. SAE architecture & training ✅

Trained in **standardized** embedding space: $x_n = (x - \mu) / \sigma$ ($\mu, \sigma$ are per-dim mean/std
over the population; saved in the checkpoint to invert downstream).

### 4.1 TopK SAE — the run variant (Gao et al. 2024, arXiv:2406.04093)
- **Encode:** TopK on the **raw pre-activations**, ReLU **after** (not relu-then-topk):
  $\text{pre} = (x - b_{\text{dec}})\cdot W_{\text{enc}} + b_{\text{enc}}$ ; $z = \mathrm{ReLU}(\mathrm{TopK}_k(\text{pre}))$.
  (Paper's form; can activate $< k$ when fewer than $k$ pre-acts are positive.)
- **Decode (linear):** $\hat{x} = z\cdot W_{\text{dec}} + b_{\text{dec}}$.
- Weights **tied at init** ($W_{\text{dec}} = W_{\text{enc}}^\top$), trained **untied**. Decoder rows
  **re-normalized to unit norm after every optimizer step** → a feature's strength lives in its activation
  $z$, not its decoder length.
- **AuxK** dead-feature revival (TopK-only): dead features (didn't fire last epoch) are trained to reconstruct
  the residual, reviving near-threshold ones. Far-negative dead features would need resampling (future).

### 4.2 JumpReLU SAE — comparison variant (Rajamanoharan et al. 2024, arXiv:2407.14435)
- $z = \text{pre}\cdot H(\text{pre} - \theta)$, per-feature **learned threshold** $\theta$, L0 sparsity. The
  hard gate has zero gradient, so $\theta$ learns via a **rectangle straight-through estimator** (`_JumpGate`).
  **Not the run variant**
  (log_theta init + L0 weight need tuning); AuxK is **TopK-only** by design.

### 4.3 Why not plain L1
L1 shrinks activation **magnitudes** — and we use those magnitudes downstream (SHAP back-distribution +
IG×SHAP), so shrinkage would bias the attribution. TopK/JumpReLU give sparsity **without** magnitude bias.

### 4.4 Config & result
`SAE_EXPANSION=4` → **3,072 features**, `K=32`, `K_aux=512`, AuxK coeff 1/32, 200 epochs, batch 256, lr 1e-3.
First real run (30,866 users): **1−FVU = 0.976, L0 = 32, dead 168/3072 (5.5%)** →
`_cache/sae/sae_topk_x4_k32.pt` (holds `state`, `mu`, `sd`, `variant`, `n_features`, `k`). The 5.5% dead rate
confirms the earlier 74–77% dead at 400 users was **data-diversity-limited**, not a bug.

---

## 5. Step 3 — feature ranking (the math) 🔧

Goal: a per-feature attribution to the risk score, **without retraining or even loading** the production
model. Two interchangeable sources for the per-dim SHAP $\phi$; same back-distribution either way.

### 5.1 Setup
For a user, the SAE gives activations $z$ (sparse, $\le k$ nonzero) and the **linear** decoder
$D = W_{\text{dec}}$ (shape $n_{\text{features}} \times 768$). The standardized embedding decomposes as:

$$
\hat{x}_{n,j} = b_{\text{dec},j} + \sum_i z_i \, D[i,j] \qquad (\text{dim } j,\ \text{standardized space})
$$

Production SHAP $\phi_j$ attributes embedding **dim $j$** to the score
($\text{score} = \text{base} + \sum_j \phi_j + \sum_{\text{tab}} \phi_{\text{tab}}$).

### 5.2 Back-distribution — the LRP z-rule
This step is **Layer-wise Relevance Propagation (LRP)** through the one linear layer that separates the SAE
features from the embedding dims the model actually scored. LRP takes the "relevance" assigned to a layer's
*outputs* and pushes it back to the *inputs*, splitting each output's relevance across the inputs **in
proportion to how much each input contributed to it**, while **conserving the total**. Mapping the pieces here:

- the **layer** is the decoder, $\hat{x}_{n,j} = b_{\text{dec},j} + \sum_i z_i\,D[i,j]$, with pre-activation
  $t_j = \sum_i z_i\,D[i,j]$;
- the **inputs** are the SAE features (feature $i$'s contribution to dim $j$ is $z_i\,D[i,j]$);
- the **output relevance** at dim $j$ is its production SHAP $\phi_j$.

The proportional split and the resulting per-feature attribution:

$$
\underbrace{w_{ij} = \frac{z_i\,D[i,j]}{t_j}}_{\text{share of feature }i\text{ in dim }j},
\qquad t_j = \sum_k z_k\,D[k,j]
$$
$$
\psi_i = \sum_j w_{ij}\,\phi_j = \sum_j \frac{z_i\,D[i,j]}{t_j}\,\phi_j
$$

This is **exactly the LRP z-rule (a.k.a. LRP-0)** of Bach et al. (2015) — in LRP notation
$R_{i\leftarrow j} = \big( z_i\,D[i,j] \big/ \sum_k z_k\,D[k,j] \big)\,R_j$ with the output relevance
$R_j = \phi_j$ and $\psi_i = \sum_j R_{i\leftarrow j}$. Montavon et al. (2017) show the z-rule is the
**deep Taylor decomposition** of a linear layer, which is what makes it the principled (not ad-hoc) choice
for the *linear* decoder.

Vectorized (per batch): $t = Z\,D$ ; $\psi = Z \odot \big( (\phi / t)\,D^\top \big)$, with $t_j$ guarded by
$|t_j| > \varepsilon$.

**Properties / why it's faithful:**
- **Conservation (LRP's defining property).** For each dim the shares sum to one,
  $\sum_i w_{ij} = \sum_i z_i\,D[i,j] / t_j = t_j / t_j = 1$, so summing over features returns the dim's SHAP
  and summing over dims gives $\sum_i \psi_i = \sum_j \phi_j$ (over dims with $|t_j| > \varepsilon$). The
  embedding's exact SHAP is **re-allocated** from 768 entangled dims onto 3,072 interpretable features — none
  invented or lost.
- **The $|t_j| > \varepsilon$ guard is the LRP-ε rule** (Bach et al. 2015). Where the reconstructed
  pre-activation $t_j \approx 0$ (features cancelling on a dim) the $1/t_j$ share is ill-posed and would blow
  up; $\varepsilon$ absorbs that sliver of relevance rather than amplifying it. The **Tikhonov-damped**
  variant ($1/t_j \to t_j/(t_j^2+\lambda)$) used to build the per-user decision-point cache is the smooth
  form of the same stabilizer.
- **Bias carries no relevance.** $b_{\text{dec}}$ is excluded from $t_j$ deliberately — per LRP the bias is
  not an input, so it is attributed to no feature (and conservation is stated w.r.t. $\sum_j \phi_j$).
- **$\sigma$ cancels:** in raw space $x_j = \mu_j + \sigma_j\,\hat{x}_{n,j}$, so feature $i$'s part of $x_j$
  is $\sigma_j\,z_i\,D[i,j]$; the ratio $w_{ij}$ cancels $\sigma_j$. So we apply $w_{ij}$ directly to the
  raw-feature SHAP $\phi_j$.
- **Baseline alignment (caveat):** SHAP's baseline is $E[x_j]$; the SAE's "$z=0$" baseline is
  $\mu_j + \sigma_j\,b_{\text{dec},j}$. Since $\mu_j = \operatorname{mean}(x_j)$ and $b_{\text{dec}} \approx 0$,
  these $\approx$ align — a footnote, not an exact identity.

> **References (cite for the method).**
> - Bach, Binder, Montavon, Klauschen, Müller & Samek (2015), *On Pixel-Wise Explanations for Non-Linear
>   Classifier Decisions by Layer-Wise Relevance Propagation*, PLoS ONE 10(7):e0130140 — origin of the
>   z-rule and the ε-stabilizer. **[primary cite]**
> - Montavon, Lapuschkin, Binder, Samek & Müller (2017), *Explaining nonlinear classification decisions
>   with deep Taylor decomposition*, Pattern Recognition 65:211–222 — z-rule = deep Taylor decomposition of
>   a linear layer (the justification for using it on the decoder).
> - Montavon, Binder, Lapuschkin, Samek & Müller (2019), *Layer-Wise Relevance Propagation: An Overview*,
>   in *Explainable AI*, LNCS 11700, Springer, 193–209 — LRP-0 / LRP-ε naming.
> - SHAP itself: Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS;
>   tree path-dependent TreeSHAP: Lundberg et al. (2020), *Nat. Mach. Intell.* 2:56–67.

### 5.3 Path B (logged SHAP) — runnable now, no binary
Use $\phi_j$ = the **logged** production SHAP at the true embedding $e$ (straight from the inference log).
Justification: the SAE reconstructs $e$ well (1−FVU = 0.976), so $\phi(e) \approx \phi(\hat{e})$. Uses the
**real production attributions**; removes the model-access blocker. **This is the path we run first.**

### 5.4 Path A (recompute) — verification, needs the binary
Use $\phi_j$ = **TreeSHAP recomputed at the reconstruction $\hat{e}$** (de-standardized $\mu + \sigma\,\hat{x}$,
with the 768 emb columns replaced and the tabular values kept). Requires loading the production LightGBM
(model owner / `mlflow.load_model`); first **confirm `TreeExplainer.expected_value` $\approx -4.9679$** to
prove it's the right artifact, then compare $\phi(\hat{e})$ vs the logged $\phi(e)$ and $\psi$ vs Path B on a
sample.

### 5.5 Ranking
Population importance of any feature = **mean over users of |contribution|**:

$$
\text{importance(SAE feature } i) = \operatorname{mean}_n \big| \psi[n,i] \big|
$$
$$
\text{importance(tabular feature)} = \operatorname{mean}_n \big| \mathrm{SHAP}[n,\text{feature}] \big|
\quad (\text{used as-is from the log})
$$

`mean_abs` drives the ranking (a feature can push risk both ways across users; signed values could cancel).
Reported alongside: `mean_signed` (net direction), `sign_consistency` (share with the dominant sign among
active users → TASK2/CHECK2 stability), `activation_rate`.

---

## 6. Unified ranking (SAE + tabular) 🔧

The production model consumes **768 embedding dims + tabular features**. The SAE re-expresses **only** the
embedding dims. So the apples-to-apples population view is:

> **3,072 SAE features  +  the model's tabular features**, ranked together by `mean_abs`.

This is valid because both are in the **same units** — mean absolute log-odds SHAP contribution — and the SAE
$\psi$'s sum back to the embedding-dim SHAP total (no double-counting). We deliberately do **not** mix in the raw
768 dims, which would double-count the embedding signal. (The old polysemantic "top raw dims" list can be
shown separately for a before/after contrast.)

Output: a `TOP N OVERALL` table (each row tagged `sae` / `tabular`) + a `TOP N SAE` drill-down.

**Generalization gate:** before trusting the ranking, the notebook checks the SAE's 1−FVU on **production**
embeddings (`payload`) ≈ 0.976. A large drop would mean our forward-pass/pooling differs from production and
the ranking can't be trusted yet.

---

## 7. Steps 4–5 — planned views

### Step 4 — derivable views 🔧

**Per-token → feature signal = activation *delta* (not IG).** `encode_hidden_states` returns every token's
hidden state in **one forward**; under `last_token` pooling token *t*'s state is the prefix `[0..t]` embedding
(row $T-1$ = the pooled vector the SAE trained on), so encoding all rows through the SAE is a single matmul —
**no per-token re-runs.** Per-token contribution $\Delta_t = z_t - z_{t-1}$; $\sum_t \Delta_t = z_{T-1}$
telescopes to the exact pooled activation §5 ranks on (the notebook asserts
$\max|\textstyle\sum \Delta - \text{pooled}| \approx 0$). Cheap enough to run for all
top-15 features over each feature's top activators; **IG (Step 5) is reserved as a validation cross-check on
the top 3–5 features only.** *Caveat:* encoding intermediate hidden states is ~in-distribution thanks to
causal + last-token pooling, but to be validated against IG on a few cases.

- **Unified ranking** (§6) ✅ in the DBX script.
- **Token-field × SAE-feature heatmap** ✅ (View A) — roll each top feature's per-token $\Delta$ up to
  transaction field buckets, per-user L1-normalized, averaged over its top activators. Shows which *field*
  drives each feature. (VIZ 2 analog, bucketed by SAE feature.)
- **Top-10 activating transactions for `sae_881`** ✅ (View B) — pooled across the feature's top activators,
  ranked by **max $|\Delta|$ over content tokens** (length-invariant — fixes the $\sum|\text{attr}|$
  long-description bias a colleague flagged; `rank="max"|"top3"|"sum"`). Token color = signed $\Delta$.
  (VIZ 5 analog.)
- **SAE-feature × SHAP direction plot** ⬜ — distribution of per-user $\psi$ for a feature (direction + magnitude +
  subgroup stability).
- **Per-user top SAE feature activations** ⬜.

Shared code lives in `utils/` (`efm_model`, `bake`, `tokens`, `viz`), imported by `ppma_sae.ipynb`;
`ppma_interp.ipynb` is left as-is (its outputs are frozen in the HTML writeup).

### Step 5 — IG × SHAP over SAE features ⬜
The SAE encoder is differentiable, so reuse the Stage-3 machinery: a forward returning a single feature's
activation, `LayerIntegratedGradients` on the token-embedding layer (target = feature *i*, `internal_batch_size=1`),
weighted by that feature's per-user $\psi$ → **token → risk, via a monosemantic feature.** (fp32 + recent-window
truncation, like the population IG in `ppma_interp.ipynb`.)

---

## 8. Faithfulness caveats (report to auditors)

- **Two chained, individually-exact attributions** — TreeSHAP assigns exact (additive) relevance to the 768
  embedding dims through the LightGBM; the LRP z-rule (§5.2; exact for the *linear* decoder) then re-allocates
  that relevance onto the SAE features. $\psi$ is therefore faithful to the SHAP it redistributes — **not** an
  independent Shapley value of the SAE feature in the full model. State the chaining plainly; don't claim a
  single end-to-end Shapley guarantee.
- **Reconstruction residual** — embedding variance the SAE doesn't capture (1−FVU) leaves a small unallocated
  SHAP remainder; report 1−FVU and avg L0 as faithfulness metrics.
- **Path B $\phi(e)$ vs $\phi(\hat{e})$** — bounded by reconstruction quality; Path A quantifies it on a sample.
- **Baseline alignment** (§5.2) — $\mu \approx E[x]$, $b_{\text{dec}} \approx 0$, so approximately aligned.
- **Dead features** (~5.5%) carry no attribution by construction.

---

## 9. Files & artifacts

| Path | Role |
|---|---|
| `ppma_sae.ipynb` | SageMaker: Steps 1–2 (extract + train SAE), Step 4 views A/B; Step 3/5 scaffolds |
| `utils/{efm_model,bake,tokens,viz}.py` | shared IO / token-machinery / rendering (used by `ppma_sae.ipynb`) |
| `databricks/sae_step3_feature_ranking.py` | DBX: Path B ranking + unified table; Path A verify stub |
| `databricks/find_ppma_lightgbm.py` | DBX: locate + verify the production booster (for Path A) |
| `_cache/sae/pop_emb_train15sh.npz` | 30,866 × 768 population embeddings (SAE training data) |
| `_cache/sae/sae_topk_x4_k32.pt` | trained TopK SAE (state + μ/σ + config) |
| `README.md` | project-level overview (baseline + SAE) |

S3 push of `_cache/sae/` deferred until a writable prefix is confirmed with the team.

---

## 10. Status summary

- ✅ Steps 1–2: embeddings extracted, healthy TopK SAE trained (1−FVU 0.976, 5.5% dead).
- 🔧 Step 3: Path B ranking + unified SAE/tabular table built; **running in DBX** (awaiting first output —
  generalization-FVU + TOP-N tables).
- ⬜ Step 3 Path A verify: gated on the model binary (owner / `mlflow.load_model`; confirm exp_value −4.9679).
- ⬜ Step 4 token-field & direction views; Step 5 IG × SHAP over SAE features.
