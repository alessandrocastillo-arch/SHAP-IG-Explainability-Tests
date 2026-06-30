---
name: autoresearch
description: Autonomous AI research loop for pre-training. The agent iterates on model architecture and training hyperparameters, submitting experiments to HyperPod, evaluating results via metrics.jsonl, and keeping or discarding changes — all without human intervention. Use when the user wants to run autonomous hyperparameter and architecture exploration for the efm-core model.
---

# autoresearch

This is an autonomous research skill for the efm-core project. You are the researcher. You modify model architecture and training hyperparameters, submit experiments to HyperPod via Slurm, evaluate results via metrics.jsonl, and keep or discard changes based on val/loss. Each experiment is a git commit on a dedicated branch. If it improves val/loss, you advance the branch. If it doesn't, you reset. You run indefinitely until the human stops you.

## Domain context (read before hypothesizing)

This is **not traditional natural-language pretraining**. The goal is to learn a latent-space representation of a user's *financial profile*. The approach: take a user's bank-transaction sequence, render each transaction as textual tokens (description, amount, merchant, date deltas, etc.), and train a language model on those token streams. The "language" the model is learning is the structure of someone's spending and income behavior over time, not English.

## Pre-requisites

Before starting the very first run, ask the user to provide the AWS_PROFILE and ENV environment variables. You will need these for the 
hyperpod access skill. Furthermore, ask the user to decide if this is for **decoder (causal LM)** or **encoder (masked LM)** remember 
this, you must not switch the objective between any two experiments. This is a very important detail.

Checkout to a new branch called `autoresearch-<model_type>` where model_type is either decoder or encoder based on the answer above.

## Setup

To set up a new experiment run:

1. **Use the following run name prefix**: autoresearch-<model_type>, when you submit a job you will get a run name unique to each 
   experiment. This will serve as the single identifier for the experiment.
2  **Read in-scope files**: Read the modifiable files and the job submission skill for full context:
   - `@code/scripts/configs/model_config.py`
   - `@code/scripts/configs/training_config.py`
   - `@code/src/efm_core/model/config.py`
   - `@code/src/efm_core/model/efm_core_model.py`
   - `@skills/submit-hyperpod-training-job/SKILL.md` — the job submission workflow. You use this, do not modify it.
   - `@code/scripts/configs/infra_config.py` — contains the experiment name for retrieving results.
6. **Apply baseline config**: Set model config to the baseline values (see Baseline Configuration below), using the encoder/decoder 
   toggle chosen in step 4. Set training config to the 1000-step budget overrides. Commit as the first change on the branch. If the
   on-disk `_DEFAULT_*` constants already match the baseline — typical for the decoder objective — there is nothing to commit; the
   branch HEAD itself is the baseline.
7. **Initialize journal.md**: Create `journal.md` in the repo root with the layout shown in Hypothesis-Driven Exploration: a short header (`# Autoresearch — autoresearch-<model_type> branch`), a `## Results` heading + the 11-column Markdown table header (header row + separator row, no data yet), and an empty `## Hypotheses` heading. Do NOT commit this file — it stays untracked.
8. **Establish baseline**: Submit the baseline config as the first experiment using the `submit-hyperpod-training-job` skill. Print the `=== BASELINE ===` pre-flight block before submission (no `### H<N>` section is needed for the baseline). Wait for completion. Append the baseline row to the `## Results` table in `journal.md` with `hypothesis_id=BASELINE`, `attempt=1/1`, `next_action=pivot`.
9. **Confirm and go**: Once the baseline is recorded, confirm setup looks good and begin the autonomous loop.

## Do not over-index on the on-disk defaults

The `_DEFAULT_*` constants in `model_config.py` / `training_config.py` are *current state*, not *best state*. Treat them as a starting point and revisit every knob each session.

## Baseline Configuration

The `_DEFAULT_*` constants in `code/scripts/configs/model_config.py` and `code/scripts/configs/training_config.py` are the source of truth for what a fresh launch will see. Read them to know the current state.

### Tunable parameters (agent CAN change)

- Architecture: `_DEFAULT_MODEL_DIM`, `_DEFAULT_FFN_DIM`, `_DEFAULT_NUM_LAYERS`, `_DEFAULT_NUM_HEADS`, `_DEFAULT_DROPOUT`, `_DEFAULT_POSITION_ENCODING`, `_DEFAULT_ROPE_THETA`, `_DEFAULT_NORM_TYPE`, `_DEFAULT_GQA_GROUP_SIZE`, `_DEFAULT_FFN_TYPE`, `_DEFAULT_TIE_WORD_EMBEDDINGS`
- Optimizer / LR: `_DEFAULT_OPTIMIZER` (`"muon_adamw"` vs `"adamw"`), `_DEFAULT_BASE_LEARNING_RATE`, `_DEFAULT_LR_STABLE_RATIO`, `_DEFAULT_LR_DECAY_STYLE`, `_DEFAULT_LR_MIN_RATIO`, `_DEFAULT_WEIGHT_DECAY`, `_DEFAULT_MAX_GRAD_NORM`
- Muon (consumed only when `_DEFAULT_OPTIMIZER == "muon_adamw"`): `_DEFAULT_MUON_LR`, `_DEFAULT_MUON_MOMENTUM`, `_DEFAULT_MUON_NS_STEPS`, `_DEFAULT_MUON_WEIGHT_DECAY`, `_DEFAULT_MUON_NESTEROV`
- Objective toggle: `_DEFAULT_IS_CAUSAL` / `_DEFAULT_MODEL_TYPE` (set once at branch creation, never change between experiments)
- `_DEFAULT_GRADIENT_ACCUMULATION_STEPS` (affects effective batch size)
- **Encoder only**: `mlm_probability` (masking fraction; default 0.15, lives in `code/scripts/configs/yaml/profile/encoder.yaml`). Ignored when `model_type=causal_lm`; never sweep on the decoder branch. Validator requires `0.0 < mlm_probability < 1.0`.

### Fixed parameters (do NOT change)

- `_DEFAULT_TOTAL_TRAINING_STEPS` (1000)
- `_DEFAULT_PER_RANK_TOKEN_BUDGET` (81920) — memory-tuned, changing it risks OOM
- `_DEFAULT_PRECISION` (`bf16`)
- `_DEFAULT_WARMUP_BASELINE_STEPS`, `_DEFAULT_WARMUP_MIN_STEPS`, `_DEFAULT_WARMUP_MAX_STEPS` — enforced by tests
- `_DEFAULT_EVAL_INTERVAL_MULTIPLIER`, `_DEFAULT_CHECKPOINT_INTERVAL_MULTIPLIER`, `_DEFAULT_LOG_INTERVAL`, `_DEFAULT_FINAL_EVAL_SPLIT`
- `_DEFAULT_LR_SCALING_MODE`, `_DEFAULT_LR_REFERENCE_GLOBAL_BATCH`, `_DEFAULT_ADAMW_BETA1`, `_DEFAULT_ADAMW_BETA2`
- `_DEFAULT_ATTN_BACKEND` (`fa2`) — image is built with specific kernels, not worth switching
- `MAX_SEQ_LEN` (see Cluster below)

### Setting up the autoresearch branch

Before the very first submission, the only edits you should need are the encoder/decoder toggle plus an optional cosine flip:

- **Encoder / masked LM**:
  - `model_config.py`: `_DEFAULT_IS_CAUSAL = False`
  - `training_config.py`: `_DEFAULT_MODEL_TYPE = "masked_lm"`
- **Decoder / causal LM**: leave both defaults as-is.

Optional, recommended for short runs: `_DEFAULT_LR_DECAY_STYLE = "cosine"`.

`train.sbatch` and `hyperpod/Makefile` are off limits — never change the number of steps or the number of GPUs.

### Cluster

Read the cluster shape from `hyperpod/train.sbatch` (`--nodes` × `--ntasks-per-node` = total GPUs). Warmup, learning rate, and effective global batch are auto-scaled by the runtime from this count — see the formulas in `training_config.py`.

A baseline 1000-step run takes ~20–50 minutes wall-clock end-to-end (training + post-train eval), depending on objective and cluster shape — roughly half the wall-clock of the previous 2000-step baseline (decoder / causal LM was ~40 min at 2000 steps on the current 3-node / 24-GPU shape; encoder / masked LM was meaningfully slower). Re-measure on the current shape if precise budgeting matters. The default `tools/wait_for_hyperpod_job.sh` budget is 120 min, which comfortably holds either; bump `--timeout-min=<larger>` if you've raised `MAX_SEQ_LEN` or otherwise expect longer.

The `train.sbatch` allocation typically carves a slice from a larger cluster — if `squeue` shows multiple `autoresearch-*` jobs running concurrently, expect FSx Lustre contention to add some variance to per-run wall-clock. That's normal cluster sharing, not a regression in your config.

`MAX_SEQ_LEN > 16384` has not been throughput-validated and can push past even the 120-min budget. If you raise it, run a ~50-step smoke first and project total wall-clock before committing to a full 1000-step run.

## Parameter Budget

Target ~100M runtime params with 768d/12L/12H. Verify architecture sanity before each submission:

```bash
PYTHONPATH=code:code/src uv run python tools/count_params.py --vocab-size 16384 --budget 100 --tolerance 15
```

The vocab size is 16384 (production tokenizer `efm-tokenizer-mlm-1776745486`).

`--tolerance 15` is a **fixed parameter — do not change it**. The baseline lands at ~87M (~74M architecture + ~13M tied embeddings), which sits below the 100M target; 15% keeps it in-band.

## In-Scope Files

**CAN modify:**

| File | Purpose | Example changes |
|---|---|---|
| `@code/scripts/configs/model_config.py` | Model architecture params | `_DEFAULT_MODEL_DIM`, `_DEFAULT_FFN_DIM`, `_DEFAULT_NUM_LAYERS`, `_DEFAULT_NUM_HEADS`, `_DEFAULT_DROPOUT`, `_DEFAULT_POSITION_ENCODING`, `_DEFAULT_ROPE_THETA`, `_DEFAULT_ATTN_BACKEND` |
| `@code/scripts/configs/training_config.py` | Training hyperparams | `_DEFAULT_BASE_LEARNING_RATE`, `_DEFAULT_WEIGHT_DECAY`, `_DEFAULT_MUON_LR`, `_DEFAULT_MUON_MOMENTUM`, `_DEFAULT_MUON_NS_STEPS`, `_DEFAULT_GRADIENT_ACCUMULATION_STEPS`, `_DEFAULT_LR_STABLE_RATIO`, `_DEFAULT_LR_MIN_RATIO`, `_DEFAULT_LR_DECAY_STYLE`, `_DEFAULT_MAX_GRAD_NORM` |
| `@code/src/efm_core/model/config.py` | `EFMConfig` class, `DeclaredModelParams` | Add new config fields for architecture experiments |
| `@code/src/efm_core/model/efm_core_model.py` | Model architecture | Attention variants, activation functions, normalization, layer structure, positional encoding |
| `@code/scripts/configs/yaml/profile/encoder.yaml` | Encoder-objective profile overrides (encoder branch only) | `mlm_probability` |

**CANNOT modify:**

| File/Area | Reason |
|---|---|
| `code/src/efm_core/data/` | Data pipeline must stay stable for comparability |
| `code/src/efm_core/contracts/` | Dataset specification is a shared contract |
| `code/scripts/train/` | Trainer, callbacks, logging must stay consistent |
| `code/jobs/` | Job launcher infrastructure |
| `hyperpod/` | Slurm/cluster infrastructure |
| `skills/` | Skill definitions |
| `pyproject.toml` | No new dependencies |

## Experimentation

**What you CAN do:**

- Modify the four in-scope files: model config, training config, `EFMConfig` class, and model architecture.
- Change architecture: model dimensions, FFN ratio, layer count, head count, activation functions, normalization, positional encoding.
- Change hyperparameters: learning rate, weight decay, warmup, schedule, gradient accumulation, dropout, Muon optimizer params.
- Add new config fields to `EFMConfig` to support architecture experiments.

**What you CANNOT do:**

- Modify any file outside the four in-scope files.
- Install new packages or add dependencies.
- Change the data pipeline, evaluation harness, trainer, callbacks, or logging.
- Change the step budget (`total_training_steps` stays at 1000).

The goal is simple: get the lowest val/loss. The step budget is fixed at 1000 steps, so you do not worry about training time — every 
experiment gets the same compute. Everything within the in-scope files is fair game.

Parameter budget is a hard constraint. Stay within ±15% of 100M total params (with the real tokenizer vocab). Check locally before every submission. If you exceed the budget, adjust the config and do not submit.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that is a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

The first run: your very first run should always be to establish the baseline. Submit the baseline config as-is and record the result.

## Retrieving Results

After each experiment completes, read metrics.jsonl from the HyperPod controller.

### Read metrics.jsonl

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'cat /fsx/runs/<RUN_NAME>/metrics.jsonl'
```

`metrics.jsonl` is written to `/fsx/runs/<RUN_NAME>/metrics.jsonl` every `log_interval` steps (default: 100). Each line is a JSON object:

```json
{"step": 100, "loss": 4.47, "grad_norm": 0.99, "learning_rate": 0.02, "tokens_seen": 1540856, "epoch": 0.05}
```

| Field | Description |
|---|---|
| `step` | Training step number |
| `loss` | Training loss at this step |
| `grad_norm` | Gradient norm at this step |
| `learning_rate` | Current LR for `param_groups[0]`. With the default MuonAdamW optimizer, group 0 is the Muon group (see `code/src/efm_core/optim/muon_adamw.py`), so this is the **Muon LR** — i.e. the LR applied to 2-D hidden matrices, the bulk of the model's params. The AdamW LR (embeddings, biases, norm scales) follows the same schedule shape but starts at `_DEFAULT_BASE_LEARNING_RATE` and is auto-scaled by `sqrt(effective_global_batch / lr_reference_global_batch)`; it is logged to MLflow but **not** to `metrics.jsonl`. When you compare `learning_rate` across experiments, you're comparing Muon trajectories. Tuning `_DEFAULT_BASE_LEARNING_RATE` will not visibly move this field. |
| `tokens_seen` | Cumulative tokens processed |
| `epoch` | Fraction of dataset seen |


### Read final val_loss from logs

`metrics.jsonl` carries training loss but NOT eval loss. The post-train final eval lands in `/fsx/runs/<RUN_NAME>/logs/out` as one line per rank:

```
 0: {'eval_loss': '5.795', 'eval_runtime': '93.1', 'eval_samples_per_second': '...', 'eval_steps_per_second': '...', 'epoch': '1'}
 1: {'eval_loss': '5.795', ...}
 ...
```

If `_DEFAULT_EVAL_INTERVAL_MULTIPLIER` is small enough that intermediate evals fire (the autoresearch defaults set it high enough that they don't), the post-train final eval is just the last block. Use `tac` to read in reverse and get only the latest block — this also sidesteps SSM output truncation (see the `hyperpod-access` skill for SSM gotchas):

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'tac /fsx/runs/<RUN_NAME>/logs/out' \
  | grep -m 1 eval_loss
```

The first match gives the final post-train eval. Pull `eval_loss` from the line with `awk` / `grep -oE` locally on the workstation.

### Analyze the experiment results

With the autoresearch defaults, only the post-train final eval fires (one `val_loss` per run) — there's no eval-loss curve to compare across the run. Mid-run analysis must come from the **training** loss in `metrics.jsonl` (one entry every `_DEFAULT_LOG_INTERVAL = 100` steps); reserve `val_loss` for the keep/discard decision.

- **Convergence speed**: compare how quickly `loss` drops in early steps (e.g. step 100-300 in `metrics.jsonl`) across experiments. Faster early convergence often predicts better final val/loss.
- **Stability**: flag experiments where `grad_norm` spikes or oscillates. Unstable gradients suggest the learning rate or optimizer params need adjustment.
- **Train-val gap**: a large gap between `final_train_loss` and `val_loss` suggests overfitting. Consider increasing dropout, weight decay, or reducing model capacity.
- **Plateau detection**: if `loss` flattens well before step 1000, the learning rate may be too low or the schedule too conservative. Try higher LR or more aggressive warmup. Note: `learning_rate` in `metrics.jsonl` is the Muon LR (see Read metrics.jsonl above); tuning `_DEFAULT_BASE_LEARNING_RATE` won't show up there, so to verify an AdamW-LR experiment landed, check the MLflow `train/adamw_decay_lr` curve or compute `_DEFAULT_BASE_LEARNING_RATE * sqrt(total_gpus / 8)` by hand against expectation.

If you want to cancel a job early based on training loss variance, use `scancel <job_id>` via the hyperpod-access skill. Log as `crash` in `journal.md`.

## Controller Command Gotchas

See the `hyperpod-access` skill for SSM gotchas (comma/quote escaping, output truncation, no shell features in `--command`).

## Hypothesis-Driven Exploration

The loop is hypothesis-driven, not a flat parameter sweep. You iterate on a hypothesis until it is accepted or rejected, but you may pivot to a new one at any time if the evidence clearly points elsewhere. This keeps the overnight transcript coherent — the goal is "we tested 6 hypotheses, kept 3" not "we ran 18 experiments."

All autoresearch state lives in a single file: **`journal.md`** at the repo root, untracked. It has two parts:

- **`## Results`** — a Markdown table at the top, one row per experiment. The flat index — for grep/query/scan across the whole loop.
- **`## Hypotheses`** — below the table. One `### H<N>: <title>` subsection per hypothesis (appended chronologically) with full reasoning, references, and per-attempt observations.

The agent updates both parts of the same file after every experiment. The user reads it end-to-end to understand the loop's thinking.

### Layout (`journal.md`)

The file starts with a short header. Then the Results table. Then the Hypotheses section with one `### H<N>` block per hypothesis. Use these exact headings — `grep` and the loop rely on them:

````markdown
# Autoresearch — autoresearch-decoder branch

To find the open hypothesis: `grep -n "Status: open" journal.md`.
To find open hypotheses: `grep -n "Status: open" journal.md`.

## Results

| commit | run_name | val_loss | final_train_loss | params_M | wall_clock_min | status | description | reasoning | hypothesis_id | attempt | next_action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a1b2c3d | autoresearch-decoder-20260428-072502Z | 0.8523 | 2.8367 | 98.5 | 62 | keep | baseline config as-is | n/a | BASELINE | 1/1 | pivot |
| b2c3d4e | autoresearch-decoder-20260428-083804Z | 0.8401 | 2.7102 | 98.5 | 58 | keep | base_lr 3e-4 -> 6e-4 | Baseline grad_norm was stable at ~1.0 throughout and loss was still falling at step 1000 with no plateau. Train-val gap of 0.08 shows no overfit. Both signals point to an undertuned LR with headroom to push higher. | H01 | 1/5 | iterate |
| c3d4e5f | autoresearch-decoder-20260428-095102Z | 0.8201 | 2.6800 | 98.5 | 61 | keep | base_lr 6e-4 -> 9e-4 | Attempt 1 improved val_loss to 0.8401 but fell short of the 0.84 threshold. Loss curve was steeper in early steps (100-300) than baseline, confirming LR headroom. Pushing to 9e-4 to test the upper end of the predicted regime. | H01 | 2/5 | accept |
| d4e5f6g | autoresearch-decoder-20260428-110000Z | 0.0000 | 0.0000 | 0.0 | 0 | crash | max_seq_len 16384 -> 32768 (no per_rank_token_budget bump) | Longer context windows expose more long-range token dependencies and have improved val_loss at similar scales in the literature. Baseline used 16384; doubling to 32768 tests whether the data has structure at that range. Forgot to bump per_rank_token_budget to match — OOM on first step. | H02 | 1/5 | reject |

## Hypotheses

### H01: base_lr undertuned at 3e-4

- Status: accepted
- Opened: 2026-04-28
- Closed: 2026-04-28
- Attempts: 2/5

#### Claim

A falsifiable claim about model behavior. 1–3 sentences. Name the parameter,
mechanism, or interaction — not just the change.

#### Rationale

Multi-paragraph reasoning. Cite **specific prior runs** (commit hash + RUN_NAME),
specific lines/steps in `metrics.jsonl`, training-curve observations (when did loss
plateau? where did grad_norm spike?), the train-val gap, and any external research
informing the hypothesis. Be thorough — this is the lab journal.

Example:

> Signals from baseline (a1b2c3d, autoresearch-decoder-20260428-072502Z):
> - `grad_norm` stable at ~1.0 through training, no spikes
> - train-val gap of 0.08 (0.6335 train, 0.7309 val) — no overfit signal
> - training loss still decreasing at step 1000 (no plateau)
>
> This pattern matches the canonical "undertuned LR" signature [1]. Karpathy's
> nanoGPT runs at similar scale used `base_lr = 6e-4` as the sweet spot, so 3e-4
> is plausibly conservative.

#### Prediction

The criterion that would accept (prediction met) or reject (prediction missed) the
hypothesis. Must be checkable from `val_loss` (or `final_train_loss` + `val_loss`)
alone — no fuzzy criteria.

> val_loss < 0.84 by step 1000 with `_DEFAULT_BASE_LEARNING_RATE` raised to 6e-4
> or 9e-4. Reject if both values give val_loss ≥ 0.84.

#### References

Optional. Papers, blog posts, internal docs, prior commits/PRs — anything that
informed the rationale. Inline links or a numbered list — your call.

1. Karpathy, *makemore* lecture series — LR undertuning signature
2. nanoGPT repo — `train_gpt.py` LR settings at similar scale
3. PR #88 — earlier LR sweep on encoder objective

#### Attempts

##### Attempt 1/5 — `iterate` — commit b2c3d4e — run autoresearch-decoder-20260428-083804Z

- Change: `_DEFAULT_BASE_LEARNING_RATE` 3e-4 → 6e-4
- Result: val_loss = 0.8401, final_train_loss = 2.7102
- Observations: train-val gap unchanged at ~0.07; loss curve steeper in first
  300 steps, suggesting more headroom. Improvement is real but didn't hit the
  0.84 threshold for accept — pushing higher.

##### Attempt 2/5 — `accept` — commit c3d4e5f — run autoresearch-decoder-20260428-095102Z

- Change: `_DEFAULT_BASE_LEARNING_RATE` 6e-4 → 9e-4
- Result: val_loss = 0.8201, final_train_loss = 2.6800
- Observations: prediction met (< 0.84). grad_norm peaks slightly higher (~1.3)
  but stable; no spikes. Closing as accepted; new branch HEAD.

### H02: larger context cuts loss

- Status: rejected
- Opened: 2026-04-29
- Closed: 2026-04-29
- Attempts: 1/5

#### Claim
[…]
````

Contracts you must preserve so `grep`/the loop can find things:

- Section headings: `## Results`, `## Hypotheses`, `### H<N>: <title>`, `#### Claim`, `#### Rationale`, `#### Prediction`, `#### References`, `#### Attempts`, `##### Attempt <N>/5 — …`. Do not rename.
- Metadata bullets under each `### H<N>` heading: `Status:`, `Opened:`, `Closed:`, `Attempts:` exactly as shown.
- Results table column order and header row stay fixed (see Logging Results).
- No literal `|` characters inside any table cell — escape as `\|` or rephrase, otherwise the row breaks.

### Hypothesis lifecycle

A hypothesis is **open** when its `### H<N>` section has `Status: open` — that is the sole authoritative signal, and `grep -n "Status: open" journal.md` is its check. Closing means editing the section's metadata to `Status: accepted` or `Status: rejected`, filling in the `Closed:` date, AND writing a closing Results-table row with the matching `next_action` — both edits land in the same loop iteration (see Logging Results), so the table and the section never disagree once a run completes. When pivoting away from an open hypothesis, close it as `rejected` before opening the new one.

`next_action` values (set per-attempt in the Results-table row):

- `iterate` — hypothesis still open. Refine the change (different value, narrower window) within the same id; next attempt += 1.
- `accept` — prediction met. Close the hypothesis section (set `Status: accepted`); the winning commit becomes the new branch HEAD.
- `reject` — prediction missed at attempt 5 (mandatory close, see Picking the next experiment #3), or a fundamental crash that won't fix (`crash + reject` in the matrix). Close the hypothesis section (set `Status: rejected`); reset to prior best. The 5-attempt cap is the discipline — no early-reject escape hatch on `discard`.
- `pivot` — used on the `BASELINE` row only. BASELINE has no `### H<N>` section, so there is nothing to close; the next experiment opens H01. Hypotheses themselves never use `pivot` — they ride out attempts until `accept` or `reject`.

### Picking the next experiment

1. If a hypothesis is open and the evidence still supports it, continue iterating: refine the same change axis — pick a different value within the predicted regime, or narrow the window. Do not bundle in unrelated changes; each attempt should test the same prediction. Append a new `##### Attempt N/5` subsection under the hypothesis section's `#### Attempts` heading.
2. To open a new hypothesis (whether or not one is currently open): if an open hypothesis exists, close it as `rejected` first, then append a new `### H<next-N>` section to `journal.md` with all required headings (Claim, Rationale, Prediction, References, Attempts) and metadata bullets (`Status: open`, etc.) **before any code edits**. Read closed sections in `journal.md` for prior conclusions first; the Experiment Ideas section is a fallback menu.
3. Forced reject at attempt 5: if a hypothesis reaches attempt 5 without acceptance, set `next_action = reject` in the closing Results-table row and `Status: rejected` in the matching `### H<N>` section. Do not extend past 5.

### Pre-flight logging

When you pick an experiment in loop step 2 — **before any code edits** — print this block to the conversation. It's a transcript-friendly summary; full reasoning lives in `journal.md` under the matching `### H<N>` section:

```
=== HYPOTHESIS H03 (attempt 2/5) ===
Journal:    journal.md → ### H03
Claim:      muon_lr undertuned at 0.02
Prediction: val_loss < 0.70 if muon_lr raised to 0.03
Change:     _DEFAULT_MUON_LR 0.02 -> 0.03
====================================
```

For the `BASELINE` row (no journal section needed):

```
=== BASELINE =======================
Journal:    n/a
Claim:      n/a (establishing baseline)
Prediction: n/a
Change:     baseline config as-is
====================================
```

The greppable contract is the line prefix `=== HYPOTHESIS ` or `=== BASELINE `; the trailing `===` and the closing fence line are decorative — match widths if you can, but the prefix is what readers and tooling key off.

## Logging Results

When an experiment is done, update `journal.md` in two places:

1. **Append a row to the `## Results` Markdown table** (the flat index — see column spec below).
2. **Append a `##### Attempt N/5` subsection** under the matching `### H<N>` hypothesis's `#### Attempts` heading with the change, raw metrics, and observations. For the `BASELINE` row, no hypothesis section needs updating.

Both edits land in the same file. Do not separate them across runs — the table row and the attempt subsection must reference the same commit and `RUN_NAME`.

### `## Results` table — 12 columns

| Column | Description |
|---|---|
| `commit` | git commit hash (short, 7 chars). |
| `run_name` | unique `RUN_NAME` from `hp-stage-code` (e.g. `autoresearch-decoder-20260428-072502Z`); ties the row back to `/fsx/runs/<RUN_NAME>/` artifacts. |
| `val_loss` | val/loss achieved (e.g. 0.8523). Use `0.0000` for crashes / cancelled jobs. |
| `final_train_loss` | last training loss from `metrics.jsonl` (e.g. 2.8367). Use `0.0000` for crashes. |
| `params_M` | parameter count in millions, round to .1f (e.g. 94.2). Use `0.0` for crashes. |
| `wall_clock_min` | wall-clock minutes end-to-end. Read from `tools/wait_for_hyperpod_job.sh`'s `elapsed_min=N` exit line when the helper exits 0/1, or compute from `train_runtime` (seconds) in `logs/out` + ~2 min for post-train eval. Use `0` for crashes that didn't start training. |
| `status` | `keep`, `discard`, or `crash`. Per-experiment outcome (does this commit advance the branch?). |
| `description` | short text of *what* this experiment changed (e.g. `muon_lr 0.02 -> 0.03`). No literal `|` characters — escape as `\|` or rephrase. |
| `reasoning` | short blurb (2–4 sentences): *why* this attempt was run — the signals, observations, or prior results that motivated it. Include specific metrics or curve behavior where relevant (e.g. step where loss plateaued, grad_norm spikes, train-val gap). No literal `|` characters. Use `n/a` for the BASELINE row. |
| `hypothesis_id` | `H01`, `H02`, … or `BASELINE` for the very first row. Matches the heading `### H01: <title>`. |
| `attempt` | `1/5`, `2/5`, … `5/5`. Use `1/1` for `BASELINE`. Same value as the `##### Attempt N/5` subsection that gets appended to the journal. |
| `next_action` | `iterate` / `accept` / `reject` / `pivot`. Drives the next loop iteration. Mirrored by the hypothesis section's `Status:` metadata when it closes. |

### `status` × `next_action`

Per-experiment `status` and per-hypothesis `next_action` are orthogonal. Common combinations:

| status | next_action | When |
|---|---|---|
| keep | accept | Lower val/loss AND prediction met → hypothesis confirmed, advance branch, close. |
| keep | iterate | Lower val/loss but prediction not yet hit → advance branch, refine within same hypothesis. |
| discard | iterate | Equal/worse val/loss, attempt < 5 → reset, refine within same hypothesis. |
| discard | reject | Equal/worse val/loss, attempt = 5 → reset, hypothesis dead, pivot. |
| crash | iterate | Trivial fix (typo, OOM at edge) → re-run with fix, same hypothesis. |
| crash | reject | Fundamental break → close hypothesis, pivot. |
| keep | pivot | `BASELINE` row only — no hypothesis to accept/reject; first real experiment opens H01. |

Tracking `wall_clock_min` lets you spot creeping cluster contention or step-time regressions across experiments — don't skip it.

Do NOT commit `journal.md`. Leave it untracked by git.

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch-decoder` or `autoresearch-encoder`).

**LOOP FOREVER:**

1. **Review state** — read `journal.md`. Scan the `## Results` table for current best val/loss and what's been tried; read closed `### H<N>` sections for prior conclusions. Determine whether a hypothesis is currently open (`grep -n "Status: open" journal.md`). Check `git log` to confirm you are on the right commit.
2. **Pick an experiment** — see Hypothesis-Driven Exploration above:
   - If a hypothesis is open and the evidence still supports it, iterate on it (same `hypothesis_id`, attempt += 1).
   - Otherwise (or if pivoting early), close the current hypothesis as `rejected` then open a new one with a one-line claim and a falsifiable prediction. Use Experiment Ideas as a menu.
   - Print the pre-flight `=== HYPOTHESIS ===` block to the conversation before any code edits.
3. **Modify in-scope files** — edit configs and/or model code to implement the experiment.
4. **Validate locally**:
   - Run `make test` to catch syntax errors, import errors, and regressions.
   - Run `PYTHONPATH=code:code/src uv run python tools/count_params.py --vocab-size 16384 --budget 100 --tolerance 15` to verify parameter count. Do not change `--vocab-size` or `--tolerance` (see Parameter Budget). If outside budget, adjust before proceeding.
5. **Git commit** — descriptive message of what changed (prefix commit with autoresearch:). Do NOT commit `journal.md`.
6. **Submit to HyperPod** — use the `submit-hyperpod-training-job` skill with the stored `AWS_PROFILE` and `ENV`. Each submission must 
   use a unique `RUN_NAME_PREFIX` which is shared across experiments then you will get a `RUN_NAME` (let `hp-stage-code` generate it).
7. **Wait for completion** — run `tools/wait_for_hyperpod_job.sh <JOB_ID> --aws-profile="$AWS_PROFILE"` (parse `<JOB_ID>` from the `Submitted batch job <id>` line emitted by `hp-submit`). If SSM truncated the submit output before that line was visible (SSM output truncates around 7-13 lines — see `hyperpod-access` skill), recover the job id by querying squeue for the run name you just used, e.g. `... --command 'scontrol show job -o' | grep -oE 'JobId=[0-9]+ JobName=<RUN_NAME>'`. Default budget is 120 min; bump `--timeout-min=<larger>` if you've raised `MAX_SEQ_LEN`. The helper polls `squeue` on the controller and exits `0` on COMPLETED, `1` on FAILED/CANCELLED/NODE_FAIL/TIMEOUT/OOM, `2` on its own wall-clock budget exceeded (it does NOT auto-scancel). Run via Bash `run_in_background: true` and let the harness's completion notification surface the result — no separate Monitor needed. On exit `0` or `1`, read `metrics.jsonl` from the controller (see Retrieving Results) to capture val/loss + the training loss curve. On exit `2`, decide whether to wait longer or `scancel` and log as a crash.
8. **Record results** — update `journal.md` in two places: append a row to the `## Results` table (all 12 columns, including `reasoning` / `hypothesis_id` / `attempt` / `next_action`), and append a `##### Attempt N/5 — <next_action> — commit <hash> — run <RUN_NAME>` subsection under the matching `### H<N>` hypothesis's `#### Attempts` heading with the change, raw metrics, and observations from the training curve. Review the training curve to inform the next experiment choice.
9. **Keep or discard, and update the hypothesis** — `status` (per-experiment) and `next_action` (per-hypothesis) together drive both the branch and the loop:
   - Lower val/loss than current best AND prediction met → `status=keep`, `next_action=accept`. Advance branch, close hypothesis.
   - Lower val/loss but prediction not yet hit → `status=keep`, `next_action=iterate`. Advance branch, refine within same hypothesis (next attempt).
   - Equal/worse val/loss, attempt < 5 → `status=discard`, `next_action=iterate`. `git reset --hard` to prior best, refine within hypothesis.
   - Equal/worse val/loss, attempt = 5 → `status=discard`, `next_action=reject`. Reset, close hypothesis, pivot to a new one.
   - Crash, trivial fix → `status=crash`, `next_action=iterate`. Same hypothesis, fix and re-run.
   - Crash, fundamental break → `status=crash`, `next_action=reject`. Close hypothesis, pivot.

## Experiment Ideas

These map to the user's experiment categories. Not a rigid agenda — a reference when deciding what to try next.

### 1. Learning rate and schedule sweep
- `base_learning_rate`: 1e-4 / 3e-4 (baseline) / 6e-4 / 1e-3
- `lr_stable_ratio`: 0.7 / 0.8 / 0.9 (baseline) / 0.95
- `lr_min_ratio`: 0.0 / 0.05 / 0.1 (baseline) / 0.2
- `lr_decay_style`: `linear` (baseline) vs `cosine` (often better for short runs)
- Primary metric: train loss + val loss curves

### 2. Muon optimizer hyperparams
- `muon_lr`: 0.01 / 0.02 (baseline) / 0.04
- `muon_momentum`: 0.90 / 0.95 (baseline) / 0.98
- `muon_ns_steps`: 3 / 5 (baseline) / 7
- Primary metric: train loss convergence speed + final val loss

### 3. Chunk size (max_seq_len)
- `max_seq_len`: 4096 / 8192 / 16384 (baseline) — see Cluster note above before raising it.
- The `per_rank_token_budget >= max_seq_len` invariant is enforced at config-load time.

### 4. Gradient accumulation (effective batch size)
- `gradient_accumulation_steps`: 1 (baseline) / 2 / 4 / 8
- Effective global batch = `gradient_accumulation_steps * total_gpus` packs.
- LR is auto-scaled by `sqrt(effective_global_batch / lr_reference_global_batch)`.

### 5. Regularization
- `weight_decay`: 0.01 / 0.05 / 0.1 (baseline) / 0.2
- `max_grad_norm`: 0.5 / 1.0 (baseline) / 2.0

### 6. GQA group size
- `num_kv_heads`: varies with `num_heads / group_size`
- Baseline is group_size=4 → `num_kv_heads = 3` (with 12 heads)
- Try MHA: `num_kv_heads = num_heads` (no grouping)

### 7. Masking probability (encoder branch only)
- `mlm_probability`: 0.10 / 0.15 (baseline) / 0.20 / 0.30 / 0.40
- BERT used 0.15; recent work (RoBERTa, T5 "span-corruption", "Should You Mask 15%?" — Wettig et al. 2023) shows higher rates (0.30–0.40) can improve representation quality on bigger models. Worth sweeping at this scale.
- Edit in `code/scripts/configs/yaml/profile/encoder.yaml` (not `config.yaml` — keep the change scoped to the encoder objective). Validator requires `0.0 < mlm_probability < 1.0`.
- Effects to watch: higher masking → fewer surviving context tokens per chunk → harder denoising task → typically slower early-step convergence but can land at lower final val_loss. Train-val gap should also shrink as the task gets harder.
- Decoder branch: do NOT sweep this — `mlm_probability` is ignored when `model_type=causal_lm`.

### Additional ideas
- Positional encoding: none (baseline) vs RoPE, theta tuning
- Depth vs width: redistribute params between `num_layers` and `model_dim`
- FFN ratio: vary `ffn_dim` relative to `model_dim`
- Dropout: 0.0 / 0.05 / 0.1 (baseline) / 0.2
- Activation functions: SwiGLU (baseline) vs GELU
- Combinations: stack winning changes together

## Error Handling

### Job Timeout
The default budget is 120 minutes (`tools/wait_for_hyperpod_job.sh`). When the helper exits `2` (wall-clock budget exceeded), pick one:
1. Re-run the helper with `--timeout-min=<larger>` if the job is still making progress and you want to wait longer.
2. `scancel <job_id>` on the controller and log as `crash` in `journal.md`.

Do not let runs sit in the queue indefinitely.

### Missing Results
Job finished but metrics.jsonl is empty or missing val_loss — treat as a crash. Check `/fsx/runs/<RUN_NAME>/logs/out` for error details.

### Crash Handling
- Trivial fix (typo, import error, shape mismatch caught locally) — fix and re-run, same experiment.
- Fundamentally broken (OOM, numerical instability) — log as crash, discard, move on.
- 3+ consecutive crashes — revert to last known-good commit, try something completely different.

### Local Validation Failures
- `make test` fails — fix before submitting. Do not submit code that fails tests.
- Param count outside ±15% of 100M (with the real tokenizer vocab) — adjust config, do not submit.

### Slurm Submission Failures
- Job fails to queue — check logs, fix if obvious infrastructure issue, otherwise skip and note.
- Do NOT get stuck debugging infrastructure. If you cannot resolve a submission issue within a few minutes, log it and move on to the next experiment idea.
- Do not run any docker, enroot, or sudo commands on the cluster controller.

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue.
Do NOT ask "should I keep going?" or "is this a good stopping point?" The human might be asleep, or gone from the computer and expects 
you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read 
the in-scope files for new angles, try combinations of previous near-misses, try more radical architectural changes, revisit discarded 
ideas from a different angle. The loop runs until the human interrupts you, period.

As a rough estimate: each experiment takes ~30 minutes (stage, submit, train 1000 steps, post-train eval, poll results). Over an 8-hour overnight session you can complete roughly 16 experiments. The user wakes up to a `journal.md` full of results and reasoning, and an advanced branch with the best configuration found.

## Red Flags

- The agent submits a job without running `make test` first
- The agent exceeds the ±15% parameter budget and submits anyway
- The agent modifies files outside the four in-scope files
- The agent asks "should I continue?" or pauses for confirmation during the loop
- The agent commits `journal.md` to git
- The agent skips the baseline and starts with experimental changes
- The agent changes `total_training_steps` from 1000
- The agent installs new dependencies or modifies `pyproject.toml`
- The agent spends more than a few minutes debugging infrastructure (Slurm, S3, controller access)
- The agent does not update `journal.md` (both the `## Results` table row and the matching `##### Attempt N/5` subsection) after an experiment completes
- The agent reuses a `RUN_NAME` from a previous experiment instead of letting `hp-stage-code` generate a new one
- The agent submits an experiment without printing the `=== HYPOTHESIS ===` pre-flight block to the conversation
- The agent has more than one open hypothesis at a time, or extends a hypothesis past attempt 5 instead of marking it `reject`
- The agent opens a new hypothesis without first closing (`accept` or `reject`) any currently open hypothesis
