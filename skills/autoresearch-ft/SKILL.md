---
name: autoresearch-ft
description: Autonomous AI research loop for fine-tuning (FT). The agent iterates on FT training hyperparameters and PEFT configuration against a fixed pretrained backbone, submitting experiments to HyperPod via `hp-submit-finetune`, evaluating results via `eval_auc`, and keeping or discarding changes — all without human intervention. Use when the user wants to run autonomous hyperparameter exploration on top of an existing pretrained EFM backbone. The search space is small by design — the architecture is frozen by the loaded backbone, the head is locked at linear, and only LoRA / DoRA PEFT is in scope.
---

# autoresearch-ft

This is an autonomous research skill for the **fine-tuning (FT)** side of the efm-core project. You are the researcher. You modify FT training hyperparameters and FT PEFT configuration, submit each experiment to HyperPod via the `submit-hyperpod-training-job` skill with `MODE=finetune` (see Submission below), evaluate results via `eval_auc`, and keep or discard changes based on best `eval_auc`. Each experiment is a git commit on a dedicated branch. If it improves best `eval_auc`, you advance the branch. If it doesn't, you reset. You run indefinitely until the human stops you.

Two scoping properties that define this loop:

- **Small search space**: architecture is fixed by the loaded pretrained backbone, so `model_dim` / `num_layers` / `num_heads` / etc. are off-limits. The head is locked at `linear`. Full FT is out of scope. Only training-side knobs and LoRA / DoRA PEFT configuration are tunable.
- **Optimization target is `eval_auc`** (greater-is-better), not `val/loss`. The signal is `trainer.state.best_metric` — the peak `eval_auc` observed across mid-run evals — which is what `load_best_model_at_end=True` picks for the exported FT artifact.

## Domain context (read before hypothesizing)

This is **not traditional natural-language fine-tuning**. The pretrained EFM backbone learned a latent-space representation of a user's *financial profile* (bank-transaction sequences as token streams). FT trains a small classification head on top of this backbone for a downstream binary-classification task. The "language" is still spending and income behavior; the FT loop is about turning the pretrained representation into a useful classifier with as few examples and as little training as possible.

## Pre-requisites

Before starting the very first run, ask the user to provide:

- `AWS_PROFILE` and `ENV` — needed by the hyperpod-access skill.
- **Pretrained backbone name** — read from `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in `code/scripts/finetune/configs/finetune_training_config.py`. This constant is human-maintained: a teammate hand-edits it to point at the pretrain run whose best-checkpoint artifact bundle should seed FT. The autoresearch loop does **not** modify it. **Before starting the loop**, read the file and verify the constant is non-empty; if it's `""`, halt and ask the user to set it in the config (and commit) before proceeding — autoresearch cannot proceed without a backbone target. Capture the resolved value for the journal `Backbone:` line.

Checkout a new branch named `autoresearch-ft` (or `autoresearch-ft-<short-alias>` if the user supplied an alias for a specific backbone). One branch + one `journal.md` covers exactly one backbone target. When the user wants to evaluate a second backbone, the human edits `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` to the new value and cuts a fresh branch with a fresh journal — never reuse a journal across backbones, since `eval_auc` numbers are not comparable across backbones and mixing them in one Results table breaks the keep/discard signal.

There is no encoder/decoder toggle — the FT objective is fixed by the loaded backbone (binary classification head + AUC metric).

## Setup

To set up a new experiment run:

1. **Use the following run name prefix**: `autoresearch-ft` (or `autoresearch-ft-<short-alias>` if the branch carries a backbone alias — keep the prefix and the branch name in lockstep). When you submit a job you will get a unique `RUN_NAME`. That serves as the single identifier for the experiment.
2. **Read in-scope files** for full context before editing or submitting:
   - `@code/scripts/finetune/configs/finetune_training_config.py`
   - `@code/scripts/finetune/configs/finetune_model_config.py`
   - `@code/scripts/finetune/entrypoint.py` — read-only reference; do not edit.
   - `@hyperpod/Makefile` — read-only; the `hp-submit-finetune` target is your submission entrypoint.
   - `@hyperpod/finetune.sbatch` — read-only; the source of truth for the Slurm shape (`--nodes`, `--ntasks-per-node`, `--gpus-per-task`) and partition. Read this file at branch setup to capture the current cluster shape — the autoresearch loop derives `total_gpus = nodes × ntasks-per-node` from these directives at submit time, and the FT runtime sqrt-scales LR against `lr_reference_global_batch=8` accordingly. Do not hardcode the shape or the partition anywhere else; if infra changes either, this is the file that moves.
3. **Verify baseline config**: per "Setting up the autoresearch branch" below, confirm the on-disk autoresearch baseline is in place. The current production defaults already match the autoresearch baseline (`_DEFAULT_TOTAL_TRAINING_STEPS = 600` in `finetune_training_config.py`, `_DEFAULT_PEFT_METHOD = "lora"` in `finetune_model_config.py`), so no edits should be needed. If either has drifted, re-set it and commit as the first change on the branch. Optionally set `_DEFAULT_LR_DECAY_STYLE = "cosine"` in the training config (commit alongside any verification edit). The 600-step budget keeps wall-clock short enough to fit many experiments per session; locking PEFT to LoRA puts the loop in lightweight-FT territory (full FT is out of scope). Do **not** edit `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` — that constant is human-maintained and was verified non-empty in Pre-requisites. Leave every other `_DEFAULT_*` constant at its checked-in value for the BASELINE run.
4. **Initialize journal.md**: create `journal.md` in the repo root with the layout shown in Hypothesis-Driven Exploration: a short header (`# Autoresearch — autoresearch-ft branch`) followed by a `Backbone:` line that records the resolved value of `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` for traceability, a `## Results` heading + the 11-column Markdown table header (header row + separator row, no data yet), and an empty `## Hypotheses` heading. Do NOT commit this file — it stays untracked.
5. **Establish baseline**: submit the baseline config as the first experiment per the Submission section below (which delegates to the `submit-hyperpod-training-job` skill with `MODE=finetune`). Print the `=== BASELINE ===` pre-flight block before submission (no `### H<N>` section is needed for the baseline). Wait for completion. Append the baseline row to the `## Results` table in `journal.md` with `hypothesis_id=BASELINE`, `attempt=1/1`, `next_action=pivot`.
6. **Confirm and go**: once the baseline is recorded, confirm setup looks good and begin the autonomous loop.

## Do not over-index on the on-disk defaults

The `_DEFAULT_*` constants in `finetune_training_config.py` and `finetune_model_config.py` are *current production defaults*, not *best autoresearch state*. Treat them as a starting point and revisit every knob each session.

## Baseline Configuration

The `_DEFAULT_*` constants in `code/scripts/finetune/configs/finetune_training_config.py` and `code/scripts/finetune/configs/finetune_model_config.py` are the source of truth for what a freshly-launched FT job sees. Read them to know the current state.

### Tunable parameters (agent CAN change)

In `finetune_training_config.py`:

- LR / schedule: `_DEFAULT_BASE_LEARNING_RATE`, `_DEFAULT_LR_DECAY_STYLE`, `_DEFAULT_LR_STABLE_RATIO`, `_DEFAULT_LR_MIN_RATIO`, `_DEFAULT_WARMUP_STEPS`
- Optimizer: `_DEFAULT_OPTIMIZER` ∈ `{"muon_adamw", "adamw"}` — paired with `_DEFAULT_PEFT_METHOD`, see "PEFT-optimizer pairing" below. Muon hyperparameters apply only when `optimizer == "muon_adamw"`: `_DEFAULT_MUON_LR`, `_DEFAULT_MUON_MOMENTUM`, `_DEFAULT_MUON_NS_STEPS`, `_DEFAULT_MUON_WEIGHT_DECAY`, `_DEFAULT_MUON_NESTEROV`. AdamW-side hyperparameters apply to both: `_DEFAULT_WEIGHT_DECAY`, `_DEFAULT_ADAMW_BETA1`/`BETA2`.
- Effective batch: `_DEFAULT_GRADIENT_ACCUMULATION_STEPS`

In `finetune_model_config.py`:

- PEFT: `_DEFAULT_PEFT_METHOD` (`"lora"` / `"dora"` only — see Fixed parameters below for why `full` is out of scope), `_DEFAULT_LORA_R`, `_DEFAULT_LORA_ALPHA`, `_DEFAULT_LORA_DROPOUT`, `_DEFAULT_LORA_TARGET_MODULES`
- Focal-loss hyperparameters: `_DEFAULT_FOCAL_GAMMA` (focusing parameter, ≥ 0) and `_DEFAULT_FOCAL_ALPHA` (positive-class weight, in `[0, 1]`). `_DEFAULT_LOSS_TYPE` itself stays locked (see Fixed parameters below).

### Fixed parameters (do NOT change)

- `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` — human-maintained constant in `finetune_training_config.py`, **out of scope for the autoresearch loop to edit**. Switching the value mid-loop breaks comparability and silently invalidates the journal's `eval_auc` numbers; if a different backbone is needed, the human edits the constant and a fresh branch + fresh journal is cut.
- `_DEFAULT_TOTAL_TRAINING_STEPS` — fixed at **600** for the loop (set once at branch setup per Setup step 3). Locking the budget keeps wall-clock and eval cadence comparable across experiments.
- `_DEFAULT_LOG_INTERVAL` (100), `_DEFAULT_EVAL_INTERVAL_MULTIPLIER` (2), `_DEFAULT_CHECKPOINT_INTERVAL_MULTIPLIER` (2) — eval/log cadence stays consistent across experiments. With 600 steps, eval fires at step 200, 400, and 600 (3 evals; no post-train test eval — the FT dataset does not carry a test split). `save_steps` matches `eval_steps` (both 200) so HF Trainer's `load_best_model_at_end` invariant holds and a checkpoint exists at every eval point — `save_total_limit=3` retains all three.
- `_DEFAULT_EARLY_STOPPING_PATIENCE` (3) — left at production default. Structurally inactive at the 600-step budget (only 3 evals fire, can't trigger a 3-patience early stop), so not a tunable axis.
- `_DEFAULT_MAX_GRAD_NORM` (1.0) — left at production default. Low-leverage on top of a frozen backbone (gradient surface is much smoother than from-scratch pretrain), not worth a hypothesis slot.
- `_DEFAULT_PER_RANK_TOKEN_BUDGET` (81_920) — matches the pretrain budget; memory-tuned for H100. Lowering it reduces packing density (fewer users per pack); raising it risks OOM if the model + activations don't fit. Leave at the production default.
- `_DEFAULT_PRECISION` (`bf16`).
- `_DEFAULT_LR_SCALING_MODE`, `_DEFAULT_LR_REFERENCE_GLOBAL_BATCH`, `_DEFAULT_ADAMW_BETA1`, `_DEFAULT_ADAMW_BETA2`.
- `_DEFAULT_EVAL_METRIC_FOR_BEST_MODEL` (`eval_auc`) — the keep/discard signal.
- `_DEFAULT_OPTIMIZER` is **paired with `_DEFAULT_PEFT_METHOD`** — see "PEFT-optimizer pairing" below. The validator accepts `"muon_adamw"` and `"adamw"`; LoRA can use either, DoRA must use `"adamw"`. Treat the pair as a single tunable.
- `_DEFAULT_HEAD_TYPE` (`binary_classification`) and `_DEFAULT_HEAD_DEPTH` (`linear`) — locked for this lightweight-FT autoresearch starting point. The 2-layer MLP head and its hidden-dim / dropout / activation fields are intentionally out of scope; revisit only when linear plateaus across the search space.
- `_DEFAULT_PEFT_METHOD` cannot be `"full"` — full FT trains every backbone parameter and is out of scope for the lightweight-FT autoresearch. Tunable values are `"lora"` and `"dora"` only. The production default is already `"lora"`; verify at branch setup (see "Setting up the autoresearch branch" below).
- `_DEFAULT_FREEZE_BACKBONE` (`False`) — only consulted when `peft_method="full"` (which is locked out above), so changing it is a no-op for the autoresearch loop. Leave at the production default.
- `_DEFAULT_LOSS_TYPE` (`"focal"`) — locked. The production FT dataset has heavy positive-class imbalance and the loss type matches that; flipping back to `"bce"` mid-loop would invalidate the journal's `eval_auc` numbers (they'd be measured against a different objective). The focal *hyperparameters* (`_DEFAULT_FOCAL_GAMMA`, `_DEFAULT_FOCAL_ALPHA`) are the tunable axes for loss-side experiments. `EFMFineTuneConfig` validation will reject inconsistent state (e.g. `loss_type="bce"` with `focal_*` set, or `loss_type="focal"` with either `focal_gamma`/`focal_alpha` empty).
- `MAX_SEQ_LEN` (inherited from pretrain `model_config.py`; the FT entrypoint does not override).
- `finetune.sbatch` and `hyperpod/Makefile` are off limits — never change the number of nodes, the partition, or the submission target.

### PEFT-optimizer pairing (paired-config invariant)

`_DEFAULT_PEFT_METHOD` and `_DEFAULT_OPTIMIZER` must be set together — they are not independent axes.

| `_DEFAULT_PEFT_METHOD` | `_DEFAULT_OPTIMIZER` | Why |
|---|---|---|
| `"lora"` | `"muon_adamw"` (default) **or** `"adamw"` | LoRA matrices are routed to Muon (2-D bucket); Newton-Schulz orthogonalization on rectangular `(r, d)` / `(d, r)` matrices produces a stable spectral-norm update that pairs cleanly with LoRA's additive `BA` reparametrization. Plain AdamW also works. |
| `"dora"` | **`"adamw"` (required)** | DoRA reparametrizes each adapted weight as `m * (W + BA) / ‖W + BA‖_col` — a per-column magnitude renormalization. Muon's Newton-Schulz orthogonalization on `A` and `B` produces a unit-spectral-norm update, which then gets renormalized away by the column-norm divisor inside DoRA's forward. Net effect: the model output stays at the base-model value across all training steps and **train + eval loss are flat** the entire run. Plain AdamW skips the orthogonalization step and lets `BA` accumulate normally. |

**When opening any DoRA hypothesis, the change must include both lines as a single commit:**

```python
# code/scripts/finetune/configs/finetune_model_config.py
_DEFAULT_PEFT_METHOD = "dora"        # was "lora"

# code/scripts/finetune/configs/finetune_training_config.py
_DEFAULT_OPTIMIZER = "adamw"         # was "muon_adamw" — REQUIRED for DoRA
```

When closing a DoRA hypothesis (accept/reject) and reverting to LoRA, also revert `_DEFAULT_OPTIMIZER` back to `"muon_adamw"` in the same commit so the next experiment doesn't silently train LoRA under plain AdamW (which is fine but obscures the comparison signal — Muon vs AdamW becomes a confound).

The `_OPTIMIZER_CHOICES` validator in `finetune_training_config.py` accepts both values, but it does **not** enforce the DoRA→AdamW pairing — that's on the agent. A `peft_method="dora"` + `optimizer="muon_adamw"` configuration will validate, submit, and land a flat-loss run that wastes compute. Always co-edit the two constants.

### Setting up the autoresearch branch

Before the very first submission, verify the autoresearch baseline is in place. The current production defaults already match the autoresearch baseline, so this step is normally a no-op:

- `finetune_training_config.py`: `_DEFAULT_TOTAL_TRAINING_STEPS = 600` (already the default).
- `finetune_model_config.py`: `_DEFAULT_PEFT_METHOD = "lora"` (already the default; full FT is out of scope for the lightweight-FT autoresearch). LoRA hyperparameters (`_DEFAULT_LORA_R`, `_DEFAULT_LORA_ALPHA`, `_DEFAULT_LORA_DROPOUT`, `_DEFAULT_LORA_TARGET_MODULES`) keep their checked-in values for the BASELINE row.

If either constant has drifted (e.g. a teammate experimented with a different value and didn't revert), re-set it and commit as the first change on the branch. Otherwise the BASELINE run is just a fresh submission with no preceding edit.

Do **not** edit `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` as part of this setup. That constant is human-maintained — a teammate hand-edits it when the team rolls a new pretrain target. Pre-requisites already verified the current value is non-empty; if you find it empty later, halt the loop and ask the user to set it in the config (the FT runtime would reject an empty value anyway, with an actionable `ValueError` that names the file to edit).

Optional, recommended for short FT runs: `_DEFAULT_LR_DECAY_STYLE = "cosine"` (cosine schedules tend to produce a cleaner endpoint than linear at sub-2k-step budgets). If you take the cosine option, commit it as part of the same baseline commit so the BASELINE row records its result.

There is no encoder/decoder toggle — the FT objective is fixed by the loaded backbone (binary classification head + AUC metric).

### Cluster

The cluster shape is **not hardcoded in this skill** — read it from `hyperpod/finetune.sbatch` at branch setup. Specifically:

- `--nodes=N` and `--ntasks-per-node=M` define the allocation; `total_gpus = N × M` (one task per GPU is the established convention, and `--gpus-per-task=1` in the sbatch enforces it).
- The partition is also set inside the sbatch and tracks infra changes; do not duplicate it.

Once you've read those directives, derive `total_gpus` and use it for everything downstream:

- **LR sqrt-scaling**: `_scaled_learning_rate` in `finetune_training_config.py` scales `_DEFAULT_BASE_LEARNING_RATE` by `sqrt(total_gpus / lr_reference_global_batch)` where `lr_reference_global_batch = 8`. At `total_gpus = 8` the factor is 1.0; at `total_gpus = 48` it is `sqrt(6) ≈ 2.45`. The runtime applies the scaling automatically — your job is just to remember it when reasoning about why the same `_DEFAULT_BASE_LEARNING_RATE` value behaves differently across cluster shapes.
- **Effective global batch**: `gradient_accumulation_steps × total_gpus`. Re-derive whenever the sbatch shape changes.
- **Warmup**: unscaled. `_warmup_steps()` returns the declared `_DEFAULT_WARMUP_STEPS` value verbatim (FT runs are short enough that pretraining's GPU-scaled warmup overshoots).

A baseline FT run at the 600-step autoresearch budget — wall-clock varies with the actual sbatch shape and the FT input pipeline. **Always measure with the BASELINE run** before projecting overnight throughput rather than relying on a number cached in this skill. The default `tools/wait_for_hyperpod_job.sh` budget is 120 min, which is generous for typical FT shapes.

Concurrency on the cluster also depends on the sbatch shape: a small allocation (e.g. 1 node) lets multiple autoresearch-ft jobs run side-by-side; a larger allocation (e.g. 6 nodes) likely consumes the full plan, so concurrent submissions queue rather than parallelize. Check `squeue` to see how a new submission lands. Either way, expect some FSx-Lustre contention variance when multiple `autoresearch-ft-*` jobs are active.

## Parameter Budget

There is no parameter budget for FT autoresearch. The architecture (model_dim, num_layers, num_heads, …) is frozen by the loaded pretrained backbone and the head is locked at `linear` (single `nn.Linear(d, 1)` — kilobytes). The **trainable** parameter count moves only with PEFT additions:

- LoRA / DoRA: `r × (in + out)` per adapted module — typically 0.1-1% of backbone params depending on `_DEFAULT_LORA_R` and `_DEFAULT_LORA_TARGET_MODULES`.

Do not run a per-experiment param-count check; `tools/count_params.py` exists in the repo but is sized for from-scratch pretraining and is not relevant to FT autoresearch. The only memory-related constraint is activation memory at the configured `_DEFAULT_PER_RANK_TOKEN_BUDGET`; if a PEFT combination ever pushes activations over the 80 GiB H100 ceiling, the run will OOM and you log it as a `crash`.

## In-Scope Files

**CAN modify:**

| File | Purpose | Example changes |
|---|---|---|
| `@code/scripts/finetune/configs/finetune_training_config.py` | FT training hyperparams | `_DEFAULT_BASE_LEARNING_RATE`, `_DEFAULT_LR_DECAY_STYLE`, `_DEFAULT_WARMUP_STEPS`, `_DEFAULT_WEIGHT_DECAY`, `_DEFAULT_MUON_LR`, `_DEFAULT_MUON_MOMENTUM`, `_DEFAULT_MUON_NS_STEPS`, `_DEFAULT_GRADIENT_ACCUMULATION_STEPS` |
| `@code/scripts/finetune/configs/finetune_model_config.py` | FT PEFT + focal-loss defaults (head shape locked at linear; full FT out of scope; loss_type locked at focal) | `_DEFAULT_PEFT_METHOD` (lora ↔ dora only), `_DEFAULT_LORA_R`, `_DEFAULT_LORA_ALPHA`, `_DEFAULT_LORA_DROPOUT`, `_DEFAULT_LORA_TARGET_MODULES`, `_DEFAULT_FOCAL_GAMMA`, `_DEFAULT_FOCAL_ALPHA` |

**CANNOT modify:**

| File/Area | Reason |
|---|---|
| `code/src/efm_finetune/` (model, head, data) | FT model definition is a shared contract; autoresearch only tunes config defaults emitted by the launchers. |
| `code/src/efm_core/` | Pretrained backbone shape is frozen by the loaded checkpoint. |
| `code/scripts/configs/` (pretrain configs) | Out of scope for FT. |
| `code/scripts/finetune/initialize/` | Data pipeline must stay stable for comparability. |
| `code/scripts/finetune/entrypoint.py` | Entrypoint orchestration is shared; consume only its emitted config surface. |
| `code/scripts/train/` (trainer, callbacks) | Trainer + callbacks shared with pretrain. |
| `code/jobs/`, `hyperpod/` | Job launcher and cluster infrastructure. |
| `skills/` | Skill definitions. |
| `pyproject.toml` | No new dependencies. |

## Experimentation

**What you CAN do:**

- Modify the two in-scope FT config files: `finetune_training_config.py` and `finetune_model_config.py`.
- Sweep training hyperparameters: LR, weight decay, warmup, schedule, gradient accumulation, optimizer (Muon) hyperparameters.
- Sweep FT PEFT configuration: PEFT method (LoRA ↔ DoRA), LoRA rank/alpha/dropout/target modules.
- Sweep focal-loss hyperparameters: `_DEFAULT_FOCAL_GAMMA` and `_DEFAULT_FOCAL_ALPHA`. `_DEFAULT_LOSS_TYPE` itself stays at `"focal"`.

**What you CANNOT do:**

- Modify any file outside the two in-scope files.
- Install new packages or add dependencies.
- Change the pretrained backbone (`_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` is human-maintained; editing it is out of the autoresearch loop's scope, see Fixed parameters).
- Change architecture (`model_dim`, `num_layers`, `num_heads`, etc. — set by the loaded backbone).
- Change the data pipeline, evaluation harness, trainer, callbacks, or logging.
- Change `_DEFAULT_TOTAL_TRAINING_STEPS` between experiments (locked at 600 for the loop, set once at branch setup).
- Change `_DEFAULT_EVAL_METRIC_FOR_BEST_MODEL` (always `eval_auc`).
- Change `_DEFAULT_HEAD_DEPTH` away from `linear` (locked for the lightweight-FT starting point).
- Set `_DEFAULT_PEFT_METHOD` to `"full"` — full FT is out of scope; only `"lora"` and `"dora"` are tunable values.
- Change `_DEFAULT_LOSS_TYPE` (locked at `"focal"`); flip the loss type would invalidate cross-experiment `eval_auc` comparability. Tune `_DEFAULT_FOCAL_GAMMA` and `_DEFAULT_FOCAL_ALPHA` instead.

The goal is simple: get the highest best `eval_auc`. Step budget is fixed for the loop, so you do not worry about training time — every experiment gets the same compute. Everything within the two in-scope files is fair game. See **Parameter Budget** above for why head + PEFT additions don't need a per-experiment param-count check.

**Simplicity criterion**: all else being equal, simpler is better. A small `eval_auc` improvement that adds ugly complexity (e.g. exotic LoRA target-module sets that don't match standard recipes) is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that is a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

The first run: your very first run should always be to establish the baseline. Submit the baseline config as-is and record the result.

## Submission

Submit each experiment through the `submit-hyperpod-training-job` skill with `MODE=finetune` and `RUN_NAME_PREFIX=autoresearch-ft` (or `autoresearch-ft-<short-alias>` to match the branch name when an alias is in use). The pretrained-backbone name is **not** an argument — that skill reads it from `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in the staged code's `finetune_training_config.py`. The submit skill owns the full stage / fetch / verify / submit flow including the `make hp-submit-finetune` invocation; this skill does not duplicate it.

After submission, capture the `Submitted batch job <id>` line from the submit output. If SSM truncated the line, recover the job id from `squeue` (see Retrieving Results below).

## Retrieving Results

After each experiment completes, read results from the HyperPod controller. FT exposes both a training-loss curve (in `metrics.jsonl`) and per-eval `eval_auc` lines (in `logs/out`); the keep/discard signal is the **max `eval_auc` across the run** (which mirrors `trainer.state.best_metric`).

### Read training-loss curve from metrics.jsonl

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'cat /fsx/runs/<RUN_NAME>/metrics.jsonl'
```

`metrics.jsonl` is written every `_DEFAULT_LOG_INTERVAL` (100) steps. Each line is a JSON object like:

```json
{"step": 100, "loss": 0.62, "grad_norm": 0.97, "learning_rate": 0.02, "tokens_seen": 540856, "epoch": 0.05}
```

Field meanings:

| Field | Description |
|---|---|
| `step` | Training step number. |
| `loss` | Training loss at this step. |
| `grad_norm` | Gradient norm at this step. |
| `learning_rate` | Current LR for `param_groups[0]`. With the default `MuonAdamW` optimizer, group 0 is the **Muon** group, so this is the Muon LR (the LR applied to 2-D hidden matrices, the bulk of the model's params). The AdamW LR (embeddings, biases, norm scales) follows the same schedule shape but starts at `_DEFAULT_BASE_LEARNING_RATE` and is auto-scaled; it is logged to MLflow but **not** to `metrics.jsonl`. So when you compare `learning_rate` across experiments, you're comparing Muon trajectories — tuning `_DEFAULT_BASE_LEARNING_RATE` alone will not visibly move this field. |
| `tokens_seen` | Cumulative tokens processed. |
| `epoch` | Fraction of dataset seen. |

### Read per-eval eval_auc lines from logs/out

`metrics.jsonl` carries training loss but NOT eval metrics. The mid-run evals land in `/fsx/runs/<RUN_NAME>/logs/out` as one line per rank, e.g.:

```
 0: {'eval_loss': '0.512', 'eval_auc': '0.8421', 'eval_runtime': '93.1', ...}
 0: {'eval_loss': '0.498', 'eval_auc': '0.8503', 'eval_runtime': '92.8', ...}
```

With `_DEFAULT_EVAL_INTERVAL_MULTIPLIER=2` and `_DEFAULT_LOG_INTERVAL=100`, eval fires every 200 steps, so a 600-step autoresearch run produces 3 `eval_auc` lines (step 200, 400, 600). There is no post-train `test_auc` — the FT dataset has no test split.

Pull all `eval_auc` lines (avoid nested-quote escapes inside `--command` — SSM mangles them):

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'grep eval_auc /fsx/runs/<RUN_NAME>/logs/out | head -n 200'
```

If you need a more selective regex (e.g. nested quoting), see the `hyperpod-access` skill for the param-via-file workaround — do **not** inline `grep -oE` patterns with embedded quotes into `--command` directly, the SSM channel will eat the escaping.

Locally on the workstation, parse `eval_auc` floats out of those lines (one regex over the captured stdout) and take `max(...)`. That value is `best_eval_auc` — the keep/discard signal (it mirrors `trainer.state.best_metric`, which is what the FT entrypoint summary calls `best_val_metric`).

### Analyze the experiment results

At the 600-step budget, eval fires three times (step 200, 400, 600). Three `eval_auc` points is a coarse trajectory — enough to read direction and detect early peaks. Use it accordingly:

- **Direction**: monotonic rise across step 200 → 400 → 600 → still improving, more compute might help. `eval_auc` peaking at step 200 or 400 then dropping at 600 → already overfitting / weight-drift past the optimum.
- **Convergence speed**: compare the step-200 `eval_auc` across experiments. Faster early lift often predicts higher peak; this is the cleanest cross-experiment signal at this budget.
- **Stability**: flag experiments where `grad_norm` (from `metrics.jsonl`) spikes or oscillates. Unstable gradients suggest the LR is too high; halve `_DEFAULT_BASE_LEARNING_RATE` or `_DEFAULT_MUON_LR` and retry.
- **Train-eval gap**: large gap between final training loss and best `eval_auc` (relative to baseline) suggests overfitting on the FT split; consider higher weight decay, lower LoRA rank, or higher LoRA dropout. (Head capacity is not a knob — head is locked at `linear`. Full FT is also out of scope.)
- **Plateau detection**: if all three eval points are within noise, the LR may be too low or the schedule too conservative; try higher LR or more aggressive warmup. Remember that `learning_rate` in `metrics.jsonl` is the Muon LR (see the field-meaning table above) — tuning `_DEFAULT_BASE_LEARNING_RATE` won't visibly move that field, so to verify an AdamW-LR experiment landed, check the MLflow `train/adamw_decay_lr` curve.

If you want to cancel a job early based on eval_auc trajectory, use `scancel <job_id>` via the hyperpod-access skill. Log as `crash` in `journal.md`.

## Controller Command Gotchas

See the `hyperpod-access` skill for SSM gotchas (comma/quote escaping, output truncation, no shell features in `--command`).

## Hypothesis-Driven Exploration

The loop is hypothesis-driven, not a flat parameter sweep. You iterate on hypotheses until each is accepted or rejected; **multiple hypotheses may be open at the same time** when each has independent merit (e.g. an LR sweep alongside a LoRA-rank sweep). Each iteration, pick the most promising open thread to advance. The goal is coherent reasoning ("we tested 4 hypotheses, kept 2"), not raw experiment count — if the open set grows past ~3 hypotheses you're losing focus, so close the weakest as `rejected` rather than letting them stagnate.

All autoresearch state lives in a single file: **`journal.md`** at the repo root, untracked. It has two parts:

- **`## Results`** — a Markdown table at the top, one row per experiment.
- **`## Hypotheses`** — below the table. One `### H<N>: <title>` subsection per hypothesis (appended chronologically) with full reasoning, references, and per-attempt observations.

The agent updates both parts of the same file after every experiment. The user reads it end-to-end to understand the loop's thinking.

### Layout (`journal.md`)

The file starts with a short header. Then the Results table. Then the Hypotheses section with one `### H<N>` block per hypothesis. Use these exact headings — `grep` and the loop rely on them:

````markdown
# Autoresearch — autoresearch-ft branch

Backbone: efm-pretrain-tayn-20260408-120000Z

To find the open hypothesis: `grep -n "Status: open" journal.md`.

## Results

| commit | run_name | best_eval_auc | final_train_loss | wall_clock_min | status | description | reasoning | hypothesis_id | attempt | next_action |
|---|---|---|---|---|---|---|---|---|---|---|
| a1b2c3d | autoresearch-ft-20260501-072502Z | 0.8421 | 0.5223 | 14 | keep | baseline config (steps=600, peft=lora) | n/a | BASELINE | 1/1 | pivot |
| b2c3d4e | autoresearch-ft-20260501-083804Z | 0.8503 | 0.4912 | 13 | keep | base_lr 1e-4 -> 3e-4 | Baseline grad_norm stable; eval_auc rose monotonically across step 200 (0.821) → 400 (0.836) → 600 (0.842), suggesting LR headroom. Pushing 3× to test the typical LoRA-FT regime. | H01 | 1/5 | accept |
| c3d4e5f | autoresearch-ft-20260501-095102Z | 0.0000 | 0.0000 | 0 | crash | peft_method lora -> dora; lora_r 16 -> 64 (no per_rank_token_budget bump) | DoRA + r=64 stacks more activation memory than r=16; forgot to verify headroom. OOM on first step. | H02 | 1/5 | reject |

## Hypotheses

### H01: base_lr undertuned at 1e-4 for LoRA

- Status: accepted
- Opened: 2026-05-01
- Closed: 2026-05-01
- Attempts: 1/5

#### Claim

A falsifiable claim about FT training behavior. 1-3 sentences. Name the parameter,
mechanism, or interaction — not just the change.

#### Rationale

Multi-paragraph reasoning. Cite **specific prior runs** (commit hash + RUN_NAME),
the step-200 / step-400 / step-600 `eval_auc` deltas, training-curve observations from
`metrics.jsonl`, and any external research informing the hypothesis.

#### Prediction

The criterion that would accept (prediction met) or reject (prediction missed) the
hypothesis. Must be checkable from `best_eval_auc` (or `best_eval_auc` + the step-500
mid-run point) alone — no fuzzy criteria. Frame the threshold as a delta from the
current branch best, not as an absolute number you guessed.

#### References

Optional. Papers, blog posts, internal docs, prior commits/PRs.

#### Attempts

##### Attempt 1/5 — `accept` — commit b2c3d4e — run autoresearch-ft-20260501-083804Z

- Change: `_DEFAULT_BASE_LEARNING_RATE` 1e-4 → 3e-4
- Result: best_eval_auc = 0.8503, final_train_loss = 0.4912
- Observations: eval_auc rose 0.821 (step 200) → 0.836 (step 400) → 0.850
  (step 600); no plateau signal at the budget. grad_norm stable at ~1.0.
  Closing as accepted; new branch HEAD.

### H02: DoRA at r=64 outperforms LoRA at r=16

[…]
````

Contracts you must preserve so `grep` and the loop can find things:

- Header line: `# Autoresearch — autoresearch-ft branch` (or the alias-suffixed variant if the branch carries one).
- The `Backbone:` line directly under the title records the resolved value of `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` for the run. Never edit it within a journal; if the human edits the config constant to point at a different backbone, cut a new branch and a new `journal.md` (see Pre-requisites).
- Section headings: `## Results`, `## Hypotheses`, `### H<N>: <title>`, `#### Claim`, `#### Rationale`, `#### Prediction`, `#### References`, `#### Attempts`, `##### Attempt <N>/5 — …`. Do not rename.
- Metadata bullets under each `### H<N>` heading: `Status:`, `Opened:`, `Closed:`, `Attempts:` exactly as shown.
- Results table is 11 columns (full spec under "Logging Results"); column order and header row stay fixed; `best_eval_auc` is mandatory.
- BASELINE row's `next_action` is `pivot`; first real experiment opens `H01`.
- No literal `|` characters inside any table cell — escape as `\|` or rephrase, otherwise the row breaks.

### Hypothesis lifecycle

A hypothesis is **open** when its `### H<N>` section has `Status: open`. Multiple may be open concurrently — `grep -n "Status: open" journal.md` returns the full open set, and the agent picks among them each iteration. Closing means editing the section's metadata to `Status: accepted` or `Status: rejected`, filling in the `Closed:` date, AND writing a closing Results-table row with the matching `next_action` — both edits land in the same loop iteration.

`next_action` values (set per-attempt in the Results-table row):

- `iterate` — hypothesis still open, refine within same id.
- `accept` — prediction met; close as accepted; winning commit becomes new branch HEAD.
- `reject` — prediction missed at attempt 5 (mandatory close), or fundamental crash. Close as rejected; reset to prior best.
- `pivot` — `BASELINE` row only.

5-attempt cap is the discipline — no early-reject escape hatch on `discard`.

### Picking the next experiment

1. **Iterate on an open hypothesis**: pick whichever open hypothesis has the most signal worth pushing on (most recent attempt was promising, or the prediction window is narrow enough that one more attempt could close it). Refine the same change axis and append a new `##### Attempt N/5` subsection under `#### Attempts`.
2. **Open a new hypothesis** when you have a fresh angle worth testing — append a new `### H<next-N>` section with all required headings and metadata bullets (`Status: open`, etc.) **before any code edits**. You do **not** need to close any existing open hypothesis first; concurrent open hypotheses are allowed (see Hypothesis-Driven Exploration above for the soft cap and focus discipline). Read closed sections in `journal.md` for prior conclusions first; the Experiment Ideas section is a fallback menu.
3. **Close an open hypothesis** as `rejected` when evidence has clearly flipped against it (e.g. two attempts in a row produced clearly worse `best_eval_auc`, or a related closed hypothesis already invalidates the claim). Don't let dead hypotheses sit open and clutter the set.
4. **Forced reject at attempt 5**: set `next_action = reject` and `Status: rejected`. Do not extend past 5 — the attempt cap is the discipline.

### Pre-flight logging

When you pick an experiment in loop step 2 — **before any code edits** — print this block to the conversation:

```
=== HYPOTHESIS H03 (attempt 2/5) ===
Journal:    journal.md → ### H03
Claim:      lora_r=16 undertuned for this backbone
Prediction: best_eval_auc improves by >= 0.005 over current branch best
            if _DEFAULT_LORA_R raised to 32
Change:     _DEFAULT_LORA_R 16 -> 32
====================================
```

For the `BASELINE` row:

```
=== BASELINE =======================
Journal:    n/a
Claim:      n/a (establishing baseline)
Prediction: n/a
Change:     baseline config as-is
====================================
```

Greppable contract: line prefix `=== HYPOTHESIS ` or `=== BASELINE `.

## Logging Results

When an experiment is done, update `journal.md` in two places:

1. **Append a row to the `## Results` Markdown table** (11-column FT spec below).
2. **Append a `##### Attempt N/5` subsection** under the matching `### H<N>` hypothesis's `#### Attempts` heading. For the `BASELINE` row, no hypothesis section needs updating.

Both edits land in the same file. Do not separate them across runs.

### `## Results` table — 11 columns

| Column | Description |
|---|---|
| `commit` | git commit hash (short, 7 chars). |
| `run_name` | unique `RUN_NAME` from `hp-stage-code` (e.g. `autoresearch-ft-20260501-072502Z`). |
| `best_eval_auc` | max `eval_auc` observed across mid-run evals (4 dp, e.g. 0.8612). Use `0.0000` for crashes / cancelled jobs. |
| `final_train_loss` | last training loss from `metrics.jsonl` (4 dp, e.g. 0.4823). Use `0.0000` for crashes. |
| `wall_clock_min` | wall-clock minutes end-to-end. Read from `tools/wait_for_hyperpod_job.sh`'s `elapsed_min=N` exit line, or compute from `train_runtime` (seconds) in `logs/out` + ~2 min for post-train eval. Use `0` for crashes that didn't start training. |
| `status` | `keep`, `discard`, or `crash`. Per-experiment outcome. |
| `description` | short text of *what* this experiment changed (e.g. `base_lr 1e-4 -> 3e-4` or `peft_method lora -> dora`). No literal `|` characters — escape as `\|` or rephrase. |
| `reasoning` | short blurb (2-4 sentences): *why* this attempt was run — signals, observations, or prior results that motivated it. Include specific metrics where relevant. No literal `|` characters. Use `n/a` for the BASELINE row. |
| `hypothesis_id` | `H01`, `H02`, … or `BASELINE`. |
| `attempt` | `1/5`, `2/5`, … `5/5`. Use `1/1` for `BASELINE`. |
| `next_action` | `iterate` / `accept` / `reject` / `pivot`. |

### `status` × `next_action`

Per-experiment `status` and per-hypothesis `next_action` are orthogonal. Common combinations:

| status | next_action | When |
|---|---|---|
| keep | accept | Higher `best_eval_auc` AND prediction met → hypothesis confirmed, advance branch, close. |
| keep | iterate | Higher `best_eval_auc` but prediction not yet hit → advance branch, refine within same hypothesis. |
| discard | iterate | Equal/lower `best_eval_auc`, attempt < 5 → reset, refine within same hypothesis. |
| discard | reject | Equal/lower `best_eval_auc`, attempt = 5 → reset, hypothesis dead, pivot. |
| crash | iterate | Trivial fix (typo, OOM at edge) → re-run with fix, same hypothesis. |
| crash | reject | Fundamental break → close hypothesis, pivot. |
| keep | pivot | `BASELINE` row only. |

Tracking `wall_clock_min` lets you spot creeping cluster contention or step-time regressions across experiments — don't skip it.

Do NOT commit `journal.md`. Leave it untracked by git.

## The Experiment Loop

The experiment runs on a dedicated branch (`autoresearch-ft`).

**Each iteration:**

1. **Review state** — read `journal.md`. Scan the `## Results` table for current best `best_eval_auc` and what's been tried; read closed `### H<N>` sections for prior conclusions. Identify the open hypothesis set (`grep -n "Status: open" journal.md`) — each is fair game to advance, and there can be more than one. Check `git log` to confirm you are on the right commit.
2. **Pick an experiment** — see Hypothesis-Driven Exploration above:
   - Iterate on whichever open hypothesis has the most signal worth pushing on (same `hypothesis_id`, attempt += 1), OR
   - Open a new hypothesis when you have a fresh angle worth testing in parallel — concurrent open hypotheses are allowed, no requirement to close existing ones first.
   - Close any open hypothesis as `rejected` when evidence has clearly turned against it (don't let dead threads sit in the open set).
   - Print the pre-flight `=== HYPOTHESIS ===` block to the conversation before any code edits.
3. **Modify in-scope files** — edit `finetune_training_config.py` and/or `finetune_model_config.py` to implement the experiment.
4. **Validate locally**:
   - Run `make test` to catch syntax errors, import errors, and regressions.
   - There is no parameter-budget command for FT (architecture is frozen). If the change touches `_DEFAULT_LORA_TARGET_MODULES`, sanity-check that the names match modules present in the loaded backbone — grep `code/src/efm_core/model/efm_core_model.py` for the `nn.Linear` definitions (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`); `code/src/efm_finetune/peft/wrap.py` carries the default target-module list and the wrapping logic.
5. **Git commit** — descriptive message of what changed (prefix commit with `autoresearch-ft:`). Do NOT commit `journal.md`.
6. **Submit to HyperPod** — invoke the `submit-hyperpod-training-job` skill with `MODE=finetune`, `RUN_NAME_PREFIX=autoresearch-ft` (or the alias suffix when one is in use), plus the stored `AWS_PROFILE` and `ENV`. The submit skill reads the locked backbone from `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in the staged config — do **not** pass it as an argument. That skill generates a unique `RUN_NAME`, stages the snapshot, fetches it on the controller, and submits via `make hp-submit-finetune`; capture the `Submitted batch job <id>` line it reports.
7. **Wait for completion** — run `tools/wait_for_hyperpod_job.sh <JOB_ID> --aws-profile="$AWS_PROFILE"` (parse `<JOB_ID>` from the `Submitted batch job <id>` line). Default budget is 120 min, which is generous for a 600-step FT autoresearch run at typical FT shapes; if the BASELINE wall-clock measurement comes in at >100 min, bump `--timeout-min=<larger>` for subsequent submissions. The helper polls `squeue` and exits `0` on COMPLETED, `1` on FAILED/CANCELLED/NODE_FAIL/TIMEOUT/OOM, `2` on its own wall-clock budget exceeded. Run via Bash `run_in_background: true` and let the harness's completion notification surface the result. On exit `0` or `1`, read `metrics.jsonl` and the `eval_auc` lines from `logs/out` (see Retrieving Results). On exit `2`, decide whether to wait longer or `scancel`.
8. **Record results** — update `journal.md` in two places: append an 11-column row to the `## Results` table, and append a `##### Attempt N/5 — <next_action> — commit <hash> — run <RUN_NAME>` subsection under the matching `### H<N>` hypothesis with the change, raw metrics (`best_eval_auc`, `final_train_loss`), and observations from the step-200 / step-400 / step-600 eval points (per the Analyze section — at this budget you have 3 mid-run evals).
9. **Keep or discard, and update the hypothesis** — driven by `best_eval_auc` (not val_loss):
   - Higher `best_eval_auc` than current best AND prediction met → `status=keep`, `next_action=accept`. Advance branch, close hypothesis.
   - Higher `best_eval_auc` but prediction not yet hit → `status=keep`, `next_action=iterate`. Advance branch, refine within same hypothesis.
   - Equal/lower `best_eval_auc`, attempt < 5 → `status=discard`, `next_action=iterate`. `git reset --hard` to prior best, refine within hypothesis.
   - Equal/lower `best_eval_auc`, attempt = 5 → `status=discard`, `next_action=reject`. Reset, close hypothesis, pivot.
   - Crash, trivial fix → `status=crash`, `next_action=iterate`.
   - Crash, fundamental break → `status=crash`, `next_action=reject`. Close hypothesis, pivot.

## Experiment Ideas

The FT search space is small by design. These categories cover most of it; treat them as a menu when picking the next experiment.

### 1. Learning rate (most common axis)

- `_DEFAULT_BASE_LEARNING_RATE`: 1e-4 (production default; **autoresearch baseline**) / 3e-4 / 1e-3 / 3e-3 — LoRA/DoRA tolerates higher LR than full FT, so the production default of 1e-4 is likely undertuned for LoRA. Push higher.
- `_DEFAULT_LR_DECAY_STYLE`: `linear` (baseline) vs `cosine` (often better for short FT runs).
- `_DEFAULT_LR_STABLE_RATIO`: 0.7 / 0.9 (baseline) / 0.95.
- `_DEFAULT_LR_MIN_RATIO`: 0.0 / 0.1 (baseline) / 0.2.
- `_DEFAULT_WARMUP_STEPS`: 50 / 100 (baseline) / 150 / 200 — at the 600-step autoresearch budget keep warmup well below half the run; FT-side warmup is unscaled (see Cluster).

### 2. PEFT method (high-leverage axis)

- `_DEFAULT_PEFT_METHOD`: `lora` (autoresearch baseline) / `dora`. **DoRA shares all LoRA hyperparameters** — there are no DoRA-specific tunables in this codebase. The only difference vs LoRA is that `peft_method="dora"` triggers `use_dora=True` on the underlying `peft.LoraConfig`, which decomposes each adapted weight into magnitude + direction (see `code/src/efm_finetune/peft/wrap.py`). Tune `_DEFAULT_LORA_*` the same way for both methods. `"full"` is out of scope for the lightweight-FT autoresearch.
- **DoRA→AdamW pairing is required** — see "PEFT-optimizer pairing" above. Any commit that flips `_DEFAULT_PEFT_METHOD` to `"dora"` MUST also set `_DEFAULT_OPTIMIZER = "adamw"` in the same commit, otherwise the run will land flat-loss. When closing a DoRA hypothesis and reverting to LoRA, also revert `_DEFAULT_OPTIMIZER` to `"muon_adamw"`.
- LoRA / DoRA hyperparameters:
  - `_DEFAULT_LORA_R`: 8 / 16 (baseline) / 32 / 64.
  - `_DEFAULT_LORA_ALPHA`: typically 2× rank — baseline 32 pairs with r=16. Move it together with rank.
  - `_DEFAULT_LORA_DROPOUT`: 0.0 / 0.05 (baseline) / 0.1 / 0.2.
  - `_DEFAULT_LORA_TARGET_MODULES`: full attention+FFN (baseline) vs attention-only (`q_proj`, `k_proj`, `v_proj`, `out_proj`) vs FFN-only (`gate_proj`, `up_proj`, `down_proj`).

### 3. Regularization

- `_DEFAULT_WEIGHT_DECAY`: 0.0 / 0.05 / 0.1 (baseline) / 0.2.

### 4. Muon optimizer hyperparams

These apply only when `_DEFAULT_OPTIMIZER == "muon_adamw"`. With `optimizer="adamw"` (required for DoRA), Muon is bypassed entirely and these constants are silently ignored at runtime — don't sweep them in DoRA experiments, the result will be a duplicate of the AdamW-only baseline.

- `_DEFAULT_MUON_LR`: 0.005 / 0.01 / 0.02 (baseline) / 0.04.
- `_DEFAULT_MUON_MOMENTUM`: 0.9 / 0.95 (baseline) / 0.98.
- `_DEFAULT_MUON_NS_STEPS`: 3 / 5 (baseline) / 7.
- `_DEFAULT_MUON_WEIGHT_DECAY`: 0.0 / 0.1 (baseline).

### 5. Effective batch size

- `_DEFAULT_GRADIENT_ACCUMULATION_STEPS`: 1 (baseline) / 2 / 4. Effective global batch = `gradient_accumulation_steps × total_gpus`, where `total_gpus` comes from `hyperpod/finetune.sbatch` (see Cluster section). LR is sqrt-scaled by the FT runtime relative to `lr_reference_global_batch=8`.

### 6. Focal-loss hyperparams

The production FT dataset has heavy positive-class imbalance, so the loss is locked at focal (see Fixed parameters). The two focal hyperparameters are tunable axes:

- `_DEFAULT_FOCAL_ALPHA`: 0.25 / 0.5 / **0.75 (baseline)** / 0.9. The positive-class weight in `[0, 1]`. Higher up-weights the rare positive class more aggressively. If the BASELINE produces strong overall `eval_auc` but the per-class confusion matrix shows the model is collapsing to "predict negative", push alpha higher. If `eval_auc` is plateauing because the model over-fits noisy positives, push alpha back toward 0.5.
- `_DEFAULT_FOCAL_GAMMA`: 0.5 / 1.0 / **2.0 (baseline)** / 3.0 / 5.0. The focusing parameter (≥ 0). 0 collapses to weighted-BCE; 2.0 is the canonical Lin et al. 2017 default and a safe baseline; 5.0 sharpens gradient on hard examples (useful when easy positives saturate quickly). Sweep only after alpha is settled — gamma's effect is conditional on the alpha-induced class weighting.
- Sequence the sweep: alpha first (one knob, monotone effect on imbalance handling), then gamma (interacts with alpha, so tuning it before alpha is wasted compute). Don't bundle alpha and gamma changes in a single attempt — that's two predictions per experiment, which violates the simplicity criterion.
- Paired-config invariant: `_DEFAULT_LOSS_TYPE` stays at `"focal"` for the entire loop; both `_DEFAULT_FOCAL_*` constants must remain set to numeric values (an empty value would fail `EFMFineTuneConfig.__post_init__`).

### Combinations

- Stack winning changes together (e.g. accepted LoRA hyperparameters + new LR sweep).
- Avoid bundling unrelated changes within a single attempt — each attempt should test one prediction.

#### Guarding against noise accumulation

Every accepted hypothesis is a single-measurement win against a single-measurement baseline, and FT runs at this shape have non-trivial run-to-run variance from cluster contention, dataloader seeding, and short-budget eval volatility. Stacking many `accept` decisions can therefore cumulate noise into the branch HEAD even when each individual attempt looked like a real improvement. Two cheap counter-measures:

- **Treat sub-noise wins as iterate, not accept.** If the `best_eval_auc` delta over the prior branch best is below ~0.005 (rough ballpark, refine once you've measured run-to-run variance on this backbone), don't promote — set `next_action=iterate` and try a more aggressive value within the same hypothesis. Promote only when the delta is clearly outside noise.
- **Re-baseline every ~5 accepted hypotheses.** Re-run the current branch HEAD as a fresh experiment (same config, new `RUN_NAME`); if its `best_eval_auc` is within noise of the original BASELINE row + the cumulative deltas you've credited along the way, the stack is real. If the re-run undershoots by more than a single accepted-step's delta, treat the stack as noise-contaminated: `git reset --hard` to the prior most-trusted commit and reopen the disputed hypotheses with stricter predictions. Log the re-baseline row in `## Results` with `hypothesis_id=BASELINE`, `attempt=N/1` (incrementing `N`), `next_action=pivot`.

## Error Handling

### Job Timeout

The default budget is 120 minutes (`tools/wait_for_hyperpod_job.sh`). When the helper exits `2`:

1. Re-run with `--timeout-min=<larger>` if the job is still making progress.
2. `scancel <job_id>` and log as `crash` in `journal.md`.

Do not let runs sit in the queue indefinitely.

### Missing Results

Job finished but `metrics.jsonl` is empty or `eval_auc` lines are missing — treat as a crash. Check `/fsx/runs/<RUN_NAME>/logs/out` and `/fsx/runs/<RUN_NAME>/logs/err` for error details.

### Crash Handling

- Trivial fix (typo, import error, shape mismatch caught locally) — fix and re-run, same experiment.
- Fundamentally broken (OOM, numerical instability, LoRA target-module name mismatch) — log as crash, discard, move on.
- 3+ consecutive crashes — revert to last known-good commit, try something completely different.

### Local Validation Failures

- `make test` fails — fix before submitting. Do not submit code that fails tests.

### Slurm Submission Failures

- Job fails to queue — check logs, fix if obvious infrastructure issue, otherwise skip and note.
- Do NOT get stuck debugging infrastructure. If you cannot resolve a submission issue within a few minutes, log it and move on.
- Do not run any docker, enroot, or sudo commands on the cluster controller.

### Backbone Resolution Failures

Two failure shapes are possible:

- **Empty constant** — `DeclaredFineTuneTrainParams.__post_init__` raises with `pretrained_efm_backbone_name is empty. ... Set _DEFAULT_PRETRAINED_EFM_BACKBONE_NAME in code/scripts/finetune/configs/finetune_training_config.py ...`. Surface that exact message to the user and stop the loop; they must hand-edit the constant and commit before the loop can resume. Do not edit the constant yourself (that's outside the autoresearch scope) and do not invent a value.
- **Wrong run name / missing artifact** — `resolve_pretrained_best_checkpoint_artifact_path` raises because the constant points at a run name that doesn't exist on FSx, or the artifact has not landed yet. Same response: surface to the human, stop the loop, do not retry by guessing alternative names.

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?" The human might be asleep, or gone from the computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combinations of previous near-misses, revisit discarded ideas from a different angle, push on the highest-leverage axes (LR + LoRA hyperparams; the LoRA↔DoRA swap is a single binary toggle once both have been tried). The loop runs until the human interrupts you, period.

Wall-clock per experiment depends on the cluster shape declared in `hyperpod/finetune.sbatch` and the FT input pipeline; measure it with the BASELINE run before projecting overnight throughput. The default `tools/wait_for_hyperpod_job.sh` 120-min budget is generous for typical FT shapes; bump it if the BASELINE measurement says you need more. The user wakes up to a `journal.md` full of results and reasoning, and an advanced branch with the best FT configuration found.

## Red Flags

- The agent submits a job without running `make test` first.
- The agent modifies files outside the two in-scope FT config files.
- The agent asks "should I continue?" or pauses for confirmation during the loop.
- The agent commits `journal.md` to git.
- The agent skips the baseline and starts with experimental changes.
- The agent changes `_DEFAULT_TOTAL_TRAINING_STEPS` away from 600 between experiments (it is locked at the autoresearch budget set at branch setup).
- The agent changes `_DEFAULT_EVAL_METRIC_FOR_BEST_MODEL` away from `eval_auc`.
- The agent edits `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` (it is human-maintained and out of the autoresearch loop's edit scope), or starts mixing two backbones in the same `journal.md` after a human did edit it.
- The agent treats `_DEFAULT_HEAD_DEPTH` as a tunable axis (it is locked at `linear` for the lightweight-FT starting point).
- The agent sets `_DEFAULT_PEFT_METHOD = "full"` — full FT is out of scope for the lightweight-FT autoresearch; only `"lora"` and `"dora"` are tunable values.
- The agent sets `_DEFAULT_PEFT_METHOD = "dora"` without also setting `_DEFAULT_OPTIMIZER = "adamw"` in the same commit. The validator does not catch this combination, but the run will produce flat train + eval loss for the entire training budget (Muon's Newton-Schulz orthogonalization vs DoRA's per-column magnitude renormalization is a pathological pair). See "PEFT-optimizer pairing" in Baseline Configuration. Symmetrically: closing a DoRA hypothesis and reverting `_DEFAULT_PEFT_METHOD` to `"lora"` without reverting `_DEFAULT_OPTIMIZER` back to `"muon_adamw"` silently swaps the optimizer underneath the loop and confounds Muon-vs-AdamW with whatever the next hypothesis is testing.
- The agent flips `_DEFAULT_LOSS_TYPE` away from `"focal"` (locked for the loop because it's matched to the heavy positive-class imbalance in the production FT dataset). Tune `_DEFAULT_FOCAL_GAMMA` / `_DEFAULT_FOCAL_ALPHA` instead — they are the focal-loss tunable axes.
- The agent bundles `_DEFAULT_FOCAL_GAMMA` and `_DEFAULT_FOCAL_ALPHA` changes into one attempt — gamma's effect is conditional on alpha's class weighting, so tuning them in the same step makes the result un-attributable. Sequence: alpha first, then gamma.
- The agent installs new dependencies or modifies `pyproject.toml`.
- The agent spends more than a few minutes debugging infrastructure (Slurm, S3, controller access).
- The agent does not update `journal.md` (both the `## Results` table row and the matching `##### Attempt N/5` subsection) after an experiment completes.
- The agent reuses a `RUN_NAME` from a previous experiment instead of letting `hp-stage-code` generate a new one.
- The agent submits an experiment without printing the `=== HYPOTHESIS ===` pre-flight block to the conversation.
- The agent extends a hypothesis past attempt 5 instead of marking it `reject` (the 5-attempt cap is the discipline).
- The agent lets the open hypothesis set grow unbounded — past ~3 concurrent open hypotheses is a sign of unfocused exploration; close the weakest as `rejected` rather than letting them stagnate.
