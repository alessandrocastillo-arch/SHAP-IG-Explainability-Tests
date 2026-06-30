---
name: submit-eval-job
description: Use when the user wants to run offline eval or extract per-user embeddings from a trained EFM checkpoint over a baked dataset split. Covers report-only eval and the embedding-export variant; covers iterating across train/val/test splits.
---

# Submit Eval Job

## Overview

Submits a SageMaker **TrainingJob** that loads a checkpoint, streams a pre-baked Arrow IPC dataset, and writes (a) an eval report (loss, anisotropy, uniformity, embedding health) and optionally (b) per-user embedding parquets. There is **no separate "embedding export" job** — embeddings are an opt-in side output of the eval job (`export.save_embeddings=true`).

Output layout:

```
report:     {report_base}/{run_name}/checkpoint-{step}/{split}/eval_report_stepN.json
embeddings: {output_base}/{run_name}/checkpoint-{step}/{split}/{host}-rank{r}-part-{NNNNN}.parquet
```

Embedding parquet schema (post-PR #253): `user_id: string`, `embedding: list<float32>`, plus any `passthrough_columns: list<T>`.

## Key facts before submitting

- **One split per submission.** Split is derived from the trailing path component of `--baked-data-uri` (must be `train`, `val`, or `test`). To produce all three splits, submit three jobs.
- **Pre-flight blocks re-runs.** The launcher fails fast if either the report or embedding S3 prefix is non-empty. To re-run a split, delete the existing prefix first.
- **Pooling auto-matches the model.** Encoder (non-causal) → first-token CLS; decoder (causal) → last real token. Read from the checkpoint's `config.json`, not the eval YAML.
- **PLR/amount-channel checkpoints require `amount_cents` in the bake.** Eval errors out if the baked shard doesn't carry the per-token `amount_cents` column. FT bake YAMLs currently don't set `per_token_cols_to_store`; re-bake or pick a non-PLR checkpoint.

## Workflow

1. **Confirm the four inputs** with the user:
   - `RUN_NAME` — the training run, e.g. `efm-decoder-pretrain-39k-ppma-lora-ft-20260512-051009Z`.
   - `CHECKPOINT_STEP` — integer; the checkpoint at `s3://ml-hyperpod-fsx-…/runs/{RUN_NAME}/checkpoints/checkpoint-{step}` must exist.
   - `BAKED_DATA_URI` — S3 prefix ending in `/train/`, `/val/`, or `/test/`. Must already be baked (see `launch-bake-job`).
   - `AWS_PROFILE` — never default from memory; ask. Usually `ds-dlprod` for prod data-lake buckets.

2. **Ask whether to save embeddings.** If yes, set `SAVE_EMBEDDINGS=1`. Without it, the job runs metrics-only and writes only the report.

3. **Ask about parallelism.** Default `INSTANCE_COUNT=1`. For >1, SageMaker shards `.arrow` files across instances via `ShardedByS3Key`; each instance writes parquets with a hostname prefix so they don't collide. There is **no cross-instance rendezvous** — MLflow logs one run per instance.

4. **Ask whether they want a one-off override or to add a permanent profile.** Per project convention, one-offs go via `--config-override` (mapped to env vars below); only commit a new YAML under `code/scripts/configs/yaml/eval/profile/` if it's a recipe worth reusing.

5. **Show the full submission command** and confirm before launching:

   ```bash
   AWS_PROFILE=<profile> make trigger-eval-job \
       RUN_NAME=<run> \
       CHECKPOINT_STEP=<step> \
       BAKED_DATA_URI=<s3-prefix-ending-in-split>/ \
       SAVE_EMBEDDINGS=1 \
       INSTANCE_COUNT=4
   ```

6. **Launch in background** (`run_in_background=true`) — eval takes tens of minutes to hours. Report the SageMaker job name.

7. **Report the output S3 paths** for both the report and (if enabled) embeddings.

## Knobs (env-var → config override)

| Make var | Effect | Notes |
|---|---|---|
| `RUN_NAME` | required | Used in both checkpoint URI and output paths |
| `CHECKPOINT_STEP` | required | Integer |
| `BAKED_DATA_URI` | required | Trailing `train\|val\|test` enforced |
| `SAVE_EMBEDDINGS=1` | `export.save_embeddings=true` | Off by default; eval still emits report |
| `INSTANCE_COUNT=N` | parallel instances | Output filenames carry hostname prefix |
| `INSTANCE_TYPE` | e.g. `ml.p4d.24xlarge` | Default `ml.g6.xlarge` |
| `MAX_BATCHES=N` | `data.max_batches=N` | Cap packs — smoke-test mode |
| `PACKS_PER_FORWARD=N` | `data.packs_per_forward=N` | Default 7; raise for GPU underutilization |
| `COMPILE_MODEL=1` | `model.compile_model=true` | `torch.compile(dynamic=True)`; first forward slow |
| `ATTN_BACKEND=sdpa` | switch off fa2 | CPU/local only |
| `PROFILE=<name>` | `+profile=<name>` | e.g. `ppma`; profile sets `run_name / checkpoint_step / baked_data_uri / passthrough_columns` |

Profile structure (`code/scripts/configs/yaml/eval/profile/ppma.yaml`):

```yaml
# @package _global_
export:
  save_embeddings: true
  passthrough_columns: [pred_times]
job:
  run_name: efm-decoder-pretrain-39k-ppma-lora-ft-20260512-051009Z
  checkpoint_step: 60000
  baked_data_uri: "s3://…/baked/…/seq16384_.../val/"
  instance_count: 16
```

`passthrough_columns`: each column must already exist as a `list<T>` field in the baked shard. The bake's `extra_cols_to_store: {pred_time: string}` writes the column as `pred_times` (pluralized) — match the bake output name exactly.

## All three splits

There is no single submission that fans out across splits. Submit three times:

```bash
for split in train val test; do
  AWS_PROFILE=ds-dlprod make trigger-eval-job \
      RUN_NAME=<run> CHECKPOINT_STEP=<step> SAVE_EMBEDDINGS=1 \
      BAKED_DATA_URI=s3://…/baked/…/${split}/
done
```

If the bake doesn't have all three splits, re-bake first (see `launch-bake-job`).

## Common Mistakes

| Mistake | Reality |
|---|---|
| Passing a `--baked-data-uri` one level too high (no trailing `train/val/test`) | Launcher errors with `trailing path component must be one of {train, val, test}` |
| Re-running a split that already has output | `_assert_no_output` fails fast; delete the prefix or change `RUN_NAME` |
| Setting `SAVE_EMBEDDINGS=1` for a PLR checkpoint with an FT bake that lacks `amount_cents` | Eval raises mid-run after expensive setup; re-bake with `per_token_cols_to_store: {amount_cents: float32}` |
| Using a profile when the user wanted a one-off override | One-offs go through `CONFIG_OVERRIDES` / env vars; new YAML only for permanent recipes |
| Expecting cross-instance metric aggregation | `torchrun --standalone` means MLflow logs one run per instance; cross-rank reduction is within-instance only |
| Forgetting that `passthrough_columns` names follow the bake's *output* names (`pred_times`, not `pred_time`) | KeyError at column lookup; check the baked shard schema first |

## Red Flags

- Defaulting `AWS_PROFILE` from memory instead of asking the user
- Launching without showing the full `make trigger-eval-job …` command first
- Submitting without running in background
- Telling the user there is a separate "embedding export" job — there isn't; it's `SAVE_EMBEDDINGS=1` on the eval job
- Creating a new YAML profile for a one-shot run instead of using `--config-override` / env vars
- Skipping the prerequisite check: the checkpoint at `checkpoint-{step}` must exist and the baked URI must be populated
