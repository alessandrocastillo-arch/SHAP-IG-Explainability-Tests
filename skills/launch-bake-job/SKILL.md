---
name: launch-bake-job
description: Use when the user wants to launch a SageMaker bake job to tokenize, pack, and write Arrow IPC shards for a dataset. Covers pretrain, finetune, and inference bake modes.
---

# Launch Bake Job

## Overview

Launches a SageMaker ProcessingJob that tokenizes + packs a dataset split into Arrow IPC FILE-format shards. Two path layouts are supported:

**Canonical layout** (pretrain and finetune bakes):

```
s3://ml-datasets-datalakeprod-us-west-2-sagemaker/
  {dataset_name}/{dataset_version}/baked/
  {tokenizer_name}/
  seq{max_seq_len}_tb{token_budget}_mc{0|1}_tdnone_bcv{PACKED_PRETRAIN_BAKE_CODE_VERSION}/
  {split}/
```

**Adhoc layout** (inference bakes — `flow_type` is non-empty):

```
s3://ml-datasets-datalakeprod-us-west-2-sagemaker/
  {dataset_name}/{flow_type}/{dataset_version}/baked/
  {tokenizer_name}/
  seq{max_seq_len}_tb{token_budget}_mc0_tdnone_bcv{PACKED_PRETRAIN_BAKE_CODE_VERSION}/
  {split}/    # always "inference" in this layout
```

## Pretrain vs Finetune vs Inference — Key Differences

| | Pretrain | Finetune | Inference |
|---|---|---|---|
| `flow_type` | `""` (must be empty) | `""` (must be empty) | required, free-form (e.g. `adhoc`, `daily`) |
| `split` | `train\|val\|test\|...` | same | `"inference"` |
| `multi_chunk` | **true** — long user histories span multiple chunks | **false** — FT requires a single chunk per user/label window | **false** — single chunk per inference row |
| `label_column` | **`""`** — never has a label | **required** — must name the scalar label column (e.g. `"labels"`) | **`""`** — inference data is unlabeled |
| Source layout | `{name}/{ver}/_raw/{split}/` | same | `{name}/{flow_type}/{ver}/` (parquet directly, no `_raw/{split}/`) |

## Workflow

1. **Check for wheels** before anything else:

   ```bash
   ls code/wheels/*.whl 2>/dev/null | head -3
   ```

   If missing or empty, ask the user to run `AWS_PROFILE=<profile> make download_wheels` before proceeding.

2. **Ask for two or three things only:**
   - the dataset name
   - the split (one of: `train`, `val`, `test`, `inference`)
   - **for inference bakes only:** the `flow_type` (currently must be `adhoc`)

   That's all the user should need to provide upfront. The launcher rejects any other combination at pre-flight.

3. **Load the baseline config** from `jobs/configs/bake/`. Use `finetune.yaml` for FT bakes, `pretrain.yaml` for pretrain bakes, and `efm_risk_ppma_inference_v1.yaml` (or the closest inference config) for inference bakes — these are the canonical starting points. If neither fits, copy the closest one.

4. **Show the full proposed config** and ask: *"Does this look right, or do you want to change anything?"* Do not ask for each field individually — present the complete config as a block and let the user correct what's wrong.

   Key defaults to apply when building from scratch:
   - `dataset_version: 1`
   - `tokenizer_s3_uri / tokenizer_name`: copy from an existing config
   - `max_seq_len: 16384`, `token_budget: 81920`
   - `multi_chunk`: **true** for pretrain, **false** for finetune
   - `label_column`: **`""`** for pretrain; ask the user for finetune (the one field you must ask for explicitly if it can't be inferred)
   - `drop_zero_loss_chunks: true`
   - `packs_per_shard: 2048`, `packs_per_ipc_batch: 64`
   - `instance_type: ml.c7i.48xlarge`, `instance_count: 48`
   - `compression: zstd`
   - `flow_type`: **`""`** for pretrain/FT (must be empty — non-empty values are only valid with `split=inference`); for inference, a free-form path-routing segment matching the upstream datagen flow (e.g. `"adhoc"`, `"daily"`) — the bake itself is identical regardless of value, the segment just keeps outputs from different flows in separate folders
   - For inference: `split=inference`, `multi_chunk=false`, `label_column=""` — all required by the pre-flight guard
   - For inference: `extra_cols_to_store` should include any *additional* columns needed to join embeddings back to source rows (e.g. `pred_time: string` for PPMA). **Do not add `user_id`** — `user_ids` is a fixed pack-schema field already populated from the source `user_id` column and exported as `user_id` per segment; passing `user_id` through `extra_cols_to_store` triggers a `user_ids` field collision and the bake aborts pre-flight.

5. **Ask for `AWS_PROFILE`** if the user hasn't provided it. Default: `ds-dlprod`.

6. **Apply any changes** the user requests, show the updated config, and confirm once more before launching.

7. **Launch:**

   ```bash
   AWS_PROFILE=<profile> uv run python -m jobs.bake_job --config jobs/configs/bake/<config>.yaml
   ```

   Run in background (`run_in_background=true`) — jobs take hours. Report the SageMaker job name on submission.

8. **Report the output S3 path** (the `packed_bake_key`) so the user can wire it into training configs.

## Common Mistakes

| Mistake | Reality |
|---|---|
| Setting `multi_chunk: true` for finetune | FT bake contract requires `multi_chunk: false`; the launcher errors pre-flight |
| Leaving `label_column: ""` for finetune | Shards will have no label column; FT training fails at load time |
| Missing `code/wheels/*.whl` | Container pip install fails with "wrong number of parts". Run `make download_wheels` first |
| Using `ds-dldev` profile | Usually lacks SageMaker execution role access; prefer `ds-dlprod` |
| Setting `multi_chunk: true` for inference | Inference bake contract requires `multi_chunk: false`; the launcher errors pre-flight |
| Leaving `label_column` non-empty for inference | The pre-flight guard rejects this; inference packs are unlabeled and use the pretrain schema |
| Missing `flow_type` for inference | The pre-flight guard rejects `split=inference` without `flow_type`; source path can't be derived |
| Placing inference source data under `_raw/{split}/` | Adhoc layout has parquet files directly at `{name}/{flow_type}/{ver}/`; the launcher won't find data under a nested `_raw/inference/` |
| Adding `user_id: string` to `extra_cols_to_store` | `user_ids` is a fixed pack-schema field already populated from the source `user_id` column; passing `user_id` through `extra_cols_to_store` produces a `user_ids` output-field collision and the bake core aborts with `ValueError` |
| Setting `flow_type` with `split != "inference"` | The pre-flight guard rejects this combination; `flow_type` is only meaningful for inference bakes (it keeps adhoc/daily/etc. outputs in separate folders) and the canonical pretrain/FT layout has no place for it |

## Red Flags

- Asking the user for every field individually instead of showing a proposed config
- Launching without verifying `code/wheels/*.whl` are present
- Setting `multi_chunk: true` for a finetune bake
- Proceeding with an empty `label_column` for finetune without flagging it
- Skipping the pre-launch confirmation
- Not running in background
- For inference: launching without confirming source data lives at `{name}/{flow_type}/{ver}/*.parquet` (no `_raw/{split}/`)
- For inference: setting `extra_cols_to_store` without identifiers that let downstream code join embeddings back to source rows
