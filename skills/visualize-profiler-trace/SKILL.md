---
name: visualize-profiler-trace
description: Use when the user wants to visualize a PyTorch profiler trace from a HyperPod training run — given a run name, downloads trace files from S3 and launches TensorBoard with the torch-tb-profiler plugin locally.
---

# Visualize Profiler Trace

## Overview

Downloads PyTorch profiler trace files for a HyperPod run from S3 by run name and opens them in TensorBoard via the `torch-tb-profiler` plugin.

## Required Inputs

- `RUN_NAME` — the training run identifier (e.g. `matteo-profiler-test-20260427-194246Z`)
- `AWS_PROFILE` — optional; only needed when the default credential chain cannot access the prod datalake bucket

## Workflow

Run the bundled helper script from the repo root:

```bash
bash skills/visualize-profiler-trace/scripts/visualize_trace.sh <RUN_NAME> [AWS_PROFILE]
```

The script:

1. Lists `s3://ml-hyperpod-fsx-datalakeprod-us-west-2/runs/<RUN_NAME>/profiler/` and exits early if nothing is there
2. Syncs all trace files to `~/profiler_logs/<RUN_NAME>/`
3. Installs (or updates) `tensorboard` with `torch-tb-profiler` and `setuptools<81` via `uv tool install`
4. Launches TensorBoard at **http://localhost:6006**

Navigate to **http://localhost:6006** and select the **PyTorch Profiler** tab.

## S3 Path Convention

```
s3://ml-hyperpod-fsx-datalakeprod-us-west-2/runs/<RUN_NAME>/profiler/
```

Files are named like `ip-10-5-13-41_152501.<timestamp>.pt.trace.json` — one per rank. `aws s3 sync` downloads all of them.

## Manual Steps

If running the script directly is not possible:

```bash
RUN_NAME=<your-run-name>

aws s3 sync \
  "s3://ml-hyperpod-fsx-datalakeprod-us-west-2/runs/$RUN_NAME/profiler/" \
  ~/profiler_logs/$RUN_NAME/

uv tool install tensorboard \
  --with torch-tb-profiler \
  --with "setuptools<81" \
  --force

tensorboard --logdir ~/profiler_logs/$RUN_NAME/
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| `--logdir ~/profiler_logs/` (parent dir, not run subdir) | Point at `~/profiler_logs/<RUN_NAME>/`; TensorBoard will misread mixed traces otherwise |
| Omitting `--with "setuptools<81"` | Causes an import error at TensorBoard startup; the pin is required |
| S3 sync returns 400 Bad Request | Verify the exact `RUN_NAME` and that profiling was enabled; check with `aws s3 ls s3://.../<RUN_NAME>/profiler/` |
| `tensorboard` not found after install | Run `uv tool update-shell` or open a new terminal so the `uv` tool bin is on `$PATH` |
