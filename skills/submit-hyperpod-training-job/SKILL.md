---
name: submit-hyperpod-training-job
description: Submit an efm-core training run to HyperPod for either pretraining or fine-tuning. The agent first decides which job mode to use (`pretrain` or `finetune`), then determines a unique run name, stages the current repo snapshot to S3, fetches that snapshot onto FSx under `/fsx/runs/<RUN_NAME>/code`, changes into that fetched tree, and runs the mode-appropriate Make target there (`make hp-submit` for pretrain, `make hp-submit-finetune` for fine-tune). For fine-tune, the pretrained-backbone source is config-driven — read from `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in `code/scripts/finetune/configs/finetune_training_config.py`, not passed as a CLI argument. Use when the user wants to launch or relaunch a training run on HyperPod after code or config changes, while reusing an existing image and existing FSx dataset/tokenizer artifacts.
---

# Submit HyperPod Training Job

## Overview

Use this skill for the repo's normal HyperPod training loop when the image already exists and the user is iterating on code or configs. The skill covers **two job modes**:

- **`pretrain`** — full from-scratch pretraining run via `make hp-submit` (pulls `code/src/efm_core/contracts/dataset_specifications/v1.py` and `_DEFAULT_TOKENIZER_JOB_NAME` from the pretrain training config).
- **`finetune`** — fine-tuning a pretrained EFM backbone via `make hp-submit-finetune` (pulls `code/src/efm_finetune/contracts/dataset.py` for the FT-specific dataset name; tokenizer is inherited from the backbone artifact, so there is no separate tokenizer-job-name to resolve; the **pretrained-backbone source is config-driven** — read from `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in `code/scripts/finetune/configs/finetune_training_config.py`, hand-maintained by the team, **not** passed at submit time).

The procedure is:

1. determine the **job mode** (`pretrain` or `finetune`)
2. determine a unique `RUN_NAME`
3. stage the current repo snapshot locally
4. use `agent-toolkit/skills/hyperpod-access` to run `ENV=<ENV> /fsx/submit/fetch_staged_changes_for_run.sh <RUN_NAME>` on the controller
5. run the mode-appropriate Make target (`make hp-submit` or `make hp-submit-finetune`) against the fetched tree using `make -C /fsx/runs/<RUN_NAME>/code <target>`, with runtime inputs derived from repo config unless the user overrides them

This skill is repo-specific. It is self-sufficient and should be followed directly without depending on `hyperpod/NEW_USER_GUIDE.md`.

## Required Inputs

Collect these before starting:

- `MODE` — `pretrain` or `finetune`. Always ask the user explicitly; do **not** infer from filenames or recent commits. The two modes use different Make targets, different dataset contracts, and different `RUN_NAME_PREFIX` conventions. Picking the wrong mode silently submits to the wrong entrypoint and wastes a job slot.
- `AWS_PROFILE` for local AWS access from the workstation
- `ENV` as `dev` or `prod`
- `RUN_NAME_PREFIX` — the human-readable prefix for this run; `hp-stage-code` appends a UTC timestamp to produce the unique `RUN_NAME`. Default shapes per mode are listed in the table below (e.g. `efm-pretrain-tayn` for pretrain, `efm-ft-fmm-tayn` for FT). Ask the user when not provided; do not invent it.
- `SLURM_MEM` only when the run must override Slurm's default memory allocation

For `MODE=finetune`, also **ask the user**:

- `CONFIG_NAME` — the Hydra profile to use (e.g. `finetune-lora-ppma`, `finetune-lora-fmm`). List the available profiles from `code/scripts/finetune/configs/yaml/profile/` before asking. Do **not** default or infer from the run name — picking the wrong profile silently uses the wrong dataset, optimizer, and learning rate and can corrupt the run. Pass as `CONFIG_NAME=<value>` in the controller-side `env` prefix when submitting.

For `MODE=finetune`, also **verify** (not collect from the user):

- `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` is non-empty in `code/scripts/finetune/configs/finetune_training_config.py`. The FT submission is config-driven; the value is hand-maintained by the team in that file. If it's empty, **stop**: tell the user to set it in the config (and commit the change) before this skill can submit. Do not fill it in yourself, do not ask the user to pass it on the command line — the Make target / sbatch script no longer accept it as an env var, and the FT runtime will reject an empty value with an actionable error anyway.

Resolve these from repo code before asking the user to provide overrides — the source files differ by mode:

| Field | Pretrain (`MODE=pretrain`) | Finetune (`MODE=finetune`) |
|---|---|---|
| `DATASET_NAME` | `code/src/efm_core/contracts/dataset_specifications/v1.py` | `code/src/efm_finetune/contracts/dataset.py` (FT contract: `efm_finetune_dataset`) |
| `DATASET_VERSION` | `code/src/efm_core/contracts/dataset_specifications/v1.py` | `code/src/efm_core/contracts/dataset_specifications/v1.py` (shared with pretrain) |
| Tokenizer | `_DEFAULT_TOKENIZER_JOB_NAME` from `code/scripts/configs/training_config.py` (HyperPod resolves the tokenizer path as `/fsx/ml-artifacts/tokenizers/<tokenizer_job_name>`) | **None to resolve** — tokenizer is inherited from the pretrained backbone artifact directory |
| Pretrained backbone | n/a | `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in `code/scripts/finetune/configs/finetune_training_config.py`. **Config-driven, hand-maintained**: verify non-empty before submit; do not pass at the CLI |
| Default `RUN_NAME_PREFIX` shape | `efm-pretrain-<owner>` (e.g. `efm-pretrain-tayn`) | `efm-ft-<task>-<owner>` (e.g. `efm-ft-fmm-tayn`) |
| Make target | `make hp-submit` | `make hp-submit-finetune` |
| FSx data root used inside the container | `/fsx/ml-datasets` (per `hyperpod/train.sbatch`) | `/fsx/ml-datasets-finetune` (per `hyperpod/finetune.sbatch`) |
| `CONFIG_NAME` (Hydra profile) | n/a | **Always ask the user** — list profiles from `code/scripts/finetune/configs/yaml/profile/` before asking. Pass as `CONFIG_NAME=<value>` in the controller-side `env` prefix. Do not default or infer. |
| Submit `--export` env vars | `RUN_NAME`, `ENV`, `CONTAINER_IMAGE_URI`, `REPO_ROOT`, `AWS_REGION`, `ML_PLATFORM_RUNTIME_ENV`, `ML_PLATFORM_VENDOR_ENV`, optional `RESUME_FROM_CHECKPOINT` | Same set as pretrain, **plus `CONFIG_NAME`**. The FT backbone is **not** an env var — it lives in the staged config file |

`hp-stage-code` generates the full `RUN_NAME` by appending a UTC timestamp and prints it at the end of its output — same behavior in both modes.

Collect these only when the derived defaults are not correct:

- `CONTAINER_IMAGE_URI` when the run must use a specific `.sqsh` image on FSx (the same image works for both modes; the Make targets share the same default)
- `DATASET_NAME` and `DATASET_VERSION` when the run should use a different dataset than the repo default
- `TOKENIZER_ARTIFACTS_PATH` (pretrain only) to override the default tokenizer path
- `RUN_NAME`
- `RESUME_FROM_CHECKPOINT`
- `MODEL_OUTPUT_DIR`
- `TRAINING_CHECKPOINTS_DIR`

## Workflow

1. Confirm the task fits this skill.

- Use this skill when the user is changing code or configs and wants to submit a training run.
- Do not rebuild the image by default for code-only or config-only changes.
- Do not use controller-side `git clone` as the primary path.

2. Ask for missing inputs.

- **First, ask for `MODE`** (`pretrain` or `finetune`) if the user did not state it. Do not guess.
- Always ask for `AWS_PROFILE` before any local AWS access from the workstation.
- Ask for `ENV` if the user did not provide it.
- When `MODE=finetune`, **ask for `CONFIG_NAME`** — list profiles from `code/scripts/finetune/configs/yaml/profile/` and ask the user to pick one. Do not default or infer it.
- When `MODE=finetune`, **read** `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` from `code/scripts/finetune/configs/finetune_training_config.py` and confirm it is non-empty. Do not ask the user to provide it on the command line. If it's empty, stop and tell the user to set it in the config file (and commit) before re-invoking this skill.
- Ask for `SLURM_MEM` only when the user wants to override the cluster default.
- Resolve dataset (and, for pretrain, tokenizer) defaults from repo code before asking the user for overrides — see the mode-specific table in Required Inputs.
- Ask only for image, dataset, tokenizer, or checkpoint overrides that are actually needed.

3. Stage the current repo snapshot locally.

`RUN_NAME` is unique for the run and scopes:
  - `s3://ml-platform-datalake<env>-us-west-2/hyperpod/runs/<RUN_NAME>/code.tar.gz`
  - `/fsx/runs/<RUN_NAME>/code`
  - `/fsx/runs/<RUN_NAME>/logs`

Run from the repo root:

```bash
make hp-stage-code RUN_NAME_PREFIX="$RUN_NAME_PREFIX" ENV="$ENV" AWS_PROFILE="$AWS_PROFILE"
```

The command prints the generated `RUN_NAME` in its output. Capture it from the line `RUN_NAME=<value>` before proceeding to the controller fetch step.

Echo the captured `RUN_NAME` back to the user before any controller-side use and confirm it ends in `Z`. `hp-stage-code` always appends a `%Y%m%d-%H%M%SZ` suffix, so a run name without the trailing `Z` is a copy-paste error and will not match the staged S3 path or the `/fsx/runs/<RUN_NAME>` directory created by the fetch step.

Preserve `AWS_CA_BUNDLE` from the caller environment when AWS CLI TLS validation requires it.

4. Use the shared controller-access skill.

- Read `agent-toolkit/skills/hyperpod-access/SKILL.md`.
- Use `agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh` for one-off controller commands.
- Do not reconstruct Session Manager access manually.
- Use the user-supplied `AWS_PROFILE` only for workstation-side discovery and Session Manager access.
- Do not assume the same named AWS profile exists on the controller.
- For controller-side `/fsx/submit/fetch_staged_changes_for_run.sh` and the submit Make target (`hp-submit` or `hp-submit-finetune`), prefer the controller's existing AWS configuration unless the user explicitly says the controller itself must use a named profile.
- The uploaded fetch script takes `RUN_NAME` as its positional input and reads `ENV` from the controller environment.

5. Resolve derived runtime values from repo code.

For **`MODE=pretrain`**:

- Read `code/src/efm_core/contracts/dataset_specifications/v1.py` and capture `DATASET_NAME` and `DATASET_VERSION`.
- Read `code/scripts/configs/training_config.py` and capture `_DEFAULT_TOKENIZER_JOB_NAME` for reference; the training code resolves the tokenizer path automatically via `tokenizer_job_name` when `--mode=hyperpod`.

For **`MODE=finetune`**:

- Read `code/src/efm_finetune/contracts/dataset.py` and capture the FT `DATASET_NAME` (currently `efm_finetune_dataset`).
- Read `code/src/efm_core/contracts/dataset_specifications/v1.py` and capture `DATASET_VERSION` (the FT side reuses the shared version field; FT does not redefine it).
- Do **not** resolve a tokenizer-job-name. The FT entrypoint (`code/scripts/finetune/entrypoint.py`) loads the tokenizer from the pretrained backbone artifacts directory under `/fsx/runs/<configured backbone>/`.
- Read `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` from `code/scripts/finetune/configs/finetune_training_config.py`. Capture the resolved string for the confirmation step in 6 and for the run summary. If empty, stop and tell the user to hand-edit the constant before this skill can submit.
- After staging (step 3), verify the configured backbone matches a run name with an artifact directory already on FSx — `ls -ld /fsx/runs/<configured backbone>/` via the `hyperpod-access` skill catches typos in the constant before Slurm accepts the job. The deeper artifact path is resolved by the FT entrypoint at run time; we only sanity-check the top-level run dir exists here.

For **both modes**:

- Capture the generated `RUN_NAME` from the `hp-stage-code` output and reuse it verbatim across fetch and submit.
- Let the training code resolve the default MLflow experiment name from `code/scripts/configs/infra_config.py`.

6. Fetch the staged bundle onto the controller, verify the fetched tree exists, then submit from the fetched tree.

The controller helper runs commands directly, not through an implicit shell. Use one controller command to fetch the staged bundle, a second controller command to verify `/fsx/runs/<RUN_NAME>/code` exists, and only then a third controller command to submit from that fetched tree. Do not run fetch and submit in parallel. The fetch and verify steps are identical for both modes; only the final submit command differs.

Fetch (both modes):

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'env ENV='"$ENV"' /fsx/submit/fetch_staged_changes_for_run.sh '"$RUN_NAME"
```

Verify (both modes):

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'ls -ld /fsx/runs/'"$RUN_NAME"'/code'
```

**Confirm before submitting** (both modes — do NOT run the submit command without this gate). Submitting a HyperPod job is a hard-to-reverse, expensive action: it claims a node slot, consumes cluster time, and writes side effects to FSx and MLflow. Before running the third controller command, echo the fully-resolved invocation back to the user and wait for explicit acknowledgement. Restate, at minimum:

- `MODE` and the corresponding Make target — e.g. "MODE=finetune → `make hp-submit-finetune`" or "MODE=pretrain → `make hp-submit`". This is the single most common mistake; if the target doesn't match the mode the user asked for, fix it now, not after the job has run.
- `RUN_NAME` generated by `hp-stage-code` (verbatim, including the trailing `Z`).
- For `MODE=finetune`: the value of `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` you read from the staged `finetune_training_config.py`. The user is acknowledging which backbone the FT job will load — if the constant is wrong, they should abort the confirmation, hand-edit the file, re-stage, and re-confirm.
- Any non-default `CONTAINER_IMAGE_URI`, `RESUME_FROM_CHECKPOINT`, or `SLURM_MEM` override that the user supplied.

Ask explicitly ("Submit with this configuration?") and wait for the user to acknowledge before running the submit command. If they notice a mismatch — wrong mode, wrong backbone in the staged config, wrong image — fix the inputs (re-edit the config + re-stage if it's the backbone) and re-confirm. **Do not proceed to the submit command without explicit acknowledgement.** When this skill is invoked from an autonomous loop that pre-declares all inputs (e.g. `autoresearch-ft`), echo the resolved invocation to the conversation transcript anyway so the human reading after the fact can audit each submission; the loop's pre-flight block (e.g. `=== HYPOTHESIS ===`) already records the change and reasoning, but the resolved Make target / RUN_NAME / staged backbone must still be printed at the moment of submission.

Submit, **`MODE=pretrain`**:

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'env RUN_NAME='"$RUN_NAME"' \
CONTAINER_IMAGE_URI='<resolved-container-image-uri>' \
make -C /fsx/runs/'"$RUN_NAME"'/code hp-submit ENV='"$ENV"'
```

Submit, **`MODE=finetune`** (the Make target is `hp-submit-finetune`; the pretrained-backbone source is read from the staged `finetune_training_config.py` and is **not** passed as an env var):

```bash
agent-toolkit/skills/hyperpod-access/scripts/run_hyperpod_controller_command.sh \
  --aws-profile "$AWS_PROFILE" \
  --command 'env RUN_NAME='"$RUN_NAME"' \
CONTAINER_IMAGE_URI='<resolved-container-image-uri>' \
make -C /fsx/runs/'"$RUN_NAME"'/code hp-submit-finetune ENV='"$ENV"'
```

7. Fill the template with resolved values before execution.

- Replace `<resolved-container-image-uri>` with the user-provided `.sqsh` image path when the run should not use the Makefile default image.
- For `MODE=finetune`, the controller-side `env ...` prefix does **not** include `PRETRAINED_EFM_BACKBONE_NAME` — the value is read from the staged `finetune_training_config.py` at FT runtime. Verify the staged value before submit instead (see step 5's FT branch).
- Do not leave placeholder strings in the executed command.
- If the user overrides dataset or tokenizer inputs, replace the derived values with the user-provided ones.
- Add optional overrides such as `SLURM_MEM`, `RESUME_FROM_CHECKPOINT`, `MODEL_OUTPUT_DIR`, or `TRAINING_CHECKPOINTS_DIR` only when they are actually needed.
- If the existence check does not show `/fsx/runs/<RUN_NAME>/code`, stop and fix the fetch step before attempting the submit Make target.

8. Avoid passing empty overrides when they are unset.

- Include the user's `AWS_PROFILE` for workstation-side commands that talk to AWS directly, such as local staging and controller discovery.
- Do not automatically pass the user's workstation `AWS_PROFILE` through to controller-side `make` commands.
- Only pass controller-side `AWS_PROFILE` when the user explicitly confirms that the controller is configured to use that named profile.
- Include `SLURM_MEM` only when the user explicitly set it.
- Include `RUN_NAME` every time; it is the primary key scoping all run artifacts.
- Include optional variables only when the user supplied them or when the default must be replaced.
- Do not rely on shared code defaults such as `.../efm-core.tar.gz`, `/fsx/code/...`, or flat `/fsx/logs/...`.
- Do not pass `ENV` as a positional argument to `/fsx/submit/fetch_staged_changes_for_run.sh`; set it in the controller command environment instead.
- When the default image path is correct, omit `CONTAINER_IMAGE_URI` and let the Make target use the Makefile-owned default image path (the same default applies to both `hp-submit` and `hp-submit-finetune`).
- If the run should use a different image on FSx, pass `CONTAINER_IMAGE_URI=/fsx/enroot/...sqsh` explicitly.
- For `MODE=finetune`, if `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` is empty in the staged config, **stop** and tell the user to set it in `code/scripts/finetune/configs/finetune_training_config.py` and commit. Do not edit the constant yourself, do not fall through to `hp-submit`, do not invent a backbone name. The FT runtime would reject the empty value anyway with an actionable error; catching it pre-submit just saves a wasted Slurm round-trip.

9. Verify that the job actually starts.

- After submission, run controller-side Slurm checks instead of assuming the job started correctly.
- Check `squeue` for the submitted job id or for the current user.
- Inspect both `/fsx/runs/<RUN_NAME>/logs/out` and `/fsx/runs/<RUN_NAME>/logs/err`.
- Use concrete log reads such as `cat`, `tail`, or `sed -n` and report the observed output.
- If the job does not appear in `squeue` or the logs show an immediate launcher failure, report that as a submission failure, not a successful launch.

10. Report the submission result and immediate next steps.

- Capture the Make-target summary (the `Submitting HyperPod job with:` block from `hp-submit`, or the `Submitting HyperPod fine-tune job with:` block from `hp-submit-finetune`) and the `Submitted batch job <id>` line.
- Tell the user where logs will appear:
  - `/fsx/runs/<RUN_NAME>/logs/out`
  - `/fsx/runs/<RUN_NAME>/logs/err`
- Include the first post-submit `squeue` check and the first `.out` / `.err` inspection in the normal verification path.
- If requested, continue with additional controller commands such as repeated `squeue`, `tail`, or `cat`.

## Specific Techniques

- Read `hyperpod/Makefile` for the exact behavior of `hp-stage-code`, `hp-submit`, and `hp-submit-finetune`. The FT target only requires `RUN_NAME` — the backbone is config-driven (see the FT row in the Required-Inputs table above) and the Make target no longer accepts `PRETRAINED_EFM_BACKBONE_NAME` as an env var.
- Read `hyperpod/train.sbatch` (pretrain) or `hyperpod/finetune.sbatch` (FT) when deciding whether dataset, tokenizer, output, or checkpoint env vars must be overridden. The two scripts differ in input data root (`/fsx/ml-datasets` vs `/fsx/ml-datasets-finetune`) and in which env vars they consume.
- For pretrain: read `code/src/efm_core/contracts/dataset_specifications/v1.py` for dataset name and version, and `code/scripts/configs/training_config.py` for `_DEFAULT_TOKENIZER_JOB_NAME`.
- For finetune: read `code/src/efm_finetune/contracts/dataset.py` for the FT dataset name (the dataset version still lives in `code/src/efm_core/contracts/dataset_specifications/v1.py` and the FT side reuses it), and read `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` from `code/scripts/finetune/configs/finetune_training_config.py` to capture the configured backbone target.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The user said 'training' so it's pretrain by default." | Always ask explicitly for `MODE`. The user's word "training" covers both pretrain and FT in this repo; guessing wrong submits to the wrong Make target. |
| "I'll just use `hp-submit` for the FT run, the entrypoint will figure it out." | No. `hp-submit` invokes `hyperpod/train.sbatch` (pretrain entrypoint, pretrain dataset root). FT requires `hp-submit-finetune`, which submits `hyperpod/finetune.sbatch`; the FT entrypoint reads its backbone target from `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in the staged `finetune_training_config.py`. |
| "The backbone name should be a CLI arg / env var like every other parameter." | It used to be — that contract was retired. The FT submission API is config-driven: hand-edit `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in `code/scripts/finetune/configs/finetune_training_config.py` and commit. The Make target / sbatch script no longer accept the env var, and the FT runtime rejects an empty constant with an actionable error. |
| "I changed only Python code, so the controller checkout is close enough." | The job runs the staged S3 snapshot, not an arbitrary checkout on FSx. Restage code before submission. |
| "The image already exists, so I can skip `hp-stage-code`." | Image reuse does not update training code. Code and image are separate inputs. |
| "The dataset path in the sbatch script is close enough." | This skill should derive dataset inputs from the mode-appropriate dataset contract first (efm_core for pretrain, efm_finetune for FT), then build the FSx paths from those values. |
| "The tokenizer path can stay generic for FT too." | FT does not consume `tokenizer_job_name` at all — the FT entrypoint loads the tokenizer from the pretrained-backbone artifacts directory, which is itself resolved from the config-driven `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME`. Do not pass `TOKENIZER_ARTIFACTS_PATH` for FT runs. |
| "I know how to reach the controller, so I can type the SSM commands directly." | Use the shared `hyperpod-access` skill and its scripts as the source of truth for controller access. |
| "The workstation `AWS_PROFILE` should also be passed into `make` on the controller." | The controller may not have that named profile configured. Use the workstation profile for Session Manager access, but let controller-side AWS calls use the controller's own working AWS configuration unless the user explicitly says otherwise. |

## Red Flags

- The agent submits without asking the user for `MODE` (`pretrain` or `finetune`).
- For `MODE=finetune`, the agent does not ask for `CONFIG_NAME` before staging or submitting. Defaulting or inferring the profile from the run name or recent commits is wrong — different profiles use different datasets, optimizers, and learning rates, and picking the wrong one silently corrupts the run.
- The agent skips the pre-submit confirmation gate (the "Confirm before submitting" paragraph in Workflow step 6) and runs the submit Make target without echoing the resolved invocation back to the user. Submission is hard-to-reverse; the confirmation is non-optional in interactive use.
- The agent uses `make hp-submit` for a fine-tune job, or `make hp-submit-finetune` for a pretrain job — the Make target must match the chosen `MODE`.
- For `MODE=finetune`, the agent edits `_DEFAULT_PRETRAINED_EFM_BACKBONE_NAME` in the config (it is human-maintained and out of this skill's edit scope), invents a backbone name to fill in an empty constant, or proceeds to submit when the constant is empty. Correct response on empty: stop and ask the user to set it in the config and commit.
- For `MODE=finetune`, the agent passes `PRETRAINED_EFM_BACKBONE_NAME` as an env var to the controller-side `env ...` prefix or to the Make invocation. The submission API no longer accepts that env var; passing it is a sign the agent is following a stale contract.
- For `MODE=finetune`, the agent reads `code/src/efm_core/contracts/dataset_specifications/v1.py` for the dataset name (it should read `code/src/efm_finetune/contracts/dataset.py` for the name and only use the efm_core spec for the shared version field).
- For `MODE=finetune`, the agent passes `TOKENIZER_ARTIFACTS_PATH` — FT inherits the tokenizer from the backbone artifact and does not consume that override.
- The agent submits without first staging the current repo snapshot.
- The agent rebuilds or republishes the image even though the task is code-only or config-only.
- The agent forces `SLURM_MEM` even though the cluster default is acceptable.
- The agent passes the workstation `AWS_PROFILE` into controller-side `/fsx/submit/fetch_staged_changes_for_run.sh` or the submit Make target without confirming that the controller has that named profile configured.
- The agent hardcodes dataset or tokenizer paths instead of deriving them from repo config first.
- The agent stages, fetches, or submits without reusing the same `RUN_NAME`.
- The agent constructs `RUN_NAME` manually instead of letting `hp-stage-code` generate it from `RUN_NAME_PREFIX`.
- The agent uses a `RUN_NAME` for fetch/submit that differs from the one printed by `hp-stage-code`.
- The agent assumes derived defaults even after the user specified overrides.
- The agent uses a controller-side git workflow instead of the staged S3 bundle.

## Verification

Confirm the workflow with observable evidence:

- Keep the stdout or stderr from `make hp-stage-code`.
- Keep the stdout or stderr from `run_hyperpod_controller_command.sh`.
- Report the Make-target summary block (`Submitting HyperPod job with:` from `hp-submit`, or `Submitting HyperPod fine-tune job with:` from `hp-submit-finetune`).
- Report the `Submitted batch job <id>` line, or the exact failure if submission did not succeed.
- Report the first `squeue` result after submission.
- Report the first observed contents from both `/fsx/runs/<RUN_NAME>/logs/out` and `/fsx/runs/<RUN_NAME>/logs/err`.
- If monitoring continues, report concrete `squeue` or log output instead of assuming the job is healthy.
