---
name: code-quality-audit
description: Audit the efm-core codebase for known anti-patterns across production code and tests. Use when the user asks for a code quality check, a cleanup audit, or wants a prioritized list of issues to address. Also use when given a PR number to scope the audit to changed files only. Produces a categorized, file-and-line-anchored report. Does not modify any files.
---

# Code Quality Audit

## Overview

Scan the codebase for recurring quality issues identified through prior audits. The output is a prioritized, actionable report anchored to specific files and line numbers. This skill is read-only — it finds and reports, never fixes.

Supports two modes:

- **Full audit** (no PR number): scans the entire codebase across all categories
- **PR mode** (PR number provided): scopes to files changed in that PR only — designed to run alongside `/review` which covers general bugs and CLAUDE.md compliance

## Inputs

- `PR` (optional) — GitHub PR number. When provided, restrict all category scans to files changed in that PR. Fetch the list with:
  ```bash
  gh pr diff --name-only <PR>
  ```
  If the PR is closed, draft, or trivially small (e.g. docs-only, single-line change), report that and stop.

Default scan directories when no PR is provided:
- `code/src/efm_core/` — model, data, validation
- `code/scripts/` — configs, train entrypoint, trainer, dataloader init, mlflow
- `code/jobs/` — SageMaker estimator
- `tests/` — unit tests
- `integration-tests/` — smoke tests, distributed tests

## Workflow

### 1. Orient

**In PR mode:** fetch the changed file list first:
```bash
gh pr diff --name-only <PR>
gh pr view <PR> --json title,state,isDraft,body
```
Stop if the PR is closed, draft, or the changed files are entirely outside the scanned directories (e.g. docs, CI config only).

**In both modes:** read the following files to understand current state before scanning:
- `ANTI_PATTERNS.md` (if present) — prior findings; skip any finding already listed there unless the code has not been fixed
- `code/scripts/configs/training_config.py` — config builders and param declarations
- `code/src/efm_core/model/config.py` — EFMConfig and DeclaredModelParams
- `code/scripts/train/entrypoint.py` — main training entry point

### 2. Scope files

- **Full audit:** use the default directory list above
- **PR mode:** use only the files returned by `gh pr diff --name-only`. If a changed file is a test, also read its corresponding source file for context (but only report findings in the changed files)

### 3. Scan each category

Launch parallel agents (or read files sequentially) to check all categories below. For each finding record:
- **Category** (from the list below)
- **Severity** — Critical / High / Medium / Low
- **File and line number(s)**
- **What the problem is** (one sentence)
- **Concrete fix** (one sentence or code snippet)

---

#### Category A — Duplicated Logic
Look for:
- Near-identical functions that differ only in a few constant values (especially config builders — `build_training_config` vs `build_local_training_config`, `build_model_config` vs `build_local_model_config`)
- Copy-pasted guards, loops, or tensor construction in `collate.py`
- Repeated `import` statements inside function bodies when the module is always available
- Duplicated port-allocation or env-construction patterns in `conftest.py`
- The same constant defined in more than one file

Files to focus on:
`code/scripts/configs/training_config.py`, `code/scripts/configs/model_config.py`, `code/src/efm_core/data/collate.py`, `code/scripts/train/initialize/initialize_dataloader.py`, `integration-tests/conftest.py`

---

#### Category B — Redundant Type Coercions
Look for:
- `int(x)`, `float(x)`, `str(x)`, `bool(x)` applied to values that are already that type at the call site
- Fields coerced both in `__init__` and in `__post_init__` of the same logical object
- Coercions applied inside model forward passes where the value was already typed at construction

Files to focus on:
`code/src/efm_core/model/config.py`, `code/src/efm_core/model/efm_core_model.py`

---

#### Category C — Dead Code and Unreachable Branches
Look for:
- `if CONSTANT == "value":` where the constant is set to a different literal on a nearby line and never changes (e.g. `_DEFAULT_MODEL_TYPE`)
- Fallback branches that re-call the same function whose result is already guarded earlier
- Guards like `if not hasattr(x, "method"):` where `x` is always the same type and always has that method
- `del local_var` at the end of a function (no-op)
- Workaround overrides with a comment that says "CPU multi-rank" or similar — verify they're still needed

Files to focus on:
`code/scripts/configs/training_config.py`, `code/scripts/train/entrypoint.py`, `code/scripts/train/initialize/initialize_dataloader.py`, `code/scripts/train/trainer.py`

---

#### Category D — Exception Handling Anti-Patterns
Look for:
- `except Exception:` that returns a null-object or silently swallows errors when `except ImportError:` is the correct scope
- `except Exception: return` with no logging or re-raise
- `assert` statements used to enforce user-facing constraints (wrong config, wrong input type) rather than internal invariants — these should be `raise ValueError`
- `assert` statements that would be disabled by `python -O` but guard runtime-critical conditions

Files to focus on:
`code/scripts/train/entrypoint.py`, `code/scripts/train/mlflow_log.py`, `code/src/efm_core/model/efm_core_model.py`

---

#### Category E — Missing or Wrong Type Annotations
Look for:
- Parameters typed as `object` instead of a concrete type or `Callable`
- Module-level variables with no type annotation (especially mutable state like `_warned_once`)
- Functions with no return type annotation
- Parameters suppressed with `# noqa: ANN001` instead of being properly typed
- Union types like `str | None` or `int | None` where `None` is never a valid value at runtime

Files to focus on:
`code/scripts/train/mlflow_log.py`, `code/scripts/train/initialize/initialize_dataloader.py`, `code/src/efm_core/model/attention_backend.py`, `code/src/efm_core/data/dataset.py`, `code/scripts/train/trainer.py`

---

#### Category F — Config Inconsistencies
Look for:
- Fields in `from_dict()` that use raw `.get()` with a default while all other fields use `require_*()` validators
- Hardcoded string literals that should be named `_LOCAL_*` constants (especially in `build_local_training_config`)
- Sentinel `None` values that carry implicit meaning not captured in the type or documented anywhere
- **Overengineered `_UNSET = object()` sentinels** — module-level `_UNSET = object()` (or any custom `object()` sentinel) used as a parameter default when `None` is never a legitimate value for that parameter. The sentinel is only justified when the function must distinguish "argument omitted" from "argument explicitly set to `None`". If `None` cannot be a real value, replace `param: T = _UNSET` with `param: T | None = None` (or just `param: T` with no default if the value is required). Flag any new instances introduced in production code.
- **Default values on required configuration kwargs** — in functions whose job is to apply or build a training-time configuration (e.g. `apply_peft`, optimizer/scheduler builders, model-construction helpers), keyword arguments like `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`, `freeze_backbone`, etc. should not carry inline defaults. Defaults silently mask config errors and create a second source of truth alongside the declared config object. Callers must pass values explicitly (or pass a config object that owns the defaults). Flag any kwarg with a literal default in these helpers.
- Public module-level constants (`ALL_CAPS`) that are only used internally and should be `_PRIVATE`
- HF config flags whose failure modes are untested

Files to focus on:
`code/scripts/configs/training_config.py`, `code/src/efm_core/model/config.py`, `code/scripts/train/entrypoint.py`, any `apply_peft` / PEFT helper module

---

#### Category G — Stale or Misleading Comments
Look for:
- Comments that say "legacy" or "old" on code that is actually the primary path
- TODO or "subsequent PR" comments referencing work that has already landed
- Two-phase init patterns documented only by a `# resolved after X` comment with no explanation of why the pattern exists
- Comments that describe what the code does rather than why

Files to focus on:
`code/src/efm_core/model/efm_core_model.py`, `code/scripts/configs/model_config.py`, `code/scripts/train/entrypoint.py`

---

#### Category H — Test Quality
This is the highest-value category. Look for:

**Silent pass anti-patterns (Critical):**
- `if m:` guards around assertions where a regex failing to match should be a test failure, not a skip
- List slices or comparisons that trivially pass on an empty list (e.g. `list[-0:]`, `len(x) > 0`)
- Checkpoint or artifact existence tests that check metadata but not weight files

**Exact-value assertions on derived or configurable quantities (High):**
- Assertions on computed step counts, row counts, epoch counts that will break whenever batch size or dataset size changes
- Assertions on specific default hyperparameter values (learning rate, seed, precision) that will break when defaults are tuned
- For each such assertion: replace with a structural invariant (e.g. `total_steps == steps_per_epoch * num_epochs`)

**Test infrastructure problems (Medium):**
- Fixture helpers defined in a test file and imported by other test files — these belong in `conftest.py`
- `import` of standard library modules inside fixture bodies instead of at module level
- Direct mutation of module-level private state (e.g. `module._flag = False`) instead of `monkeypatch.setattr`
- `sys.modules` patching that affects the global import system for the entire test session

**Stale fixture defaults (High):**
- Check `tests/_declared_model_params_fixture.py` — verify `rope_theta` matches the current production default in `code/scripts/configs/model_config.py`
- Check any fixture that hardcodes model shape parameters and verify they match current defaults

**Distributed test invariants (Medium):**
- Hardcoded per-rank sequence counts — verify they are derived from the fake dataset size and document the derivation
- Missing existence checks before reading probe output files
- Sharding assertions that check token-level content instead of index-level partition correctness

Files to focus on:
`tests/test_pretraining_configs.py`, `tests/_declared_model_params_fixture.py`, `tests/test_sequence_builder.py`, `tests/test_attention_backend.py`, `integration-tests/test_smoke.py`, `integration-tests/test_distributed.py`, `integration-tests/conftest.py`

---

#### Category I — God Functions and Structural Issues
Look for:
- Functions longer than ~100 lines that handle more than one concern (config building, IO, validation, dispatch all in one function)
- Functions with two-phase initialization (variable set to `None`, then reassigned 20+ lines later)
- Hardcoded infrastructure constants (ARNs, paths, magic numbers) that should be named constants

Files to focus on:
`code/scripts/train/entrypoint.py`, `code/jobs/estimator.py`

---

### 3. De-duplicate and prioritize

Before reporting:
- Check if the issue is already tracked in `ANTI_PATTERNS.md` — if it's there and unresolved, include it; if it's there and resolved (code has changed), skip it
- Assign severity:
  - **Critical** — test silently passes on broken behavior, data correctness risk
  - **High** — wrong behavior at runtime, swallowed errors, stale fixture wrong values
  - **Medium** — duplication, maintainability debt, structural issues
  - **Low** — style, naming, annotations

### 4. Report format

**Full audit mode** — output a markdown report with:
1. **Summary table** — count of findings by severity and category
2. **Critical findings** — full detail first
3. **High findings** — full detail
4. **Medium / Low** — grouped by category, briefer

Each finding:
```
### [SEVERITY] Short title
File: path/to/file.py:line
Problem: one sentence
Fix: one sentence or short code snippet
```

**PR mode** — confidence-score each finding before reporting:
- Score each finding 0–100 on whether it was *introduced or worsened* by this PR (not pre-existing issues unrelated to the change)
- Drop findings scored below 60
- Output only findings scored 60+, ordered by score descending
- Format each finding to be compatible with `/review` output:
  ```
  [SEVERITY] Short title

  path/to/file.py:line — Problem description. Suggested fix.
  ```
- Keep output brief; avoid restating context the `/review` comment already covers
- Do not post a GitHub comment — output to the conversation only; the user decides whether to post it

## What to Skip

Do not report:
- Anything the user has explicitly said is planned for a future PR (check conversation context)
- Test assertions on values that are intentionally contract-pinned (ask if unclear)
- Type annotation gaps in third-party subclasses where the parent type is not annotatable (e.g. HF Trainer callback signatures that are typed `Any` upstream)
- `uv run` or other toolchain invocations — these are not anti-patterns

## Red Flags

- Reporting a finding without a file path and line number
- Reporting something as "dead code" without verifying the constant's value
- Marking a test assertion as wrong without checking whether the value is intentionally pinned
- Modifying any file (this skill is read-only)
- Running `make`, `uv run`, or any build/test command

## Verification

The audit is complete when:
- All 9 categories have been checked
- Every finding has a file path, line number, severity, and fix
- The summary table is present
- No files were modified
