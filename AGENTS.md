# AGENTS.md

This file governs agent behavior in this repository.

## Skills

- Before inventing a workflow, check `skills/` for a relevant repo-specific skill.
- If no repo-specific skill applies, check `agent-toolkit/skills/` for a relevant shared skill.
- If a skill applies, follow the relevant `SKILL.md` and use any bundled helper scripts from that skill.
- Treat `agent-toolkit/AGENTS.md` as guidance for maintaining the toolkit repository, not as the governing instruction file for this repository.

## Repository-specific guidance

- Keep repository-specific behavior, constraints, and conventions in this file or other files in this repository.
- Add new shared operational workflows to `agent-toolkit` when they are reusable across repositories.

## Tests

Do not add `sys.path` manipulation (`REPO_ROOT / CODE_ROOT / SRC_ROOT` blocks) to test files. `pyproject.toml` already declares `pythonpath = ["code", "code/src"]` under `[tool.pytest.ini_options]`, so pytest resolves imports automatically. Run tests via `uv run pytest` or `make` — never patch `sys.path` at module level.

## Directory layout

### `../efm-token-decoder/tools/`

One-off scripts for manual operations against live infrastructure (S3 inspection, tokenizer patching, etc.). These are **not** part of the training pipeline and are never invoked during training or CI. Run them ad hoc from within the `tools/` directory so that relative imports between scripts resolve correctly.

### `integration-tests/`

End-to-end smoke tests and their supporting artifacts.

- `resources/data/` — a committed fake dataset that conforms to the training data schema (7 train / 4 val / 2 test rows, each split across 2 parquet shards: `part-0.parquet`, `part-1.parquet`, plus `stats/stats.json`). Do not regenerate unless the schema changes; if you must, run `uv run python tools/generate_fake_dataset.py`.
- `resources/tokenizer/` — a committed tokenizer artifact used by local smoke-test runs.
- `output/` — gitignored; written by `make smoke`.
- `probes/` — standalone scripts designed to run under `torchrun`. Each probe exercises one distributed behavior (sharding, allreduce, etc.) in isolation, writes per-rank JSON output to a caller-supplied directory, and exits. Probes are not pytest tests — they are invoked by the `run_distributed` fixture in `conftest.py`. This pattern is modelled after PyTorch's distributed test suite (https://github.com/pytorch/pytorch/tree/main/test/distributed) and Megatron-LM's lightweight parallelism probes (https://github.com/NVIDIA/Megatron-LM/tree/main/tests).

Run a local end-to-end training pass with `make smoke`. This wires together local-mode config, the fake dataset, and a short training run without connecting to any cloud infrastructure. Both training phases (initial and resume) run once via a session-scoped pytest fixture (`smoke_run` in `conftest.py`); test functions in `integration-tests/` receive the shared run directory and captured stdout/stderr and assert on artifacts, loss curves, and resume correctness.

The `run_distributed` fixture (also in `conftest.py`) is a session-scoped factory that launches any probe under `torchrun --nproc_per_node=2` with a fresh OS-assigned port. Tests in `test_distributed.py` call it, read the per-rank JSON files, and assert cross-rank invariants (e.g. partitions are non-empty and disjoint).
