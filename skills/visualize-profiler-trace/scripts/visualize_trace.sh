#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <RUN_NAME> [AWS_PROFILE]}"
AWS_PROFILE="${2:-}"
LOCAL_DIR="$HOME/profiler_logs/$RUN_NAME"
S3_PATH="s3://ml-hyperpod-fsx-datalakeprod-us-west-2/runs/$RUN_NAME/profiler/"

AWS_CMD=(aws)
if [[ -n "$AWS_PROFILE" ]]; then
  AWS_CMD=(aws --profile "$AWS_PROFILE")
fi

echo "Checking $S3_PATH ..."
"${AWS_CMD[@]}" s3 ls "$S3_PATH" || {
  echo "ERROR: No profiler output found at $S3_PATH"
  echo "Verify the run name is correct and profiling was enabled."
  exit 1
}

mkdir -p "$LOCAL_DIR"
echo "Downloading traces to $LOCAL_DIR ..."
"${AWS_CMD[@]}" s3 sync "$S3_PATH" "$LOCAL_DIR/"

echo "Installing tensorboard with torch-tb-profiler ..."
uv tool install tensorboard --with torch-tb-profiler --with "setuptools<81" --force

# Patch TC_Allowlist to recognise flash-attn v2/v3 and nvJETPACK kernels.
# torch-tb-profiler uses substring name-matching (not hardware counters), so
# these kernels show "No" without this fix. No upstream issue filed as of 2026-04.
TC_FILE="$(uv tool run tensorboard -- -c "import torch_tb_profiler.profiler.tensor_core as m, inspect; print(inspect.getfile(m))" 2>/dev/null || true)"
# Fall back to locating the file directly under the uv tools tree
if [[ -z "$TC_FILE" || ! -f "$TC_FILE" ]]; then
  TC_FILE="$(find "$(uv tool dir)/tensorboard" -name "tensor_core.py" -path "*/torch_tb_profiler/*" 2>/dev/null | head -1)"
fi
if [[ -f "$TC_FILE" ]]; then
  python3 - "$TC_FILE" <<'PYEOF'
import sys, re
path = sys.argv[1]
src = open(path).read()
marker = "'c1688']"
additions = (
    "\n                 # Flash Attention v2 (flash-attn)\n"
    "                 'flash_fwd_kernel', 'flash_bwd_',\n"
    "                 # Flash Attention v3 / CUTLASS SM90 (wgmma)\n"
    "                 'FlashAttnFwd', 'FlashAttnBwd', 'enable_sm90',\n"
    "                 # nvJETPACK fused attention (Hopper)\n"
    "                 'nvjet_tst']"
)
patched = src.replace(marker, additions, 1)
if patched == src:
    print("TC_Allowlist patch: already applied or marker not found, skipping.")
else:
    open(path, 'w').write(patched)
    print("TC_Allowlist patch: applied flash-attn / nvjet patterns.")
PYEOF
fi

echo ""
echo "Launching TensorBoard. Open: http://localhost:6006"
echo "Select the 'PyTorch Profiler' tab."
echo "Press Ctrl+C to stop."
uv tool run tensorboard --logdir "$LOCAL_DIR"
