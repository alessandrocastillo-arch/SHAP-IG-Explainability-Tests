---
name: efm-endpoint-loadtest
description: Latency/throughput load-test of EFM embedding serving variants (torchlib / TRT-LLM bf16 / TRT-LLM fp8-gemm) on SageMaker real-time endpoints, across instance types (ml.g6 = L4, ml.g6e = L40S). Use when the user wants to measure or compare endpoint inference latency, throughput, or the cost of a serving backend or quantization mode. Covers building artifacts, deploying the endpoint matrix, driving it with Locust, rendering an HTML report, and tearing everything down. Not for correctness/parity (use efm_inference.parity) or saturated in-process throughput (use efm_inference.bench).
---

# efm-endpoint-loadtest

Measure end-to-end request latency of the EFM embedding model served from
SageMaker real-time endpoints, and compare serving variants × instance types.
The model is the **last-token embedding** of the finetuned decoder backbone;
one request = one user sequence, `request_output_len=1`.

Three serving variants (artifacts the deploy scripts consume):

| variant | backend | image | model dir | request builder |
|---|---|---|---|---|
| `torchlib` | Triton `pytorch` (TorchScript) | `…/tritonserver:26.05-libtorch` | `efm_libtorch` | `prepare_requests.libtorch_request` |
| `trt` (bf16) | Triton `tensorrtllm` | `…/tritonserver:26.05` | `tensorrt_llm` | `prepare_requests.kserve_request` |
| `trt-fp8` (fp8-gemm) | Triton `tensorrtllm` | `…/tritonserver:26.05` | `tensorrt_llm` | `prepare_requests.kserve_request` |

The load-test code lives in the `efm_inference` package; this skill is the
runbook + the HTML report builders. All host-side commands run in the standalone
inference env (`make inference_env` once; the `python -m` calls below assume
`PYTHONPATH=code/src` and `uv run --project code/src/efm_inference --no-sync`).

## Key facts / gotchas (read first — these cost real time)

- **One engine runs on both g6 and g6e.** L4 and L40S are both SM 8.9 (Ada), so
  a single built TRT engine (and the libtorch `model.pt`) is portable across the
  two instance types — build once, deploy to both. No per-instance rebuild.
- **Merge LoRA on the host, not in the container.** The finetune checkpoints are
  unmerged LoRA. The TRT/libtorch builds' `--ckpt` path calls
  `export.load_merged_backbone`, which needs `peft`+`torchao` to fold the
  adapter — but the serving/build container ships a `torchao` too old for peft's
  LoRA path (peft 0.19 needs torchao ≥0.16; image has 0.15). Run
  `efm_inference.merge` on the **host** (its env has torchao ≥0.16) to produce a
  merged checkpoint, then build engines from the **merged** checkpoint
  (no peft needed in-container).
- **Studio docker rejects symlinked bind mounts.** Mounting `code/src` fails
  because `efm_inference/.venv` contains symlinks. `rsync -a --exclude='.venv'
  --exclude='__pycache__' code/src/ <clean>/` to a symlink-free copy and mount
  that. Also pass `--network sagemaker` and a `TMPDIR` on a mounted volume
  (engine builds need multi-GB scratch).
- **fp8 calibration needs the merged checkpoint, not a backbone tar.**
  `calibrate.py --ckpt <merged-ckpt>` builds the eager backbone via
  `load_merged_backbone` (the actual served weights). The legacy
  `--backbone-tar` path needs a cached pretrained-backbone `model.tar.gz` that
  may not be present; the merged-ckpt path is equivalent and self-contained.
- **fp8-gemm only quantizes the linear GEMMs; attention (FMHA) + KV cache stay
  bf16.** So its speedup is largest where the GEMMs dominate (mid lengths, ~4k)
  and **dilutes at long context** where O(L²) attention dominates — measured
  ~+15–27% single-stream, only ~16% at 16k. For a 16k attention-bound embedding
  workload, fp8-gemm is the wrong knob (you'd need fp8 attention / `fp8-full`).
- **The workload is batch-1, prefill-bound.** Throughput is flat across client
  concurrency (latency scales ~linearly), so single-stream (`-u 1`) is the
  cleanest latency signal; higher concurrency just measures queueing.

## Prerequisites

1. **Check for existing artifacts first.** The model artifacts live under
   **`s3://sagemaker-us-west-2-185993409072/efm/`**. Look here before building
   anything — the three variants are usually already present and reusable across
   deploys:

   ```bash
   aws s3 ls s3://sagemaker-us-west-2-185993409072/efm/ --recursive
   # efm/libtorch/model.tar.gz   (torchlib)
   # efm/trt_bf16/model.tar.gz   (trt bf16)
   # efm/trt_fp8/model.tar.gz    (trt fp8-gemm)
   ```

   If all three are present, skip the build section entirely and go straight to
   the deploy matrix with these `--model-data-url`s. Only rebuild (below) when
   the checkpoint changed or an artifact is missing.
2. **Token sample** — `python -m efm_inference.parity.make_sample --rows 0`
   writes `$EFM_MODELS_DIR/sample_tokens.npz` (real packed user sequences; the
   load uses these, not synthetic tokens).
3. **ECR images** present: `…/ml-models/tritonserver:26.05` (trtllm) and
   `:26.05-libtorch` (see `code/src/efm_inference/sagemaker/`).

## Build the artifacts (GPU; only if missing from S3 — see Prerequisites)

Merge on host, then build engines/exports from the merged checkpoint in the
trtllm container (symlink-free mount + `--network sagemaker`). See
`code/src/efm_inference/README.md` and `sagemaker/README.md` for the canonical
recipes. Outline:

```bash
# host: fold LoRA
python -m efm_inference.merge --ckpt <ft-ckpt> --out <ckpt>-merged

# container (trtllm image): bf16 engine straight from merged ckpt
python -m efm_inference.triton.tensorrtllm.build --ckpt <ckpt>-merged --dtype bf16 --out <engine_bf16>

# container: fp8 — export → calibrate (--ckpt!) → build
python -m efm_inference.export --ckpt <ckpt>-merged --out <hf_export> --dtype bf16
python -m efm_inference.triton.tensorrtllm.calibrate --hf-dir <hf_export> --ckpt <ckpt>-merged \
    --calib-tokens <baked.arrow> --qformat fp8 --num-samples 128 --out <hf_fp8>
python -m efm_inference.triton.tensorrtllm.build --hf-dir <hf_fp8> --quant fp8-gemm --out <engine_fp8>

# torchlib: trace to model.pt (container)
python -m efm_inference.triton.pytorch.build --ckpt <ckpt>-merged --dtype bf16 --out <torchscript>

# host: pack + upload each as a SageMaker model.tar.gz
python -m efm_inference.scripts.build_model_artifact --engine-dir <engine_*> \
    --out model.tar.gz --s3-uri s3://<bucket>/efm/<variant>/model.tar.gz
python -m efm_inference.scripts.build_model_artifact --model-class libtorch \
    --source-dir <torchscript> --out model.tar.gz --s3-uri s3://<bucket>/efm/libtorch/model.tar.gz
```

## Deploy the endpoint matrix

For each (variant × instance) deploy a single-instance endpoint with
`create_endpoint.py` (`--no-wait`, then poll to `InService`). trt variants use
the `:26.05` image; torchlib uses `:26.05-libtorch`. The `serve` shim
auto-detects the single model dir.

```bash
python -m efm_inference.scripts.create_endpoint --name efm-trt-fp8-g6 \
    --image <reg>/ml-models/tritonserver:26.05 \
    --model-data-url s3://<bucket>/efm/trt_fp8/model.tar.gz \
    --role <execution-role-arn> --instance-type ml.g6.2xlarge --no-wait
```

Instance types: `ml.g6.2xlarge` (L4) and `ml.g6e.2xlarge` (L40S).

## Run the load test

`locustfile.py` is the driver — a custom Locust `User` over boto3
`invoke_endpoint` (SageMaker isn't plain HTTP, so it's not an `HttpUser`), using
the same KServe-v2 bodies as `prepare_requests` drawn from the parity token
sample. Closed-loop, time-boxed, `-u` = client concurrency. The `--csv` prefix
**must** be `results/locust/efm-<variant>-<inst>-u<N>` (`<variant>` ∈
`libtorch|trt-bf16|trt-fp8`, `<inst>` ∈ `g6|g6e`) so the report builder can key
each run by model × instance × concurrency:

```bash
PYTHONPATH=code/src uv run --project code/src/efm_inference --no-sync --with locust \
  locust -f code/src/efm_inference/scripts/locustfile.py --headless \
  --endpoint efm-trt-fp8-g6 --model-class tensorrtllm \
  -u 8 -r 8 -t 25s --csv results/locust/efm-trt-fp8-g6-u8
```

Use `--model-class libtorch` for the torchlib endpoints (different IO). Sweep
`-u 1,4,8…` as separate runs (one `--csv` prefix each); `-u 1` is the cleanest
single-stream latency signal for this batch-1, prefill-bound workload.

## Report

`build_report.py` reads the Locust per-run CSVs and writes a self-contained HTML
report (pass the results dir as the one positional arg):

```bash
python skills/efm-endpoint-loadtest/build_report.py results   # -> results/efm_latency_report.html
```

It consumes `<results>/locust/*_stats.csv` (the Aggregated row per run) and emits
the latency (p50/p90/p99 ms) and throughput (req/s) matrix across
model × instance × concurrency, plus the single-stream bar chart and the
g6-vs-g6e speedup table.

## Teardown (always, when done — endpoints bill continuously)

```bash
for e in <endpoint names>; do
  aws sagemaker delete-endpoint        --endpoint-name $e        --region us-west-2
  aws sagemaker delete-endpoint-config --endpoint-config-name $e --region us-west-2
  aws sagemaker delete-model           --model-name $e           --region us-west-2
done
```

Verify with `aws sagemaker list-endpoints`. The S3 `model.tar.gz` artifacts are
cheap and worth keeping for a fast redeploy.
