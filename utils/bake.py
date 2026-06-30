"""Pull and iterate the packed tokenized bake + the tokenizer runtime.

The bake stores users as packed varlen segments (multi_chunk=False => each user is one contiguous
segment; `user_ids` lists only real segments, trailing pad skipped). Same Arrow-IPC reader both notebooks
use. The bake that MATCHES the model is interface_config_id 225ef2ba / tokenizer 1776745486.
"""
from __future__ import annotations

import os
from glob import glob

import pyarrow as pa
from efm_core.data.tokenizer_runtime import load_pretrained_tokenizer_runtime

# Defaults (overridable per call). VAL bake (1 shard ~2k users) for quick iteration; TRAIN bake is large.
VAL_BAKE_S3 = ("s3://ml-datasets-datalakeprod-us-west-2-sagemaker/efm_risk_ppma_finetune_dataset/3/"
               "baked/efm-tokenizer-mlm-1776745486/seq16384_tb16384_mc0_tdnone_bcv6/val")
TOK_S3 = ("s3://ml-hyperpod-fsx-datalakeprod-us-west-2/runs/"
          "efm-ppma-ft-decoder100m-h100-20260519-142518Z/outputs/model")   # vocab 16386, matches 1776745486
BAKE_LOCAL_DEFAULT = "/home/sagemaker-user/Develop/Explainability/_cache/efm_ppma_bake_1776745486_val"
TOK_LOCAL_DEFAULT = "/home/sagemaker-user/Develop/Explainability/_cache/efm_tokenizer"


def pull_bake(n_shards: int = 1, local: str = BAKE_LOCAL_DEFAULT, s3: str = VAL_BAKE_S3):
    os.makedirs(local, exist_ok=True)
    if not os.path.exists(f"{local}/bake_manifest.json"):
        os.system(f"aws s3 cp {s3}/bake_manifest.json {local}/bake_manifest.json")
    for i in range(n_shards):
        f = f"merged-algo-1-{i:06d}.arrow"
        if not os.path.exists(f"{local}/{f}"):
            os.system(f"aws s3 cp {s3}/{f} {local}/{f}")
    return sorted(glob(f"{local}/*.arrow"))


def pull_tokenizer(local: str = TOK_LOCAL_DEFAULT, s3: str = TOK_S3):
    os.makedirs(f"{local}/tokenizer", exist_ok=True)
    for rel in ("interface_config.json", "tokenizer/tokenizer.json"):
        if not os.path.exists(f"{local}/{rel}"):
            os.system(f"aws s3 cp {s3}/{rel} {local}/{rel}")
    return load_pretrained_tokenizer_runtime(local)


def iter_user_sequences_from_bake(arrow_files, max_users=None, max_len=None):
    """Yield (user_id, token_ids, label, pred_time). Reads Arrow IPC directly."""
    n = 0
    for f in arrow_files:
        reader = pa.ipc.open_file(pa.memory_map(f, "r"))
        for bi in range(reader.num_record_batches):
            for row in reader.get_record_batch(bi).to_pylist():
                cu = row["cu_seqlens"]
                for si, uid in enumerate(row["user_ids"]):
                    ids = row["input_ids"][cu[si]:cu[si + 1]]
                    if max_len and len(ids) > max_len:
                        continue
                    yield uid, ids, row["labels_per_segment"][si], row["pred_times"][si]
                    n += 1
                    if max_users and n >= max_users:
                        return
