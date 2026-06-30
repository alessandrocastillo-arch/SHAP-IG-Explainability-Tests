"""Load the fine-tuned EFM checkpoint and run forward passes.

Identical model-load path as both notebooks (LoRA mid-run ckpt -> apply_peft -> load strict=False ->
merge). Forward helpers are FORWARD-ONLY (no grad) and run in bf16: fp32 at 16k tokens OOMs the A10G
because the SDPA math kernel materializes the O(T^2) score matrix (~12GB); bf16 takes the mem-efficient
kernel and the precision is fine for the pooled embedding. (Captum IG, which needs fp32 + a recency
window, lives in the notebooks, not here.)

`per_token_hidden_states` is the key addition for SAE per-token attribution: `encode_hidden_states`
returns the full (1, T, D) per-token states in ONE forward, and under `last_token` pooling the pooled
vector is exactly the last row. So encoding every row through the SAE is a single matmul — no per-token
re-runs — and each row is a valid same-distribution input (the prefix's last-token pooled embedding).
"""
from __future__ import annotations

import os
import torch

from safetensors.torch import load_file as load_safetensors
from efm_finetune.model.config import EFMFineTuneConfig
from efm_finetune.model.finetune_model import EFMForFineTuning
from efm_finetune.peft.wrap import apply_peft
from efm_finetune.head.pooling import pool_sequence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Defaults (overridable per call); match ppma_interp.ipynb / ppma_sae.ipynb config.
FT_CHECKPOINT_S3 = ("s3://ml-hyperpod-fsx-datalakeprod-us-west-2/runs/"
                    "efm-ppma-ft-decoder100m-h100-20260519-142518Z/checkpoints/checkpoint-120000")
FT_CHECKPOINT_LOCAL = "/home/sagemaker-user/Develop/Explainability/_cache/efm_ft_ckpt"
POOL_METHOD = "last_token"
ATTN_BACKEND = "sdpa"


def pull_checkpoint(local: str = FT_CHECKPOINT_LOCAL, s3: str = FT_CHECKPOINT_S3) -> str:
    os.makedirs(local, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        if not os.path.exists(f"{local}/{f}"):
            rc = os.system(f"aws s3 cp {s3}/{f} {local}/{f}")
            if rc != 0:
                raise RuntimeError(f"aws s3 cp failed for {f} (rc={rc})")
    return local


def load_ft_model(local: str = FT_CHECKPOINT_LOCAL, device: str = DEVICE,
                  dtype: torch.dtype = torch.bfloat16, attn_backend: str = ATTN_BACKEND):
    """Recreate the PEFT wrapper, load strict=False, merge LoRA into the backbone, return eval() model."""
    pull_checkpoint(local)
    cfg = EFMFineTuneConfig.from_pretrained(local, attn_backend=attn_backend, dropout=0.0)
    m = EFMForFineTuning(cfg)
    if cfg.peft_method in ("lora", "dora"):
        m.model = apply_peft(m.model, cfg)
    result = m.load_state_dict(load_safetensors(f"{local}/model.safetensors"), strict=False)
    if result.missing_keys:        # missing backbone keys => silently re-init'd => noise; fail loud
        raise RuntimeError(f"missing state_dict keys (backbone would be re-init'd): {result.missing_keys[:8]}")
    m.merge_peft_into_backbone()
    return m.to(dtype).to(device).eval()


def build_user_pack(token_ids, device: str = DEVICE):
    """One user's token-id list -> (input_ids, cu_seqlens, position_ids, max_seqlen) as a SINGLE segment,
    so last-token pooling yields that user's production embedding."""
    T = len(token_ids)
    return (torch.tensor(token_ids, dtype=torch.long, device=device).view(1, T),
            torch.tensor([0, T], dtype=torch.int32, device=device),
            torch.arange(T, dtype=torch.int32, device=device).view(1, T), T)


@torch.no_grad()
def pooled_embedding(model, token_ids, *, pool_method: str = POOL_METHOD,
                     eos_token_id=None, device: str = DEVICE):
    """The 768-d pooled embedding that feeds the LightGBM scorer (no grad, full sequence)."""
    ids, cu, pos, T = build_user_pack(token_ids, device)
    hidden = model.model.encode_hidden_states(input_ids=ids, cu_seqlens=cu, position_ids=pos, max_seqlen=T)
    pooled = pool_sequence(hidden, cu, is_causal=bool(model.config.is_causal),
                           pool_method=pool_method, input_ids=ids, eos_token_id=eos_token_id)
    return pooled.squeeze(0).float().detach().cpu().numpy()        # (768,)


@torch.no_grad()
def per_token_hidden_states(model, token_ids, *, device: str = DEVICE):
    """Full per-token hidden states (T, 768) from ONE forward. Under `last_token` pooling, row -1 is the
    pooled embedding; every row t is the prefix [0..t]'s last-token representation (causal model), i.e. a
    valid same-distribution input to an SAE trained on pooled embeddings."""
    ids, cu, pos, T = build_user_pack(token_ids, device)
    hidden = model.model.encode_hidden_states(input_ids=ids, cu_seqlens=cu, position_ids=pos, max_seqlen=T)
    return hidden.view(-1, hidden.shape[-1]).float().detach().cpu().numpy()   # (T, 768)
