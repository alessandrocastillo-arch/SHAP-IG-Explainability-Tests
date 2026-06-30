"""Rendering for the per-token interpretability views, shared by both notebooks.

`field_x_signal_heatmap` — VIZ 2 analog: a field-bucket × column (dim or SAE feature) heatmap of a rolled-up
signed signal.

`top_transactions_text` — VIZ 5 analog: per-feature/-dim top-N transactions with inline token coloring.
The ranking is signal-agnostic AND length-invariant by default: a colleague flagged that the original
Σ|attr| score rewards transactions with long descriptions (more tokens -> bigger sum). So `rank` defaults
to "max" (the single strongest token), with "top3" (mean of the 3 strongest) as a robustness option and
"sum" kept only for backward-comparison. Ranking ignores structural/[bos]/[eos] tokens so a field marker
can't win the slot; token COLORS still carry sign and cover every token.
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RWG = LinearSegmentedColormap.from_list("red_white_green", ["#b2182b", "#f7f7f7", "#1a9850"])


def field_x_signal_heatmap(M, row_labels, col_labels, *, title="", value_label="Σ signed signal",
                           ax=None, annot_thresh=0.005, xlabel="column"):
    """M: (n_rows × n_cols) signed matrix. Green=+, red=−, white=0, symmetric color scale."""
    mx = float(np.abs(M).max()) or 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(0.55 * len(col_labels) + 3.0, 0.42 * len(row_labels) + 1.8))
    im = ax.imshow(M, cmap=RWG, norm=TwoSlopeNorm(vcenter=0.0, vmin=-mx, vmax=mx), aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(c) for c in col_labels], fontsize=8, rotation=45, ha="left")
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if abs(M[r, c]) > annot_thresh:
                ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=6)
    if title:
        ax.set_title(title, fontsize=9)
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=value_label)
    return ax


def field_content_heatmap(values, labels, field_labels, *, title="", value_label="Σ signed Δ (across users)",
                          annotate_values=False):
    """Content-level drill-down grid: columns = fields, rows = within-field rank (1..N). values[r, c] = signed
    aggregated signal of the r-th ranked content token in field c; labels[r][c] = its rendered name ("" for an
    empty slot, drawn as white/0). Green=+, red=−, symmetric scale. Decomposes field_x_signal_heatmap one
    level deeper for a single feature. If annotate_values, the raw value is printed under each token name."""
    values = np.asarray(values, dtype=float)
    R, C = values.shape
    mx = float(np.abs(values).max()) or 1.0
    _, ax = plt.subplots(figsize=(0.95 * C + 3.0, (0.62 if annotate_values else 0.5) * R + 1.8))
    im = ax.imshow(values, cmap=RWG, norm=TwoSlopeNorm(vcenter=0.0, vmin=-mx, vmax=mx), aspect="auto")
    ax.set_xticks(range(C)); ax.set_xticklabels([str(f) for f in field_labels], fontsize=8, rotation=45, ha="left")
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top"); ax.set_xlabel("token field")
    ax.set_yticks(range(R)); ax.set_yticklabels([f"#{r + 1}" for r in range(R)], fontsize=8)
    ax.set_ylabel("within-field rank")
    for r in range(R):
        for c in range(C):
            if labels[r][c]:
                if annotate_values:                              # token name above, raw value beneath
                    ax.text(c, r - 0.16, labels[r][c], ha="center", va="center", fontsize=6)
                    ax.text(c, r + 0.22, f"{values[r, c]:+.1f}", ha="center", va="center", fontsize=5, alpha=0.85)
                else:
                    ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=6)
    if title:
        ax.set_title(title, fontsize=9)
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=value_label)
    return ax


def _txn_rank_key(absvals, rank):
    """Length-invariant ranking of a transaction from the |signal| of its CONTENT tokens."""
    if absvals.size == 0:
        return 0.0
    if rank == "max":
        return float(absvals.max())
    if rank == "top3":
        return float(np.sort(absvals)[-3:].mean())
    if rank == "sum":                               # legacy / length-biased — kept for comparison only
        return float(absvals.sum())
    raise ValueError(f"unknown rank={rank!r}; expected 'max' | 'top3' | 'sum'")


def score_transactions(per_token, token_ids, ctx, *, rank="max"):
    """Length-invariant per-transaction ranking score from a per-token signal. Returns a list of
    {idx (1-based), n, start, end, key} where `key` is the rank score over CONTENT tokens (structural /
    [bos] / [eos] excluded so a field marker can't win the slot)."""
    from .tokens import txn_bounds
    pt = np.asarray(per_token, dtype=np.float64)
    bounds, N = txn_bounds(token_ids, ctx.txn_sep_id)
    exclude = ctx.struct_ids | ctx.bos_eos_ids
    out = []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        absvals = np.array([abs(pt[i]) for i in range(a, b) if token_ids[i] not in exclude])
        out.append({"idx": k + 1, "n": N, "start": a, "end": b, "key": _txn_rank_key(absvals, rank)})
    return out


def render_transactions(items, ctx, *, color_scale=None, header=None):
    """Render a list of transaction spans as inline-colored token tables (captum visualize_text).

    items: list of dicts, each {token_ids: <slice>, per_token: <1D slice>, pred_class, true_class,
           attr_class, attr_score, pred_prob (optional), conv_delta (optional)}. `color_scale` normalizes
           token color across ALL items (pass a shared scale for cross-user comparability; default =
           max |per_token| over the items). Selection of WHICH spans is the caller's job."""
    from captum.attr import visualization as viz
    from IPython.display import HTML, display
    from .tokens import raw_token

    scale = color_scale or max((float(np.abs(it["per_token"]).max()) for it in items if len(it["per_token"])),
                               default=1.0) or 1.0
    records = []
    for it in items:
        pt = np.asarray(it["per_token"], dtype=np.float64)
        toks = [raw_token(ctx.id2tok.get(t, str(t)), t in ctx.struct_ids, ctx.amount_edges)
                for t in it["token_ids"]]
        records.append(viz.VisualizationDataRecord(
            word_attributions=torch.tensor(pt / scale, dtype=torch.float),
            pred_prob=round(it.get("pred_prob", float(np.abs(pt).sum())), 3),
            pred_class=it["pred_class"], true_class=it["true_class"], attr_class=it["attr_class"],
            attr_score=round(it["attr_score"], 4), raw_input_ids=toks,
            convergence_score=round(it["conv_delta"], 4) if it.get("conv_delta") is not None else float("nan"),
        ))
    if header:
        display(HTML(f"<h3 style='font-family:monospace;margin-top:18px'>{header}</h3>"))
    return viz.visualize_text(records)


def top_transactions_text(per_token, token_ids, ctx, *, feature_name, label=None, uid=None,
                          rank="max", top_txn=10, conv_delta=None, title=None):
    """VIZ 5 analog for ONE user: top-`top_txn` transactions of `per_token`, ranked length-invariantly."""
    pt = np.asarray(per_token, dtype=np.float64)
    scored = score_transactions(pt, token_ids, ctx, rank=rank)
    N = scored[0]["n"] if scored else 0
    top = sorted(sorted(scored, key=lambda s: -s["key"])[:top_txn], key=lambda s: s["idx"])  # chronological
    total_abs = float(np.abs(pt).sum()) or 1.0
    items = [{
        "token_ids": token_ids[s["start"]:s["end"]], "per_token": pt[s["start"]:s["end"]],
        "pred_class": f"txn {s['idx']}/{N}",
        "true_class": (f"user {uid}" + (f" (L{int(label)})" if label is not None else "")) if uid is not None else "",
        "attr_class": feature_name, "attr_score": s["key"], "conv_delta": conv_delta,
        "pred_prob": float(np.abs(pt[s["start"]:s["end"]]).sum()) / total_abs,
    } for s in top]
    hdr = title or (f"{feature_name} — top-{top_txn} transactions by {rank} |signal| "
                    f"(token color = signed signal: green +, red −; per-user scale)")
    return render_transactions(items, ctx, header=hdr)
