"""Token -> transaction/field machinery and raw-value rendering.

Signal-agnostic: every function takes a per-token signal vector (IG attribution to a dim, SAE per-token
feature activation, occlusion delta, ...) plus the user's token_ids. No model, no plotting here.

Read token semantics from the token's OWN name prefix ([cat_*]=category, [bank_name_*]=bank, [amt_bin_*]
=amount, [month_*]/[dom_*]/[weekday_*]=time), NOT from its positional slot — this bake's field slots don't
always match the current sequence_builder template.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from efm_core.data.sequence_builder import (
    BOS_TOKEN, EOS_TOKEN, TXN_SEP_TOKEN, STRUCTURAL_TOKENS,
)

_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]        # dt.weekday(): 0=Mon .. 6=Sun
_VAL_PREFIXES = ("cat_", "bank_name_", "bank_", "type_")          # bank_name_ before bank_ (longest first)


@dataclass
class TokenContext:
    """Derived lookups from a tokenizer runtime, computed once and passed to the helpers below."""
    vocab: dict                 # token -> id
    id2tok: dict                # id -> token
    struct_list: list           # ordered structural marker names (STRUCTURAL_TOKENS)
    struct_ids: frozenset       # ids of all structural markers
    txn_sep_id: int
    bos_eos_ids: frozenset
    amount_edges: list          # interface_config["amount_bin_edges_cents"]


def token_context(runtime) -> TokenContext:
    v = runtime.vocab
    struct_list = list(STRUCTURAL_TOKENS)
    return TokenContext(
        vocab=v,
        id2tok={i: t for t, i in v.items()},
        struct_list=struct_list,
        struct_ids=frozenset(v[t] for t in struct_list if t in v),
        txn_sep_id=v[TXN_SEP_TOKEN],
        bos_eos_ids=frozenset(v[t] for t in (BOS_TOKEN, EOS_TOKEN) if t in v),
        amount_edges=runtime.interface_config["amount_bin_edges_cents"],
    )


def txn_bounds(token_ids, txn_sep_id):
    """Transaction span boundaries from [txn_sep] positions. Returns (bounds, n_txn)."""
    seps = [i for i, t in enumerate(token_ids) if t == txn_sep_id]
    bounds = [0] + [s + 1 for s in seps] + [len(token_ids)]
    return bounds, len(bounds) - 1


def txn_of_pos_map(token_ids, txn_sep_id):
    """position -> 1-based transaction index (1=oldest .. N=newest)."""
    bounds, _ = txn_bounds(token_ids, txn_sep_id)
    return {i: k + 1 for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) for i in range(a, b)}


def aggregate_tokens_to_transactions(per_token, token_ids, txn_sep_id, exclude_ids=frozenset()):
    """Per-transaction signal: split on [txn_sep], sum per-token signal within each transaction EXCLUDING
    exclude_ids (e.g. [bos]/[eos]). Returns [{idx, n, start, end, attr}] in sequence order (idx 1-based)."""
    bounds, n = txn_bounds(token_ids, txn_sep_id)
    spans = []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        s = float(sum(per_token[i] for i in range(a, b) if token_ids[i] not in exclude_ids))
        spans.append({"idx": k + 1, "n": n, "start": a, "end": b, "attr": s})
    return spans


def top_content_tokens(per_token, token_ids, structural_ids, txn_of_pos, k=6):
    """Top-k individual tokens by |signal|, EXCLUDING structural markers. Returns
    [(pos, txn_idx, token_id, signal)]."""
    out = []
    for i in np.argsort(-np.abs(per_token)):
        if token_ids[i] in structural_ids:
            continue
        out.append((int(i), txn_of_pos[int(i)], token_ids[int(i)], float(per_token[i])))
        if len(out) >= k:
            break
    return out


def structural_field_matrix(token_ids, per_token_cols, struct_list, vocab):
    """Roll per-token signal up to field buckets. `per_token_cols` is a list of 1D arrays (one per column,
    each length T). cell[s, c] = sum over non-structural tokens following marker s (its field values) of
    per_token_cols[c], accumulated across all of s's occurrences. Each non-structural token is credited to
    the most recent preceding structural marker. Returns M of shape (len(struct_list), len(per_token_cols))."""
    row_of = {vocab[t]: r for r, t in enumerate(struct_list) if t in vocab}
    M = np.zeros((len(struct_list), len(per_token_cols)))
    for c, pt in enumerate(per_token_cols):
        cur = None
        for i, t in enumerate(token_ids):
            if t in row_of:          # structural marker -> owning field for what follows
                cur = row_of[t]
            elif cur is not None:    # non-structural value -> credit to the owning field marker
                M[cur, c] += pt[i]
    return M


def field_content_signal(token_ids, per_token, struct_list, vocab):
    """One level deeper than structural_field_matrix: instead of pooling a field's values, keep the CONTENT
    token identity. Same crediting rule (each non-structural token -> the most recent preceding structural
    marker = its field). Returns {(field_name, content_id): (summed_signal, occurrence_count)} for ONE user.
    The count lets callers form a per-occurrence MEAN (Σ/n) instead of a raw sum, so ubiquitous tokens (e.g.
    the amount sign, present in every txn) don't dominate by frequency alone (View C)."""
    field_of_id = {vocab[t]: t for t in struct_list if t in vocab}
    out, cur = {}, None
    for tid, s in zip(token_ids, per_token):
        if tid in field_of_id:       # structural marker -> owning field for the values that follow
            cur = field_of_id[tid]
        elif cur is not None:        # content value -> credit to current field, keyed by content id
            k = (cur, int(tid))
            ssum, cnt = out.get(k, (0.0, 0))
            out[k] = (ssum + float(s), cnt + 1)
    return out


def money(cents) -> str:
    d = cents / 100.0
    if d >= 1000:
        return f"${d / 1000:.0f}k"
    if d >= 10:
        return f"${d:.0f}"
    return ("$%.2f" % d).rstrip("0").rstrip(".")


def raw_token(name, is_struct, amount_edges) -> str:
    """Render a token as its pre-tokenization value. Structural field markers keep brackets; value tokens
    render bare (numerics -> raw values: amt_bin -> dollar range, month/dom/weekday -> names, prefixes
    stripped). The exact pre-bin amount is not in the bake (amount_cents masked) and the model only saw the
    bin, so the bin RANGE is the faithful raw value."""
    if is_struct:
        return name                                              # keep brackets: [amount], [merchant], ...
    s = name[1:-1] if len(name) > 1 and name.startswith("[") and name.endswith("]") else name
    if s.startswith("amt_bin_"):
        tail = s[len("amt_bin_"):]
        if tail.isdigit() and 0 <= int(tail) < len(amount_edges) - 1:
            lo, hi = amount_edges[int(tail)], amount_edges[int(tail) + 1]
            return f"≥{money(lo)}" if hi >= 1e12 else f"{money(lo)}–{money(hi)}"
        return s.replace("amt_bin_", "amt:")                     # underflow / overflow
    if s.startswith("amt_sign_"):
        return "−" if s.endswith("neg") else "+"
    if s.startswith("month_"):
        t = s[len("month_"):]
        return _MONTHS[int(t)] if t.isdigit() and 0 < int(t) < 13 else s
    if s.startswith("dom_"):
        t = s[len("dom_"):]
        return t.lstrip("0") if t.isdigit() else s
    if s.startswith("weekday_"):
        t = s[len("weekday_"):]
        return _WDAYS[int(t)] if t.isdigit() and int(t) < 7 else s
    for pre in _VAL_PREFIXES:
        if s.startswith(pre):
            return s[len(pre):] or s                             # cat_groceries->groceries, bank_unk->unk
    return s
