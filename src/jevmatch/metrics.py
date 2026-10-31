"""Metrics over a full score matrix (spec section 8).

`scores` maps (source_col, target_col) -> float for every m x n pair.
Pairs a method did not score are treated as -inf (tied at the bottom).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

NEG = -math.inf


def full_matrix(scores: dict, src_cols, tgt_cols) -> dict:
    return {(s, t): scores.get((s, t), NEG) for s in src_cols for t in tgt_cols}


def recall_at_gt(scores: dict, gt: set) -> dict:
    """Recall@|GT| with expected, pessimistic and optimistic tie handling."""
    k = len(gt)
    if k == 0:
        return {"expected": math.nan, "pessimistic": math.nan, "optimistic": math.nan}
    vals = sorted(scores.values(), reverse=True)
    t = vals[k - 1] if len(vals) >= k else NEG
    above = [p for p, v in scores.items() if v > t]
    tied = [p for p, v in scores.items() if v == t]
    r = k - len(above)
    a_hit = sum(p in gt for p in above)
    t_hit = sum(p in gt for p in tied)
    t_miss = len(tied) - t_hit
    exp = a_hit + (r * t_hit / len(tied) if tied else 0)
    pes = a_hit + max(0, r - t_miss)
    opt = a_hit + min(r, t_hit)
    return {"expected": exp / k, "pessimistic": pes / k, "optimistic": opt / k}


def per_source_rank(scores: dict, gt: set) -> dict:
    """MRR and P@1 per source column with expected tie handling.

    Expected first-hit rank in a tie group of size g holding c hits that
    starts after b items is b + (g + 1) / (c + 1); RR uses 1 / E[rank].
    """
    by_src = defaultdict(list)
    for (s, t), v in scores.items():
        by_src[s].append((v, (s, t) in gt))
    rr, p1 = [], []
    for s, items in by_src.items():
        if not any(hit for _, hit in items):
            continue
        groups = defaultdict(lambda: [0, 0])
        for v, hit in items:
            groups[v][0] += 1
            groups[v][1] += hit
        before = 0
        first = True
        for v in sorted(groups, reverse=True):
            g, c = groups[v]
            if first:
                p1.append(c / g)
                first = False
            if c:
                rr.append(1.0 / (before + (g + 1) / (c + 1)))
                break
            before += g
    return {
        "mrr": float(np.mean(rr)) if rr else math.nan,
        "p_at_1": float(np.mean(p1)) if p1 else math.nan,
    }


def thresholded_prf(scores: dict, gt: set, tau: float) -> dict:
    pred = {p for p, v in scores.items() if v >= tau}
    tp = len(pred & gt)
    prec = tp / len(pred) if pred else math.nan
    rec = tp / len(gt) if gt else math.nan
    f1 = 2 * prec * rec / (prec + rec) if pred and tp else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_pred": len(pred)}


def hungarian_f1(scores: dict, gt: set, src_cols, tgt_cols) -> dict:
    """One-to-one F1 after the max-score assignment (pairs with score > 0)."""
    si = {c: i for i, c in enumerate(src_cols)}
    ti = {c: j for j, c in enumerate(tgt_cols)}
    m = np.zeros((len(src_cols), len(tgt_cols)))
    for (s, t), v in scores.items():
        m[si[s], ti[t]] = v if math.isfinite(v) else 0.0
    rows, cols = linear_sum_assignment(m, maximize=True)
    pred = {(src_cols[r], tgt_cols[c]) for r, c in zip(rows, cols) if m[r, c] > 0}
    tp = len(pred & gt)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else math.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def all_metrics(scores: dict, gt: set, src_cols, tgt_cols, probabilistic: bool) -> dict:
    full = full_matrix(scores, src_cols, tgt_cols)
    out = {f"recall_gt_{k}": v for k, v in recall_at_gt(full, gt).items()}
    out.update(per_source_rank(full, gt))
    out.update({f"h1_{k}": v for k, v in hungarian_f1(full, gt, src_cols, tgt_cols).items()})
    if probabilistic:
        for tau in (0.5, 0.7):
            out.update({f"t{int(tau * 10)}_{k}": v for k, v in thresholded_prf(full, gt, tau).items()})
    return out
