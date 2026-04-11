"""
Secondary workflow metrics for the retained Phase 1 entropy-analysis path.

These utilities remain available for capture/analysis experiments, but they are
not part of the primary pruning-runtime entrypoint.
"""

import numpy as np
from typing import Optional


def normalize_to_distribution(scores: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize scores to a valid probability distribution along the given axis.

    Args:
        scores: array of non-negative values
        axis: axis along which to normalize
        eps: small constant to avoid division by zero

    Returns:
        Normalized probability distribution (sums to 1 along axis)
    """
    scores = np.maximum(scores, 0)  # ensure non-negative
    total = scores.sum(axis=axis, keepdims=True)
    total = np.maximum(total, eps)
    return scores / total


def shannon_entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute Shannon entropy H(p) = -sum(p * log2(p)).

    Args:
        probs: probability distribution (should sum to 1 along axis)
        axis: axis along which to compute entropy

    Returns:
        Entropy values. Shape is probs.shape with the specified axis removed.
    """
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=axis)


def renyi_entropy(probs: np.ndarray, alpha: float = 2.0, axis: int = -1) -> np.ndarray:
    """
    Compute Rényi entropy H_α(p) = 1/(1-α) * log2(sum(p^α)).

    For α=2, this is related to collision entropy.
    More sensitive to concentrated distributions than Shannon entropy.

    Args:
        probs: probability distribution
        alpha: order of Rényi entropy (must not be 1)
        axis: axis along which to compute

    Returns:
        Rényi entropy values.
    """
    if abs(alpha - 1.0) < 1e-6:
        return shannon_entropy(probs, axis=axis)

    p = np.clip(probs, 1e-12, 1.0)
    return (1.0 / (1.0 - alpha)) * np.log2(np.sum(p ** alpha, axis=axis))


def gini_coefficient(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute Gini coefficient measuring inequality of the distribution.
    G = 0 means perfectly uniform, G → 1 means maximally concentrated.

    Args:
        scores: non-negative values (need not be normalized)
        axis: axis along which to compute

    Returns:
        Gini coefficient values in [0, 1).
    """
    # Sort along axis
    sorted_scores = np.sort(scores, axis=axis)
    n = sorted_scores.shape[axis]
    if n <= 1:
        return np.zeros(sorted_scores.shape[:axis] + sorted_scores.shape[axis+1:])

    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    indices = np.arange(1, n + 1)
    # Reshape indices for broadcasting
    shape = [1] * sorted_scores.ndim
    shape[axis] = n
    indices = indices.reshape(shape)

    total = np.sum(sorted_scores, axis=axis, keepdims=True)
    total = np.maximum(total, 1e-12)

    weighted_sum = np.sum(indices * sorted_scores, axis=axis)
    total_sum = total.squeeze(axis)

    return (2.0 * weighted_sum) / (n * total_sum) - (n + 1.0) / n


def topk_concentration(scores: np.ndarray, k: int = 64, axis: int = -1) -> np.ndarray:
    """
    Compute Top-K concentration: fraction of total mass in the top K values.

    Args:
        scores: non-negative values
        k: number of top elements
        axis: axis along which to compute

    Returns:
        Concentration ratio in [0, 1].
    """
    axis = axis if axis >= 0 else scores.ndim + axis
    n = scores.shape[axis]
    k = int(k)

    if n == 0 or k <= 0:
        out_shape = scores.shape[:axis] + scores.shape[axis + 1:]
        return np.zeros(out_shape, dtype=np.float64)

    k = min(k, n)
    if k == n:
        topk_sum = np.sum(scores, axis=axis)
        total_sum = np.maximum(np.sum(scores, axis=axis), 1e-12)
        return topk_sum / total_sum

    # Partition to find top-k values
    # np.partition is O(n) vs O(n log n) for full sort
    neg_scores = -scores  # negate because partition gives smallest
    partitioned = np.partition(neg_scores, k - 1, axis=axis)
    # Take the first k elements (which are the k largest after negation)
    slices = [slice(None)] * scores.ndim
    slices[axis] = slice(0, k)
    topk_values = -partitioned[tuple(slices)]

    topk_sum = np.sum(topk_values, axis=axis)
    total_sum = np.maximum(np.sum(scores, axis=axis), 1e-12)

    return topk_sum / total_sum


def matrix_rank(attn_matrix: np.ndarray, tol_ratio: float = 0.01) -> dict:
    """
    Compute the numerical rank of a 2D attention matrix.

    Uses SVD to determine the effective rank. A singular value is considered
    significant if it exceeds tol_ratio * max_singular_value.

    Args:
        attn_matrix: [L_t, L_v] text→vision attention matrix (head-averaged)
        tol_ratio: threshold ratio relative to the largest singular value

    Returns:
        dict with keys:
            - attn_rank: effective numerical rank
            - attn_rank_ratio: rank / min(L_t, L_v), i.e. fraction of full rank
            - attn_sv_top1_ratio: top singular value / sum, measures dominance
    """
    sv = np.linalg.svd(attn_matrix.astype(np.float32), compute_uv=False)
    tol = tol_ratio * sv[0]
    rank = int(np.sum(sv > tol))
    min_dim = min(attn_matrix.shape)
    sv_sum = float(sv.sum()) if sv.sum() > 0 else 1e-12
    return {
        "attn_rank": rank,
        "attn_rank_ratio": rank / max(min_dim, 1),
        "attn_sv_top1_ratio": float(sv[0]) / sv_sum,
    }


def compute_all_metrics(
    prune_scores: np.ndarray,
    topk_values: Optional[list] = None,
    tv_attn: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all entropy/distribution metrics from prune scores.

    Args:
        prune_scores: [L_v] importance scores per visual token (one layer, one sample)
        topk_values: list of K values for top-k concentration (default: [32, 64, 128])
        tv_attn: optional [L_t, L_v] attention matrix for rank computation

    Returns:
        dict with keys: shannon, renyi_2, gini, topk_{k} for each k,
                        and attn_rank, attn_rank_ratio, attn_sv_top1_ratio if tv_attn is provided
    """
    if topk_values is None:
        topk_values = [32, 64, 128]

    # Normalize to probability distribution
    probs = normalize_to_distribution(prune_scores)

    results = {
        "shannon": float(shannon_entropy(probs)),
        "renyi_2": float(renyi_entropy(probs, alpha=2.0)),
        "gini": float(gini_coefficient(prune_scores)),
    }

    for k in topk_values:
        results[f"topk_{k}"] = float(topk_concentration(prune_scores, k=k))

    if tv_attn is not None:
        results.update(matrix_rank(tv_attn))

    return results


def compute_layer_metrics_batch(
    prune_scores_per_layer: dict,
    topk_values: Optional[list] = None,
) -> dict:
    """
    Compute metrics for all layers of one sample.

    Args:
        prune_scores_per_layer: {layer_idx: np.ndarray of shape [L_v]}

    Returns:
        {layer_idx: {metric_name: value}}
    """
    results = {}
    for layer_idx, scores in prune_scores_per_layer.items():
        results[layer_idx] = compute_all_metrics(scores, topk_values)
    return results
