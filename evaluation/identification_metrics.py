"""Open-set-free closed-set 1:N identification metrics."""
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rank_accuracy(scores, query_labels, gallery_labels=None, topk=(1, 5)):
    """Rank-k identification accuracy.

    Args:
        scores: ``[N, G]`` similarity scores (higher = more similar).
        query_labels: ``[N]`` identity label of each query.
        gallery_labels: ``[G]`` identity label of each gallery column. Defaults
            to ``0..G-1`` (prototypes are sorted by label).
        topk: tuple of k values.

    Returns:
        dict with ``rank{k}`` values in percent and ``n``.
    """
    scores = _to_numpy(scores)
    query_labels = _to_numpy(query_labels).astype(np.int64)
    if scores.ndim != 2 or scores.shape[0] != len(query_labels):
        raise ValueError(f"bad shapes: scores={scores.shape}, labels={query_labels.shape}")
    G = scores.shape[1]
    if gallery_labels is None:
        gallery_labels = np.arange(G)
    else:
        gallery_labels = _to_numpy(gallery_labels).astype(np.int64)
        if len(gallery_labels) != G:
            raise ValueError("gallery_labels length must match score columns")

    order = np.argsort(-scores, axis=1)  # [N, G]
    out = {"n": int(len(query_labels))}
    for k in topk:
        k_eff = min(int(k), G)
        top = gallery_labels[order[:, :k_eff]]  # [N, k]
        hit = (top == query_labels[:, None]).any(axis=1)
        out[f"rank{k}"] = float(hit.mean() * 100.0)
    return out


def identification_accuracy(scores, query_labels, gallery_labels=None):
    return rank_accuracy(scores, query_labels, gallery_labels, topk=(1,))["rank1"]


def top1_predictions(scores, gallery_labels=None):
    scores = _to_numpy(scores)
    order = np.argsort(-scores, axis=1)
    idx = order[:, 0]
    if gallery_labels is None:
        return idx
    return _to_numpy(gallery_labels)[idx]
