"""Official Gallery matching protocol (nearest registration image).

The official CompNet/CCNet ``inference.py`` does **not** average registration
features. For every query image it compares against *every* registration image
and keeps the nearest one:

    cosdis = dot(query_feature, gallery_feature)
    dis = arccos(clip(cosdis, -1, 1)) / pi
    gallery_id = label(argmin(dis))

``arccos`` is monotone decreasing, so argmin(distance) == argmax(cosine).
This module implements that protocol exactly (with max-cosine ranking) so its
clean metrics can be compared against the prototype protocol.
"""
import torch
import torch.nn.functional as F


def rank_accuracy_image_gallery(
    probe_features,
    probe_labels,
    gallery_features,
    gallery_labels,
    topk=(1, 5),
    chunk_size=512,
):
    """Rank-k with the official per-image gallery, aggregated per identity.

    Every identity's score is the **maximum** cosine over its registration
    images; identities are then ranked. This matters for k > 1: taking the
    top-k gallery *images* directly can let several images of one identity
    crowd out the true identity. For k=1 both definitions coincide, which is
    why the official ``inference.py`` (Rank-1 only) is unaffected.

    Ties are broken arbitrarily by ``torch.topk``, matching the arbitrary tie
    order of the official ``argmin`` loop.

    Returns dict with ``rank{k}`` percentages and ``n``.
    """
    pf = F.normalize(torch.as_tensor(probe_features, dtype=torch.float32), dim=1)
    gf = F.normalize(torch.as_tensor(gallery_features, dtype=torch.float32), dim=1)
    pl = torch.as_tensor(probe_labels, dtype=torch.long)
    gl = torch.as_tensor(gallery_labels, dtype=torch.long)
    if len(pf) != len(pl) or len(gf) != len(gl):
        raise ValueError("feature/label length mismatch")
    if len(gf) == 0:
        raise ValueError("empty gallery")

    unique_labels, inverse = torch.unique(gl, sorted=True, return_inverse=True)
    num_ids = len(unique_labels)
    kmax = min(max(int(k) for k in topk), num_ids)

    hits = {int(k): 0 for k in topk}
    for start in range(0, len(pf), chunk_size):
        sims = pf[start:start + chunk_size] @ gf.T  # [b, G]
        b = sims.shape[0]
        # Per-identity maximum over registration images.
        id_scores = torch.full(
            (b, num_ids), -torch.inf, dtype=sims.dtype, device=sims.device
        )
        id_scores.scatter_reduce_(
            1, inverse.view(1, -1).expand(b, -1), sims, reduce="amax",
            include_self=True,
        )
        _, top_idx = torch.topk(id_scores, k=kmax, dim=1)
        top_labels = unique_labels[top_idx]  # [b, kmax]
        query = pl[start:start + chunk_size, None]
        for k in topk:
            kk = min(int(k), num_ids)
            hits[int(k)] += int((top_labels[:, :kk] == query).any(dim=1).sum())

    out = {"n": int(len(pf))}
    for k in topk:
        out[f"rank{k}"] = float(hits[int(k)] / len(pf) * 100.0)
    return out


def nearest_gallery_identity(probe_features, gallery_features, gallery_labels,
                             chunk_size=512):
    """Identity of the nearest registration image for each query."""
    pf = F.normalize(torch.as_tensor(probe_features, dtype=torch.float32), dim=1)
    gf = F.normalize(torch.as_tensor(gallery_features, dtype=torch.float32), dim=1)
    gl = torch.as_tensor(gallery_labels, dtype=torch.long)
    preds = []
    for start in range(0, len(pf), chunk_size):
        sims = pf[start:start + chunk_size] @ gf.T
        preds.append(gl[sims.argmax(dim=1)])
    return torch.cat(preds)
