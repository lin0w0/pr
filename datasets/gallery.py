"""Gallery prototype construction and 1:N cosine classification."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from utils.paths import resolve_path


@torch.no_grad()
def extract_features(adapter, dataset, batch_size=256, progress=None):
    """Extract L2-normalized features for every sample in ``dataset``.

    Returns:
        features: ``[N, D]`` on CPU
        labels:   ``[N]`` on CPU
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for i, (x, y) in enumerate(loader):
        feats.append(adapter.extract_features(x).cpu())
        labels.append(y.cpu())
        if progress is not None:
            progress(i + 1, len(loader))
    return torch.cat(feats), torch.cat(labels)


def prototypes_from_features(features, labels, num_identities=None):
    """Per-identity mean feature, re-normalized.

    Implements the specification exactly: normalize each registration feature
    (already done by the adapter), average, then re-normalize the prototype.
    """
    features = F.normalize(features.float(), p=2, dim=1)
    labels = labels.to(torch.long)
    unique = torch.unique(labels, sorted=True)
    if num_identities is not None and len(unique) != num_identities:
        raise ValueError(f"expected {num_identities} identities, found {len(unique)}")
    counts = torch.stack([(labels == l).sum() for l in unique])
    if counts.min() != counts.max():
        raise ValueError(f"identities have unequal registration counts: {counts.tolist()}")
    protos = []
    for label in unique:
        mean = features[labels == label].mean(dim=0)
        protos.append(F.normalize(mean, p=2, dim=0))
    return torch.stack(protos), unique


def build_gallery(adapter, gallery_dataset, batch_size=256, num_identities=None, progress=None):
    """Extract gallery features and build prototypes.

    Returns:
        dict with ``prototypes`` ``[G, D]``, ``labels`` ``[G]``, and raw features.
    """
    features, labels = extract_features(adapter, gallery_dataset, batch_size, progress=progress)
    protos, unique = prototypes_from_features(features, labels, num_identities=num_identities)
    return {
        "prototypes": protos,
        "labels": unique,
        "features": features,
        "feature_labels": labels,
    }


def save_gallery_cache(path, gallery, meta):
    abs_path = resolve_path(path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        abs_path,
        prototypes=gallery["prototypes"].numpy(),
        labels=gallery["labels"].numpy(),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )
    return abs_path


def load_gallery_cache(path, expected_meta):
    abs_path = resolve_path(path)
    if not abs_path.is_file():
        return None
    with np.load(abs_path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta != json.loads(json.dumps(expected_meta, sort_keys=True)):
            return None
        return {
            "prototypes": torch.from_numpy(data["prototypes"]),
            "labels": torch.from_numpy(data["labels"]),
        }


class GalleryClassifier:
    """Closed-set 1:N classifier over gallery prototypes.

    ``logits = temperature_scale * cosine(query_features, prototypes)``.
    The default ``temperature_scale=30`` matches the scale of the official
    ArcFace logits (``30 * cosine``), so attack losses operate on the same
    numerical scale as the models were trained with.
    """

    def __init__(self, prototypes, labels=None, temperature_scale=30.0, device=None):
        self.prototypes = F.normalize(prototypes.float(), p=2, dim=1)
        self.labels = labels if labels is not None else torch.arange(len(self.prototypes))
        self.temperature_scale = float(temperature_scale)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.prototypes = self.prototypes.to(device)
        self.labels = self.labels.to(device)
        return self

    def scores(self, features):
        return F.normalize(features.float(), p=2, dim=1) @ self.prototypes.T

    def logits(self, features):
        return self.temperature_scale * self.scores(features)

    def predict(self, features):
        return self.labels[self.scores(features).argmax(dim=1)]

    def __len__(self):
        return len(self.prototypes)
