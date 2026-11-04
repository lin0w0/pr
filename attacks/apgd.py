"""AutoAttack APGD wrapper for the Gallery-prototype identity logits.

The installed ``autoattack`` package is used as the verified APGD implementation.
Its ``predict`` callable receives raw [0,1] images and must return logits; our
wrapper applies the official differentiable preprocessing and the frozen
backbone, then the Gallery prototype classifier.
"""
import torch

from autoattack.autopgd_base import APGDAttack


class _GalleryAPGDModel(torch.nn.Module):
    def __init__(self, adapter, classifier, scale):
        super().__init__()
        self.adapter = adapter
        self.classifier = classifier
        self.scale = float(scale)

    def forward(self, x):
        features = self.adapter.extract_features(x)
        return self.scale * self.classifier.scores(features)


def apgd_linf(adapter, classifier, x, labels, eps, steps=50, restarts=1,
              seed=0, loss="ce", scale=30.0, verbose=False):
    """Run AutoAttack APGD (Linf) and return adversarial images."""
    model = _GalleryAPGDModel(adapter, classifier, scale)
    attack = APGDAttack(
        model,
        n_iter=int(steps),
        norm="Linf",
        n_restarts=int(restarts),
        eps=float(eps),
        seed=int(seed),
        loss=loss,
        verbose=verbose,
        device=adapter.device,
    )
    x_adv = attack.perturb(x.to(adapter.device), labels.to(adapter.device))
    return x_adv.detach()
