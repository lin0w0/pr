"""CW-L2 untargeted margin attack using the Gallery prototype interface."""
import torch
import torch.nn.functional as F

from spectral.losses import margin_per_sample


def cw_l2(adapter, classifier, x, labels, c=10.0, steps=100, lr=0.01,
          kappa=0.0, scale=10.0):
    """Minimize ||delta||_2^2 + c * max(margin + kappa, 0) via tanh parameterization."""
    x = x.to(adapter.device)
    labels = labels.to(adapter.device)
    w = torch.atanh((x * 2.0 - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    w = w.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=float(lr))
    best_adv = x.clone()
    best_obj = torch.full((x.shape[0],), float("inf"), device=adapter.device)
    for _ in range(int(steps)):
        xa = 0.5 * (torch.tanh(w) + 1.0)
        feats = F.normalize(adapter.extract_features(xa), dim=1)
        logits = float(scale) * (feats @ classifier.prototypes.T)
        margin = margin_per_sample(logits, labels)
        l2 = (xa - x).reshape(xa.shape[0], -1).pow(2).sum(dim=1)
        # margin = max_other - s_y; larger margin = stronger attack.
        # Minimize L2 - c * margin (CW-style max-margin with L2 penalty).
        obj = l2 - float(c) * (margin - float(kappa))
        opt.zero_grad()
        obj.sum().backward()
        opt.step()
        with torch.no_grad():
            better = obj < best_obj
            if better.any():
                best_obj[better] = obj[better].detach()
                best_adv[better] = xa[better].detach()
    return best_adv.detach()
