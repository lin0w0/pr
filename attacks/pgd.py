"""PGD (L-infinity) on the Gallery-prototype identity logits."""
import torch

from attacks.base import gallery_logits, loss_per_sample, project_linf_box


def pgd_linf(adapter, classifier, x, labels, eps, alpha, steps,
             random_start=True, seed=0, loss="margin", scale=10.0,
             restarts=1, return_info=False):
    """Maximize the identity loss under ||delta||inf <= eps.

    Returns ``x_adv`` (and optional per-restart info). The best restart is
    selected by the final attack loss (higher is stronger).
    """
    device = adapter.device
    x = x.to(device)
    labels = labels.to(device)
    best = None
    best_loss = None
    infos = []
    for restart in range(int(restarts)):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + restart)
        if random_start:
            delta = (torch.rand(x.shape, generator=gen, device=device, dtype=x.dtype) * 2 - 1) * eps
        else:
            delta = torch.zeros_like(x)
        delta = project_linf_box(delta, x, eps)
        last_loss = None
        for _ in range(int(steps)):
            x_adv = (x + delta).detach().requires_grad_(True)
            logits = gallery_logits(adapter, classifier, x_adv, scale)
            loss_batch = loss_per_sample(loss, logits, labels)
            grad, = torch.autograd.grad(loss_batch.sum(), x_adv)
            delta = project_linf_box(delta + float(alpha) * grad.sign(), x, eps)
            last_loss = loss_batch.detach()
        x_adv = (x + delta).detach()
        score = float(last_loss.mean().item())
        infos.append({"restart": restart, "final_loss": score})
        if best_loss is None or score > best_loss:
            best_loss = score
            best = x_adv
    if return_info:
        return best, {"restarts": infos, "best_loss": best_loss}
    return best
