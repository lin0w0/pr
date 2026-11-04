"""Original CSR (Contrastive Spectral Rectification) for palmprint backbones.

Faithful to the paper Algorithm 1 / official FastCSR:
  1. x_low = Gaussian LPF(x)
  2. C(x) = cos(f(x), f(x_low)); if C >= tau, pass through
  3. L_rec = cos(f(x+delta), f(x_low)) - lambda * cos(f(x+delta), f(x))
     sign-PGD on delta, L-infinity budget, pixel clamp
  4. Greedy selection over delta_0..delta_{N-1}

No identity labels, Gallery logits or classifier are used.
"""
import torch
import torch.nn.functional as F


class OriginalCSR:
    def __init__(self, adapter, radius=22.857, lam=1.0, steps=3,
                 eps=4.0 / 255.0, alpha=2.0 / 255.0, pixel_clamp=True):
        self.adapter = adapter
        self.device = adapter.device
        self.radius = float(radius)
        self.lam = float(lam)
        self.steps = int(steps)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.pixel_clamp = bool(pixel_clamp)

    # ------------------------------------------------------------------ LPF
    def lowpass(self, x):
        """Official Gaussian LPF: fft2 -> fftshift -> mask -> ifftshift."""
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=x.device) - cy
        xx = torch.arange(W, device=x.device) - cx
        yg, xg = torch.meshgrid(y, xx, indexing="ij")
        dist_sq = yg ** 2 + xg ** 2
        mask = torch.exp(-dist_sq / (2.0 * self.radius ** 2))
        filtered = fft_shift * mask.view(1, 1, H, W)
        out = torch.fft.ifft2(torch.fft.ifftshift(filtered)).real
        return torch.clamp(out, 0.0, 1.0)

    # --------------------------------------------------------------- features
    def features(self, x):
        return F.normalize(self.adapter.extract_features(x), p=2, dim=1)

    @torch.no_grad()
    def score(self, x):
        """Detection score C(x) = cos(f(x), f(LPF(x)))."""
        x = x.to(self.device)
        f0 = self.features(x)
        fl = self.features(self.lowpass(x))
        return (f0 * fl).sum(dim=1)

    def score_differentiable(self, x, sigmas=None):
        """Detector score used by adaptive attacks (single radius)."""
        f0 = self.features(x)
        fl = self.features(self.lowpass(x))
        return (f0 * fl).sum(dim=1)

    # --------------------------------------------------------------- rectify
    def rectify(self, x, return_records=False):
        """Greedy CSR rectification for every sample in ``x``."""
        x = x.to(self.device)
        B = x.shape[0]
        with torch.no_grad():
            orig_feats = self.features(x)
            anchor = self.features(self.lowpass(x))

        delta = torch.zeros_like(x)
        xp = x.clone()
        best = x.clone()
        best_score = torch.full((B,), -float("inf"), device=self.device)
        best_step = torch.full((B,), -1, dtype=torch.long, device=self.device)
        records = []

        for t in range(self.steps):
            feats = self.features(xp)
            sim_lpf = (feats * anchor).sum(dim=1)
            sim_adv = (feats * orig_feats).sum(dim=1)
            lrec = sim_lpf - self.lam * sim_adv

            with torch.no_grad():
                improve = lrec > best_score
                if improve.any():
                    best[improve] = xp[improve].detach()
                    best_score[improve] = lrec[improve].detach()
                    best_step[improve] = t
                for j in range(B):
                    records.append({
                        "step": t,
                        "local_index": j,
                        "sim_lpf": float(sim_lpf[j]),
                        "sim_adv": float(sim_adv[j]),
                        "lrec": float(lrec[j]),
                        "selected": False,
                        "l2": float((xp[j] - x[j]).pow(2).sum().sqrt()),
                        "linf": float((xp[j] - x[j]).abs().max()),
                    })

            if t == self.steps - 1:
                break

            xp_req = xp.detach().clone().requires_grad_(True)
            feats = self.features(xp_req)
            sim_lpf = (feats * anchor).sum(dim=1)
            sim_adv = (feats * orig_feats).sum(dim=1)
            lrec = sim_lpf - self.lam * sim_adv
            loss = -lrec.sum()
            grad, = torch.autograd.grad(loss, xp_req)
            # loss = -L_rec; gradient ascent on L_rec is xp - alpha * sign(grad)
            xp_new = xp_req - self.alpha * grad.sign()
            delta = torch.clamp(xp_new - x, -self.eps, self.eps)
            xp = torch.clamp(x + delta, 0.0, 1.0).detach() if self.pixel_clamp \
                else (x + delta).detach()

        if return_records:
            for r in records:
                if int(best_step[r["local_index"]]) == r["step"]:
                    r["selected"] = True
            return best.detach(), records
        return best.detach()

    def process(self, x, tau):
        """Detection gate + rectification for flagged samples.

        Returns:
            defended x, detection mask, records, scores.
        """
        x = x.to(self.device)
        scores = self.score(x)
        mask = scores < float(tau)
        defended = x.clone()
        records = []
        if mask.any():
            idx = torch.where(mask)[0]
            rect, recs = self.rectify(x[idx], return_records=True)
            defended[idx] = rect
            for r in recs:
                r["sample_index"] = int(idx[r["local_index"]])
                records.append(r)
        return defended.detach(), mask, records, scores

    @torch.no_grad()
    def detect(self, x, tau):
        return self.score(x) < float(tau)
