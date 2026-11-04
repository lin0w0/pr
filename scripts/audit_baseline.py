"""Phase 0 baseline sanity: full Tongji 600-ID Rank-1/5 and EER with official weights.

Protocol (official): gallery = session1 (10 images/ID), probe = session2 (10 images/ID).
Prototype gallery = mean of the 10 L2-normalized gallery features, re-normalized.
Verification EER uses all probe x gallery image pairs (genuine if same label).
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


compnet = _load_module("cn", os.path.join(ROOT, "official_project", "compnet", "CompNet", "models", "compnet.py")).compnet
ccnet = _load_module("xn", os.path.join(ROOT, "official_project", "CCNet", "models", "ccnet.py")).ccnet
DEVICE = torch.device("cuda:0")


def load_models():
    ck = torch.load(os.path.join(ROOT, "weights", "CompNet", "compnet.pth"), map_location="cpu", weights_only=True)
    net_c = compnet(num_classes=600)
    net_c.load_state_dict(ck, strict=True)
    ckx = torch.load(os.path.join(ROOT, "weights", "CCNet", "ccnet.pth"), map_location="cpu", weights_only=True)
    ckx = {k: v for k, v in ckx.items() if not k.startswith(("cb1.att", "cb2.att", "cb3.att"))}
    net_x = ccnet(num_classes=600, weight=0.8)
    net_x.load_state_dict(ckx, strict=True)
    for net in (net_c, net_x):
        net.to(DEVICE).eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return net_c, net_x


def normalize_like_official(x):
    B, C, H, W = x.shape
    flat = x.reshape(B, C, H * W)
    idx = flat > 0
    counts = idx.sum(dim=(1, 2)).clamp_min(1)
    means = (flat * idx).sum(dim=(1, 2)) / counts
    sq = (((flat - means[:, None, None]) ** 2) * idx).sum(dim=(1, 2))
    stds = torch.sqrt(sq / (counts - 1).clamp_min(1))
    out = (flat - means[:, None, None]) / (stds[:, None, None] + 1e-6)
    return torch.where(idx, out, flat).view(B, C, H, W)


def feats(net, session, names, batch):
    tf = T.Compose([T.Resize(128), T.ToTensor()])
    out = []
    with torch.no_grad():
        for i in range(0, len(names), batch):
            xs = torch.stack([tf(Image.open(os.path.join(ROOT, "data", "TongjI", "ROI", session, n)).convert("L"))
                              for n in names[i:i + batch]]).to(DEVICE)
            out.append(net.getFeatureCode(normalize_like_official(xs)).cpu())
    return F.normalize(torch.cat(out), dim=1)


def compute_metrics(fg, fp, gal_labels, probe_labels):
    n_ids = int(gal_labels.max().item()) + 1
    protos = F.normalize(fg.view(n_ids, 10, -1).mean(1), dim=1)
    sims = (fp @ protos.T).numpy()
    pred = sims.argmax(1)
    labs = probe_labels.numpy()
    rank1 = float((pred == labs).mean() * 100)
    top5 = float(np.mean([labs[i] in np.argsort(-sims[i])[:5] for i in range(len(labs))]) * 100)

    # verification: all probe x gallery image pairs (chunked)
    gl = gal_labels.numpy()
    pl = probe_labels.numpy()
    gen, imp = [], []
    for i in range(0, len(fp), 256):
        s = (fp[i:i + 256] @ fg.T).numpy()
        m = pl[i:i + 256, None] == gl[None, :]
        gen.append(s[m])
        imp.append(s[~m])
    gen = np.concatenate(gen)
    imp = np.concatenate(imp)
    # EER via sorted arrays: TPR(t)=P(gen>=t), FPR(t)=P(imp>=t)
    thresholds = np.unique(gen)
    gs = np.sort(gen)
    ims = np.sort(imp)
    tpr = 1.0 - np.searchsorted(gs, thresholds, side="left") / len(gs)
    fpr = 1.0 - np.searchsorted(ims, thresholds, side="left") / len(ims)
    idx = np.argmin(np.abs(tpr - (1 - fpr)))
    eer = float((fpr[idx] + (1 - tpr[idx])) / 2 * 100)

    def tar_at_far(target):
        ok = fpr <= target
        return float(tpr[ok].max() * 100) if ok.any() else float("nan")

    return {"rank1": round(rank1, 3), "rank5": round(top5, 3), "eer_pct": round(eer, 4),
            "tar_at_far_1pct": round(tar_at_far(0.01), 3), "tar_at_far_0.1pct": round(tar_at_far(0.001), 3),
            "n_genuine_pairs": int(len(gen)), "n_impostor_pairs": int(len(imp)),
            "score_range": [round(float(gen.min()), 4), round(float(gen.max()), 4)]}


def main():
    torch.manual_seed(0)
    net_c, net_x = load_models()
    names = [f"{i:05d}.bmp" for i in range(1, 6001)]
    gal_labels = torch.arange(600).repeat_interleave(10)
    probe_labels = gal_labels.clone()
    report = {"n_gallery": len(names), "n_probe": len(names)}
    for nm, net, batch in [("compnet", net_c, 256), ("ccnet", net_x, 64)]:
        t0 = time.time()
        fg = feats(net, "session1", names, batch)
        fp = feats(net, "session2", names, batch)
        t_feat = time.time() - t0
        m = compute_metrics(fg, fp, gal_labels, probe_labels)
        m["feature_extraction_s"] = round(t_feat, 1)
        m["feature_dim"] = int(fg.shape[1])
        report[nm] = m
        print(nm, json.dumps(m, ensure_ascii=False), flush=True)
        del fg, fp
        torch.cuda.empty_cache()
    with open(os.path.join(ROOT, "scripts", "audit_baseline_result.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
