"""Phase 0 audit probe 2: CCNet strict load (filtering documented extras),
real-data forward/gradient checks, and a small empirical Rank-1 validation.

Run:
    D:/code/research/envs/pr/python.exe scripts/audit_probe2.py
"""
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPNET_DIR = os.path.join(ROOT, "official_project", "compnet", "CompNet")
CCNET_DIR = os.path.join(ROOT, "official_project", "CCNet")


def _load_module(mod_name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


compnet = _load_module("cn_model", os.path.join(COMPNET_DIR, "models", "compnet.py")).compnet
ccnet = _load_module("xn_model", os.path.join(CCNET_DIR, "models", "ccnet.py")).ccnet

REPORT = {}
EXTRA_KEYS = frozenset(
    f"cb{b}.att_{o}.{l}.{p}"
    for b in (1, 2, 3)
    for o in (1, 2)
    for l in ("conv0", "conv_spatial", "conv1")
    for p in ("weight", "bias")
)


def section(name):
    def deco(fn):
        try:
            REPORT[name] = fn()
            print(f"[OK] {name}", flush=True)
        except Exception as e:
            REPORT[name] = {"error": repr(e), "trace": traceback.format_exc()}
            print(f"[FAIL] {name}: {e!r}", flush=True)
        return fn

    return deco


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_models(device):
    ck_c = torch.load(os.path.join(ROOT, "weights", "CompNet", "compnet.pth"), map_location="cpu", weights_only=True)
    net_c = compnet(num_classes=600)
    missing_c, unexpected_c = net_c.load_state_dict(ck_c, strict=True)

    ck_x = torch.load(os.path.join(ROOT, "weights", "CCNet", "ccnet.pth"), map_location="cpu", weights_only=True)
    net_x = ccnet(num_classes=600, weight=0.8)
    extras_in_ckpt = {k for k in ck_x if k.startswith(("cb1.att", "cb2.att", "cb3.att"))}
    expected_extras = set(EXTRA_KEYS)
    filtered = {k: v for k, v in ck_x.items() if k not in expected_extras}
    net_state = set(net_x.state_dict())
    missing_x, unexpected_x = net_x.load_state_dict(filtered, strict=True)
    info = {
        "compnet": {"missing": list(missing_c), "unexpected": list(unexpected_c), "strict": True},
        "ccnet": {
            "extras_in_checkpoint": len(extras_in_ckpt),
            "extras_all_documented": extras_in_ckpt == expected_extras,
            "active_keys_loaded": len(filtered),
            "missing": list(missing_x),
            "unexpected": list(unexpected_x),
            "strict_after_filter": True,
            "model_keys_not_in_ckpt": sorted(net_state - set(filtered)),
        },
    }
    net_c.to(device).eval()
    net_x.to(device).eval()
    for net in (net_c, net_x):
        for p in net.parameters():
            p.requires_grad_(False)
    return net_c, net_x, info


def load_batch(session, names, size=128, device="cuda:0"):
    """Load raw [0,1] tensors with official preprocessing (Resize, ToTensor, NormSingleROI)."""
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    xs = []
    for n in names:
        p = os.path.join(ROOT, "data", "TongjI", "ROI", session, n)
        xs.append(tf(Image.open(p).convert("L")))
    x = torch.stack(xs).to(device)
    # official NormSingleROI: z-score over pixels > 0, per sample, unbiased std
    B, C, H, W = x.shape
    flat = x.view(B, C, H * W)
    idx = flat > 0
    t = flat[idx]  # (N_nonzero,) flattened in row-major sample order
    # per-sample mean/std is required -> compute from reversed mask counts
    counts = idx.sum(dim=(1, 2))
    sums = (flat * idx).sum(dim=(1, 2))
    means = sums / counts.clamp_min(1)
    sq = ((flat - means[:, None, None]) ** 2 * idx).sum(dim=(1, 2))
    stds = torch.sqrt(sq / (counts.clamp_min(1) - 1).clamp_min(1))
    flat_n = (flat - means[:, None, None]) / (stds[:, None, None] + 1e-6)
    xn = torch.where(idx, flat_n, flat).view(B, C, H, W)
    return x, xn


@section("checkpoint_meta")
def _meta():
    out = {}
    for nm, rel in [("compnet", "weights/CompNet/compnet.pth"), ("ccnet", "weights/CCNet/ccnet.pth")]:
        p = os.path.join(ROOT, rel)
        out[nm] = {"path": rel, "sha256": sha256(p), "size_bytes": os.path.getsize(p)}
    return out


@section("preprocess_equivalence")
def _prep_eq():
    # compare torch-side normalization against official PIL pipeline on 20 real images
    sys.path.insert(0, COMPNET_DIR)
    import importlib.util

    spec = importlib.util.spec_from_file_location("cnds", os.path.join(COMPNET_DIR, "models", "dataset.py"))
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    tf = T.Compose([T.Resize(128), T.ToTensor(), ds.NormSingleROI(outchannels=1)])
    names = [f"{i:05d}.bmp" for i in range(1, 21)]
    ref = torch.stack([tf(Image.open(os.path.join(ROOT, "data", "TongjI", "ROI", "session1", n)).convert("L")) for n in names])
    _, mine = load_batch("session1", names)
    mine = mine.cpu()
    return {"max_abs_diff": float((ref - mine).abs().max()), "allclose": bool(torch.allclose(ref, mine, atol=1e-6))}


@section("forward_grad")
def _fg():
    device = torch.device("cuda:0")
    net_c, net_x, info = build_models(device)
    names = ["00001.bmp", "00011.bmp", "06000.bmp", "00002.bmp"]
    raw, x = load_batch("session1", names)
    out = {"build_load": info, "input_shape": list(x.shape),
           "raw_range": [float(raw.min()), float(raw.max())],
           "norm_mean": float(x.mean()), "norm_std": float(x.std())}

    for nm, net, feat_dim in [("compnet", net_c, 512), ("ccnet", net_x, 2048)]:
        with torch.no_grad():
            f = net.getFeatureCode(x)
            logits = net(x, None)
            if isinstance(logits, tuple):
                logits = logits[0]
            t0 = time.time()
            for _ in range(10):
                net.getFeatureCode(x)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 10 * 1000
        out[nm] = {
            "feature_dim": int(f.shape[1]), "expected_dim": feat_dim,
            "feature_norm": float(f.norm(dim=1).mean()),
            "logits_shape": list(logits.shape),
            "forward_ms_batch4": round(ms, 2),
        }
        xg = x.clone().requires_grad_(True)
        logits_g = net(xg, None)
        if isinstance(logits_g, tuple):
            logits_g = logits_g[0]
        loss_g = F.cross_entropy(logits_g, torch.arange(x.shape[0], device=device) % 600)
        g, = torch.autograd.grad(loss_g, xg)
        out[nm]["input_grad_finite"] = bool(torch.isfinite(g).all())
        out[nm]["input_grad_absmax"] = float(g.abs().max())
        out[nm]["input_grad_nonzero_frac"] = float((g != 0).float().mean())
        out[nm]["param_grads_after"] = sum(1 for p in net.parameters() if p.grad is not None)

    # SGM path for CCNet too
    for nm, net in [("compnet", net_c), ("ccnet", net_x)]:
        A = torch.fft.rfft2(x, norm="ortho")
        amp, phase = A.abs(), A.angle()
        Av = amp.detach().requires_grad_(True)
        xr = torch.fft.irfft2(Av * torch.exp(1j * phase), s=x.shape[-2:], norm="ortho")
        logits = net(xr, None)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits, torch.zeros(x.shape[0], dtype=torch.long, device=device))
        g, = torch.autograd.grad(loss, Av)
        out[nm]["sgm_grad_finite"] = bool(torch.isfinite(g).all())
        out[nm]["sgm_grad_absmax"] = float(g.abs().max())
        out[nm]["sgm_grad_shape"] = list(g.shape)
    return out


@section("rank1_small")
def _rank1():
    device = torch.device("cuda:0")
    net_c, net_x, _ = build_models(device)
    n_ids = 100
    gal_names = [f"{pid * 10 + k:05d}.bmp" for pid in range(n_ids) for k in range(1, 11)]
    probe_names = list(gal_names)
    gal_labels = torch.arange(n_ids).repeat_interleave(10)

    out = {"n_ids": n_ids, "gallery_images": len(gal_names), "protocol": "proto = mean of 10 session1 images, probe = each session2 image"}
    labels = torch.arange(n_ids).repeat_interleave(10)

    def run(nm, net, batch):
        def feats(session, names):
            fs = []
            for i in range(0, len(names), batch):
                _, x = load_batch(session, names[i:i + batch])
                with torch.no_grad():
                    fs.append(net.getFeatureCode(x).cpu())
            return torch.cat(fs)

        fg = feats("session1", gal_names)
        fp = feats("session2", probe_names)
        protos = F.normalize(fg.view(n_ids, 10, -1).mean(1), dim=1)
        fp = F.normalize(fp, dim=1)
        pred = (fp @ protos.T).argmax(1)
        rank1 = (pred == labels).float().mean().item() * 100
        accs = []
        for i in range(0, len(probe_names), batch):
            _, x = load_batch("session2", probe_names[i:i + batch])
            with torch.no_grad():
                logits = net(x, None)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred_l = logits.argmax(1).cpu()
            accs.append((pred_l == labels[i:i + batch]).float().mean().item())
        return {"rank1_prototype": round(rank1, 3),
                "closedset_acc_arcface": round(float(np.mean(accs)) * 100, 3),
                "batch_size": batch}

    out["compnet"] = run("compnet", net_c, 256)
    del net_c
    torch.cuda.empty_cache()
    out["ccnet"] = run("ccnet", net_x, 64)
    return out


if __name__ == "__main__":
    print(json.dumps(REPORT, indent=2, ensure_ascii=False))
    with open(os.path.join(ROOT, "scripts", "audit_probe2_result.json"), "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False)
