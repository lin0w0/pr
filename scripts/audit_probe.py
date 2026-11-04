"""Phase 0 audit probe: verify checkpoints, interfaces, gradients on real data.

Run:
    D:/code/research/envs/pr/python.exe scripts/audit_probe.py
"""
import json
import os
import sys
import time
import traceback
import warnings

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


_ds = _load_module("compnet_dataset", os.path.join(COMPNET_DIR, "models", "dataset.py"))
_cn = _load_module("compnet_model", os.path.join(COMPNET_DIR, "models", "compnet.py"))
_xn = _load_module("ccnet_model", os.path.join(CCNET_DIR, "models", "ccnet.py"))
NormSingleROI = _ds.NormSingleROI
compnet = _cn.compnet
ccnet = _xn.ccnet

REPORT = {}
warnings.filterwarnings("default")


def section(name):
    def deco(fn):
        try:
            REPORT[name] = fn()
            print(f"[OK] {name}")
        except Exception as e:
            REPORT[name] = {"error": repr(e), "trace": traceback.format_exc()}
            print(f"[FAIL] {name}: {e!r}")
        return fn

    return deco


def load_state(path):
    info = {}
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        info["weights_only=True"] = "ok"
    except Exception as e:
        info["weights_only=True"] = repr(e)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        info["weights_only=False"] = "ok (fallback)"
    info["type"] = type(sd).__name__
    info["n_keys"] = len(sd)
    info["n_params"] = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    info["nbytes_MB"] = round(sum(v.numel() * v.element_size() for v in sd.values() if hasattr(v, "numel")) / 1e6, 1)
    info["dtypes"] = sorted({str(v.dtype) for v in sd.values() if hasattr(v, "dtype")})
    info["first_keys"] = list(sd.keys())[:6]
    info["classifier_shapes"] = {k: list(v.shape) for k, v in sd.items() if "arc" in k.lower()}
    return sd, info


@section("env")
def _env():
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }


@section("dataset")
def _dataset():
    out = {}
    for s in ["session1", "session2"]:
        d = os.path.join(ROOT, "data", "TongjI", "ROI", s)
        fs = sorted(os.listdir(d))
        labels = [(int(f[:5]) - 1) // 10 for f in fs]
        sizes = set()
        for f in fs[::100]:
            with Image.open(os.path.join(d, f)) as im:
                sizes.add((im.size, im.mode))
        out[s] = {
            "n": len(fs),
            "unique_ids": len({int(f[:5]) for f in fs}),
            "unique_labels": len(set(labels)),
            "per_label": len(fs) // len(set(labels)),
            "image_sizes": sorted(map(str, sizes)),
            "first": fs[0],
            "last": fs[-1],
        }
    out["same_filenames"] = sorted(os.listdir(os.path.join(ROOT, "data", "TongjI", "ROI", "session1"))) == sorted(
        os.listdir(os.path.join(ROOT, "data", "TongjI", "ROI", "session2"))
    )
    return out


@section("checkpoints")
def _checkpoints():
    out = {}
    for name, path in [("compnet", os.path.join(ROOT, "weights", "CompNet", "compnet.pth")),
                       ("ccnet", os.path.join(ROOT, "weights", "CCNet", "ccnet.pth"))]:
        sd, info = load_state(path)
        info["size_MB"] = round(os.path.getsize(path) / 1e6, 1)
        out[name] = info
    return out


@section("model_build_and_load")
def _build_and_load():
    out = {}
    for name in ["compnet", "ccnet"]:
        path = os.path.join(ROOT, "weights", "CompNet" if name == "compnet" else "CCNet", f"{name}.pth")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        t0 = time.time()
        if name == "compnet":
            net = compnet(num_classes=600)
        else:
            net = ccnet(num_classes=600, weight=0.8)
        build_s = time.time() - t0
        t0 = time.time()
        missing, unexpected = net.load_state_dict(sd, strict=True)
        out[name] = {
            "strict_load": "ok",
            "missing": list(missing),
            "unexpected": list(unexpected),
            "load_s": round(time.time() - t0, 3),
            "build_s": round(build_s, 3),
            "n_params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2),
        }
    return out


@section("preprocess_and_forward")
def _forward():
    device = torch.device("cuda:0")
    img_path = os.path.join(ROOT, "data", "TongjI", "ROI", "session1", "00001.bmp")
    pil = Image.open(img_path).convert("L")
    tf = T.Compose([T.Resize(128), T.ToTensor(), NormSingleROI(outchannels=1)])
    x = tf(pil).unsqueeze(0).to(device)
    raw = T.ToTensor()(pil).unsqueeze(0).to(device)

    out = {"input_shape": list(x.shape), "raw_range": [float(raw.min()), float(raw.max())],
           "preproc_range": [float(x.min()), float(x.max())],
           "preproc_mean": float(x.mean()), "preproc_std": float(x.std())}

    sd_c = torch.load(os.path.join(ROOT, "weights", "CompNet", "compnet.pth"), map_location="cpu", weights_only=False)
    net = compnet(num_classes=600)
    net.load_state_dict(sd_c)
    net.to(device).eval()
    sd_x = torch.load(os.path.join(ROOT, "weights", "CCNet", "ccnet.pth"), map_location="cpu", weights_only=False)
    net2 = ccnet(num_classes=600, weight=0.8)
    net2.load_state_dict(sd_x)
    net2.to(device).eval()

    with torch.no_grad():
        f1 = net.getFeatureCode(x)
        f2 = net2.getFeatureCode(x)
        logits1 = net(x, None)
        logits2, fe2 = net2(x, None)
    out["compnet"] = {
        "feature_dim": int(f1.shape[1]),
        "feature_norm": float(f1.norm(dim=1).item()),
        "logits_shape": list(logits1.shape),
        "logits_range": [float(logits1.min()), float(logits1.max())],
    }
    out["ccnet"] = {
        "getFeatureCode_dim": int(f2.shape[1]),
        "getFeatureCode_norm": float(f2.norm(dim=1).item()),
        "forward_fe_dim": int(fe2.shape[1]),
        "forward_fe_norm": float(fe2.norm(dim=1).item()),
        "logits_shape": list(logits2.shape),
        "logits_range": [float(logits2.min()), float(logits2.max())],
    }

    # is forward logits == 30 * cosine?
    with torch.no_grad():
        cos = F.linear(F.normalize(net.getFeatureCode(x)), F.normalize(net.arclayer.weight))
        out["compnet"]["logits_equals_30cos_featurecode"] = bool(torch.allclose(logits1, 30 * cos, atol=1e-5))

    # timing
    for nm, m in [("compnet", net), ("ccnet", net2)]:
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(20):
                m.getFeatureCode(x)
        torch.cuda.synchronize()
        out[nm]["forward_ms_batch1"] = round((time.time() - t0) / 20 * 1000, 2)

    # input gradient
    for nm, m in [("compnet", net), ("ccnet", net2)]:
        xg = x.clone().requires_grad_(True)
        feat = m.getFeatureCode(xg)
        loss = feat.pow(2).sum()
        g, = torch.autograd.grad(loss, xg)
        out[nm]["input_grad_ok"] = bool(torch.isfinite(g).all() and g.abs().sum() > 0)
        out[nm]["input_grad_absmax"] = float(g.abs().max())
        out[nm]["param_grads_after_input_grad"] = sum(
            1 for p in m.parameters() if p.grad is not None
        )

    # requires_grad freeze check
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net2.parameters():
        p.requires_grad_(False)
    xg = x.clone().requires_grad_(True)
    net.getFeatureCode(xg).sum().backward()
    out["compnet"]["frozen_param_grads"] = sum(1 for p in net.parameters() if p.grad is not None)
    return out


@section("fft_chain")
def _fft():
    import torch.fft as fft
    device = torch.device("cuda:0")
    img_path = os.path.join(ROOT, "data", "TongjI", "ROI", "session1", "00001.bmp")
    raw = T.ToTensor()(Image.open(img_path).convert("L")).unsqueeze(0).to(device)
    H, W = raw.shape[-2:]
    out = {"H": H, "W": W}

    spec = fft.rfft2(raw, norm="ortho")
    rec = fft.irfft2(spec, s=(H, W), norm="ortho")
    out["roundtrip_max_abs_err"] = float((rec - raw).abs().max())

    A = spec.abs()
    phase = spec.angle()
    A_var = A.detach().requires_grad_(True)
    X_var = A_var * torch.exp(1j * phase)
    x_rec = fft.irfft2(X_var, s=(H, W), norm="ortho")
    out["amp_recon_max_abs_err"] = float((x_rec - raw).abs().max())
    out["recon_requires_grad"] = bool(x_rec.requires_grad)

    # band mask shape / symmetry check
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.rfftfreq(W, device=device)
    rho = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / (0.5 ** 2 + 0.5 ** 2) ** 0.5
    out["rho_max"] = float(rho.max())
    out["mask_shape"] = list(rho.shape)

    # build compnet, gradient of gallery-logit CE wrt amplitude
    sd = torch.load(os.path.join(ROOT, "weights", "CompNet", "compnet.pth"), map_location="cpu", weights_only=False)
    net = compnet(num_classes=600)
    net.load_state_dict(sd)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    logits = net(x_rec, None)
    loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
    g, = torch.autograd.grad(loss, A_var)
    out["sgm_grad_ok"] = bool(torch.isfinite(g).all() and g.abs().sum() > 0)
    out["sgm_grad_shape"] = list(g.shape)
    out["sgm_grad_absmax"] = float(g.abs().max())
    out["n_zeros_in_grad"] = int((g == 0).sum())

    # grad wrt image through preprocessing normalization (z-score over all pixels)
    xg = raw.clone().requires_grad_(True)
    mu = xg.mean(dim=(1, 2, 3), keepdim=True)
    sdv = xg.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    xn = (xg - mu) / (sdv + 1e-6)
    logits = net(xn, None)
    loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
    g2, = torch.autograd.grad(loss, xg)
    out["pixel_grad_ok"] = bool(torch.isfinite(g2).all() and g2.abs().sum() > 0)
    out["pixel_grad_absmax"] = float(g2.abs().max())
    return out


if __name__ == "__main__":
    print(json.dumps(REPORT, indent=2, ensure_ascii=False))
    with open(os.path.join(ROOT, "scripts", "audit_probe_result.json"), "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False)
