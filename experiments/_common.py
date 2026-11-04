"""Shared Phase 2 context: split, adapters, gallery, adversarial cache."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from attacks.apgd import apgd_linf
from attacks.pgd import pgd_linf
from datasets.dataset_split import ensure_splits
from datasets.gallery import (
    GalleryClassifier,
    build_gallery,
    load_gallery_cache,
    save_gallery_cache,
)
from datasets.tongji_dataset import TongjiDataset
from models import build_adapter
from utils.config import load_config, load_dataset_config, load_model_config
from utils.paths import PROJECT_ROOT, resolve_path
from utils.seed import set_deterministic, set_seed

PHASE2_ROOT = PROJECT_ROOT / "results" / "phase2"
CACHE_DIR = PHASE2_ROOT / "cache"


def load_spectral_config(path="configs/spectral.yaml"):
    return load_config(path)


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paths_hash(samples):
    return _sha256_text("\n".join(s.path for s in samples))


def build_context(model_name, device="cuda:0", max_ids=None, verbose=True):
    """Build split, adapter, gallery prototypes and raw validation tensors.

    ``max_ids`` restricts only the validation/query set (used for pilots); the
    gallery always contains all 600 identities.
    """
    ds_cfg = load_dataset_config()
    scfg = load_spectral_config()
    set_seed(ds_cfg["runtime"]["seed"])
    set_deterministic(True)

    split, split_meta, _ = ensure_splits(ds_cfg)
    val = split["validation"]
    if max_ids is not None:
        val = [s for s in val if s.label < int(max_ids)]
    if not val:
        raise ValueError("empty validation subset")

    mcfg = load_model_config(model_name)
    adapter = build_adapter(model_name, device=device, config=mcfg)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gallery_meta = {
        "model": model_name,
        "checkpoint_sha256": adapter.checkpoint_info["sha256"],
        "split": split_meta["name"],
        "n_gallery": len(split["gallery"]),
        "num_identities": int(ds_cfg["dataset"]["num_identities"]),
    }
    gallery_path = CACHE_DIR / f"gallery_{model_name}.npz"
    cached = load_gallery_cache(gallery_path, gallery_meta)
    if cached is None:
        if verbose:
            print(f"[{model_name}] building gallery cache...", flush=True)
        gallery_ds = TongjiDataset(split["gallery"], input_size=ds_cfg["dataset"]["input_size"])
        gallery = build_gallery(
            adapter, gallery_ds,
            batch_size=int(mcfg["batch"]["inference"]),
            num_identities=int(ds_cfg["dataset"]["num_identities"]),
        )
        save_gallery_cache(gallery_path, gallery, gallery_meta)
        cached = {"prototypes": gallery["prototypes"], "labels": gallery["labels"]}
    classifier = GalleryClassifier(
        cached["prototypes"], cached["labels"], temperature_scale=1.0
    ).to(adapter.device)

    ds = TongjiDataset(val, input_size=ds_cfg["dataset"]["input_size"])
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)

    return {
        "model": model_name,
        "device": torch.device(device),
        "dataset_cfg": ds_cfg,
        "spectral_cfg": scfg,
        "model_cfg": mcfg,
        "split": split,
        "split_meta": split_meta,
        "samples": val,
        "dataset": ds,
        "xs": xs,
        "ys": ys,
        "adapter": adapter,
        "classifier": classifier,
        "gallery_meta": gallery_meta,
        "paths_hash": paths_hash(val),
    }


@torch.no_grad()
def extract_features(adapter, x, batch_size=64):
    feats = []
    for start in range(0, len(x), batch_size):
        feats.append(adapter.extract_features(x[start:start + batch_size]).cpu())
    return torch.cat(feats)


@torch.no_grad()
def predict_labels(adapter, classifier, x, batch_size=64):
    preds = []
    for start in range(0, len(x), batch_size):
        feats = adapter.extract_features(x[start:start + batch_size])
        preds.append(classifier.predict(feats).cpu())
    return torch.cat(preds)


@torch.no_grad()
def rank_hits(prototypes, features, labels, ks=(1, 5)):
    """Per-sample rank-k hits against Gallery prototypes (labels 0..G-1)."""
    import torch.nn.functional as F

    feats = F.normalize(features.float(), dim=1)
    protos = F.normalize(prototypes.float(), dim=1)
    scores = protos @ feats.T  # [G, N]
    pred = scores.argmax(dim=0)
    gallery_labels = torch.arange(scores.shape[0], device=scores.device)
    out = {"pred": pred.cpu()}
    for k in ks:
        kk = min(int(k), scores.shape[0])
        top = scores.topk(kk, dim=0).indices
        out[f"rank{k}"] = (gallery_labels[top] == labels.to(scores.device)[None, :]).any(dim=0).cpu()
    return out


def _attack_fn(name):
    if name == "pgd":
        return pgd_linf
    if name == "apgd":
        return apgd_linf
    raise ValueError(f"unknown attack {name!r}")


def adversarial_cache_path(ctx, attack_name, attack_cfg):
    key = _sha256_text(json.dumps({
        "model": ctx["model"],
        "checkpoint": ctx["adapter"].checkpoint_info["sha256"],
        "split": ctx["split_meta"]["name"],
        "paths": ctx["paths_hash"],
        "attack": attack_name,
        "cfg": attack_cfg,
    }, sort_keys=True))[:16]
    return CACHE_DIR / f"adv_{ctx['model']}_{attack_name}_{key}.npz"


def get_adversarial(ctx, attack_name, verbose=True):
    """Load or generate cached adversarial images for the current context."""
    scfg = ctx["spectral_cfg"]
    attack_cfg = dict(scfg["attacks"][attack_name])
    batch_cfg = attack_cfg.pop("batch_size", 16)
    if isinstance(batch_cfg, dict):
        batch = int(batch_cfg.get(ctx["model"], 16))
    else:
        batch = int(batch_cfg)
    path = adversarial_cache_path(ctx, attack_name, attack_cfg)
    meta_path = path.with_suffix(".json")

    if path.is_file() and meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        data = np.load(path)
        x_adv = torch.from_numpy(data["x_adv"])
        if verbose:
            print(f"[{ctx['model']}] loaded {attack_name} cache: ASR={meta.get('asr', float('nan')):.3f}")
        return x_adv, meta

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fn = _attack_fn(attack_name)
    t0 = time.time()
    out = []
    n = len(ctx["xs"])
    for start in range(0, n, batch):
        xb = ctx["xs"][start:start + batch]
        yb = ctx["ys"][start:start + batch]
        if attack_name == "pgd":
            xa = fn(
                ctx["adapter"], ctx["classifier"], xb, yb,
                eps=attack_cfg["eps"], alpha=attack_cfg["alpha"],
                steps=attack_cfg["steps"], random_start=attack_cfg["random_start"],
                seed=attack_cfg["seed"] + start, loss=attack_cfg["loss"],
                scale=attack_cfg["scale"], restarts=attack_cfg.get("restarts", 1),
            )
        else:  # apgd
            xa = fn(
                ctx["adapter"], ctx["classifier"], xb, yb,
                eps=attack_cfg["eps"], steps=attack_cfg["steps"],
                restarts=attack_cfg.get("restarts", 1),
                seed=attack_cfg["seed"] + start, loss=attack_cfg["loss"],
                scale=attack_cfg["scale"],
            )
        out.append(xa.cpu())
        if verbose:
            print(f"[{ctx['model']}] {attack_name} {min(start + batch, n)}/{n}", flush=True)
    x_adv = torch.cat(out, dim=0)

    clean_pred = predict_labels(ctx["adapter"], ctx["classifier"], ctx["xs"],
                                batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
    adv_pred = predict_labels(ctx["adapter"], ctx["classifier"], x_adv,
                              batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
    correct = clean_pred == ctx["ys"]
    success = correct & (adv_pred != ctx["ys"])
    asr = float(success.sum().item() / max(int(correct.sum().item()), 1))
    meta = {
        "model": ctx["model"],
        "attack": attack_name,
        "attack_cfg": attack_cfg,
        "split": ctx["split_meta"]["name"],
        "n": int(n),
        "paths_hash": ctx["paths_hash"],
        "checkpoint_sha256": ctx["adapter"].checkpoint_info["sha256"],
        "clean_accuracy": float(correct.float().mean().item()),
        "asr": asr,
        "unconditional_error_rate": float((adv_pred != ctx["ys"]).float().mean().item()),
        "batch_size": int(batch),
        "runtime_s": round(time.time() - t0, 1),
        "linf_max": float((x_adv - ctx["xs"]).abs().max().item()),
        "l2_mean": float((x_adv - ctx["xs"]).pow(2).sum(dim=(1, 2, 3)).sqrt().mean().item()),
    }
    np.savez(path, x_adv=x_adv.numpy(), labels=ctx["ys"].numpy(),
             clean_pred=clean_pred.numpy(), adv_pred=adv_pred.numpy())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[{ctx['model']}] {attack_name} ASR={asr:.3f} in {meta['runtime_s']}s -> {path.name}")
    return x_adv, meta


def gaussian_noise(x, sigma, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=gen, dtype=x.dtype) * float(sigma)
    return torch.clamp(x + noise, 0.0, 1.0)


def condition_array(ctx, condition):
    if condition == "clean":
        return ctx["xs"], {"condition": "clean"}
    if condition == "gaussian":
        noise_cfg = ctx["spectral_cfg"]["noise"]
        x = gaussian_noise(ctx["xs"], noise_cfg["sigma"], noise_cfg["seed"])
        return x, {"condition": "gaussian", "noise_sigma": noise_cfg["sigma"],
                   "noise_seed": noise_cfg["seed"]}
    if condition in ("pgd", "apgd"):
        x, meta = get_adversarial(ctx, condition)
        return x, meta
    raise ValueError(f"unknown condition {condition!r}")


def save_csv(path, rows, fieldnames=None):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to save")
    fields = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path
