"""Phase 1 environment / data / checkpoint self-check.

Usage:
    D:/code/research/envs/pr/python.exe scripts/check_environment.py [--skip-models]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from datasets.dataset_split import ensure_splits, scan_dataset, summarize  # noqa: E402
from utils.config import load_dataset_config, load_model_config  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append({"check": name, "status": "PASS", "detail": detail})
        print(f"[PASS] {name}" + (f": {detail}" if detail else ""))
    except Exception as e:
        RESULTS.append({"check": name, "status": "FAIL", "detail": repr(e)})
        print(f"[FAIL] {name}: {e!r}")


def _env():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available() else None,
    }
    return info


def _packages():
    out = {}
    for mod in ["numpy", "scipy", "sklearn", "matplotlib", "cv2", "yaml", "h5py",
                "pandas", "openpyxl", "autoattack", "pytest"]:
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "installed")
        except Exception:
            out[mod] = "MISSING"
    required = ["numpy", "scipy", "sklearn", "matplotlib", "cv2", "yaml"]
    missing = [m for m in required if out[m] == "MISSING"]
    if missing:
        raise RuntimeError(f"required packages missing: {missing}")
    return out


def _dataset(cfg):
    samples = scan_dataset(cfg)
    labels = sorted({s.label for s in samples})
    return {
        "n_images": len(samples),
        "n_identities": len(labels),
        "sessions": sorted({s.session for s in samples}),
        "per_identity_per_session": len(samples) // (len(labels) * len({s.session for s in samples})),
    }


def _splits(cfg):
    split, meta, out = ensure_splits(cfg)
    return {"dir": str(out), "protocol": meta["protocol"], **summarize(split)}


def _checkpoints():
    out = {}
    for name in ("compnet", "ccnet"):
        cfg = load_model_config(name)
        p = (ROOT / cfg["checkpoint"]["path"]).resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        import hashlib

        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        expected = cfg["checkpoint"]["sha256"]
        if actual != expected:
            raise ValueError(f"{name}: sha256 {actual} != {expected}")
        out[name] = {"path": str(p), "sha256": actual[:16] + "..."}
    return out


def _models(device):
    from models import build_adapter

    out = {}
    for name in ("compnet", "ccnet"):
        adapter = build_adapter(name, device=device)
        out[name] = adapter.validate()
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    print("=== environment ===")
    check("python/torch/cuda", _env)
    check("packages", _packages)
    print("=== data ===")
    check("dataset scan", lambda: _dataset(cfg))
    check("splits", lambda: _splits(cfg))
    print("=== checkpoints ===")
    check("checkpoint hashes", _checkpoints)
    if not args.skip_models:
        print("=== models ===")
        check("adapter validation", lambda: _models(args.device))

    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    print()
    if failed:
        print(f"FAILED: {len(failed)} check(s)")
        for r in failed:
            print(f"  - {r['check']}: {r['detail']}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
