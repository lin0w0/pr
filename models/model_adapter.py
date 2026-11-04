"""Unified, frozen adapter around the official CompNet / CCNet implementations.

The official network files are never modified. They are imported by absolute
file path so the local ``models`` package cannot shadow them.
"""
import gc
import hashlib
import importlib.util
import sys
import time

import torch
import torch.nn.functional as F

from datasets.preprocessing import OfficialROIPreprocess
from utils.paths import resolve_path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_official_class(module_path, class_name, alias):
    abs_path = resolve_path(module_path)
    spec = importlib.util.spec_from_file_location(alias, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"{abs_path} has no class {class_name!r}")
    return getattr(module, class_name)


class PalmprintModelAdapter:
    """Common interface for a frozen palmprint recognition backbone.

    Contract:
        features      -> ``[B, D]``, L2-normalized
        model logits  -> ``[B, num_classes]`` (ArcFace scale already applied)
        preprocessing -> differentiable replica of the official pipeline
    """

    name = "base"

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.name = config["name"]
        self.num_classes = int(config["num_classes"])
        self.expected_feature_dim = int(config["feature_dim"])
        self.input_size = int(config["input_size"])
        self.device = torch.device(device)

        self.preprocess = OfficialROIPreprocess(self.input_size)

        cls = _load_official_class(
            config["module_path"], config["class_name"], alias=f"_official_{self.name}"
        )
        init_kwargs = dict(config.get("init", {}))
        self.net = cls(**init_kwargs)
        self.checkpoint_info = self._load_checkpoint()
        self.net = self._move_to_device(self.net)
        self.net.eval()
        self.freeze()

        self.feature_dim = self.expected_feature_dim

    # ------------------------------------------------------------------ setup
    def _load_checkpoint(self):
        ck_cfg = self.config["checkpoint"]
        path = resolve_path(ck_cfg["path"])
        actual_sha = _sha256(path)
        expected_sha = ck_cfg.get("sha256")
        if expected_sha and actual_sha.lower() != expected_sha.lower():
            raise ValueError(f"{path} sha256 mismatch: {actual_sha} != {expected_sha}")

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(f"{path} does not contain a state_dict")

        stripped = {}
        for key, value in state.items():
            new_key = key
            while new_key.startswith("module."):
                new_key = new_key[len("module."):]
            stripped[new_key] = value

        prefixes = tuple(ck_cfg.get("ignore_prefixes", []))
        ignored = {k for k in stripped if prefixes and k.startswith(prefixes)}
        expected_ignored = int(ck_cfg.get("expected_ignored_keys", 0))
        if len(ignored) != expected_ignored:
            raise ValueError(
                f"{self.name}: ignored {len(ignored)} checkpoint keys, expected "
                f"{expected_ignored}. Keys: {sorted(ignored)[:10]}"
            )
        filtered = {k: v for k, v in stripped.items() if k not in ignored}

        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{self.name}: checkpoint mismatch after filtering. "
                f"missing={missing}, unexpected={unexpected}"
            )
        return {
            "path": str(path),
            "sha256": actual_sha,
            "n_active_tensors": len(filtered),
            "n_ignored_tensors": len(ignored),
        }

    def _move_to_device(self, net, retries=3):
        """Move weights to the device, retrying transient Windows OOM errors.

        On Windows/WDDM a large single allocation (CCNet's 206 MiB ``fc.weight``)
        has been observed to fail intermittently even with >6 GiB free. The
        failure is transient in practice; a fresh cache + retry succeeds.
        """
        for attempt in range(retries):
            try:
                return net.to(self.device)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or attempt == retries - 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(2.0)
        return net

    def freeze(self):
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        return self

    def num_parameters(self):
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------ core inference
    def preprocess_raw(self, x):
        """Raw ``[B,1,H,W]`` in ``[0,1]`` -> differentiable official preprocessing."""
        return self.preprocess(x.to(self.device))

    def features_from_preprocessed(self, z):
        """Preprocessed tensor -> L2-normalized features ``[B, D]``."""
        return F.normalize(self.net.getFeatureCode(z), p=2, dim=1)

    def extract_features(self, x_raw):
        """Raw images -> L2-normalized features (differentiable)."""
        return self.features_from_preprocessed(self.preprocess_raw(x_raw))

    def model_logits_from_preprocessed(self, z):
        out = self.net(z, None)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def get_model_logits(self, x_raw):
        return self.model_logits_from_preprocessed(self.preprocess_raw(x_raw))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def normalize_features(features):
        return F.normalize(features, p=2, dim=1)

    @staticmethod
    def get_similarity(features1, features2):
        """Cosine similarity matrix ``[N1, N2]`` between normalized features."""
        f1 = F.normalize(features1, p=2, dim=1)
        f2 = F.normalize(features2, p=2, dim=1)
        return f1 @ f2.T

    @torch.no_grad()
    def validate(self):
        """Run a random-input shape/normalization self-check."""
        x = torch.rand(2, 1, self.input_size, self.input_size, device=self.device) * 0.8 + 0.1
        feats = self.extract_features(x)
        logits = self.get_model_logits(x)
        if tuple(feats.shape) != (2, self.expected_feature_dim):
            raise RuntimeError(f"feature shape {tuple(feats.shape)} != (2, {self.expected_feature_dim})")
        if tuple(logits.shape) != (2, self.num_classes):
            raise RuntimeError(f"logits shape {tuple(logits.shape)} != (2, {self.num_classes})")
        norms = feats.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            raise RuntimeError(f"features are not unit-norm: {norms.tolist()}")
        if not torch.isfinite(feats).all() or not torch.isfinite(logits).all():
            raise RuntimeError("non-finite outputs")
        return {
            "name": self.name,
            "feature_dim": int(feats.shape[1]),
            "num_classes": int(logits.shape[1]),
            "feature_norm_max_err": float((norms - 1).abs().max()),
            "n_params": self.num_parameters(),
            "checkpoint": self.checkpoint_info,
            "device": str(self.device),
        }

    def __repr__(self):
        return (f"{type(self).__name__}(name={self.name!r}, D={self.feature_dim}, "
                f"classes={self.num_classes}, device={self.device})")
