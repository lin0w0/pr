"""YAML config loading with project-root-relative paths."""
from pathlib import Path

import yaml

from utils.paths import PROJECT_ROOT, resolve_path


def load_config(path):
    """Load a YAML config file.

    The returned dict gets two private keys:
      ``_config_path``: absolute path of the YAML file
      ``_config_sha256``: sha256 of the file content
    """
    import hashlib

    abs_path = resolve_path(path)
    with open(abs_path, "r", encoding="utf-8") as f:
        raw = f.read()
    cfg = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config {abs_path} must contain a YAML mapping")
    cfg["_config_path"] = str(abs_path)
    cfg["_config_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return cfg


def load_dataset_config(path="configs/dataset.yaml"):
    return load_config(path)


def load_model_config(name, path=None):
    path = path or f"configs/{name}.yaml"
    cfg = load_config(path)
    if cfg.get("name") != name:
        raise ValueError(f"Config {path} declares name={cfg.get('name')!r}, expected {name!r}")
    return cfg


def dump_config(cfg, path):
    """Write a config copy (without private keys) for reproducibility."""
    abs_path = resolve_path(path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(abs_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
    return abs_path
