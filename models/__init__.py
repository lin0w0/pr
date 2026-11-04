from models.model_adapter import PalmprintModelAdapter
from models.compnet_adapter import CompNetAdapter
from models.ccnet_adapter import CCNetAdapter

ADAPTERS = {
    "compnet": CompNetAdapter,
    "ccnet": CCNetAdapter,
}


def build_adapter(name, device="cuda:0", config=None):
    """Build an adapter from ``configs/<name>.yaml`` or an explicit config dict."""
    if name not in ADAPTERS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(ADAPTERS)}")
    if config is None:
        from utils.config import load_model_config

        config = load_model_config(name)
    return ADAPTERS[name](config, device=device)


__all__ = [
    "PalmprintModelAdapter",
    "CompNetAdapter",
    "CCNetAdapter",
    "ADAPTERS",
    "build_adapter",
]
