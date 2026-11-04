"""Project-wide path resolution.

All config paths are stored relative to the project root so scripts can be
launched from any working directory.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    """Resolve a config path relative to the project root.

    Absolute paths are returned unchanged.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def ensure_dir(path):
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
