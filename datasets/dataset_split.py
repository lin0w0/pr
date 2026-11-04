"""Deterministic Tongji split construction (protocol A).

Protocol A (confirmed by the user):
    gallery    = session1, all 10 images per identity
    validation = session2, images 0-4 per identity
    test       = session2, images 5-9 per identity

Image indices are inclusive and 0-based within each identity's sorted filenames.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone

from datasets.tongji_dataset import Sample, read_split_csv
from utils.paths import PROJECT_ROOT, resolve_path


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(cfg):
    """Scan the ROI directories and return sorted :class:`Sample` objects.

    Also validates that every identity has the configured number of images and
    that labels are contiguous from 0.
    """
    ds = cfg["dataset"]
    root = resolve_path(ds["root"])
    sessions = list(ds["sessions"])
    ext = ds["file_extension"].lower()
    expected_per_id = int(ds["images_per_identity"])

    samples = []
    for session in sessions:
        session_dir = root / session
        if not session_dir.is_dir():
            raise FileNotFoundError(f"missing session directory: {session_dir}")
        files = sorted(p for p in session_dir.iterdir() if p.suffix.lower() == ext)
        if not files:
            raise FileNotFoundError(f"no {ext} files in {session_dir}")

        per_label = {}
        for p in files:
            raw_id = int(p.stem[:5])
            label = (raw_id - 1) // 10
            per_label.setdefault(label, []).append((raw_id, p))

        for label, items in per_label.items():
            items.sort(key=lambda t: t[0])
            if len(items) != expected_per_id:
                raise ValueError(
                    f"{session} label {label} has {len(items)} images, "
                    f"expected {expected_per_id}"
                )
            raw_ids = [r for r, _ in items]
            if raw_ids != list(range(raw_ids[0], raw_ids[0] + expected_per_id)):
                raise ValueError(f"{session} label {label} raw ids are not contiguous: {raw_ids}")
            for index_in_id, (raw_id, p) in enumerate(items):
                rel = p.relative_to(PROJECT_ROOT).as_posix()
                samples.append(Sample(
                    path=rel,
                    label=label,
                    raw_id=raw_id,
                    session=session,
                    index_in_id=index_in_id,
                ))

    samples.sort(key=lambda s: (s.session, s.raw_id))
    _validate(samples, cfg)
    return samples


def _validate(samples, cfg):
    ds = cfg["dataset"]
    n_ids = int(ds["num_identities"])
    labels = sorted({s.label for s in samples})
    if labels != list(range(n_ids)):
        raise ValueError(f"expected labels 0..{n_ids - 1}, got {labels[:5]}...{labels[-5:]}")
    for session in ds["sessions"]:
        sess = [s for s in samples if s.session == session]
        if len({s.label for s in sess}) != n_ids:
            raise ValueError(f"session {session} does not cover all {n_ids} identities")
    paths = [s.path for s in samples]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate sample paths")


def _select(samples, spec, name):
    session = spec["session"]
    indices = spec["image_indices"]
    if indices == "all":
        selected = [s for s in samples if s.session == session]
    else:
        if not (isinstance(indices, (list, tuple)) and len(indices) == 2):
            raise ValueError(f"split.{name}.image_indices must be 'all' or [lo, hi]")
        lo, hi = int(indices[0]), int(indices[1])
        if not (0 <= lo <= hi):
            raise ValueError(f"invalid inclusive range for split.{name}: {indices}")
        selected = [s for s in samples if s.session == session and lo <= s.index_in_id <= hi]
    if not selected:
        raise ValueError(f"split {name!r} selected zero samples")
    selected.sort(key=lambda s: (s.label, s.index_in_id))
    return selected


def make_split(samples, cfg):
    """Build gallery / validation / test lists and check for leakage."""
    split_cfg = cfg["split"]
    split = {
        "gallery": _select(samples, split_cfg["gallery"], "gallery"),
        "validation": _select(samples, split_cfg["validation"], "validation"),
        "test": _select(samples, split_cfg["test"], "test"),
    }

    path_sets = {k: {s.path for s in v} for k, v in split.items()}
    for a in split:
        for b in split:
            if a < b and path_sets[a] & path_sets[b]:
                raise ValueError(f"image leakage between {a} and {b}")
    for name in ("validation", "test"):
        if {s.label for s in split[name]} != {s.label for s in split["gallery"]}:
            raise ValueError(f"{name} does not cover every gallery identity")
    return split


def save_split(split, cfg, out_dir=None):
    """Persist the split CSVs plus a metadata/JSON manifest."""
    split_cfg = cfg["split"]
    out = resolve_path(out_dir or f"{split_cfg['output_dir']}/{split_cfg['name']}")
    out.mkdir(parents=True, exist_ok=True)

    fields = ["path", "label", "raw_id", "session", "index_in_id"]
    files = {}
    for name, samples in split.items():
        path = out / f"{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "path": s.path,
                    "label": s.label,
                    "raw_id": s.raw_id,
                    "session": s.session,
                    "index_in_id": s.index_in_id,
                })
        try:
            stored = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            stored = str(path)
        files[name] = {
            "file": stored,
            "n": len(samples),
            "sha256": _sha256_file(path),
        }

    meta = {
        "protocol": split_cfg["protocol"],
        "name": split_cfg["name"],
        "seed": split_cfg["seed"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg["dataset"],
        "split": split_cfg,
        "files": files,
    }
    with open(out / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out, meta


def load_split(cfg):
    """Load an existing split and verify it against its manifest."""
    split_cfg = cfg["split"]
    out = resolve_path(f"{split_cfg['output_dir']}/{split_cfg['name']}")
    meta_path = out / "split_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing split manifest: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    split = {}
    for name in ("gallery", "validation", "test"):
        info = meta["files"][name]
        csv_path = resolve_path(info["file"])
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing split file: {csv_path}")
        if _sha256_file(csv_path) != info["sha256"]:
            raise ValueError(f"split file hash mismatch: {csv_path}")
        split[name] = read_split_csv(csv_path)
        if len(split[name]) != info["n"]:
            raise ValueError(f"split row count mismatch for {name}")
    return split, meta, out


def ensure_splits(cfg, force=False):
    """Load the split if present, otherwise build it deterministically."""
    if not force:
        try:
            split, meta, out = load_split(cfg)
            return split, meta, out
        except FileNotFoundError:
            pass
    samples = scan_dataset(cfg)
    split = make_split(samples, cfg)
    out, meta = save_split(split, cfg)
    return split, meta, out


def summarize(split):
    return {name: {
        "n": len(samples),
        "labels": len({s.label for s in samples}),
        "sessions": sorted({s.session for s in samples}),
        "images_per_label": len(samples) // len({s.label for s in samples}),
    } for name, samples in split.items()}
