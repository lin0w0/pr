"""Build and persist the deterministic Tongji split (protocol A).

Usage:
    D:/code/research/envs/pr/python.exe scripts/build_splits.py [--force]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_split import ensure_splits, scan_dataset, summarize  # noqa: E402
from utils.config import load_dataset_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build Tongji gallery/validation/test splits")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild even if a valid split exists")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    if args.force:
        samples = scan_dataset(cfg)
        print(f"scanned {len(samples)} images")
    split, meta, out = ensure_splits(cfg, force=args.force)
    summary = summarize(split)
    print(f"split: {meta['name']} (protocol {meta['protocol']})")
    print(f"output: {out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for name, info in meta["files"].items():
        print(f"  {name:10s} n={info['n']:5d} sha256={info['sha256'][:16]}...")


if __name__ == "__main__":
    main()
