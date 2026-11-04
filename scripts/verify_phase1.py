"""Phase 1 acceptance run: reproduce the clean baselines on the confirmed split.

Usage:
    D:/code/research/envs/pr/python.exe scripts/verify_phase1.py --model all
    ... --model ccnet --max-ids 100          # smoke run
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.dataset_split import ensure_splits  # noqa: E402
from datasets.gallery import build_gallery, extract_features  # noqa: E402
from datasets.tongji_dataset import TongjiDataset  # noqa: E402
from evaluation.identification_metrics import rank_accuracy  # noqa: E402
from evaluation.verification_metrics import pairwise_scores, verification_metrics  # noqa: E402
from models import build_adapter  # noqa: E402
from utils.config import load_dataset_config, load_model_config  # noqa: E402
from utils.seed import set_deterministic, set_seed  # noqa: E402


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def subsample(samples, max_ids):
    if max_ids is None:
        return samples
    keep = {s.label for s in samples if s.label < max_ids}
    if len(keep) != max_ids:
        raise ValueError(f"requested {max_ids} ids, found {len(keep)}")
    return [s for s in samples if s.label < max_ids]


@torch.no_grad()
def model_logits_accuracy(adapter, dataset, batch_size):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    for x, y in loader:
        pred = adapter.get_model_logits(x).argmax(dim=1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / total * 100.0


def evaluate_split(adapter, gallery, split_name, samples, batch_size, temperature_scale):
    ds = TongjiDataset(samples, input_size=adapter.input_size)
    t0 = time.time()
    feats, labels = extract_features(adapter, ds, batch_size=batch_size)
    feat_s = time.time() - t0

    scores = gallery["prototypes"] @ feats.T  # both normalized -> cosine, [N, G]
    rank = rank_accuracy(scores.T, labels, gallery["labels"])
    t0 = time.time()
    genuine, impostor = pairwise_scores(
        feats, gallery["features"], labels, gallery["feature_labels"]
    )
    ver = verification_metrics(genuine, impostor)
    ver_s = time.time() - t0
    del genuine, impostor

    return {
        "split": split_name,
        "n": len(samples),
        "feature_extraction_s": round(feat_s, 2),
        "verification_s": round(ver_s, 2),
        "identification": rank,
        "verification": ver,
        "scores_are_cosine": True,
        "temperature_scale": temperature_scale,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-ids", type=int, default=None, help="smoke run over the first N identities")
    parser.add_argument("--out", default="results/phase1_baseline.json")
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset_config)
    set_seed(cfg["runtime"]["seed"])
    set_deterministic(bool(cfg["runtime"]["deterministic"]))

    split, split_meta, split_dir = ensure_splits(cfg)
    if args.max_ids:
        n_ids = min(args.max_ids, cfg["dataset"]["num_identities"])
        split = {k: subsample(v, n_ids) for k, v in split.items()}
    else:
        n_ids = cfg["dataset"]["num_identities"]

    # sanity: all three splits cover the same identities, no path overlap
    for a in split:
        for b in split:
            if a < b:
                assert not ({s.path for s in split[a]} & {s.path for s in split[b]})

    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": args.device,
        "n_identities": n_ids,
        "split": {
            "name": split_meta["name"],
            "protocol": split_meta["protocol"],
            "files": split_meta["files"],
            "dir": str(split_dir),
        },
        "models": {},
    }

    gallery_ds = TongjiDataset(split["gallery"], input_size=cfg["dataset"]["input_size"])
    for name in models:
        model_cfg = load_model_config(name)
        batch = int(model_cfg["batch"]["inference"])
        t0 = time.time()
        adapter = build_adapter(name, device=args.device, config=model_cfg)
        load_s = time.time() - t0
        print(f"\n[{name}] loaded in {load_s:.2f}s -> {adapter}")

        t0 = time.time()
        gallery = build_gallery(
            adapter, gallery_ds, batch_size=batch, num_identities=n_ids,
            progress=None,
        )
        gallery_s = time.time() - t0
        print(f"[{name}] gallery built in {gallery_s:.2f}s, prototypes {tuple(gallery['prototypes'].shape)}")

        entry = {
            "checkpoint": adapter.checkpoint_info,
            "feature_dim": adapter.feature_dim,
            "num_classes": adapter.num_classes,
            "batch_size": batch,
            "load_s": round(load_s, 2),
            "gallery_s": round(gallery_s, 2),
            "gallery_shape": list(gallery["prototypes"].shape),
            "splits": {},
        }
        for split_name in ("validation", "test"):
            res = evaluate_split(
                adapter, gallery, split_name, split[split_name], batch,
                temperature_scale=cfg["gallery"]["temperature_scale"],
            )
            res["model_logits_accuracy_pct"] = round(
                model_logits_accuracy(adapter, TongjiDataset(split[split_name], input_size=adapter.input_size), batch), 4
            )
            entry["splits"][split_name] = res
            print(f"[{name}] {split_name}: rank1={res['identification']['rank1']:.3f}% "
                  f"rank5={res['identification']['rank5']:.3f}% "
                  f"eer={res['verification']['eer_pct']:.4f}% "
                  f"arcface_acc={res['model_logits_accuracy_pct']:.3f}%")

        report["models"][name] = entry
        del adapter, gallery
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")

    # Phase 1 gate: rank-1 must be 100% on both splits for both models
    failures = []
    for name, entry in report["models"].items():
        for split_name, res in entry["splits"].items():
            if res["identification"]["rank1"] < 99.999:
                failures.append(f"{name}/{split_name} rank1={res['identification']['rank1']}")
    if failures:
        print("PHASE 1 GATE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PHASE 1 GATE PASSED (Rank-1 = 100%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
