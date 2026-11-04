"""Phase 1 review: compare Gallery protocols, recompute EER, calibrate thresholds.

Writes only under results/phase1_review/ and never touches splits/weights/models.

Prototype protocol:
    prototype = normalize(mean(normalize(f(registration images))))
    scores    = probe @ prototypes.T
Official protocol (official inference.py):
    scores    = probe @ every registration image.T ; rank-1 = nearest image
EER:
    both protocols, complete ROC over genuine union impostor scores.
Threshold calibration:
    validation scores -> threshold at target FAR -> apply unchanged to test.
"""
import argparse
import csv
import gc
import json
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
from evaluation.official_matching import rank_accuracy_image_gallery  # noqa: E402
from evaluation.verification_metrics import (  # noqa: E402
    at_threshold,
    evaluate_at_threshold,
    pairwise_scores,
    threshold_at_far,
    verification_metrics,
)
from models import build_adapter  # noqa: E402
from utils.config import load_dataset_config, load_model_config  # noqa: E402
from utils.seed import set_deterministic, set_seed  # noqa: E402

OUT_DIR = ROOT / "results" / "phase1_review"


def legacy_eer(genuine, impostor):
    """Reproduction of the pre-review EER (thresholds from genuine only)."""
    g = np.sort(np.asarray(genuine, dtype=np.float64))
    i = np.sort(np.asarray(impostor, dtype=np.float64))
    th = np.unique(g)
    tpr = 1.0 - np.searchsorted(g, th, side="left") / len(g)
    fpr = 1.0 - np.searchsorted(i, th, side="left") / len(i)
    idx = int(np.argmin(np.abs(tpr - (1.0 - fpr))))
    return {
        "eer_pct": round(float((fpr[idx] + (1.0 - tpr[idx])) / 2.0 * 100.0), 6),
        "eer_threshold": float(th[idx]),
    }


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
    return [s for s in samples if s.label < max_ids]


def euler_report(genuine, impostor):
    ver = verification_metrics(genuine, impostor)
    new = {
        k: ver[k] for k in (
            "eer_pct", "eer_threshold", "eer_mode", "eer_discrete_pct",
            "eer_discrete_threshold", "auc", "n_genuine", "n_impostor",
            "genuine_mean", "impostor_mean", "score_min", "score_max",
        )
    }
    old = legacy_eer(genuine, impostor)
    report = {
        "new": new,
        "legacy": old,
        "delta_pct": round(new["eer_pct"] - old["eer_pct"], 6),
    }
    return report, ver


def evaluate_protocols(adapter, gallery, split, batch):
    """Return metrics and score arrays for one split."""
    ds = TongjiDataset(split, input_size=adapter.input_size)
    feats, labels = extract_features(adapter, ds, batch_size=batch)
    protos = gallery["prototypes"]
    proto_labels = gallery["labels"]

    t0 = time.time()
    proto_scores = protos @ feats.T  # [G, N]
    proto_rank = rank_accuracy(proto_scores.T, labels, proto_labels)
    proto_gen, proto_imp = pairwise_scores(feats, protos, labels, proto_labels)
    proto_eer, proto_ver = euler_report(proto_gen, proto_imp)
    proto_time = time.time() - t0

    t0 = time.time()
    off_rank = rank_accuracy_image_gallery(
        feats, labels, gallery["features"], gallery["feature_labels"], topk=(1, 5)
    )
    off_gen, off_imp = pairwise_scores(feats, gallery["features"], labels, gallery["feature_labels"])
    off_eer, off_ver = euler_report(off_gen, off_imp)
    off_time = time.time() - t0

    return {
        "prototype": {
            "rank1": proto_rank["rank1"],
            "rank5": proto_rank["rank5"],
            "eer": proto_eer,
            "verification": proto_ver,
            "elapsed_s": round(proto_time, 2),
        },
        "official": {
            "rank1": off_rank["rank1"],
            "rank5": off_rank["rank5"],
            "eer": off_eer,
            "verification": off_ver,
            "elapsed_s": round(off_time, 2),
        },
        "_scores": {
            "prototype": (proto_gen, proto_imp),
            "official": (off_gen, off_imp),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default=None, help="suffix for output filenames")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    cfg = load_dataset_config()
    set_seed(cfg["runtime"]["seed"])
    set_deterministic(bool(cfg["runtime"]["deterministic"]))
    split, split_meta, _ = ensure_splits(cfg)
    n_ids = cfg["dataset"]["num_identities"]
    if args.max_ids:
        n_ids = min(args.max_ids, n_ids)
        split = {k: subsample(v, n_ids) for k, v in split.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split_meta["name"],
        "n_identities": n_ids,
        "max_ids": args.max_ids,
        "device": args.device,
        "protocols": {
            "prototype": "mean of per-image normalized registration features, re-normalized",
            "official": "nearest of all registration images (official inference.py)",
        },
        "models": {},
    }
    csv_rows = []

    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    gallery_ds = TongjiDataset(split["gallery"], input_size=cfg["dataset"]["input_size"])
    for name in models:
        mcfg = load_model_config(name)
        batch = int(mcfg["batch"]["inference"])
        adapter = build_adapter(name, device=args.device, config=mcfg)
        print(f"\n[{name}] gallery...", flush=True)
        t0 = time.time()
        gallery = build_gallery(adapter, gallery_ds, batch_size=batch, num_identities=n_ids)
        print(f"[{name}] gallery built in {time.time() - t0:.1f}s", flush=True)

        entry = {
            "checkpoint": adapter.checkpoint_info,
            "feature_dim": adapter.feature_dim,
            "n_gallery_images": len(gallery["features"]),
            "splits": {},
            "calibration": {},
        }
        scores_by_split = {}
        for split_name in ("validation", "test"):
            print(f"[{name}] {split_name}...", flush=True)
            t0 = time.time()
            res = evaluate_protocols(adapter, gallery, split[split_name], batch)
            res["total_s"] = round(time.time() - t0, 2)
            scores_by_split[split_name] = res.pop("_scores")
            entry["splits"][split_name] = res
            for protocol in ("prototype", "official"):
                p = res[protocol]
                csv_rows.append({
                    "model": name,
                    "split": split_name,
                    "protocol": protocol,
                    "rank1": p["rank1"],
                    "rank5": p["rank5"],
                    "eer_new_pct": p["eer"]["new"]["eer_pct"],
                    "eer_new_mode": p["eer"]["new"]["eer_mode"],
                    "eer_old_pct": p["eer"]["legacy"]["eer_pct"],
                    "eer_delta_pct": p["eer"]["delta_pct"],
                    "auc": p["eer"]["new"]["auc"],
                    "n_genuine": p["eer"]["new"]["n_genuine"],
                    "n_impostor": p["eer"]["new"]["n_impostor"],
                })
                print(f"  {protocol:9s} rank1={p['rank1']:.3f}% rank5={p['rank5']:.3f}% "
                      f"EER_new={p['eer']['new']['eer_pct']:.6f}% "
                      f"EER_old={p['eer']['legacy']['eer_pct']:.6f}%", flush=True)

        # ---- validation-calibrated thresholds, applied unchanged to test ----
        val = scores_by_split["validation"]
        test = scores_by_split["test"]
        for protocol in ("prototype", "official"):
            vg, vi = val[protocol]
            tg, ti = test[protocol]
            oracle_key = {0.01: "tar_at_far_1pct", 0.001: "tar_at_far_0.1pct"}
            for target in (0.01, 0.001):
                key = f"far_{target:g}"
                cal = threshold_at_far(vg, vi, target)
                test_achieved = evaluate_at_threshold(tg, ti, cal["threshold"])
                entry["calibration"].setdefault(protocol, {})[key] = {
                    "calibration_on_validation": cal,
                    "validation_at_threshold": at_threshold(vg, vi, cal["threshold"]),
                    "test_at_validation_threshold": test_achieved,
                    "test_oracle_tar_at_far_pct": entry["splits"]["test"][protocol]["verification"][oracle_key[target]],
                    "test_oracle_note": "threshold chosen from the test ROC; reported for reference only",
                }
                print(f"  calib {protocol} FAR<={target:g}: thr={cal['threshold']:.6f} "
                      f"valFAR={cal['calibration_far']:.6f} valTAR={cal['calibration_tar']:.6f} "
                      f"-> testFAR={test_achieved['FAR']:.6f} testTAR={test_achieved['TAR']:.6f}",
                      flush=True)

        report["models"][name] = entry
        # Drop all large score arrays before the next model. Python locals would
        # otherwise keep ~150 MB alive while CCNet allocates its own ~18M arrays.
        del val, test, vg, vi, tg, ti
        del adapter, gallery, scores_by_split
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = OUT_DIR / f"gallery_eer_review{suffix}.json"
    csv_path = OUT_DIR / f"gallery_eer_review{suffix}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
