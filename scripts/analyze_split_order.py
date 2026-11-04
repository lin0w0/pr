"""Phase 1 review: test whether session2 image order biases the val/test split.

Protocol A uses session2 indices 0-4 for validation and 5-9 for test. This
script measures, on real data:

  * per-index raw intensity mean/std,
  * per-index cosine similarity to the identity's gallery prototype,
  * per-index Rank-1,
  * Spearman drift across indices,
  * validation-vs-test distribution tests (KS, paired by identity),
  * available ordering metadata (PIL info / file mtime).

It does NOT change the split. Writes under results/phase1_review/.
"""
import argparse
import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402

from datasets.dataset_split import ensure_splits, scan_dataset  # noqa: E402
from datasets.gallery import build_gallery, extract_features  # noqa: E402
from datasets.tongji_dataset import TongjiDataset  # noqa: E402
from models import build_adapter  # noqa: E402
from utils.config import load_dataset_config, load_model_config  # noqa: E402
from utils.paths import resolve_path  # noqa: E402
from utils.seed import set_deterministic, set_seed  # noqa: E402

OUT_DIR = ROOT / "results" / "phase1_review"


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


def raw_stats(samples):
    means, stds = [], []
    for s in samples:
        with Image.open(resolve_path(s.path)) as im:
            a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        means.append(float(a.mean()))
        stds.append(float(a.std()))
    return np.asarray(means), np.asarray(stds)


def per_index(values, indices):
    out = {}
    for idx in range(10):
        v = values[indices == idx]
        out[int(idx)] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "median": float(np.median(v)),
            "min": float(v.min()),
            "max": float(v.max()),
        }
    return out


def paired_split_test(values, labels, indices, val_range=(0, 4), test_range=(5, 9)):
    """Per-identity mean(val) vs mean(test), paired tests."""
    diffs = []
    for label in np.unique(labels):
        m = labels == label
        val = values[m & (indices >= val_range[0]) & (indices <= val_range[1])]
        test = values[m & (indices >= test_range[0]) & (indices <= test_range[1])]
        diffs.append(float(val.mean() - test.mean()))
    diffs = np.asarray(diffs)
    n = len(diffs)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    # normal-approximation 95% CI for the paired mean difference
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    t = stats.ttest_rel(
        np.array([values[(labels == l) & (indices <= 4)].mean() for l in np.unique(labels)]),
        np.array([values[(labels == l) & (indices >= 5)].mean() for l in np.unique(labels)]),
    )
    try:
        w = stats.wilcoxon(diffs)
    except ValueError:
        w = None
    return {
        "n_identities": int(n),
        "mean_val_minus_test": mean,
        "std": float(diffs.std(ddof=1)),
        "ci95_normal": [float(ci[0]), float(ci[1])],
        "paired_t_stat": float(t.statistic),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_stat": float(w.statistic) if w is not None else None,
        "wilcoxon_p": float(w.pvalue) if w is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_dataset_config()
    set_seed(cfg["runtime"]["seed"])
    set_deterministic(bool(cfg["runtime"]["deterministic"]))
    split, split_meta, _ = ensure_splits(cfg)
    all_samples = scan_dataset(cfg)
    session2 = [s for s in all_samples if s.session == "session2"]
    session2.sort(key=lambda s: (s.label, s.index_in_id))
    labels = np.array([s.label for s in session2], dtype=np.int64)
    indices = np.array([s.index_in_id for s in session2], dtype=np.int64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"session2 samples: {len(session2)}")
    raw_mean, raw_std = raw_stats(session2)
    gc.collect()
    print("raw stats done")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split_meta["name"],
        "n_session2": len(session2),
        "question": "Does session2 index order (0-4 vs 5-9) bias validation/test?",
        "ordering_metadata": {},
        "raw": {},
        "models": {},
    }

    # ---- ordering metadata: there is no capture timestamp in the BMP files ----
    sample_paths = [session2[i].abs_path for i in (0, 1, len(session2) // 2, len(session2) - 1)]
    info = []
    for p in sample_paths:
        with Image.open(p) as im:
            info.append({"path": p.name, "info_keys": sorted(im.info.keys()), "exif": bool(im.getexif())})
        st = p.stat()
        info[-1]["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    report["ordering_metadata"] = {
        "samples": info,
        "filename_order_is_only_order": True,
        "note": "BMP files carry no capture timestamp; filenames are the only ordering available. "
                "The official genText script also sorts by filename.",
    }
    report["raw"]["mean_intensity_per_index"] = per_index(raw_mean, indices)
    report["raw"]["std_intensity_per_index"] = per_index(raw_std, indices)
    sp = stats.spearmanr(indices, raw_mean)
    report["raw"]["spearman_index_vs_mean_intensity"] = {
        "rho": float(sp.statistic), "p": float(sp.pvalue)
    }

    gallery_ds = TongjiDataset(split["gallery"], input_size=cfg["dataset"]["input_size"])
    ds = TongjiDataset(session2, input_size=cfg["dataset"]["input_size"])
    rows = []
    for name in ("compnet", "ccnet"):
        mcfg = load_model_config(name)
        batch = int(mcfg["batch"]["inference"])
        adapter = build_adapter(name, device=args.device, config=mcfg)
        print(f"[{name}] gallery + session2 features...")
        gallery = build_gallery(adapter, gallery_ds, batch_size=batch, num_identities=600)
        feats, feat_labels = extract_features(adapter, ds, batch_size=batch)
        assert np.array_equal(feat_labels.numpy(), labels)

        protos = gallery["prototypes"]
        sims = (F.normalize(feats, dim=1) * protos[feat_labels]).sum(dim=1).numpy()
        scores = feats @ protos.T
        pred = scores.argmax(dim=1).numpy()
        correct = (pred == feat_labels.numpy()).astype(np.float64)

        sp = stats.spearmanr(indices, sims)
        ks = stats.ks_2samp(sims[indices <= 4], sims[indices >= 5])
        paired = paired_split_test(sims, labels, indices)
        entry = {
            "similarity_to_own_prototype_per_index": per_index(sims, indices),
            "rank1_per_index": {str(i): float(correct[indices == i].mean() * 100.0) for i in range(10)},
            "spearman_index_vs_similarity": {"rho": float(sp.statistic), "p": float(sp.pvalue)},
            "ks_val_vs_test_similarity": {"statistic": float(ks.statistic), "p": float(ks.pvalue)},
            "paired_val_vs_test": paired,
        }
        report["models"][name] = entry
        for idx in range(10):
            rows.append({
                "model": name,
                "index_in_id": idx,
                "n": 600,
                "similarity_mean": entry["similarity_to_own_prototype_per_index"][idx]["mean"],
                "similarity_std": entry["similarity_to_own_prototype_per_index"][idx]["std"],
                "rank1_pct": entry["rank1_per_index"][str(idx)],
                "raw_mean_intensity": report["raw"]["mean_intensity_per_index"][idx]["mean"],
                "raw_std_intensity": report["raw"]["std_intensity_per_index"][idx]["mean"],
            })
        print(f"[{name}] rho={sp.statistic:.4f} p={sp.pvalue:.3g} "
              f"KS={ks.statistic:.4f} p={ks.pvalue:.3g} "
              f"paired diff={paired['mean_val_minus_test']:.5f} p={paired['paired_t_p']:.3g}",
              flush=True)
        del adapter, gallery, feats, sims, scores, pred, correct
        gc.collect()
        torch.cuda.empty_cache()

    with open(OUT_DIR / "split_order_analysis.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "split_order_analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("compnet", "ccnet"):
        e = report["models"][name]["similarity_to_own_prototype_per_index"]
        ys = [e[i]["mean"] for i in range(10)]
        err = [e[i]["std"] / np.sqrt(600) for i in range(10)]
        axes[0].errorbar(range(10), ys, yerr=err, marker="o", capsize=3, label=name)
    axes[0].axvline(4.5, color="k", linestyle="--", linewidth=1)
    axes[0].set_xlabel("index within identity (session2)")
    axes[0].set_ylabel("cosine to gallery prototype")
    axes[0].set_title("Feature similarity vs index")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    ys = [report["raw"]["mean_intensity_per_index"][i]["mean"] for i in range(10)]
    err = [report["raw"]["mean_intensity_per_index"][i]["std"] / np.sqrt(600) for i in range(10)]
    axes[1].errorbar(range(10), ys, yerr=err, marker="s", color="tab:green", capsize=3)
    axes[1].axvline(4.5, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("index within identity (session2)")
    axes[1].set_ylabel("mean pixel intensity")
    axes[1].set_title("Raw intensity vs index")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "split_order_drift.png", dpi=160)
    plt.close(fig)

    print(f"wrote {OUT_DIR / 'split_order_analysis.json'}")
    print(f"wrote {OUT_DIR / 'split_order_analysis.csv'}")
    print(f"wrote {OUT_DIR / 'split_order_drift.png'}")


if __name__ == "__main__":
    main()
