"""Phase 1 review / Phase 2 preparation: CE vs Margin SGM stability.

For each model, loss (CE, Margin) and Gallery logit scale (1, 10, 30), compute
the gradient of the identity loss w.r.t. the spectral amplitude (phase fixed)
on the same real validation images, then report:

  * gradient L1 (mean / sum / std), L2, max
  * fraction of near-zero-gradient samples
  * low / middle / high band energy ratios

Writes under results/phase1_review/. Uses validation images only.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from datasets.dataset_split import ensure_splits  # noqa: E402
from datasets.gallery import GalleryClassifier, build_gallery  # noqa: E402
from datasets.preprocessing import load_raw_image  # noqa: E402
from datasets.tongji_dataset import TongjiDataset  # noqa: E402
from models import build_adapter  # noqa: E402
from spectral.frequency import DEFAULT_BANDS, radial_frequency  # noqa: E402
from spectral.losses import LOSSES  # noqa: E402
from spectral.sgm import amplitude_gradient, gradient_metrics  # noqa: E402
from utils.config import load_dataset_config, load_model_config  # noqa: E402
from utils.paths import resolve_path  # noqa: E402
from utils.seed import set_deterministic, set_seed  # noqa: E402

OUT_DIR = ROOT / "results" / "phase1_review"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--n-ids", type=int, default=20, help="validation identities (5 images each)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_dataset_config()
    set_seed(cfg["runtime"]["seed"])
    set_deterministic(bool(cfg["runtime"]["deterministic"]))
    split, split_meta, _ = ensure_splits(cfg)

    val = [s for s in split["validation"] if s.label < args.n_ids]
    xs = torch.stack([load_raw_image(s.path, cfg["dataset"]["input_size"]) for s in val])
    ys = torch.tensor([s.label for s in val], dtype=torch.long)
    n = len(val)
    print(f"validation images: {n} ({args.n_ids} identities)")

    scales = (1, 10, 30)
    losses = ("ce", "margin")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split_meta["name"],
        "n_images": n,
        "n_identities": args.n_ids,
        "batch_size": args.batch_size,
        "logit_scales": list(scales),
        "losses": list(losses),
        "bands": DEFAULT_BANDS,
        "models": {},
        "reconstruction_check": {},
    }
    aggregate_rows = []
    per_sample_rows = []

    gallery_ds = TongjiDataset(split["gallery"], input_size=cfg["dataset"]["input_size"])

    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    for name in models:
        mcfg = load_model_config(name)
        adapter = build_adapter(name, device=args.device, config=mcfg)
        print(f"[{name}] gallery...", flush=True)
        gallery = build_gallery(
            adapter, gallery_ds,
            batch_size=int(mcfg["batch"]["inference"]),
            num_identities=cfg["dataset"]["num_identities"],
        )
        assert torch.equal(gallery["labels"], torch.arange(len(gallery["prototypes"])))
        classifier = GalleryClassifier(
            gallery["prototypes"], gallery["labels"], temperature_scale=1.0
        ).to(adapter.device)

        # amplitude/phase round-trip on real images
        spec = torch.fft.rfft2(xs, norm="ortho")
        rec = torch.fft.irfft2(spec, s=xs.shape[-2:], norm="ortho")
        recon_err = float((rec - xs).abs().max())
        report["reconstruction_check"][name] = recon_err

        rho = radial_frequency(xs.shape[-2], xs.shape[-1], device="cpu")
        collected = {}
        for loss_name in losses:
            loss_fn = LOSSES[loss_name]
            for scale in scales:
                grads, losses_all = [], []
                for start in range(0, n, args.batch_size):
                    xb = xs[start:start + args.batch_size]
                    yb = ys[start:start + args.batch_size]
                    grad, per_loss, _ = amplitude_gradient(
                        adapter, classifier, xb, yb, loss_fn,
                        logit_scale=scale, returns="all",
                    )
                    grads.append(grad.cpu())
                    losses_all.append(per_loss.cpu())
                grad = torch.cat(grads, dim=0)
                all_losses = torch.cat(losses_all, dim=0)
                metrics = gradient_metrics(grad, rho)
                per = metrics.pop("per_sample")
                metrics["mean_loss"] = float(all_losses.mean().item())
                metrics["loss_std"] = float(all_losses.std(unbiased=False).item())
                metrics["grad_shape"] = list(grad.shape)
                metrics["logit_scale"] = scale
                metrics["loss"] = loss_name
                collected[(loss_name, scale)] = (metrics, per, all_losses)

        model_max_l1 = max(
            float(per["l1"].max().item()) for _, per, _ in collected.values()
        )
        entry = {loss_name: {} for loss_name in losses}
        for (loss_name, scale), (metrics, per, all_losses) in collected.items():
            relative = per["l1"] < (0.01 * model_max_l1)
            metrics["model_max_l1"] = model_max_l1
            metrics["near_zero_relative_lt_1pct_of_model_max_fraction"] = float(
                relative.float().mean().item()
            )
            metrics["l1_relative_to_model_max_mean"] = float(
                (per["l1"].mean() / max(model_max_l1, 1e-30)).item()
            )
            entry[loss_name][str(scale)] = metrics

            aggregate_rows.append({
                "model": name,
                "loss": loss_name,
                "logit_scale": scale,
                "n_samples": metrics["n_samples"],
                "mean_loss": metrics["mean_loss"],
                "grad_l1_mean": metrics["grad_l1_mean"],
                "grad_l1_sum": metrics["grad_l1_sum"],
                "grad_l1_std": metrics["grad_l1_std"],
                "grad_l2_mean": metrics["grad_l2_mean"],
                "grad_max": metrics["grad_max"],
                "finite_fraction": metrics["finite_fraction"],
                "near_zero_1e-12": metrics["near_zero_l1_lt_1e-12_fraction"],
                "near_zero_1e-10": metrics["near_zero_l1_lt_1e-10_fraction"],
                "near_zero_relative_lt_1pct_of_model_max": metrics[
                    "near_zero_relative_lt_1pct_of_model_max_fraction"],
                "l1_relative_to_model_max_mean": metrics["l1_relative_to_model_max_mean"],
                "low_ratio_aggregate": metrics["band_ratio_aggregate"]["low"],
                "middle_ratio_aggregate": metrics["band_ratio_aggregate"]["middle"],
                "high_ratio_aggregate": metrics["band_ratio_aggregate"]["high"],
                "low_ratio_mean": metrics["band_ratio_mean"]["low"],
                "middle_ratio_mean": metrics["band_ratio_mean"]["middle"],
                "high_ratio_mean": metrics["band_ratio_mean"]["high"],
            })
            for i in range(n):
                per_sample_rows.append({
                    "model": name,
                    "loss": loss_name,
                    "logit_scale": scale,
                    "sample_index": i,
                    "label": int(ys[i]),
                    "loss_value": float(all_losses[i]),
                    "grad_l1": float(per["l1"][i]),
                    "grad_l2": float(per["l2"][i]),
                    "grad_max": float(per["max"][i]),
                    "low_ratio": float(per["band_ratio"]["low"][i]),
                    "middle_ratio": float(per["band_ratio"]["middle"][i]),
                    "high_ratio": float(per["band_ratio"]["high"][i]),
                })
            print(f"[{name}] {loss_name:6s} scale={scale:2d} "
                  f"L1={metrics['grad_l1_mean']:.6g} max={metrics['grad_max']:.6g} "
                  f"near0(abs)={metrics['near_zero_l1_lt_1e-10_fraction']:.3f} "
                  f"near0(rel<1%max)={metrics['near_zero_relative_lt_1pct_of_model_max_fraction']:.3f} "
                  f"bands(lo/mid/hi)={metrics['band_ratio_aggregate']['low']:.3f}/"
                  f"{metrics['band_ratio_aggregate']['middle']:.3f}/"
                  f"{metrics['band_ratio_aggregate']['high']:.3f}", flush=True)
        report["models"][name] = entry
        del adapter, gallery, classifier, collected
        torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sgm_loss_stability.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "sgm_loss_stability.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with open(OUT_DIR / "sgm_per_sample.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"\nwrote {OUT_DIR / 'sgm_loss_stability.json'}")
    print(f"wrote {OUT_DIR / 'sgm_loss_stability.csv'}")
    print(f"wrote {OUT_DIR / 'sgm_per_sample.csv'}")


if __name__ == "__main__":
    main()
