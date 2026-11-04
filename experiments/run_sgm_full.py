"""Experiment B: SGM frequency-gradient analysis on clean/PGD/APGD images.

Primary loss: Margin, scale=10. Control: CE, scale=10.
Gradients are per-sample (spectral.sgm.amplitude_gradient, min_batch=4), then
absolute values are averaged across samples.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluation.paired_stats import independent_summary, paired_summary  # noqa: E402
from experiments._common import (  # noqa: E402
    PHASE2_ROOT,
    build_context,
    condition_array,
    save_csv,
    save_json,
)
from spectral.frequency import band_masks, radial_frequency  # noqa: E402
from spectral.losses import LOSSES  # noqa: E402
from spectral.sgm import amplitude_gradient  # noqa: E402

BANDS = ("low", "middle", "high")


def full_spectrum_map(half):
    """Mirror an rfft2 half-spectrum to a symmetric full [H, W] map."""
    if half.shape[-1] * 2 - 2 != half.shape[-2]:
        raise ValueError("expected square image and rfft2 half-spectrum")
    right = half[:, 1:-1].flip(-1)
    return torch.cat([half, right], dim=-1)


def save_heatmap(mean_abs, out_dir, tag, condition, loss_name, scale, max_l1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = mean_abs[0, 0].detach().cpu()
    full = full_spectrum_map(half)
    raw_max = float(full.max())
    norm = full / (raw_max + 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im0 = axes[0].imshow(full.numpy(), cmap="viridis")
    axes[0].set_title(f"{tag} | raw mean |grad| (max={raw_max:.3g})", fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(norm.numpy(), cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("normalized by its own max", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("fx bin")
        ax.set_ylabel("fy bin")
    fig.tight_layout()
    fig.savefig(out_dir / f"sgm_{condition}_{loss_name}_s{scale:g}.png", dpi=150)
    plt.close(fig)

    np.save(out_dir / f"sgm_{condition}_{loss_name}_s{scale:g}_half.npy", half.numpy())
    np.save(out_dir / f"sgm_{condition}_{loss_name}_s{scale:g}_full.npy", full.numpy())
    return {"raw_max": raw_max, "global_max_sample_l1": max_l1}


def radial_profile(mean_abs, rho, bins=50):
    half = mean_abs[0, 0].detach().cpu()
    r = rho.detach().cpu()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    flat_r = r.reshape(-1).numpy()
    flat_g = half.reshape(-1).numpy()
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (flat_r >= lo) & (flat_r < hi) if i < bins - 1 else (flat_r >= lo) & (flat_r <= hi)
        if m.any():
            rows.append({
                "rho_low": lo, "rho_high": hi,
                "rho_center": (lo + hi) / 2.0,
                "mean_abs_grad": float(flat_g[m].mean()),
                "n_bins": int(m.sum()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conditions", default="clean,pgd,apgd")
    parser.add_argument("--losses", default="margin,ce")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    loss_names = [l.strip() for l in args.losses.split(",") if l.strip()]
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]

    for model_name in models:
        ctx = build_context(model_name, device=args.device, max_ids=args.max_ids)
        scfg = ctx["spectral_cfg"]
        batch = args.batch_size or int(scfg["sgm"]["batch_size"])
        min_batch = int(scfg["sgm"]["min_batch"])
        scale_map = {
            "margin": float(scfg["sgm"]["primary"]["scale"]),
            "ce": float(scfg["sgm"]["control"]["scale"]),
        }
        out_dir = PHASE2_ROOT / "sgm" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        H, W = ctx["xs"].shape[-2:]
        rho = radial_frequency(H, W, device="cpu")
        masks = band_masks(rho, bands=scfg["bands"])

        print(f"\n[{model_name}] n={len(ctx['xs'])} batch={batch}", flush=True)
        per_sample_rows = []
        summary_rows = []
        radial_rows = []
        heat_meta = {}
        model_max_l1 = 0.0
        collected = {}

        for condition in conditions:
            x_cond, cond_meta = condition_array(ctx, condition)
            for loss_name in loss_names:
                loss_fn = LOSSES[loss_name]
                scale = scale_map[loss_name]
                t0 = time.time()
                sum_abs = None
                l1_all, l2_all, max_all = [], [], []
                band_energy_all = {b: [] for b in BANDS}
                loss_all = []
                for start in range(0, len(x_cond), batch):
                    xb = x_cond[start:start + batch]
                    yb = ctx["ys"][start:start + batch]
                    grad, losses, _ = amplitude_gradient(
                        ctx["adapter"], ctx["classifier"], xb, yb, loss_fn,
                        logit_scale=scale, returns="all", min_batch=min_batch,
                    )
                    grad = grad.detach().cpu()
                    abs_grad = grad.abs()
                    batch_sum = abs_grad.sum(0, keepdim=True)
                    sum_abs = batch_sum if sum_abs is None else sum_abs + batch_sum
                    l1 = abs_grad.sum(dim=(1, 2, 3))
                    l2 = torch.sqrt((abs_grad ** 2).sum(dim=(1, 2, 3)))
                    mx = abs_grad.amax(dim=(1, 2, 3))
                    l1_all.append(l1); l2_all.append(l2); max_all.append(mx)
                    loss_all.append(losses.detach().cpu())
                    for b in BANDS:
                        band_energy_all[b].append((abs_grad * masks[b]).sum(dim=(1, 2, 3)))
                l1_all = torch.cat(l1_all); l2_all = torch.cat(l2_all)
                max_all = torch.cat(max_all); loss_all = torch.cat(loss_all)
                band_energy = {b: torch.cat(v) for b, v in band_energy_all.items()}
                total_energy = sum(band_energy.values())
                band_ratio = {b: band_energy[b] / (total_energy + 1e-30) for b in BANDS}
                mean_abs = (sum_abs / len(x_cond))
                model_max_l1 = max(model_max_l1, float(l1_all.max().item()))

                # per-sample rows
                for i in range(len(x_cond)):
                    row = {
                        "model": model_name,
                        "condition": condition,
                        "loss": loss_name,
                        "scale": scale,
                        "sample_index": i,
                        "label": int(ctx["ys"][i]),
                        "loss_value": float(loss_all[i]),
                        "grad_l1": float(l1_all[i]),
                        "grad_l2": float(l2_all[i]),
                        "grad_max": float(max_all[i]),
                    }
                    for b in BANDS:
                        row[f"{b}_energy"] = float(band_energy[b][i])
                        row[f"{b}_ratio"] = float(band_ratio[b][i])
                    per_sample_rows.append(row)

                agg = {
                    "model": model_name,
                    "condition": condition,
                    "loss": loss_name,
                    "scale": scale,
                    "n": int(len(x_cond)),
                    "loss_mean": float(loss_all.mean()),
                    "loss_std": float(loss_all.std(unbiased=False)),
                    "grad_l1_mean": float(l1_all.mean()),
                    "grad_l1_sum": float(l1_all.sum()),
                    "grad_l1_std": float(l1_all.std(unbiased=False)),
                    "grad_l2_mean": float(l2_all.mean()),
                    "grad_max": float(max_all.max()),
                    "grad_max_mean": float(max_all.mean()),
                    "near_zero_abs_1e-12": float((l1_all < 1e-12).float().mean()),
                    "near_zero_abs_1e-10": float((l1_all < 1e-10).float().mean()),
                    "band_ratio_low_mean": float(band_ratio["low"].mean()),
                    "band_ratio_middle_mean": float(band_ratio["middle"].mean()),
                    "band_ratio_high_mean": float(band_ratio["high"].mean()),
                    "band_ratio_low_aggregate": float(band_energy["low"].sum() / (total_energy.sum() + 1e-30)),
                    "band_ratio_middle_aggregate": float(band_energy["middle"].sum() / (total_energy.sum() + 1e-30)),
                    "band_ratio_high_aggregate": float(band_energy["high"].sum() / (total_energy.sum() + 1e-30)),
                    "runtime_s": round(time.time() - t0, 2),
                }
                # identity-paired CI on L1 and band ratios
                pairs = paired_summary(l1_all.numpy(), ctx["ys"].numpy())
                agg.update({f"l1_{k}": v for k, v in pairs.items()})
                for b in BANDS:
                    bp = paired_summary(
                        band_ratio[b].numpy(), ctx["ys"].numpy()
                    )
                    agg[f"band_ratio_{b}_paired_mean"] = bp["paired_mean"]
                    agg[f"band_ratio_{b}_paired_ci_low"] = bp["paired_ci_low"]
                    agg[f"band_ratio_{b}_paired_ci_high"] = bp["paired_ci_high"]
                summary_rows.append(agg)
                collected[(condition, loss_name)] = {
                    "agg": agg, "l1": l1_all, "band_ratio": band_ratio,
                    "mean_abs": mean_abs,
                }
                radial_rows.extend([
                    {"model": model_name, "condition": condition, "loss": loss_name,
                     "scale": scale, **r}
                    for r in radial_profile(mean_abs, rho, bins=int(scfg["sgm"]["radial_bins"]))
                ])
                print(f"[{model_name}] {condition:7s} {loss_name:6s} s={scale:g}: "
                      f"L1={agg['grad_l1_mean']:.4g} max={agg['grad_max']:.4g} "
                      f"bands={agg['band_ratio_low_aggregate']:.3f}/"
                      f"{agg['band_ratio_middle_aggregate']:.3f}/"
                      f"{agg['band_ratio_high_aggregate']:.3f} "
                      f"({agg['runtime_s']}s)", flush=True)

        # relative near-zero and heatmaps after model-wide max is known
        for (condition, loss_name), c in collected.items():
            agg = c["agg"]
            scale = agg["scale"]
            agg["model_max_l1"] = model_max_l1
            agg["near_zero_rel_lt_1pct_model_max"] = float(
                (c["l1"] < 0.01 * max(model_max_l1, 1e-30)).float().mean()
            )
            agg["l1_relative_to_model_max_mean"] = float(
                c["l1"].mean() / max(model_max_l1, 1e-30)
            )
            heat_meta[f"{condition}_{loss_name}"] = save_heatmap(
                c["mean_abs"], out_dir, model_name, condition, loss_name, scale,
                model_max_l1,
            )

        save_csv(out_dir / "per_sample.csv", per_sample_rows)
        save_csv(out_dir / "summary.csv", summary_rows)
        save_csv(out_dir / "radial_profile.csv", radial_rows)
        save_json(out_dir / "sgm_meta.json", {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "n": len(ctx["xs"]),
            "conditions": conditions,
            "losses": loss_names,
            "scales": scale_map,
            "batch_size": batch,
            "min_batch": min_batch,
            "checkpoint": ctx["adapter"].checkpoint_info,
            "split": ctx["split_meta"]["name"],
            "paths_hash": ctx["paths_hash"],
            "heatmap_meta": heat_meta,
        })
        print(f"[{model_name}] wrote {out_dir}", flush=True)
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
