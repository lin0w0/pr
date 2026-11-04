"""Experiment A: Progressive Frequency Attenuation.

For clean / Gaussian noise / PGD / APGD images, compute
C(x, sigma) = cosine(f(x), f(G_sigma(x))) for a sweep of Gaussian low-pass
sigmas. Smaller sigma is a stronger low-pass.

All filtering happens on the raw 128x128 ROI in [0,1]; the official
preprocessing and frozen model are applied afterwards.
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
from spectral.gaussian_filter import (  # noqa: E402
    filter_response_table,
    gaussian_lowpass,
)
from spectral.frequency import radial_frequency  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def attenuation_for_condition(ctx, x_cond, sigmas, batch_size=32):
    adapter = ctx["adapter"]
    device = ctx["device"]
    rho = radial_frequency(x_cond.shape[-2], x_cond.shape[-1],
                           device=device, dtype=torch.float32)
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(x_cond), batch_size):
            xb = x_cond[start:start + batch_size].to(device)
            base = F.normalize(adapter.extract_features(xb), dim=1)
            for sigma in sigmas:
                low = gaussian_lowpass(xb, sigma, rho=rho)
                feat = F.normalize(adapter.extract_features(low), dim=1)
                cons = (base * feat).sum(dim=1).cpu()
                low_cpu = low.cpu()
                for j in range(len(xb)):
                    rows.append({
                        "condition": None,
                        "sample_index": start + j,
                        "label": int(ctx["ys"][start + j]),
                        "sigma": float(sigma),
                        "consistency": float(cons[j]),
                        "filtered_min": float(low_cpu[j].min()),
                        "filtered_max": float(low_cpu[j].max()),
                    })
    return rows, time.time() - t0


def summarize(rows, condition):
    keys = sorted({float(r["sigma"]) for r in rows})
    out = []
    for sigma in keys:
        sel = [r for r in rows if float(r["sigma"]) == sigma]
        values = np.array([r["consistency"] for r in sel], dtype=np.float64)
        groups = np.array([r["label"] for r in sel])
        ind = independent_summary(values)
        pair = paired_summary(values, groups)
        out.append({
            "condition": condition,
            "sigma": sigma,
            **ind,
            **pair,
        })
    return out


def save_examples(ctx, condition, sigmas, out_dir, n_examples=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adapter = ctx["adapter"]
    device = ctx["device"]
    x_cond, _ = condition_array(ctx, condition)
    rho = radial_frequency(128, 128, device=device, dtype=torch.float32)
    idxs = np.linspace(0, len(x_cond) - 1, n_examples).astype(int)
    show_sigmas = [s for s in sigmas if s in (0.05, 0.2, 0.5, 1.0)] or list(sigmas[:4])
    out_dir.mkdir(parents=True, exist_ok=True)
    for row_i, idx in enumerate(idxs):
        xb = x_cond[idx:idx + 1].to(device)
        images = [xb]
        labels = ["raw"]
        for sigma in show_sigmas:
            images.append(gaussian_lowpass(xb, sigma, rho=rho))
            labels.append(f"sigma={sigma:g}")
        fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
        for ax, im, title in zip(axes, images, labels):
            arr = im[0, 0].detach().cpu().numpy()
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{ctx['model']} | {condition} | sample {idx}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{condition}_sample{row_i:02d}.png", dpi=140)
        plt.close(fig)


def plot_curves(all_summary, model, out_dir, sigmas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in ("clean", "gaussian", "pgd", "apgd"):
        sel = [r for r in all_summary if r["condition"] == condition]
        if not sel:
            continue
        xs = [r["sigma"] for r in sel]
        ys = [r["paired_mean"] for r in sel]
        lo = [r["paired_mean"] - r["paired_ci_low"] for r in sel]
        hi = [r["paired_ci_high"] - r["paired_mean"] for r in sel]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=condition)
    ax.set_xlabel("Gaussian sigma (smaller = stronger low-pass)")
    ax.set_ylabel("cosine(f(x), f(LPF_sigma(x)))")
    ax.set_title(f"{model}: progressive frequency attenuation")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "attenuation_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conditions", default="clean,gaussian,pgd,apgd")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    for model_name in models:
        ctx = build_context(model_name, device=args.device, max_ids=args.max_ids)
        scfg = ctx["spectral_cfg"]
        sigmas = [float(s) for s in scfg["sigma"]["values"]]
        out_dir = Path(args.out_dir) if args.out_dir else PHASE2_ROOT / "attenuation" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{model_name}] n={len(ctx['xs'])} sigmas={sigmas}", flush=True)

        response_rows = filter_response_table(
            sigmas, scfg["sigma"]["response_rho_points"]
        )
        response_csv = []
        for row in response_rows:
            for rho, val in zip(row["rho_values"], row["response"]):
                response_csv.append({
                    "sigma": row["sigma"], "rho": rho, "response": val,
                    "cutoff_rho_half_gain": row["cutoff_rho_half_gain"],
                })
        save_csv(out_dir / "filter_response.csv", response_csv)

        all_rows = []
        all_summary = []
        condition_meta = {}
        for condition in conditions:
            x_cond, meta = condition_array(ctx, condition)
            condition_meta[condition] = meta
            print(f"[{model_name}] condition {condition} ...", flush=True)
            t0 = time.time()
            rows, elapsed = attenuation_for_condition(
                ctx, x_cond, sigmas, batch_size=args.batch_size
            )
            for r in rows:
                r["condition"] = condition
                r["model"] = model_name
            all_rows.extend(rows)
            summ = summarize(rows, condition)
            all_summary.extend(summ)
            print(f"[{model_name}] {condition}: {elapsed:.1f}s", flush=True)

        fields = ["model", "condition", "sample_index", "label", "sigma",
                  "consistency", "filtered_min", "filtered_max"]
        save_csv(out_dir / "per_sample.csv", all_rows, fieldnames=fields)
        save_csv(out_dir / "summary.csv", all_summary)
        plot_curves(all_summary, model_name, out_dir, sigmas)
        for condition in conditions:
            save_examples(ctx, condition, sigmas, out_dir / "examples")

        save_json(out_dir / "run_meta.json", {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "n": len(ctx["xs"]),
            "max_ids": args.max_ids,
            "conditions": conditions,
            "sigmas": sigmas,
            "condition_meta": condition_meta,
            "checkpoint": ctx["adapter"].checkpoint_info,
            "split": ctx["split_meta"]["name"],
            "paths_hash": ctx["paths_hash"],
            "config": {
                "spectral": scfg,
                "dataset": ctx["dataset_cfg"],
                "model": ctx["model_cfg"],
            },
        })
        print(f"[{model_name}] wrote {out_dir}", flush=True)
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
