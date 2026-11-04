"""Figures for Experiment A (progressive frequency attenuation)."""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PHASE2 = ROOT / "results" / "phase2"
CONDITIONS = ("clean", "gaussian", "pgd", "apgd")
COLORS = {"clean": "tab:blue", "gaussian": "tab:green", "pgd": "tab:orange", "apgd": "tab:red"}


def load_summary(model):
    path = PHASE2 / "attenuation" / model / "summary.csv"
    if not path.is_file():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def plot_model(ax, rows, model):
    for condition in CONDITIONS:
        sel = [r for r in rows if r["condition"] == condition]
        if not sel:
            continue
        x = np.array([float(r["sigma"]) for r in sel])
        y = np.array([float(r["paired_mean"]) for r in sel])
        lo = np.array([float(r["paired_ci_low"]) for r in sel])
        hi = np.array([float(r["paired_ci_high"]) for r in sel])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=2,
                    color=COLORS[condition], label=condition)
    ax.set_xlabel("Gaussian sigma (smaller = stronger low-pass)")
    ax.set_ylabel("cosine(f(x), f(LPF_sigma(x)))")
    ax.set_title(model)
    ax.set_ylim(-0.05, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="compnet,ccnet")
    parser.add_argument("--out-dir", default=str(PHASE2 / "figures"))
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        plot_model(ax, load_summary(model), model)
    fig.tight_layout()
    fig.savefig(out / "attenuation_models.png", dpi=160)
    plt.close(fig)

    # Cross-model clean comparison
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for model, style in zip(models, ("-o", "-s")):
        rows = [r for r in load_summary(model) if r["condition"] == "clean"]
        if not rows:
            continue
        x = np.array([float(r["sigma"]) for r in rows])
        y = np.array([float(r["paired_mean"]) for r in rows])
        ax.plot(x, y, style, label=model)
    ax.set_xlabel("Gaussian sigma")
    ax.set_ylabel("clean consistency")
    ax.set_ylim(-0.05, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "attenuation_clean_model_comparison.png", dpi=160)
    plt.close(fig)

    # Filter responses
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.0), squeeze=False)
    for ax, model in zip(axes[0], models):
        path = PHASE2 / "attenuation" / model / "filter_response.csv"
        if not path.is_file():
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        sigmas = sorted({float(r["sigma"]) for r in rows})
        cmap = plt.get_cmap("viridis")
        for i, sigma in enumerate(sigmas):
            sel = [r for r in rows if float(r["sigma"]) == sigma]
            rho = [float(r["rho"]) for r in sel]
            gain = [float(r["response"]) for r in sel]
            ax.plot(rho, gain, color=cmap(i / max(len(sigmas) - 1, 1)),
                    label=f"sigma={sigma:g}")
        ax.set_xlabel("normalized radial frequency rho")
        ax.set_ylabel("G_sigma(rho)")
        ax.set_title(f"{model} filter response")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "filter_response.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
