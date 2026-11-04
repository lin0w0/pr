"""Figures for Experiment B (SGM): raw/normalized heatmaps, radial profiles,
band-energy bars and clean-vs-adversarial comparisons."""
import argparse
import csv
import json
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
CONDITIONS = ("clean", "pgd", "apgd")
MODELS = ("compnet", "ccnet")


def load_summary(model):
    path = PHASE2 / "sgm" / model / "summary.csv"
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.is_file() else []


def load_map(model, condition, loss, scale):
    path = PHASE2 / "sgm" / model / f"sgm_{condition}_{loss}_s{scale:g}_full.npy"
    if not path.is_file():
        return None
    return np.fft.fftshift(np.load(path))


def heatmap_grid(loss, scale, normalized, out):
    fig, axes = plt.subplots(len(MODELS), len(CONDITIONS),
                             figsize=(4.2 * len(CONDITIONS), 4.0 * len(MODELS)),
                             squeeze=False)
    for i, model in enumerate(MODELS):
        for j, condition in enumerate(CONDITIONS):
            ax = axes[i][j]
            arr = load_map(model, condition, loss, scale)
            if arr is None:
                ax.axis("off")
                continue
            vmax = float(arr.max())
            shown = arr / (vmax + 1e-30) if normalized else arr
            im = ax.imshow(shown, cmap="viridis",
                           vmin=0, vmax=1 if normalized else None)
            ax.set_title(f"{model} | {condition}\nraw max={vmax:.3g}" if not normalized
                         else f"{model} | {condition}\nraw max={vmax:.3g}",
                         fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xlabel("fx (fftshift)")
            ax.set_ylabel("fy")
    fig.suptitle(f"SGM {loss} scale={scale} "
                 f"({'normalized by own max' if normalized else 'raw mean |grad|'})")
    fig.tight_layout()
    fig.savefig(out / f"sgm_{loss}_s{scale}_{'normalized' if normalized else 'raw'}.png",
                dpi=150)
    plt.close(fig)


def radial_plot(loss, out):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4.0), squeeze=False)
    for ax, model in zip(axes[0], MODELS):
        path = PHASE2 / "sgm" / model / "radial_profile.csv"
        if not path.is_file():
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        for condition in CONDITIONS:
            sel = [r for r in rows if r["condition"] == condition and r["loss"] == loss]
            if not sel:
                continue
            rho = [float(r["rho_center"]) for r in sel]
            val = [float(r["mean_abs_grad"]) for r in sel]
            ax.plot(rho, val, label=condition)
        ax.set_xlabel("normalized radial frequency rho")
        ax.set_ylabel("mean |dL/dA|")
        ax.set_yscale("log")
        ax.set_title(f"{model} | {loss}")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"sgm_{loss}_radial_profiles.png", dpi=160)
    plt.close(fig)


def band_bars(loss, out):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4.0), squeeze=False)
    bands = ("low", "middle", "high")
    for ax, model in zip(axes[0], MODELS):
        rows = {r["condition"]: r for r in load_summary(model) if r["loss"] == loss}
        x = np.arange(len(bands))
        width = 0.25
        for k, condition in enumerate(CONDITIONS):
            r = rows.get(condition)
            if not r:
                continue
            vals = [float(r[f"band_ratio_{b}_aggregate"]) for b in bands]
            ax.bar(x + (k - 1) * width, vals, width, label=condition)
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.set_ylabel("fraction of gradient energy")
        ax.set_title(f"{model} | {loss}")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"sgm_{loss}_band_energy.png", dpi=160)
    plt.close(fig)


def model_comparison(loss, out):
    fig, ax = plt.subplots(figsize=(6, 4.0))
    for model, style in zip(MODELS, ("-o", "-s")):
        path = PHASE2 / "sgm" / model / "radial_profile.csv"
        if not path.is_file():
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        sel = [r for r in rows if r["condition"] == "clean" and r["loss"] == loss]
        rho = [float(r["rho_center"]) for r in sel]
        val = [float(r["mean_abs_grad"]) for r in sel]
        ax.plot(rho, val, style, label=f"{model} (raw magnitude)")
    ax.set_xlabel("rho")
    ax.set_ylabel("mean |dL/dA|")
    ax.set_yscale("log")
    ax.set_title(f"Clean SGM comparison | {loss}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"sgm_{loss}_clean_model_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PHASE2 / "figures"))
    parser.add_argument("--scale", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for loss in ("margin", "ce"):
        heatmap_grid(loss, args.scale, False, out)
        heatmap_grid(loss, args.scale, True, out)
        radial_plot(loss, out)
        band_bars(loss, out)
        model_comparison(loss, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
