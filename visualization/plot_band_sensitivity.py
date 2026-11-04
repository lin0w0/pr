"""Figures for Experiments C/D: band feature shift, Rank-1, ASR and residuals."""
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
BANDS = ("low", "middle", "high")
MODELS = ("compnet", "ccnet")


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.is_file() else []


def bars(ax, values_by_model, ylabel, title, ylim=None):
    x = np.arange(len(BANDS))
    width = 0.35
    for i, (model, vals) in enumerate(values_by_model.items()):
        ax.bar(x + (i - (len(values_by_model) - 1) / 2) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PHASE2 / "figures"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = {
        m: load(PHASE2 / "band_sensitivity" / m / "summary.csv") for m in MODELS
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    shift = {}
    rank1 = {}
    outside = {}
    for m in MODELS:
        by_band = {r["band"]: r for r in summaries[m]}
        shift[m] = [float(by_band[b]["feature_shift_mean"]) for b in BANDS]
        rank1[m] = [float(by_band[b]["rank1_pct"]) for b in BANDS]
        outside[m] = [float(by_band[b]["band_outside_max"]) for b in BANDS]
    bars(axes[0], shift, "mean 1 - cosine", "Band feature shift")
    bars(axes[1], rank1, "Rank-1 (%)", "Identification Rank-1 after band perturbation")
    for m in MODELS:
        axes[2].plot(BANDS, outside[m], marker="o", label=m)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("max band-outside energy ratio")
    axes[2].set_title("Dykstra band residual")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "band_sensitivity_summary.png", dpi=160)
    plt.close(fig)

    attacks = {m: load(PHASE2 / "band_attacks" / m / "summary.csv") for m in MODELS}
    for objective in ("untargeted", "targeted"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        asr, rank1a, fail = {}, {}, {}
        for m in MODELS:
            rows = {r["band"]: r for r in attacks[m] if r["objective"] == objective}
            asr[m] = [float(rows[b]["asr_clean_correct_pct"]) for b in BANDS]
            rank1a[m] = [float(rows[b]["rank1_pct"]) for b in BANDS]
            fail[m] = [float(rows[b]["projection_failed_pct"]) for b in BANDS]
        bars(axes[0], asr, "ASR (%)", f"{objective} band ASR", ylim=(0, 100))
        bars(axes[1], rank1a, "Rank-1 (%)", f"{objective} band Rank-1", ylim=(0, 100))
        bars(axes[2], fail, "projection failed (%)", f"{objective} projection failures",
             ylim=(0, 100))
        fig.tight_layout()
        fig.savefig(out / f"band_attacks_{objective}.png", dpi=160)
        plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
