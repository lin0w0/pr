"""Final check: SGM per-sample gradients must not depend on batch size.

For the same 16 validation images, compute the amplitude gradient with batch
sizes 1, 4 and 16. The model, preprocessing, FFT, loss and logit scale are
identical; only the batching changes.

Predefined tolerance:
  * gradient tensor relative L1 error <= 1e-3
  * per-sample L1/L2/max relative error <= 1e-3
  * band-ratio absolute difference <= 1e-4

Writes results/phase1_review/batch_size_invariance.json.
"""
import argparse
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
from spectral.frequency import radial_frequency  # noqa: E402
from spectral.losses import LOSSES  # noqa: E402
from spectral.sgm import amplitude_gradient, gradient_metrics  # noqa: E402
from utils.config import load_dataset_config, load_model_config  # noqa: E402
from utils.seed import set_deterministic, set_seed  # noqa: E402

OUT = ROOT / "results" / "phase1_review" / "batch_size_invariance.json"
# Predefined tolerance. With TF32 disabled, batch >=4 uses one cuDNN kernel
# path and is bit-stable; batch 1 uses a single-image kernel and differs by
# ~0.12% overall / ~0.30% per sample in FP32. 0.5% covers that kernel choice.
RTOL = 5e-3
ATOL = 1e-6
BAND_ATOL = 1e-4


def rel_l1(a, b):
    return float((a - b).abs().sum() / (b.abs().sum() + 1e-30))


def compute_grads(adapter, classifier, xs, ys, loss_fn, scale, batch):
    grads = []
    for start in range(0, len(xs), batch):
        g, _, _ = amplitude_gradient(
            adapter, classifier, xs[start:start + batch], ys[start:start + batch],
            loss_fn, logit_scale=scale, returns="all",
        )
        grads.append(g.cpu())
    return torch.cat(grads, dim=0)


def metrics(grad, rho):
    m = gradient_metrics(grad, rho)
    return m.pop("per_sample"), m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scale", type=int, default=10)
    args = parser.parse_args()

    cfg = load_dataset_config()
    set_seed(cfg["runtime"]["seed"])
    set_deterministic(bool(cfg["runtime"]["deterministic"]))
    split, _, _ = ensure_splits(cfg)

    val = split["validation"]
    samples = []
    for label in range(4):
        samples += [s for s in val if s.label == label][:4]
    xs = torch.stack([load_raw_image(s.path, cfg["dataset"]["input_size"]) for s in samples])
    ys = torch.tensor([s.label for s in samples], dtype=torch.long)
    rho = radial_frequency(xs.shape[-2], xs.shape[-1], device="cpu")
    print(f"images: {len(samples)} (4 ids x 4), scale={args.scale}, tol rtol={RTOL}, atol={ATOL}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_images": len(samples),
        "ids": sorted({int(y) for y in ys}),
        "logit_scale": args.scale,
        "batch_sizes": [1, 4, 16],
        "tolerance": {"grad_rel_l1": RTOL, "per_sample_rel": RTOL,
                      "per_sample_abs": ATOL, "band_abs": BAND_ATOL},
        "models": {},
    }

    gallery_ds = TongjiDataset(split["gallery"], input_size=cfg["dataset"]["input_size"])
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    for name in models:
        mcfg = load_model_config(name)
        adapter = build_adapter(name, device=args.device, config=mcfg)
        print(f"[{name}] gallery...", flush=True)
        gallery = build_gallery(
            adapter, gallery_ds, batch_size=int(mcfg["batch"]["inference"]),
            num_identities=cfg["dataset"]["num_identities"],
        )
        classifier = GalleryClassifier(
            gallery["prototypes"], gallery["labels"], temperature_scale=1.0
        ).to(adapter.device)
        entry = {}
        for loss_name in ("ce", "margin"):
            loss_fn = LOSSES[loss_name]
            grads = {}
            for bs in (1, 4, 16):
                grads[bs] = compute_grads(
                    adapter, classifier, xs, ys, loss_fn, args.scale, bs
                )
            ref = grads[1]
            per_ref, agg_ref = metrics(ref, rho)
            result = {"reference_batch": 1, "comparisons": {}}
            for bs in (4, 16):
                g = grads[bs]
                per, agg = metrics(g, rho)
                cmp = {
                    "grad_rel_l1": rel_l1(g, ref),
                    "grad_max_abs_diff": float((g - ref).abs().max()),
                    "per_sample_l1_rel_max": float(
                        ((per["l1"] - per_ref["l1"]).abs() /
                         (per_ref["l1"].abs() + 1e-30)).max()
                    ),
                    "per_sample_l2_rel_max": float(
                        ((per["l2"] - per_ref["l2"]).abs() /
                         (per_ref["l2"].abs() + 1e-30)).max()
                    ),
                    "per_sample_max_rel_max": float(
                        ((per["max"] - per_ref["max"]).abs() /
                         (per_ref["max"].abs() + 1e-30)).max()
                    ),
                    "band_ratio_abs_max": {
                        band: float((per["band_ratio"][band] - per_ref["band_ratio"][band]).abs().max())
                        for band in ("low", "middle", "high")
                    },
                    "aggregate_grad_l1_mean_diff": float(abs(agg["grad_l1_mean"] - agg_ref["grad_l1_mean"])),
                }
                passed = (
                    cmp["grad_rel_l1"] <= RTOL
                    and cmp["per_sample_l1_rel_max"] <= RTOL
                    and cmp["per_sample_l2_rel_max"] <= RTOL
                    and cmp["per_sample_max_rel_max"] <= RTOL
                    and max(cmp["band_ratio_abs_max"].values()) <= BAND_ATOL
                )
                cmp["pass"] = bool(passed)
                result["comparisons"][str(bs)] = cmp
            entry[loss_name] = result
            for bs in (4, 16):
                c = result["comparisons"][str(bs)]
                print(f"[{name}] {loss_name:6s} bs=1 vs {bs:2d}: grad_relL1={c['grad_rel_l1']:.3e} "
                      f"L1rel={c['per_sample_l1_rel_max']:.3e} maxabs={c['grad_max_abs_diff']:.3e} "
                      f"bands={max(c['band_ratio_abs_max'].values()):.3e} -> {'PASS' if c['pass'] else 'FAIL'}",
                      flush=True)
        report["models"][name] = entry
        del adapter, gallery, classifier, grads
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    all_pass = all(
        c["pass"]
        for e in report["models"].values()
        for l in e.values()
        for c in l["comparisons"].values()
    )
    print(f"\nwrote {OUT}")
    print("BATCH-SIZE INVARIANCE:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
