"""Equal-L2 budget band-limited attacks (Task 1.5).

For each model, all three bands use the same per-sample L2 radius, chosen as
the mean L2 achieved by the original low-band L-infinity attack. No L-infinity
cap; projection is Dykstra onto band ∩ L2 ball ∩ pixel box.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from attacks.band_limited import band_limited_margin_l2  # noqa: E402
from evaluation.paired_stats import paired_summary  # noqa: E402
from experiments._common import (  # noqa: E402
    build_context,
    extract_features,
    rank_hits,
    save_csv,
    save_json,
)
from spectral.band_projection import band_mask  # noqa: E402
from spectral.frequency import radial_frequency  # noqa: E402

OUT = ROOT / "results" / "phase2_supplement" / "equal_l2"
PHASE2 = ROOT / "results" / "phase2"


def original_low_l2(model):
    path = PHASE2 / "band_attacks" / model / "summary.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        if r["band"] == "low" and r["objective"] == "untargeted":
            return float(r["l2_mean"])
    raise FileNotFoundError(f"low-band L2 not found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--max-ids", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    for model in models:
        ctx = build_context(model, device=args.device, max_ids=args.max_ids)
        scfg = ctx["spectral_cfg"]
        bcfg = scfg["attacks"]["band"]
        batch = args.batch_size or (64 if model == "compnet" else 32)
        radius = original_low_l2(model)
        out = OUT / model
        out.mkdir(parents=True, exist_ok=True)
        rho = radial_frequency(128, 128, device="cpu")
        xs, ys = ctx["xs"], ctx["ys"]
        protos = ctx["classifier"].prototypes.detach().cpu()
        with torch.no_grad():
            clean_feats = extract_features(
                ctx["adapter"], xs,
                batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
            clean_pred = ctx["classifier"].predict(clean_feats.to(ctx["device"])).cpu()
        clean_rank = rank_hits(protos, clean_feats, ys)
        rows, summary = [], []
        print(f"\n[{model}] equal-L2 n={len(xs)} radius={radius:.4f}", flush=True)
        for band in ("low", "middle", "high"):
            mask = band_mask(rho, scfg["bands"][band]).to(ctx["device"])
            t0 = time.time()
            for start in range(0, len(xs), batch):
                xb = xs[start:start + batch]
                yb = ys[start:start + batch]
                xa, delta, metrics = band_limited_margin_l2(
                    ctx["adapter"], ctx["classifier"], xb, yb, mask,
                    radius=radius, alpha=bcfg["alpha"], steps=bcfg["steps"],
                    objective="untargeted", scale=bcfg["scale"],
                    random_start=False, seed=bcfg["seed"] + start,
                    dykstra_iters=30, dykstra_tol=bcfg["dykstra_tol"],
                )
                with torch.no_grad():
                    feats = F.normalize(ctx["adapter"].extract_features(xa), dim=1).cpu()
                base = clean_feats[start:start + batch]
                shift = 1.0 - (base * feats).sum(dim=1)
                rank = rank_hits(protos, feats, ys[start:start + batch])
                for j in range(len(xb)):
                    idx = start + j
                    clean_ok = bool(clean_pred[idx] == ys[idx])
                    adv_pred = int(rank["pred"][j])
                    outside = float(metrics["band_outside_energy_ratio"][j])
                    boxv = float(metrics["box_violation"][j])
                    l2exc = float(metrics["l2_excess"][j])
                    failed = (outside > bcfg["max_band_outside_ratio"]
                              or boxv > bcfg["max_box_violation"]
                              or l2exc > 1e-6)
                    rows.append({
                        "model": model, "band": band, "sample_index": idx,
                        "label": int(ys[idx]),
                        "clean_pred": int(clean_pred[idx]), "adv_pred": adv_pred,
                        "clean_correct": clean_ok,
                        "success": bool(clean_ok and adv_pred != int(ys[idx])),
                        "feature_shift": float(shift[j]),
                        "rank1": bool(rank["rank1"][j]),
                        "rank5": bool(rank["rank5"][j]),
                        "l2": float(metrics["l2"][j]),
                        "l2_excess": l2exc,
                        "linf": float(metrics["linf"][j]),
                        "band_outside_ratio": outside,
                        "box_violation": boxv,
                        "band_residual_linf": float(metrics["band_residual_linf"][j]),
                        "dykstra_iterations": int(metrics["iterations"][j]),
                        "dykstra_converged": bool(metrics["converged"][j]),
                        "projection_failed": bool(failed),
                    })
            sel = [r for r in rows if r["band"] == band]
            shift = np.array([r["feature_shift"] for r in sel])
            groups = np.array([r["label"] for r in sel])
            pair = paired_summary(shift, groups)
            n_cc = sum(r["clean_correct"] for r in sel)
            n_s = sum(r["success"] for r in sel)
            summary.append({
                "model": model, "band": band, "n": len(sel),
                "l2_budget": radius,
                "n_clean_correct": n_cc,
                "asr_clean_correct_pct": 100.0 * n_s / max(n_cc, 1),
                "rank1_pct": 100.0 * sum(r["rank1"] for r in sel) / len(sel),
                "feature_shift_mean": float(shift.mean()),
                **pair,
                "l2_mean": float(np.mean([r["l2"] for r in sel])),
                "l2_max": float(np.max([r["l2"] for r in sel])),
                "linf_mean": float(np.mean([r["linf"] for r in sel])),
                "linf_max": float(np.max([r["linf"] for r in sel])),
                "band_outside_max": float(np.max([r["band_outside_ratio"] for r in sel])),
                "box_violation_max": float(np.max([r["box_violation"] for r in sel])),
                "l2_excess_max": float(np.max([r["l2_excess"] for r in sel])),
                "band_residual_linf_max": float(np.max([r["band_residual_linf"] for r in sel])),
                "dykstra_converged_pct": 100.0 * sum(r["dykstra_converged"] for r in sel) / len(sel),
                "projection_failed_pct": 100.0 * sum(r["projection_failed"] for r in sel) / len(sel),
                "runtime_s": round(time.time() - t0, 2),
            })
            s = summary[-1]
            print(f"[{model}] {band:6s}: L2={s['l2_mean']:.4f} "
                  f"ASR={s['asr_clean_correct_pct']:.1f}% rank1={s['rank1_pct']:.1f}% "
                  f"shift={s['feature_shift_mean']:.4f} outside={s['band_outside_max']:.2e} "
                  f"fail={s['projection_failed_pct']:.1f}%", flush=True)
        save_csv(out / "per_sample.csv", rows)
        save_csv(out / "summary.csv", summary)
        save_json(out / "run_meta.json", {
            "model": model, "n": len(xs), "l2_budget": radius,
            "max_ids": args.max_ids, "batch_size": batch,
            "cfg": bcfg, "bands": scfg["bands"],
            "split": ctx["split_meta"]["name"], "paths_hash": ctx["paths_hash"],
        })
        print(f"[{model}] wrote {out}", flush=True)
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
