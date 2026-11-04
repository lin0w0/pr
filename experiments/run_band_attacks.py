"""Experiment D: band-limited untargeted / targeted adversarial attacks.

Same epsilon, steps, alpha and seed for every band. Every perturbed image is
checked for band-outside energy, box violation and L-infinity excess; failed
projections are reported, never silently treated as strict band-limited attacks.
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
import torch.nn.functional as F  # noqa: E402

from attacks.band_limited import band_limited_margin  # noqa: E402
from evaluation.paired_stats import paired_summary  # noqa: E402
from experiments._common import (  # noqa: E402
    PHASE2_ROOT,
    build_context,
    extract_features,
    rank_hits,
    save_csv,
    save_json,
)
from spectral.band_projection import band_mask  # noqa: E402
from spectral.frequency import radial_frequency  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bands", default="low,middle,high")
    parser.add_argument("--objectives", default="untargeted,targeted")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]

    for model_name in models:
        ctx = build_context(model_name, device=args.device, max_ids=args.max_ids)
        scfg = ctx["spectral_cfg"]
        bcfg = scfg["attacks"]["band"]
        out_dir = PHASE2_ROOT / "band_attacks" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        rho = radial_frequency(ctx["xs"].shape[-2], ctx["xs"].shape[-1], device="cpu")
        xs, ys = ctx["xs"], ctx["ys"]
        num_ids = int(ctx["classifier"].prototypes.shape[0])
        targets = (ys + 1) % num_ids
        protos = ctx["classifier"].prototypes.detach().cpu()

        with torch.no_grad():
            clean_feats = extract_features(
                ctx["adapter"], xs,
                batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
            clean_pred = ctx["classifier"].predict(clean_feats.to(ctx["device"])).cpu()
        clean_rank = rank_hits(protos, clean_feats, ys)
        rows = []
        summary = []

        print(f"\n[{model_name}] D: n={len(xs)} bands={bands} eps={bcfg['eps']}", flush=True)
        for band_name in bands:
            mask = band_mask(rho, scfg["bands"][band_name]).to(ctx["device"])
            for objective in objectives:
                t0 = time.time()
                for start in range(0, len(xs), args.batch_size):
                    xb = xs[start:start + args.batch_size]
                    yb = ys[start:start + args.batch_size]
                    tb = targets[start:start + args.batch_size]
                    xa, delta, metrics = band_limited_margin(
                        ctx["adapter"], ctx["classifier"], xb, yb, mask,
                        eps=bcfg["eps"], alpha=bcfg["alpha"], steps=bcfg["steps"],
                        objective=objective, targets=tb, scale=bcfg["scale"],
                        random_start=bcfg["random_start"], seed=bcfg["seed"] + start,
                        dykstra_iters=bcfg["dykstra_iters"],
                        dykstra_tol=bcfg["dykstra_tol"],
                    )
                    with torch.no_grad():
                        feats = F.normalize(ctx["adapter"].extract_features(xa), dim=1).cpu()
                    base = clean_feats[start:start + args.batch_size]
                    shift = 1.0 - (base * feats).sum(dim=1)
                    rank = rank_hits(protos, feats, ys[start:start + args.batch_size])
                    for j in range(len(xb)):
                        idx = start + j
                        label = int(ys[idx])
                        target = int(tb[j])
                        adv_pred = int(rank["pred"][j])
                        clean_ok = bool(clean_pred[idx] == ys[idx])
                        success = bool(
                            clean_ok and (
                                (objective == "untargeted" and adv_pred != label)
                                or (objective == "targeted" and adv_pred == target)
                            )
                        )
                        outside = float(metrics["band_outside_energy_ratio"][j])
                        boxv = float(metrics["box_violation"][j])
                        linf_excess = float(metrics["linf_excess"][j])
                        failed = (
                            outside > bcfg["max_band_outside_ratio"]
                            or boxv > bcfg["max_box_violation"]
                            or linf_excess > bcfg["max_linf_excess"]
                        )
                        rows.append({
                            "model": model_name,
                            "band": band_name,
                            "objective": objective,
                            "sample_index": idx,
                            "label": label,
                            "target": target,
                            "clean_pred": int(clean_pred[idx]),
                            "adv_pred": adv_pred,
                            "clean_correct": clean_ok,
                            "success": success,
                            "feature_shift": float(shift[j]),
                            "rank1": bool(rank["rank1"][j]),
                            "rank5": bool(rank["rank5"][j]),
                            "l2": float(delta[j].pow(2).sum().sqrt()),
                            "linf": float(metrics["linf"][j]),
                            "linf_excess": linf_excess,
                            "band_outside_ratio": outside,
                            "box_violation": boxv,
                            "band_residual_linf": float(metrics["band_residual_linf"][j]),
                            "dykstra_iterations": int(metrics["iterations"][j]),
                            "dykstra_converged": bool(metrics["converged"][j]),
                            "projection_failed": bool(failed),
                        })
                sel = [r for r in rows if r["band"] == band_name and r["objective"] == objective]
                shift = np.array([r["feature_shift"] for r in sel])
                groups = np.array([r["label"] for r in sel])
                pair = paired_summary(shift, groups)
                n_cc = sum(r["clean_correct"] for r in sel)
                n_success = sum(r["success"] for r in sel)
                # Per-identity ASR over clean-correct samples, then bootstrap
                # over identities (paired units).
                per_id_asr = []
                for label in np.unique(groups):
                    rows_l = [r for r in sel if r["label"] == label and r["clean_correct"]]
                    if rows_l:
                        per_id_asr.append(
                            sum(r["success"] for r in rows_l) / len(rows_l)
                        )
                asr_paired = paired_summary(
                    np.array(per_id_asr), np.arange(len(per_id_asr))
                )
                r1_paired = paired_summary(
                    np.array([float(r["rank1"]) for r in sel]), groups
                )
                summary.append({
                    "model": model_name,
                    "band": band_name,
                    "objective": objective,
                    "n": len(sel),
                    "n_clean_correct": n_cc,
                    "asr_clean_correct_pct": 100.0 * n_success / max(n_cc, 1),
                    "asr_paired_mean_pct": 100.0 * asr_paired["paired_mean"],
                    "asr_paired_ci_low_pct": 100.0 * asr_paired["paired_ci_low"],
                    "asr_paired_ci_high_pct": 100.0 * asr_paired["paired_ci_high"],
                    "n_identities_with_clean_correct": int(asr_paired["n_identities"]),
                    "success_unconditional_pct": 100.0 * n_success / len(sel),
                    "rank1_pct": 100.0 * sum(r["rank1"] for r in sel) / len(sel),
                    "rank5_pct": 100.0 * sum(r["rank5"] for r in sel) / len(sel),
                    "rank1_paired_mean_pct": 100.0 * r1_paired["paired_mean"],
                    "rank1_paired_ci_low_pct": 100.0 * r1_paired["paired_ci_low"],
                    "rank1_paired_ci_high_pct": 100.0 * r1_paired["paired_ci_high"],
                    "unconditional_error_pct": 100.0 * sum(
                        not r["rank1"] for r in sel) / len(sel),
                    "feature_shift_mean": float(shift.mean()),
                    "feature_shift_std": float(shift.std(ddof=1)),
                    **pair,
                    "l2_mean": float(np.mean([r["l2"] for r in sel])),
                    "linf_mean": float(np.mean([r["linf"] for r in sel])),
                    "linf_max": float(np.max([r["linf"] for r in sel])),
                    "band_outside_mean": float(np.mean([r["band_outside_ratio"] for r in sel])),
                    "band_outside_max": float(np.max([r["band_outside_ratio"] for r in sel])),
                    "box_violation_max": float(np.max([r["box_violation"] for r in sel])),
                    "band_residual_linf_max": float(np.max([r["band_residual_linf"] for r in sel])),
                    "dykstra_converged_pct": 100.0 * sum(
                        r["dykstra_converged"] for r in sel) / len(sel),
                    "projection_failed_pct": 100.0 * sum(
                        r["projection_failed"] for r in sel) / len(sel),
                    "runtime_s": round(time.time() - t0, 2),
                })
                s = summary[-1]
                print(f"[{model_name}] {band_name:6s} {objective:10s}: "
                      f"ASR={s['asr_clean_correct_pct']:.1f}% "
                      f"rank1={s['rank1_pct']:.1f}% shift={s['feature_shift_mean']:.4f} "
                      f"outside={s['band_outside_max']:.2e} fail={s['projection_failed_pct']:.1f}%",
                      flush=True)

        save_csv(out_dir / "per_sample.csv", rows)
        save_csv(out_dir / "summary.csv", summary)
        save_json(out_dir / "run_meta.json", {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "n": len(xs),
            "bands": bands,
            "objectives": objectives,
            "target_rule": "(label + 1) mod num_identities",
            "band_cfg": bcfg,
            "bands_definition": scfg["bands"],
            "checkpoint": ctx["adapter"].checkpoint_info,
            "split": ctx["split_meta"]["name"],
            "paths_hash": ctx["paths_hash"],
        })
        print(f"[{model_name}] wrote {out_dir}", flush=True)
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
