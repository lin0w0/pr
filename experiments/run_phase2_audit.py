"""Phase 2 supplementary audit (Task 1).

Outputs only under results/phase2_supplement/; original Phase 2 CSVs untouched.

  A. Corrected Gaussian filter response (-3 dB vs half-gain).
  B. SGM band-average gradient magnitude and one-sided rfft2 weights.
  C. Dynamic >0 mask vs fixed full-ROI mask.
  D. Attack projection residual audit.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from datasets.preprocessing import fixed_mask_normalize  # noqa: E402
from experiments._common import (  # noqa: E402
    build_context,
    condition_array,
    get_adversarial,
    save_csv,
    save_json,
)
from spectral.band_projection import band_mask  # noqa: E402
from spectral.fft_utils import one_sided_weights  # noqa: E402
from spectral.frequency import radial_frequency  # noqa: E402
from spectral.gaussian_filter import filter_response_table, gaussian_lowpass  # noqa: E402
from utils.config import load_config  # noqa: E402

OUT = ROOT / "results" / "phase2_supplement" / "audit"
PHASE2 = ROOT / "results" / "phase2"
BANDS = ("low", "middle", "high")


def audit_filter_response(scfg):
    sigmas = [float(s) for s in scfg["sigma"]["values"]]
    rho_points = [float(r) for r in scfg["sigma"]["response_rho_points"]]
    rows = filter_response_table(sigmas, rho_points)
    long_rows, summary = [], []
    for row in rows:
        for rho, val, db in zip(row["rho_values"], row["response"], row["gain_db"]):
            long_rows.append({
                "sigma": row["sigma"], "rho": rho, "response": val, "gain_db": db,
            })
        summary.append({
            "sigma": row["sigma"],
            "rho_half_gain": row["rho_half_gain"],
            "rho_minus3db": row["rho_minus3db"],
            "note_half_gain_is_minus6.02db_not_minus3db": True,
        })
    save_csv(OUT / "filter_response_corrected.csv", long_rows)
    save_csv(OUT / "filter_response_cutoffs.csv", summary)
    return {"n_sigmas": len(sigmas), "rows": len(long_rows)}


def audit_fft_weights():
    H, W = 128, 128
    w = one_sided_weights(H, W)
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(2, 1, H, W, generator=rng)
    spec = torch.fft.rfft2(x, norm="ortho")
    weighted = (spec.abs().pow(2) * w).sum(dim=(-2, -1))
    unweighted = spec.abs().pow(2).sum(dim=(-2, -1))
    real = x.pow(2).sum(dim=(-2, -1))
    out = {
        "H": H, "W": W,
        "weights_shape": list(w.shape),
        "weight_sum": float(w.sum()),
        "half_spectrum_bins": int(w.numel()),
        "equivalent_full_bins": float(w.sum()),
        "interior_weight": float(w[0, 1]),
        "axis_weight": float(w[0, 0]),
        "nyquist_weight": float(w[0, -1]),
        "parseval_weighted_max_abs_err": float((weighted - real).abs().max()),
        "parseval_unweighted_max_abs_err": float((unweighted - real).abs().max()),
        "unweighted_over_weighted_ratio": float((unweighted / weighted).mean()),
    }
    save_json(OUT / "rfft_weights.json", out)
    rows = []
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            rows.append({"fy_index": i, "fx_index": j, "weight": float(w[i, j])})
    save_csv(OUT / "rfft_weights_grid.csv", rows)
    return out


def audit_sgm_band_weighting(scfg):
    rho = radial_frequency(128, 128)
    weights = one_sided_weights(128, 128)
    masks = {b: band_mask(rho, scfg["bands"][b]) for b in BANDS}
    rows = []
    for model in ("compnet", "ccnet"):
        for condition in ("clean", "pgd", "apgd"):
            for loss in ("margin", "ce"):
                path = PHASE2 / "sgm" / model / f"sgm_{condition}_{loss}_s10_half.npy"
                if not path.is_file():
                    continue
                half = torch.from_numpy(np.load(path))
                total_w = float((half * weights).sum())
                total_u = float(half.sum())
                for b in BANDS:
                    m = masks[b]
                    u = float((half * m).sum())
                    w = float((half * m * weights).sum())
                    n_half = int(m.sum())
                    n_eq = float((m * weights).sum())
                    rows.append({
                        "model": model, "condition": condition, "loss": loss,
                        "scale": 10.0, "band": b,
                        "n_half_bins": n_half, "n_equivalent_full_bins": n_eq,
                        "unweighted_energy": u, "weighted_energy": w,
                        "unweighted_ratio": u / (total_u + 1e-30),
                        "weighted_ratio": w / (total_w + 1e-30),
                        "unweighted_mean_abs_grad": u / max(n_half, 1),
                        "weighted_mean_abs_grad": w / max(n_eq, 1e-30),
                    })
    save_csv(OUT / "sgm_band_weighting.csv", rows)
    return {"n_rows": len(rows)}


def audit_mask(scfg, device):
    results = []
    for model in ("compnet", "ccnet"):
        ctx = build_context(model, device=device, max_ids=100)
        adapter = ctx["adapter"]
        batch = 64 if model == "compnet" else 32
        rho = radial_frequency(128, 128, device=adapter.device)
        for condition in ("clean", "pgd", "apgd"):
            x_cond, _ = condition_array(ctx, condition)
            cos_all, cons_dyn = [], {s: [] for s in (0.05, 0.1, 0.2)}
            cons_fix = {s: [] for s in (0.05, 0.1, 0.2)}
            pred_agree, nonpos_raw = [], []
            nonpos_low = {s: [] for s in (0.05, 0.1, 0.2)}
            with torch.no_grad():
                for start in range(0, len(x_cond), batch):
                    xb = x_cond[start:start + batch].to(adapter.device)
                    z_dyn = adapter.preprocess_raw(xb)
                    f_dyn = F.normalize(adapter.net.getFeatureCode(z_dyn), dim=1)
                    z_fix = fixed_mask_normalize(xb)
                    f_fix = F.normalize(adapter.net.getFeatureCode(z_fix), dim=1)
                    cos_all.append((f_dyn * f_fix).sum(dim=1).cpu())
                    pred_agree.append(
                        (ctx["classifier"].predict(f_dyn) ==
                         ctx["classifier"].predict(f_fix)).float().cpu()
                    )
                    nonpos_raw.append((xb <= 0).float().sum(dim=(1, 2, 3)).cpu())
                    for sigma in (0.05, 0.1, 0.2):
                        low = gaussian_lowpass(xb, sigma, rho=rho)
                        f_dyn_low = F.normalize(adapter.extract_features(low), dim=1)
                        f_fix_low = F.normalize(
                            adapter.net.getFeatureCode(fixed_mask_normalize(low)), dim=1
                        )
                        cons_dyn[sigma].append((f_dyn * f_dyn_low).sum(dim=1).cpu())
                        cons_fix[sigma].append((f_fix * f_fix_low).sum(dim=1).cpu())
                        nonpos_low[sigma].append((low <= 0).float().sum(dim=(1, 2, 3)).cpu())
            cos_all = torch.cat(cos_all)
            pred_agree = torch.cat(pred_agree)
            nonpos_raw = torch.cat(nonpos_raw)
            results.append({
                "model": model, "condition": condition, "sigma": -1.0,
                "n_nonpositive_pixels_mean": float(nonpos_raw.mean()),
                "feature_cosine_mean": float(cos_all.mean()),
                "feature_cosine_min": float(cos_all.min()),
                "prediction_agreement": float(pred_agree.mean()),
            })
            for sigma in (0.05, 0.1, 0.2):
                cd = torch.cat(cons_dyn[sigma])
                cf = torch.cat(cons_fix[sigma])
                results.append({
                    "model": model, "condition": condition, "sigma": sigma,
                    "n_nonpositive_pixels_mean": float(torch.cat(nonpos_low[sigma]).mean()),
                    "consistency_dynamic": float(cd.mean()),
                    "consistency_fixed": float(cf.mean()),
                    "consistency_difference": float((cd - cf).mean()),
                    "consistency_corr": float(torch.corrcoef(torch.stack([cd, cf]))[0, 1]),
                })
        del ctx
        torch.cuda.empty_cache()
    fields = [
        "model", "condition", "sigma", "n_nonpositive_pixels_mean",
        "feature_cosine_mean", "feature_cosine_min", "prediction_agreement",
        "consistency_dynamic", "consistency_fixed", "consistency_difference",
        "consistency_corr",
    ]
    save_csv(OUT / "mask_comparison.csv", results, fieldnames=fields)
    return {"n_rows": len(results)}


def _as_bool(value):
    return str(value).strip().lower() in ("true", "1", "1.0")


def audit_global_attacks(device):
    """Check L-infinity / pixel-box constraints for PGD and APGD caches."""
    rows = []
    for model in ("compnet", "ccnet"):
        ctx = build_context(model, device=device, max_ids=100)
        for attack in ("pgd", "apgd"):
            x_adv, meta = get_adversarial(ctx, attack)
            delta = x_adv - ctx["xs"]
            linf = delta.abs().amax(dim=(1, 2, 3))
            xa = x_adv
            box = torch.clamp(xa - 1.0, min=0).amax(dim=(1, 2, 3)) + \
                torch.clamp(-xa, min=0).amax(dim=(1, 2, 3))
            l2 = delta.reshape(delta.shape[0], -1).norm(dim=1)
            rows.append({
                "model": model, "attack": attack, "n": len(x_adv),
                "eps": meta["attack_cfg"]["eps"],
                "linf_max": float(linf.max()),
                "linf_excess_max": float((linf - meta["attack_cfg"]["eps"]).clamp_min(0).max()),
                "box_violation_max": float(box.max()),
                "l2_mean": float(l2.mean()),
                "x_min": float(x_adv.min()), "x_max": float(x_adv.max()),
                "asr": meta["asr"],
            })
        del ctx
        torch.cuda.empty_cache()
    save_csv(OUT / "global_attack_constraints.csv", rows)
    return {"n_rows": len(rows)}


def audit_attack_constraints(scfg):
    rows = []
    for exp in ("band_sensitivity", "band_attacks"):
        for model in ("compnet", "ccnet"):
            path = PHASE2 / exp / model / "per_sample.csv"
            if not path.is_file():
                continue
            data = list(csv.DictReader(open(path, encoding="utf-8")))
            if exp == "band_sensitivity":
                groups = {(r["band"], "feature_shift") for r in data}
            else:
                groups = {(r["band"], r["objective"]) for r in data}
            for band, objective in groups:
                sel = [r for r in data if r["band"] == band and
                       (exp == "band_sensitivity" or r["objective"] == objective)]
                rows.append({
                    "experiment": exp, "model": model, "band": band,
                    "objective": objective, "n": len(sel),
                    "linf_max": max(float(r["linf"]) for r in sel),
                    "linf_excess_max": max(float(r["linf_excess"]) for r in sel),
                    "band_outside_max": max(float(r["band_outside_ratio"]) for r in sel),
                    "band_outside_mean": float(np.mean([float(r["band_outside_ratio"]) for r in sel])),
                    "box_violation_max": max(float(r["box_violation"]) for r in sel),
                    "band_residual_linf_max": max(float(r["band_residual_linf"]) for r in sel),
                    "converged_pct": 100.0 * sum(_as_bool(r["dykstra_converged"]) for r in sel) / len(sel),
                    "failed_pct": 100.0 * sum(_as_bool(r["projection_failed"]) for r in sel) / len(sel),
                })
    save_csv(OUT / "attack_constraint_audit.csv", rows)
    return {"n_rows": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    scfg = load_config("configs/spectral.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "filter_response": audit_filter_response(scfg),
        "rfft_weights": audit_fft_weights(),
        "sgm_band_weighting": audit_sgm_band_weighting(scfg),
        "mask_comparison": audit_mask(scfg, args.device),
        "global_attack_constraints": audit_global_attacks(args.device),
        "attack_constraints": audit_attack_constraints(scfg),
    }
    save_json(OUT / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
