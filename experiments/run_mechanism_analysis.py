"""Phase 3 preliminary mechanism analysis (Task 2).

Outputs under results/phase2_supplement/mechanism/:
  * clean identity margins (full validation),
  * per-band gradient projection norms / directional derivatives (full),
  * post-band-attack margins and success/failure distributions (100 ids),
  * feature-shift vs attack-success relationship (existing full D CSVs).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import pointbiserialr  # noqa: E402

from attacks.band_limited import band_limited_margin  # noqa: E402
from evaluation.paired_stats import paired_summary  # noqa: E402
from experiments._common import (  # noqa: E402
    build_context,
    extract_features,
    save_csv,
    save_json,
)
from spectral.band_projection import band_mask, project_band  # noqa: E402
from spectral.frequency import radial_frequency  # noqa: E402
from spectral.losses import margin_per_sample  # noqa: E402

OUT = ROOT / "results" / "phase2_supplement" / "mechanism"
PHASE2 = ROOT / "results" / "phase2"
BANDS = ("low", "middle", "high")


def margins_from_features(features, labels, protos, scale=10.0):
    scores = F.normalize(features, dim=1) @ protos.T
    rows = torch.arange(len(labels), device=scores.device)
    s_y = scores[rows, labels.to(scores.device)]
    masked = scores.clone()
    masked[rows, labels.to(scores.device)] = -torch.inf
    other = masked.max(dim=1).values
    return (s_y - other) * scale, scores


def clean_margins(ctx):
    feats = extract_features(
        ctx["adapter"], ctx["xs"],
        batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
    protos = ctx["classifier"].prototypes.detach().cpu()
    cos_margin, scores = margins_from_features(feats, ctx["ys"], protos, scale=1.0)
    logit_margin = cos_margin * 10.0
    pred = scores.argmax(dim=1)
    rows = []
    for i in range(len(feats)):
        rows.append({
            "model": ctx["model"], "sample_index": i,
            "label": int(ctx["ys"][i]), "pred": int(pred[i]),
            "correct": bool(pred[i] == ctx["ys"][i]),
            "cos_margin": float(cos_margin[i]),
            "logit_margin_scale10": float(logit_margin[i]),
        })
    save_csv(OUT / f"clean_margins_{ctx['model']}.csv", rows)
    pair = paired_summary(
        logit_margin.numpy(), ctx["ys"].numpy()
    )
    vals = logit_margin.numpy()
    summary = {
        "model": ctx["model"], "n": len(vals),
        "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        **pair,
    }
    return summary


def gradient_bands(ctx, batch=16):
    rho = radial_frequency(128, 128, device="cpu")
    masks = {b: band_mask(rho, ctx["spectral_cfg"]["bands"][b]).to(ctx["device"])
             for b in BANDS}
    protos = ctx["classifier"].prototypes
    rows = []
    for start in range(0, len(ctx["xs"]), batch):
        xb = ctx["xs"][start:start + batch].to(ctx["device"]).detach().requires_grad_(True)
        yb = ctx["ys"][start:start + batch].to(ctx["device"])
        feats = F.normalize(ctx["adapter"].extract_features(xb), dim=1)
        scores = 10.0 * (feats @ protos.T)
        loss = margin_per_sample(scores, yb).sum()
        grad, = torch.autograd.grad(loss, xb)
        for b in BANDS:
            gb = project_band(grad, masks[b])
            gnorm = grad.flatten(1).norm(dim=1)
            bnorm = gb.flatten(1).norm(dim=1)
            cosine = (grad * gb).flatten(1).sum(1) / (gnorm * bnorm + 1e-30)
            for j in range(len(xb)):
                rows.append({
                    "model": ctx["model"], "band": b,
                    "sample_index": start + j,
                    "label": int(ctx["ys"][start + j]),
                    "grad_norm": float(gnorm[j]),
                    "band_grad_norm": float(bnorm[j]),
                    "band_projection_ratio": float(bnorm[j] / (gnorm[j] + 1e-30)),
                    "directional_derivative": float(bnorm[j]),
                    "cosine_full_band": float(cosine[j]),
                })
    save_csv(OUT / f"gradient_bands_{ctx['model']}.csv", rows)
    summary = []
    for b in BANDS:
        sel = [r for r in rows if r["band"] == b]
        ratio = np.array([r["band_projection_ratio"] for r in sel])
        bn = np.array([r["band_grad_norm"] for r in sel])
        cos = np.array([r["cosine_full_band"] for r in sel])
        groups = np.array([r["label"] for r in sel])
        pr = paired_summary(ratio, groups)
        summary.append({
            "model": ctx["model"], "band": b, "n": len(sel),
            "grad_norm_mean": float(np.mean([r["grad_norm"] for r in sel])),
            "band_grad_norm_mean": float(bn.mean()),
            "band_grad_norm_std": float(bn.std(ddof=1)),
            "band_projection_ratio_mean": float(ratio.mean()),
            "band_projection_ratio_std": float(ratio.std(ddof=1)),
            **{f"ratio_{k}": v for k, v in pr.items()},
            "cosine_mean": float(cos.mean()),
            "cosine_std": float(cos.std(ddof=1)),
            "directional_derivative_mean": float(np.mean([r["directional_derivative"] for r in sel])),
        })
    save_csv(OUT / f"gradient_bands_summary_{ctx['model']}.csv", summary)
    return summary


def attack_margins(ctx, max_ids=100, batch=None):
    idxs = torch.nonzero(ctx["ys"] < max_ids).flatten()
    xs = ctx["xs"][idxs]
    ys = ctx["ys"][idxs]
    batch = batch or (64 if ctx["model"] == "compnet" else 32)
    protos = ctx["classifier"].prototypes
    with torch.no_grad():
        clean_feats = extract_features(
            ctx["adapter"], xs,
            batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
        clean_scores = F.normalize(clean_feats.to(ctx["device"]), dim=1) @ protos.T
        clean_pred = clean_scores.argmax(dim=1).cpu()
        clean_margin, _ = margins_from_features(clean_feats, ys, protos.cpu(), scale=10.0)
    rho = radial_frequency(128, 128, device="cpu")
    bcfg = ctx["spectral_cfg"]["attacks"]["band"]
    rows = []
    for band in BANDS:
        mask = band_mask(rho, ctx["spectral_cfg"]["bands"][band]).to(ctx["device"])
        for start in range(0, len(xs), batch):
            xb = xs[start:start + batch]
            yb = ys[start:start + batch]
            xa, delta, metrics = band_limited_margin(
                ctx["adapter"], ctx["classifier"], xb, yb, mask,
                eps=bcfg["eps"], alpha=bcfg["alpha"], steps=bcfg["steps"],
                objective="untargeted", scale=bcfg["scale"],
                random_start=bcfg["random_start"], seed=bcfg["seed"] + start,
                dykstra_iters=bcfg["dykstra_iters"], dykstra_tol=bcfg["dykstra_tol"],
            )
            with torch.no_grad():
                feats = F.normalize(ctx["adapter"].extract_features(xa), dim=1)
                adv_margin, adv_scores = margins_from_features(
                    feats.cpu(), ys[start:start + batch], protos.cpu(), scale=10.0
                )
                adv_pred = adv_scores.argmax(dim=1)
                adv_feats_cpu = feats.cpu()
            base = clean_feats[start:start + batch]
            shift = 1.0 - (base * adv_feats_cpu).sum(dim=1)
            for j in range(len(xb)):
                idx = int(idxs[start + j])
                clean_ok = bool(clean_pred[start + j] == ys[start + j])
                pred = int(adv_pred[j])
                rows.append({
                    "model": ctx["model"], "band": band, "sample_index": idx,
                    "label": int(ys[start + j]),
                    "clean_margin": float(clean_margin[start + j]),
                    "adv_margin": float(adv_margin[j]),
                    "margin_drop": float(clean_margin[start + j] - adv_margin[j]),
                    "clean_correct": clean_ok,
                    "adv_pred": pred,
                    "success": bool(clean_ok and pred != int(ys[start + j])),
                    "feature_shift": float(shift[j]),
                    "l2": float(delta[j].pow(2).sum().sqrt()),
                    "linf": float(metrics["linf"][j]),
                    "band_outside_ratio": float(metrics["band_outside_energy_ratio"][j]),
                })
    save_csv(OUT / f"attack_margins_{ctx['model']}.csv", rows)
    summary = []
    for band in BANDS:
        sel = [r for r in rows if r["band"] == band]
        succ = [r for r in sel if r["success"]]
        fail = [r for r in sel if r["clean_correct"] and not r["success"]]
        def stats(vals):
            a = np.array(vals)
            return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                    "median": float(np.median(a)),
                    "p25": float(np.percentile(a, 25)),
                    "p75": float(np.percentile(a, 75))}
        summary.append({
            "model": ctx["model"], "band": band, "n": len(sel),
            "n_success": len(succ), "n_clean_correct_fail": len(fail),
            "clean_margin_all": stats([r["clean_margin"] for r in sel]),
            "adv_margin_all": stats([r["adv_margin"] for r in sel]),
            "margin_drop_all": stats([r["margin_drop"] for r in sel]),
            "clean_margin_success": stats([r["clean_margin"] for r in succ]) if succ else None,
            "adv_margin_success": stats([r["adv_margin"] for r in succ]) if succ else None,
            "clean_margin_failure": stats([r["clean_margin"] for r in fail]) if fail else None,
            "adv_margin_failure": stats([r["adv_margin"] for r in fail]) if fail else None,
            "feature_shift_success": stats([r["feature_shift"] for r in succ]) if succ else None,
            "feature_shift_failure": stats([r["feature_shift"] for r in fail]) if fail else None,
        })
    save_csv(OUT / f"attack_margins_summary_{ctx['model']}.csv", summary)
    return summary


def feature_shift_success():
    rows = []
    for model in ("compnet", "ccnet"):
        path = PHASE2 / "band_attacks" / model / "per_sample.csv"
        data = list(csv.DictReader(open(path, encoding="utf-8")))
        for band in BANDS:
            for objective in ("untargeted", "targeted"):
                sel = [r for r in data if r["band"] == band and r["objective"] == objective]
                shift = np.array([float(r["feature_shift"]) for r in sel])
                success = np.array([1 if r["success"].lower() == "true" else 0 for r in sel])
                if success.min() == success.max():
                    auc = float("nan")
                    corr = float("nan")
                else:
                    auc = float(roc_auc_score(success, shift))
                    corr = float(pointbiserialr(success, shift).statistic)
                s = shift[success == 1] if success.any() else np.array([np.nan])
                f = shift[success == 0] if (~success.astype(bool)).any() else np.array([np.nan])
                rows.append({
                    "model": model, "band": band, "objective": objective,
                    "n": len(sel), "n_success": int(success.sum()),
                    "feature_shift_success_mean": float(np.nanmean(s)),
                    "feature_shift_failure_mean": float(np.nanmean(f)),
                    "auc_shift_predicts_success": auc,
                    "pointbiserial_corr": corr,
                })
    save_csv(OUT / "feature_shift_success.csv", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attack-max-ids", type=int, default=100)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    all_summary = {}
    for model in models:
        ctx = build_context(model, device=args.device, max_ids=None)
        print(f"\n[{model}] clean margins...", flush=True)
        cm = clean_margins(ctx)
        print(f"[{model}] gradient bands...", flush=True)
        gb = gradient_bands(ctx)
        print(f"[{model}] attack margins ({args.attack_max_ids} ids)...", flush=True)
        am = attack_margins(ctx, max_ids=args.attack_max_ids)
        all_summary[model] = {
            "clean_margin": cm,
            "gradient_bands": gb,
            "attack_margins": am,
        }
        del ctx
        torch.cuda.empty_cache()
    fs = feature_shift_success()
    save_json(OUT / "mechanism_summary.json", {
        "models": all_summary,
        "feature_shift_success": fs,
    })
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
