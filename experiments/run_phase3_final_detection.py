"""Phase 3 final detection validation: A1, A2, A5.

Outputs under results/phase3_final/. Original results untouched.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

from attacks.band_limited import band_limited_margin  # noqa: E402
from defenses.detection import detector_scores  # noqa: E402
from evaluation.detection_metrics import (  # noqa: E402
    auc_lower_is_adv,
    rates_at_threshold,
    threshold_at_fpr,
)
from experiments._common import (  # noqa: E402
    build_context,
    get_adversarial,
    predict_labels,
    save_csv,
    save_json,
)
from spectral.band_projection import band_mask  # noqa: E402
from spectral.frequency import radial_frequency  # noqa: E402
from utils.config import load_config  # noqa: E402

OUT = ROOT / "results" / "phase3_final"
PHASE2 = ROOT / "results" / "phase2"
BANDS = ("low", "middle", "high")


def load_cfg():
    return load_config("configs/phase3_final.yaml")


def load_thresholds():
    return json.load(open(ROOT / "results/phase3_detection_preliminary/thresholds_and_summary.json", encoding="utf-8"))


def detector_specs(scfg):
    return {
        "single": ([float(scfg["detection"]["single_sigma"]["compnet"])], "mean"),
        "multi": ([float(s) for s in scfg["detection"]["multi_sigmas"]],
                  scfg["detection"]["multi_method"]),
    }


def scores_for(adapter, x, scfg, model, batch=32):
    sig_single = [float(scfg["detection"]["single_sigma"][model])]
    sig_multi = [float(s) for s in scfg["detection"]["multi_sigmas"]]
    method = scfg["detection"]["multi_method"]
    out = {"single": [], "multi": []}
    for start in range(0, len(x), batch):
        xb = x[start:start + batch]
        out["single"].append(detector_scores(adapter, xb, sig_single, "mean").cpu())
        out["multi"].append(detector_scores(adapter, xb, sig_multi, method).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def band_samples(ctx, model, scfg, indices, out_dir):
    """Generate or load band-limited PGD samples for the selected indices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = ctx["xs"][indices]
    ys = ctx["ys"][indices]
    bcfg = scfg_spectral["attacks"]["band"]
    bands = scfg["attacks"]["band"]["bands"]
    out = {}
    for band in bands:
        cache = out_dir / f"{model}_{band}.npz"
        meta_path = out_dir / f"{model}_{band}.json"
        if cache.is_file():
            data = np.load(cache)
            x_adv = torch.from_numpy(data["x_adv"])
            meta = json.load(open(meta_path, encoding="utf-8"))
        else:
            mask = band_mask(
                radial_frequency(128, 128), scfg_spectral["bands"][band]
            ).to(ctx["device"])
            chunks = []
            batch = 64 if model == "compnet" else 32
            for start in range(0, len(xs), batch):
                xa, _, _ = band_limited_margin(
                    ctx["adapter"], ctx["classifier"],
                    xs[start:start + batch], ys[start:start + batch], mask,
                    eps=bcfg["eps"], alpha=bcfg["alpha"], steps=bcfg["steps"],
                    objective="untargeted", scale=bcfg["scale"],
                    random_start=bcfg["random_start"],
                    seed=bcfg["seed"] + start,
                    dykstra_iters=bcfg["dykstra_iters"],
                    dykstra_tol=bcfg["dykstra_tol"],
                )
                chunks.append(xa.cpu())
            x_adv = torch.cat(chunks)
            pred = predict_labels(ctx["adapter"], ctx["classifier"], x_adv,
                                  batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
            clean_pred = predict_labels(ctx["adapter"], ctx["classifier"], xs,
                                        batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
            cc = clean_pred == ys
            success = cc & (pred != ys)
            meta = {
                "model": model, "band": band, "n": len(xs),
                "asr": float(success.sum() / max(int(cc.sum()), 1)),
                "clean_accuracy": float(cc.float().mean()),
                "cfg": bcfg,
            }
            np.savez(cache, x_adv=x_adv.numpy(), labels=ys.numpy())
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        out[band] = (x_adv, meta)
    return out


def perturb_one(x, kind, param):
    """x: [1,1,H,W] raw [0,1]."""
    if kind == "gaussian_noise":
        g = torch.Generator().manual_seed(1234)
        noise = torch.randn(x.shape, generator=g) * float(param)
        return torch.clamp(x + noise, 0, 1)
    if kind == "gaussian_blur":
        sigma = float(param)
        k = int(2 * max(1, round(3 * sigma)) + 1)
        return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    if kind == "brightness_delta":
        return torch.clamp(x + float(param), 0, 1)
    if kind == "contrast_factor":
        return torch.clamp((x - 0.5) * float(param) + 0.5, 0, 1)
    if kind == "jpeg_quality":
        arr = (x[0, 0].numpy() * 255).round().clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=int(param))
        buf.seek(0)
        im = Image.open(buf).convert("L")
        a = np.array(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).view(1, 1, *a.shape)
    if kind == "rotation_degrees":
        return TF.rotate(x, float(param), interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    if kind == "translation_pixels":
        p = float(param)
        return TF.affine(x, angle=0.0, translate=[p, p], scale=1.0, shear=[0.0, 0.0],
                         interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    raise ValueError(kind)


def run_a1(ctx, model, scfg, thresholds, indices):
    band_data = band_samples(ctx, model, scfg, indices, OUT / "attacks")
    xs = ctx["xs"][indices]
    ys = ctx["ys"][indices]
    clean = scores_for(ctx["adapter"], xs, scfg, model)
    attacks = {}
    for band, (x_adv, meta) in band_data.items():
        attacks[f"band_{band}"] = (x_adv, meta["asr"])
    for attack in ("pgd", "apgd"):
        x_adv, meta = get_adversarial(ctx, attack)
        attacks[f"global_{attack.upper()}"] = (x_adv[indices], meta["asr"])
    rows = []
    for name, (x_adv, asr) in attacks.items():
        sc = scores_for(ctx["adapter"], x_adv, scfg, model)
        for det in ("single", "multi"):
            clean_s = clean[det]
            adv_s = sc[det]
            tau1 = thresholds[model][det]["threshold_fpr1"]
            tau5 = thresholds[model][det]["threshold_fpr5"]
            r1 = rates_at_threshold(clean_s, adv_s, tau1)
            r5 = rates_at_threshold(clean_s, adv_s, tau5)
            rows.append({
                "model": model, "attack": name, "detector": det,
                "n": len(adv_s), "attack_asr": asr,
                "auc": auc_lower_is_adv(clean_s, adv_s),
                "tau_fpr1": tau1, "tpr_fpr1": r1["TPR"], "fpr_fpr1": r1["FPR"],
                "tau_fpr5": tau5, "tpr_fpr5": r5["TPR"], "fpr_fpr5": r5["FPR"],
            })
    save_csv(OUT / f"a1_attack_detection_{model}.csv", rows)
    return rows


def run_a2(ctx, model, scfg, thresholds, indices, pcfg):
    xs = ctx["xs"][indices]
    rows = []
    variants = []
    for v in pcfg["gaussian_noise"]:
        variants.append(("gaussian_noise", v))
    for v in pcfg["gaussian_blur_sigma"]:
        variants.append(("gaussian_blur", v))
    for v in pcfg["brightness_delta"]:
        variants.append(("brightness_delta", v))
    for v in pcfg["contrast_factor"]:
        variants.append(("contrast_factor", v))
    for v in pcfg["jpeg_quality"]:
        variants.append(("jpeg_quality", v))
    for v in pcfg["rotation_degrees"]:
        variants.append(("rotation_degrees", v))
    for v in pcfg["translation_pixels"]:
        variants.append(("translation_pixels", v))
    for kind, param in variants:
        xp = torch.cat([perturb_one(xs[i:i + 1], kind, param) for i in range(len(xs))])
        sc = scores_for(ctx["adapter"], xp, scfg, model)
        for det in ("single", "multi"):
            s = sc[det]
            for target in (1, 5):
                tau = thresholds[model][det][f"threshold_fpr{target}"]
                rows.append({
                    "model": model, "perturbation": kind, "param": float(param),
                    "detector": det, "target_fpr_pct": target,
                    "tau": tau, "fpr": float((s <= tau).mean()),
                    "score_mean": float(s.mean()), "score_std": float(s.std()),
                })
    save_csv(OUT / f"a2_nonadversarial_fpr_{model}.csv", rows)
    return rows


def load_attenuation(model):
    rows = list(csv.DictReader(open(PHASE2 / "attenuation" / model / "per_sample.csv", encoding="utf-8")))
    labels, scores = {}, {}
    for r in rows:
        idx = int(r["sample_index"])
        labels[idx] = int(r["label"])
        scores.setdefault(r["condition"], {}).setdefault(float(r["sigma"]), {})[idx] = float(r["consistency"])
    return labels, scores


def run_a5(scfg):
    rows = []
    for model in ("compnet", "ccnet"):
        labels, scores = load_attenuation(model)
        idxs = np.array(sorted(labels))
        labels_arr = np.array([labels[i] for i in idxs])
        sig_single = [float(scfg["detection"]["single_sigma"][model])]
        sig_multi = [float(s) for s in scfg["detection"]["multi_sigmas"]]
        method = scfg["detection"]["multi_method"]
        def sscore(cond):
            if len(sig_single) == 1:
                return np.array([scores[cond][sig_single[0]][i] for i in idxs])
            return np.array([scores[cond][sig_single[0]][i] for i in idxs])
        def mscore(cond):
            mat = np.stack([np.array([scores[cond][s][i] for i in idxs]) for s in sig_multi], axis=1)
            return mat.mean(axis=1) if method == "mean" else mat.min(axis=1)
        for det, fn in (("single", sscore), ("multi", mscore)):
            clean = fn("clean")
            pgd = fn("pgd")
            apgd = fn("apgd")
            for fold in range(5):
                test = (labels_arr % 5) == fold
                train = ~test
                for target in (1, 5):
                    tau = threshold_at_fpr(clean[train], target / 100.0)
                    for attack_name, adv in (("pgd", pgd), ("apgd", apgd), ("combined", np.r_[pgd[test], apgd[test]])):
                        if attack_name == "combined":
                            clean_test = np.r_[clean[test], clean[test]]
                            y_clean = clean[test]
                        else:
                            clean_test = clean[test]
                            y_clean = clean[test]
                        # AUC for combined uses duplicated clean to balance; per-attack uses clean_test
                        if attack_name == "combined":
                            auc = auc_lower_is_adv(np.r_[clean[test], clean[test]], adv)
                        else:
                            auc = auc_lower_is_adv(clean_test, adv)
                        r = rates_at_threshold(clean[test], adv, tau)
                        rows.append({
                            "model": model, "detector": det, "fold": fold,
                            "target_fpr_pct": target, "attack": attack_name,
                            "tau": tau, "auc": auc,
                            "tpr": r["TPR"], "fpr": r["FPR"],
                            "n_test_ids": int(test.sum()),
                        })
    save_csv(OUT / "a5_threshold_stability.csv", rows)
    # summary mean/std
    summary = []
    keys = sorted({(r["model"], r["detector"], r["target_fpr_pct"], r["attack"]) for r in rows})
    for model, det, target, attack in keys:
        sel = [r for r in rows if r["model"] == model and r["detector"] == det
               and r["target_fpr_pct"] == target and r["attack"] == attack]
        for metric in ("auc", "tpr", "fpr"):
            vals = np.array([r[metric] for r in sel], dtype=float)
            se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            summary.append({
                "model": model, "detector": det, "target_fpr_pct": target,
                "attack": attack, "metric": metric, "n_folds": len(vals),
                "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                "ci_low": float(vals.mean() - 1.96 * se),
                "ci_high": float(vals.mean() + 1.96 * se),
            })
    save_csv(OUT / "a5_threshold_stability_summary.csv", summary)
    return rows, summary


scfg_spectral = None


def main():
    global scfg_spectral
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    scfg = load_cfg()
    scfg_spectral = load_config("configs/spectral.yaml")
    thresholds = load_thresholds()
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    OUT.mkdir(parents=True, exist_ok=True)
    for model in models:
        ctx = build_context(model, device=args.device, max_ids=None)
        ys = ctx["ys"]
        start = int(scfg["evaluation"]["identity_start"])
        end = int(scfg["evaluation"]["identity_end"])
        indices = torch.nonzero((ys >= start) & (ys <= end)).flatten()
        print(f"[{model}] A1/A2 eval indices: {len(indices)}", flush=True)
        run_a1(ctx, model, scfg, thresholds, indices)
        run_a2(ctx, model, scfg, thresholds, indices, scfg["perturbations"])
        del ctx
        torch.cuda.empty_cache()
    run_a5(scfg)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
