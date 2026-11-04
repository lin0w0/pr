"""Task D: Original CSR baseline on PGD / APGD / low-band PGD."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from defenses.csr_original import OriginalCSR  # noqa: E402
from experiments._common import (  # noqa: E402
    build_context,
    get_adversarial,
    predict_labels,
    save_csv,
    save_json,
)
from utils.config import load_config  # noqa: E402

OUT = ROOT / "results" / "csr_original"
PHASE3 = ROOT / "results" / "phase3_final"


def rank_metrics(feats, labels, protos):
    feats = F.normalize(feats, dim=1)
    scores = feats @ protos.T  # [N, G]
    pred = scores.argmax(dim=1)
    order = scores.topk(min(5, scores.shape[1]), dim=1).indices
    gl = torch.arange(scores.shape[1], device=scores.device)
    top = gl[order]
    rank1 = (top[:, 0] == labels).float()
    rank5 = (top == labels[:, None]).any(dim=1).float()
    return pred.cpu(), rank1.cpu(), rank5.cpu()


def load_low_band(model, eval_indices, labels_eval):
    data = np.load(PHASE3 / "attacks" / f"{model}_low.npz")
    x_all = torch.from_numpy(data["x_adv"])
    y_all = data["labels"]
    # band cache covers validation identities 100..299 in order; first 500 rows
    # correspond to identities 100..199.
    x = x_all[:len(eval_indices)]
    y = torch.from_numpy(y_all[:len(eval_indices)]).long()
    if not torch.equal(y, labels_eval):
        raise ValueError("low-band cache label order mismatch")
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tau-mode", choices=["calibrated", "original", "both"], default="both")
    args = parser.parse_args()
    ccfg = load_config("configs/csr_original.yaml")["csr"]
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    OUT.mkdir(parents=True, exist_ok=True)

    for model in models:
        ctx = build_context(model, device=args.device, max_ids=None)
        xs, ys = ctx["xs"], ctx["ys"]
        cal_idx = torch.nonzero(ys < 100).flatten()
        ev_idx = torch.nonzero((ys >= 100) & (ys <= 199)).flatten()
        x_eval, y_eval = xs[ev_idx], ys[ev_idx]
        protos = ctx["classifier"].prototypes.detach().cpu()
        adapter = ctx["adapter"]
        csr = OriginalCSR(
            adapter, radius=ccfg["radius"], lam=ccfg["lambda"],
            steps=ccfg["steps"], eps=ccfg["eps"], alpha=ccfg["alpha"],
            pixel_clamp=ccfg["pixel_clamp"],
        )
        batch = 32 if model == "ccnet" else 64

        # calibrate tau on clean calibration ids
        with torch.no_grad():
            cal_scores = torch.cat([
                csr.score(xs[cal_idx][i:i + batch])
                for i in range(0, len(cal_idx), batch)
            ]).cpu()
        tau5 = float(torch.quantile(cal_scores, 0.05))
        tau1 = float(torch.quantile(cal_scores, 0.01))
        fpr085 = float((cal_scores <= 0.85).float().mean())
        taus = {}
        if args.tau_mode in ("calibrated", "both"):
            taus["calibrated_tau5"] = tau5
        if args.tau_mode in ("original", "both"):
            taus["original_tau085"] = 0.85

        # attack inputs
        x_pgd, _ = get_adversarial(ctx, "pgd")
        x_apgd, _ = get_adversarial(ctx, "apgd")
        x_low = load_low_band(model, ev_idx, y_eval)
        attacks = {
            "clean": (x_eval, y_eval),
            "pgd": (x_pgd[ev_idx], y_eval),
            "apgd": (x_apgd[ev_idx], y_eval),
            "low_band_pgd": (x_low, y_eval),
        }

        # clean reference features
        with torch.no_grad():
            clean_feats = F.normalize(adapter.extract_features(x_eval), dim=1).cpu()
        base_pred, base_r1, base_r5 = rank_metrics(clean_feats, y_eval, protos)
        clean_correct = base_pred == y_eval

        # No defense and LPF baseline
        summary_rows, per_sample_rows = [], []
        for attack, (x_in, y_in) in attacks.items():
            with torch.no_grad():
                f0 = F.normalize(adapter.extract_features(x_in), dim=1).cpu()
            pred, r1, r5 = rank_metrics(f0, y_in, protos)
            summary_rows.append({
                "model": model, "defense": "none", "attack": attack, "tau": None,
                "rank1_pct": float(r1.mean() * 100), "rank5_pct": float(r5.mean() * 100),
                "recovery_pct": None, "clean_mis_correction_pct": 0.0,
                "feature_cos_to_clean": 1.0 if attack == "clean" else float((F.normalize(f0, dim=1) * clean_feats).sum(1).mean()),
                "l2": 0.0, "linf": 0.0, "detection_rate": None,
                "trigger_rate": None, "time_ms": 0.0,
            })
            for i in range(len(x_in)):
                per_sample_rows.append({
                    "model": model, "defense": "none", "attack": attack,
                    "tau": None, "sample_index": int(ev_idx[i]), "label": int(y_in[i]),
                    "pred": int(pred[i]), "correct": bool(pred[i] == y_in[i]),
                    "detected": None,
                })

        # Gaussian LPF baseline (same CSR LPF radius)
        t0 = time.time()
        with torch.no_grad():
            low_imgs = torch.cat([csr.lowpass(x_eval[i:i + batch]) for i in range(0, len(x_eval), batch)])
            low_feats = F.normalize(adapter.extract_features(low_imgs), dim=1).cpu()
        lpf_time = (time.time() - t0) / len(x_eval) * 1000
        pred, r1, r5 = rank_metrics(low_feats, y_eval, protos)
        summary_rows.append({
            "model": model, "defense": "gaussian_lpf", "attack": "clean", "tau": None,
            "rank1_pct": float(r1.mean() * 100), "rank5_pct": float(r5.mean() * 100),
            "recovery_pct": None,
            "clean_mis_correction_pct": float((pred != y_eval).float().mean() * 100),
            "feature_cos_to_clean": float((low_feats * clean_feats).sum(1).mean()),
            "l2": float((low_imgs - x_eval).pow(2).sum(dim=(1, 2, 3)).sqrt().mean()),
            "linf": float((low_imgs - x_eval).abs().amax(dim=(1, 2, 3)).mean()),
            "detection_rate": None, "trigger_rate": None, "time_ms": lpf_time,
        })
        for i in range(len(x_eval)):
            per_sample_rows.append({
                "model": model, "defense": "gaussian_lpf", "attack": "clean",
                "tau": None, "sample_index": int(ev_idx[i]), "label": int(y_eval[i]),
                "pred": int(pred[i]), "correct": bool(pred[i] == y_eval[i]),
                "detected": None,
            })
        for attack, (x_in, y_in) in attacks.items():
            if attack == "clean":
                continue
            with torch.no_grad():
                low = torch.cat([csr.lowpass(x_in[i:i + batch]) for i in range(0, len(x_in), batch)])
                fl = F.normalize(adapter.extract_features(low), dim=1).cpu()
            pred, r1, r5 = rank_metrics(fl, y_in, protos)
            base_pred_attack, base_r1_attack, _ = rank_metrics(
                F.normalize(adapter.extract_features(x_in), dim=1).cpu(), y_in, protos)
            nd_pred = base_pred_attack
            # clean-correct samples that the attack breaks without defense
            lost = clean_correct & (nd_pred != y_in)
            rec = lost & (pred == y_in)
            summary_rows.append({
                "model": model, "defense": "gaussian_lpf", "attack": attack, "tau": None,
                "rank1_pct": float(r1.mean() * 100), "rank5_pct": float(r5.mean() * 100),
                "recovery_pct": float(rec.sum() / max(int(lost.sum()), 1) * 100),
                "clean_mis_correction_pct": None,
                "feature_cos_to_clean": float((fl * clean_feats).sum(1).mean()),
                "l2": float((low - x_in).pow(2).sum(dim=(1, 2, 3)).sqrt().mean()),
                "linf": float((low - x_in).abs().amax(dim=(1, 2, 3)).mean()),
                "detection_rate": None, "trigger_rate": None, "time_ms": lpf_time,
            })
            for i in range(len(x_in)):
                per_sample_rows.append({
                    "model": model, "defense": "gaussian_lpf", "attack": attack,
                    "tau": None, "sample_index": int(ev_idx[i]), "label": int(y_in[i]),
                    "pred": int(pred[i]), "correct": bool(pred[i] == y_in[i]),
                    "detected": None,
                })

        # CSR for each tau
        for tau_name, tau in taus.items():
            t0 = time.time()
            defended_parts, masks, recs_all = [], [], []
            scores_all = []
            for i in range(0, len(x_eval), batch):
                d, mask, recs, scores = csr.process(x_eval[i:i + batch], tau)
                defended_parts.append(d.cpu())
                masks.append(mask.cpu())
                scores_all.append(scores.cpu())
                for r in recs:
                    r["batch_start"] = i
                    recs_all.append(r)
            defended_clean = torch.cat(defended_parts)
            mask_clean = torch.cat(masks)
            csr_time = (time.time() - t0) / len(x_eval) * 1000
            with torch.no_grad():
                fdef = F.normalize(adapter.extract_features(defended_clean), dim=1).cpu()
            pred, r1, r5 = rank_metrics(fdef, y_eval, protos)
            mis = float((pred != y_eval).float().mean() * 100)
            summary_rows.append({
                "model": model, "defense": f"csr_{tau_name}", "attack": "clean",
                "tau": tau,
                "rank1_pct": float(r1.mean() * 100), "rank5_pct": float(r5.mean() * 100),
                "recovery_pct": None, "clean_mis_correction_pct": mis,
                "feature_cos_to_clean": float((fdef * clean_feats).sum(1).mean()),
                "l2": float((defended_clean - x_eval).pow(2).sum(dim=(1, 2, 3)).sqrt().mean()),
                "linf": float((defended_clean - x_eval).abs().amax(dim=(1, 2, 3)).mean()),
                "detection_rate": float(mask_clean.float().mean() * 100),
                "trigger_rate": float(mask_clean.float().mean() * 100),
                "time_ms": csr_time,
            })
            for i in range(len(x_eval)):
                per_sample_rows.append({
                    "model": model, "defense": f"csr_{tau_name}", "attack": "clean",
                    "tau": tau, "sample_index": int(ev_idx[i]), "label": int(y_eval[i]),
                    "pred": int(pred[i]), "correct": bool(pred[i] == y_eval[i]),
                    "detected": bool(mask_clean[i]),
                })
            for attack, (x_in, y_in) in attacks.items():
                if attack == "clean":
                    continue
                t0 = time.time()
                defended_parts, masks, recs_attack = [], [], []
                for i in range(0, len(x_in), batch):
                    d, mask, recs, scores = csr.process(x_in[i:i + batch], tau)
                    defended_parts.append(d.cpu())
                    masks.append(mask.cpu())
                    for r in recs:
                        r["batch_start"] = i
                        recs_attack.append(r)
                defended = torch.cat(defended_parts)
                mask = torch.cat(masks)
                t_ms = (time.time() - t0) / len(x_in) * 1000
                with torch.no_grad():
                    fa = F.normalize(adapter.extract_features(defended), dim=1).cpu()
                    nd_feats = F.normalize(adapter.extract_features(x_in), dim=1).cpu()
                pred, r1, r5 = rank_metrics(fa, y_in, protos)
                nd_pred, nd_r1, _ = rank_metrics(nd_feats, y_in, protos)
                lost = clean_correct & (nd_pred != y_in)
                rec = lost & (pred == y_in)
                summary_rows.append({
                    "model": model, "defense": f"csr_{tau_name}", "attack": attack,
                    "tau": tau,
                    "rank1_pct": float(r1.mean() * 100), "rank5_pct": float(r5.mean() * 100),
                    "recovery_pct": float(rec.sum() / max(int(lost.sum()), 1) * 100),
                    "clean_mis_correction_pct": None,
                    "feature_cos_to_clean": float((fa * clean_feats).sum(1).mean()),
                    "l2": float((defended - x_in).pow(2).sum(dim=(1, 2, 3)).sqrt().mean()),
                    "linf": float((defended - x_in).abs().amax(dim=(1, 2, 3)).mean()),
                    "detection_rate": float(mask.float().mean() * 100),
                    "trigger_rate": float(mask.float().mean() * 100),
                    "time_ms": t_ms,
                })
                for i in range(len(x_in)):
                    per_sample_rows.append({
                        "model": model, "defense": f"csr_{tau_name}", "attack": attack,
                        "tau": tau, "sample_index": int(ev_idx[i]), "label": int(y_in[i]),
                        "pred": int(pred[i]), "correct": bool(pred[i] == y_in[i]),
                        "detected": bool(mask[i]),
                    })
                if recs_attack:
                    rec_rows = [{
                        "model": model, "tau_name": tau_name, "attack": attack,
                        "sample_index": int(ev_idx[r["sample_index"] + r["batch_start"]]),
                        "step": r["step"], "sim_lpf": r["sim_lpf"], "sim_adv": r["sim_adv"],
                        "lrec": r["lrec"], "selected": r["selected"],
                        "l2": r["l2"], "linf": r["linf"],
                    } for r in recs_attack]
                    save_csv(OUT / model / f"csr_{tau_name}_{attack}_records.csv", rec_rows)

        save_csv(OUT / model / "summary.csv", summary_rows)
        save_csv(OUT / model / "per_sample.csv", per_sample_rows)
        save_json(OUT / model / "meta.json", {
            "model": model, "csr_config": ccfg,
            "tau5": tau5, "tau1": tau1, "fpr_at_0.85_on_clean_calibration": fpr085,
            "n_eval": len(x_eval), "eval_identity_range": [100, 199],
            "baseline_clean_rank1": float(base_r1.mean() * 100),
        })
        print(f"[{model}] CSR done; tau5={tau5:.4f}, tau1={tau1:.4f}, fpr@.85={fpr085:.3f}", flush=True)
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
