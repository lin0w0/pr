"""A3: CW-L2 attack and fixed-detector evaluation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attacks.cw import cw_l2
from defenses.detection import detector_scores
from evaluation.detection_metrics import auc_lower_is_adv, rates_at_threshold
from experiments._common import build_context, predict_labels, save_csv, save_json
from utils.config import load_config

OUT = ROOT / "results" / "phase3_final"


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


def run_cw(adapter, classifier, x, y, c, steps, lr, kappa, batch):
    out = []
    for start in range(0, len(x), batch):
        out.append(cw_l2(adapter, classifier, x[start:start + batch],
                         y[start:start + batch], c=c, steps=steps, lr=lr,
                         kappa=kappa, scale=10.0).cpu())
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config("configs/phase3_final.yaml")
    thresholds = json.load(open(ROOT / cfg["detection"]["threshold_source"], encoding="utf-8"))
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    OUT.mkdir(parents=True, exist_ok=True)
    for model in models:
        ctx = build_context(model, device=args.device, max_ids=None)
        xs, ys = ctx["xs"], ctx["ys"]
        cal = torch.nonzero(ys < int(cfg["attacks"]["cw"]["calibration_identity_end"]) + 1).flatten()
        ev = torch.nonzero((ys >= int(cfg["attacks"]["cw"]["evaluation_identity_start"])) &
                           (ys <= int(cfg["attacks"]["cw"]["evaluation_identity_end"]))).flatten()
        batch = 16 if model == "ccnet" else 32
        cwcfg = cfg["attacks"]["cw"]
        clean_pred = predict_labels(ctx["adapter"], ctx["classifier"], xs[ev],
                                    batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
        best = None
        cal_rows = []
        for c in cwcfg["c_candidates"]:
            xa = run_cw(ctx["adapter"], ctx["classifier"], xs[cal], ys[cal], c,
                        cwcfg["steps"], cwcfg["lr"], cwcfg["kappa"], batch)
            pred = predict_labels(ctx["adapter"], ctx["classifier"], xa,
                                  batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
            asr = float((pred != ys[cal]).float().mean())
            l2 = float((xa - xs[cal]).reshape(len(xa), -1).norm(dim=1).mean())
            cal_rows.append({"model": model, "c": c, "cal_asr": asr, "cal_l2_mean": l2})
            if best is None or (asr, -l2) > (best["cal_asr"], -best["cal_l2_mean"]):
                best = cal_rows[-1]
        save_csv(OUT / f"a3_cw_calibration_{model}.csv", cal_rows)
        xa = run_cw(ctx["adapter"], ctx["classifier"], xs[ev], ys[ev], best["c"],
                    cwcfg["steps"], cwcfg["lr"], cwcfg["kappa"], batch)
        pred = predict_labels(ctx["adapter"], ctx["classifier"], xa,
                              batch_size=int(ctx["model_cfg"]["batch"]["inference"]))
        cc = clean_pred == ys[ev]
        success = cc & (pred != ys[ev])
        asr = float(success.sum() / max(int(cc.sum()), 1))
        clean_s = scores_for(ctx["adapter"], xs[ev], cfg, model)
        adv_s = scores_for(ctx["adapter"], xa, cfg, model)
        det_rows = []
        for det in ("single", "multi"):
            for target in (1, 5):
                tau = thresholds[model][det][f"threshold_fpr{target}"]
                r = rates_at_threshold(clean_s[det], adv_s[det], tau)
                det_rows.append({
                    "model": model, "attack": "CW-L2", "detector": det,
                    "target_fpr_pct": target, "tau": tau,
                    "auc": auc_lower_is_adv(clean_s[det], adv_s[det]),
                    "tpr": r["TPR"], "fpr": r["FPR"],
                    "n": len(xa), "cw_c": best["c"],
                })
        save_csv(OUT / f"a3_cw_detection_{model}.csv", det_rows)
        np.savez(OUT / f"a3_cw_samples_{model}.npz",
                 x_adv=xa.numpy(), labels=ys[ev].numpy(), indices=ev.numpy())
        save_json(OUT / f"a3_cw_stats_{model}.json", {
            "model": model, "chosen_c": best["c"], "calibration": cal_rows,
            "eval_asr": asr, "eval_n": len(xa),
            "l2_mean": float((xa - xs[ev]).reshape(len(xa), -1).norm(dim=1).mean()),
            "l2_max": float((xa - xs[ev]).reshape(len(xa), -1).norm(dim=1).max()),
            "linf_mean": float((xa - xs[ev]).abs().amax(dim=(1, 2, 3)).mean()),
            "linf_max": float((xa - xs[ev]).abs().amax(dim=(1, 2, 3)).max()),
            "config": cwcfg,
        })
        print(f"[{model}] CW c={best['c']} ASR={asr:.3f}", flush=True)
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
