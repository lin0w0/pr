"""A4: adaptive attack against spectral-consistency detection."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attacks.adaptive_detection import adaptive_pgd
from defenses.detection import detector_scores
from experiments._common import build_context, get_adversarial, predict_labels, save_csv, save_json
from utils.config import load_config

OUT = ROOT / "results" / "phase3_final"


def det_scores(adapter, x, scfg, model, det, batch=32):
    if det == "single":
        sig = [float(scfg["detection"]["single_sigma"][model])]
        method = "mean"
    else:
        sig = [float(s) for s in scfg["detection"]["multi_sigmas"]]
        method = scfg["detection"]["multi_method"]
    out = []
    for start in range(0, len(x), batch):
        out.append(detector_scores(adapter, x[start:start + batch], sig, method).cpu())
    return torch.cat(out).numpy(), sig, method


def run_adapt(adapter, classifier, x, y, init, scfg, model, det, lam, acfg, batch):
    sig, method = (([float(scfg["detection"]["single_sigma"][model])], "mean")
                   if det == "single" else
                   ([float(s) for s in scfg["detection"]["multi_sigmas"]],
                    scfg["detection"]["multi_method"]))
    out = []
    for start in range(0, len(x), batch):
        out.append(adaptive_pgd(
            adapter, classifier, x[start:start + batch], y[start:start + batch],
            init[start:start + batch], sig, method, lam,
            acfg["eps"], acfg["alpha"], acfg["steps"], scale=10.0,
        ).cpu())
    return torch.cat(out)


def metrics(adapter, classifier, x, y, xa, tau, scfg, model, det):
    with torch.no_grad():
        clean_pred = predict_labels(adapter, classifier, x,
                                    batch_size=int(adapter.config["batch"]["inference"])
                                    if isinstance(adapter.config, dict) else 64)
        pred = predict_labels(adapter, classifier, xa,
                              batch_size=int(adapter.config["batch"]["inference"])
                              if isinstance(adapter.config, dict) else 64)
    cc = (clean_pred == y).numpy()
    success = cc & (pred != y).numpy()
    scores, _, _ = det_scores(adapter, xa, scfg, model, det)
    detected = (scores <= tau)
    escape = ~detected
    n_cc = max(int(cc.sum()), 1)
    return {
        "clean_correct": int(cc.sum()),
        "success": int(success.sum()),
        "asr": float(success.sum() / n_cc),
        "detection_rate": float(detected.mean()),
        "escape_rate": float(escape.mean()),
        "asr_evade": float((success & escape).sum() / n_cc),
        "asr_given_escape": float((success & escape).sum() / max(int(escape.sum()), 1)),
        "detected_success_rate": float((success & detected).sum() / n_cc),
        "unconditional_error": float((pred != y).float().mean()),
        "scores": scores,
        "success_mask": success,
        "escape_mask": escape,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config("configs/phase3_final.yaml")
    thresholds = json.load(open(ROOT / cfg["detection"]["threshold_source"], encoding="utf-8"))
    acfg = cfg["attacks"]["adaptive"]
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    OUT.mkdir(parents=True, exist_ok=True)
    for model in models:
        ctx = build_context(model, device=args.device, max_ids=None)
        xs, ys = ctx["xs"], ctx["ys"]
        x_pgd, pgd_meta = get_adversarial(ctx, "pgd")
        cal = torch.nonzero(ys < int(acfg["calibration_identity_end"]) + 1).flatten()
        ev = torch.nonzero((ys >= int(acfg["evaluation_identity_start"])) &
                           (ys <= int(acfg["evaluation_identity_end"]))).flatten()
        batch = 16 if model == "ccnet" else 32
        summary, rows = [], []
        for det in ("single", "multi"):
            # calibration: choose lambda maximizing ASR that also evades
            best = None
            cal_results = []
            for lam in acfg["lambdas"]:
                xa = run_adapt(ctx["adapter"], ctx["classifier"], xs[cal], ys[cal],
                               x_pgd[cal], cfg, model, det, lam, acfg, batch)
                tau = thresholds[model][det]["threshold_fpr1"]
                m = metrics(ctx["adapter"], ctx["classifier"], xs[cal], ys[cal], xa,
                            tau, cfg, model, det)
                cal_results.append({"lambda": lam, "cal_asr": m["asr"],
                                    "cal_asr_evade": m["asr_evade"],
                                    "cal_detection_rate": m["detection_rate"]})
                if best is None or (m["asr_evade"], m["asr"]) > (best["cal_asr_evade"], best["cal_asr"]):
                    best = cal_results[-1]
            # eval chosen lambda
            lam = best["lambda"]
            xa = run_adapt(ctx["adapter"], ctx["classifier"], xs[ev], ys[ev],
                           x_pgd[ev], cfg, model, det, lam, acfg, batch)
            for target in (1, 5):
                tau = thresholds[model][det][f"threshold_fpr{target}"]
                m = metrics(ctx["adapter"], ctx["classifier"], xs[ev], ys[ev], xa,
                            tau, cfg, model, det)
                # original PGD metrics at same tau
                mo = metrics(ctx["adapter"], ctx["classifier"], xs[ev], ys[ev],
                             x_pgd[ev], tau, cfg, model, det)
                summary.append({
                    "model": model, "detector": det, "target_fpr_pct": target,
                    "lambda": lam, "tau": tau,
                    "original_asr": mo["asr"],
                    "adaptive_asr": m["asr"],
                    "detection_rate": m["detection_rate"],
                    "escape_rate": m["escape_rate"],
                    "asr_evade": m["asr_evade"],
                    "asr_given_escape": m["asr_given_escape"],
                    "detected_success_rate": m["detected_success_rate"],
                    "clean_correct": m["clean_correct"],
                    "calibration": [r for r in cal_results if r["lambda"] == lam][0],
                })
                for j in range(len(ev)):
                    rows.append({
                        "model": model, "detector": det, "target_fpr_pct": target,
                        "lambda": lam, "sample_index": int(ev[j]),
                        "label": int(ys[ev[j]]),
                        "adaptive_success": bool(m["success_mask"][j]),
                        "detected": bool(not m["escape_mask"][j]),
                        "score": float(m["scores"][j]),
                        "l2": float((xa[j] - xs[ev[j]]).pow(2).sum().sqrt()),
                        "linf": float((xa[j] - xs[ev[j]]).abs().max()),
                    })
            print(f"[{model}] {det} lambda={lam} ASR_evade={summary[-2]['asr_evade']:.3f}", flush=True)
        save_csv(OUT / f"a4_adaptive_summary_{model}.csv", summary)
        save_csv(OUT / f"a4_adaptive_per_sample_{model}.csv", rows)
        save_json(OUT / f"a4_adaptive_meta_{model}.json", {
            "model": model, "cfg": acfg, "n_eval": int(len(ev)),
            "n_cal": int(len(cal)), "thresholds_source": cfg["detection"]["threshold_source"],
        })
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
