"""Phase 3 preliminary spectral-consistency detection (Task 3).

Validation is split by identity:
  calibration identities 0..99   -> choose sigma / multi-scale method / thresholds
  evaluation identities 100..599 -> held-out detection evaluation

Score = cosine(f(x), f(LPF(x))); lower means more adversarial.
Thresholds are lower-tail quantiles of clean calibration scores.
The test split is never read.
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.detection_metrics import (  # noqa: E402
    auc_lower_is_adv,
    rates_at_threshold,
    roc_lower_is_adv,
    threshold_at_fpr,
)
from experiments._common import save_csv, save_json  # noqa: E402

PHASE2 = ROOT / "results" / "phase2"
OUT = ROOT / "results" / "phase3_detection_preliminary"
SIGMAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
CAL_MAX_ID = 100
CONDITIONS = ("clean", "pgd", "apgd")


def load_attenuation(model):
    path = PHASE2 / "attenuation" / model / "per_sample.csv"
    data = list(csv.DictReader(open(path, encoding="utf-8")))
    labels, scores = {}, {}
    for r in data:
        idx = int(r["sample_index"])
        labels[idx] = int(r["label"])
        scores.setdefault(r["condition"], {}).setdefault(float(r["sigma"]), {})[idx] = float(r["consistency"])
    indices = np.array(sorted(labels))
    return labels, scores, indices


def matrix(scores, condition, sigmas, indices):
    return np.stack([np.array([scores[condition][float(s)][i] for i in indices])
                     for s in sigmas], axis=1)


def evaluate_threshold(clean_cal, adv_cal, clean_eval, adv_eval, target_fpr):
    tau = threshold_at_fpr(clean_cal, target_fpr)
    return {
        "target_fpr": target_fpr,
        "threshold": tau,
        "calibration": rates_at_threshold(clean_cal, adv_cal, tau),
        "evaluation": rates_at_threshold(clean_eval, adv_eval, tau),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sgm_rows, multi_rows, score_rows = [], [], []
    thresholds = {}
    for model in ("compnet", "ccnet"):
        labels, scores, indices = load_attenuation(model)
        cal_idx = np.array([i for i in indices if labels[i] < CAL_MAX_ID])
        eval_idx = np.array([i for i in indices if labels[i] >= CAL_MAX_ID])
        print(f"[{model}] cal={len(cal_idx)} eval={len(eval_idx)}", flush=True)

        # ---------------- single scale ----------------
        single = {}
        for sigma in SIGMAS:
            ccal = scores["clean"][float(sigma)]
            pcal = scores["pgd"][float(sigma)]
            acal = scores["apgd"][float(sigma)]
            clean_cal = np.array([ccal[i] for i in cal_idx])
            adv_cal = np.array([pcal[i] for i in cal_idx] + [acal[i] for i in cal_idx])
            clean_eval = np.array([ccal[i] for i in eval_idx])
            adv_eval = np.array([pcal[i] for i in eval_idx] + [acal[i] for i in eval_idx])
            auc_cal = auc_lower_is_adv(clean_cal, adv_cal)
            auc_eval = auc_lower_is_adv(clean_eval, adv_eval)
            r1 = evaluate_threshold(clean_cal, adv_cal, clean_eval, adv_eval, 0.01)
            r5 = evaluate_threshold(clean_cal, adv_cal, clean_eval, adv_eval, 0.05)
            row = {
                "model": model, "sigma": sigma,
                "cal_auc": auc_cal, "eval_auc": auc_eval,
                "cal_tpr_at_fpr1": r1["calibration"]["TPR"],
                "cal_tpr_at_fpr5": r5["calibration"]["TPR"],
                "eval_tpr_at_fpr1": r1["evaluation"]["TPR"],
                "eval_fpr_at_fpr1": r1["evaluation"]["FPR"],
                "eval_tpr_at_fpr5": r5["evaluation"]["TPR"],
                "eval_fpr_at_fpr5": r5["evaluation"]["FPR"],
                "threshold_fpr1": r1["threshold"],
                "threshold_fpr5": r5["threshold"],
                "eval_tpr_pgd_at_fpr1": rates_at_threshold(
                    clean_eval, np.array([scores["pgd"][float(sigma)][i] for i in eval_idx]),
                    r1["threshold"])["TPR"],
                "eval_tpr_apgd_at_fpr1": rates_at_threshold(
                    clean_eval, np.array([scores["apgd"][float(sigma)][i] for i in eval_idx]),
                    r1["threshold"])["TPR"],
            }
            sgm_rows.append(row)
            single[sigma] = row
        best_sigma = max(SIGMAS, key=lambda s: (single[s]["cal_auc"], single[s]["cal_tpr_at_fpr1"]))

        # ---------------- multi scale ----------------
        combos = {
            "mean_all": ("mean", SIGMAS),
            "mean_0.05_0.1_0.2": ("mean", [0.05, 0.1, 0.2]),
            "mean_0.1_0.15_0.2": ("mean", [0.1, 0.15, 0.2]),
            "min_all": ("min", SIGMAS),
            "min_0.05_0.1_0.2": ("min", [0.05, 0.1, 0.2]),
        }
        multi = {}
        for name, (method, sigset) in combos.items():
            mats = {c: matrix(scores, c, sigset, indices) for c in CONDITIONS}
            score_by_cond = {}
            for c in CONDITIONS:
                arr = mats[c]
                s = arr.mean(axis=1) if method == "mean" else arr.min(axis=1)
                score_by_cond[c] = {int(indices[k]): float(s[k]) for k in range(len(indices))}
            clean_cal = np.array([score_by_cond["clean"][i] for i in cal_idx])
            pgd_cal = np.array([score_by_cond["pgd"][i] for i in cal_idx])
            apgd_cal = np.array([score_by_cond["apgd"][i] for i in cal_idx])
            adv_cal = np.r_[pgd_cal, apgd_cal]
            clean_eval = np.array([score_by_cond["clean"][i] for i in eval_idx])
            pgd_eval = np.array([score_by_cond["pgd"][i] for i in eval_idx])
            apgd_eval = np.array([score_by_cond["apgd"][i] for i in eval_idx])
            adv_eval = np.r_[pgd_eval, apgd_eval]
            auc_cal = auc_lower_is_adv(clean_cal, adv_cal)
            auc_eval = auc_lower_is_adv(clean_eval, adv_eval)
            r1 = evaluate_threshold(clean_cal, adv_cal, clean_eval, adv_eval, 0.01)
            r5 = evaluate_threshold(clean_cal, adv_cal, clean_eval, adv_eval, 0.05)
            row = {
                "model": model, "method": method, "name": name,
                "sigmas": sigset,
                "cal_auc": auc_cal, "eval_auc": auc_eval,
                "cal_tpr_at_fpr1": r1["calibration"]["TPR"],
                "cal_tpr_at_fpr5": r5["calibration"]["TPR"],
                "eval_tpr_at_fpr1": r1["evaluation"]["TPR"],
                "eval_fpr_at_fpr1": r1["evaluation"]["FPR"],
                "eval_tpr_at_fpr5": r5["evaluation"]["TPR"],
                "eval_fpr_at_fpr5": r5["evaluation"]["FPR"],
                "threshold_fpr1": r1["threshold"],
                "threshold_fpr5": r5["threshold"],
                "eval_tpr_pgd_at_fpr1": rates_at_threshold(clean_eval, pgd_eval, r1["threshold"])["TPR"],
                "eval_tpr_apgd_at_fpr1": rates_at_threshold(clean_eval, apgd_eval, r1["threshold"])["TPR"],
            }
            multi_rows.append(row)
            multi[name] = row
        best_multi = max(combos, key=lambda n: (multi[n]["cal_auc"], multi[n]["cal_tpr_at_fpr1"]))

        # ---------------- chosen-configuration scores and plots ----------------
        single_clean_eval = np.array([scores["clean"][float(best_sigma)][i] for i in eval_idx])
        single_adv_eval = np.r_[
            np.array([scores["pgd"][float(best_sigma)][i] for i in eval_idx]),
            np.array([scores["apgd"][float(best_sigma)][i] for i in eval_idx]),
        ]
        method, sigset = combos[best_multi]
        mats = {c: matrix(scores, c, sigset, indices) for c in CONDITIONS}
        ms = {}
        for c in CONDITIONS:
            arr = mats[c]
            s = arr.mean(axis=1) if method == "mean" else arr.min(axis=1)
            ms[c] = {int(indices[k]): float(s[k]) for k in range(len(indices))}
        multi_clean_eval = np.array([ms["clean"][i] for i in eval_idx])
        multi_adv_eval = np.r_[
            np.array([ms["pgd"][i] for i in eval_idx]),
            np.array([ms["apgd"][i] for i in eval_idx]),
        ]
        for split, idxs in (("calibration", cal_idx), ("evaluation", eval_idx)):
            for c in CONDITIONS:
                for i in idxs:
                    score_rows.append({
                        "model": model, "split": split, "condition": c,
                        "sample_index": int(i), "label": labels[i],
                        "single_score": scores[c][float(best_sigma)][i],
                        "multi_score": ms[c][i],
                    })
        thresholds[model] = {
            "single": {"sigma": best_sigma, **single[best_sigma]},
            "multi": {"name": best_multi, "method": method, "sigmas": sigset,
                      **multi[best_multi]},
            "calibration_identity_max": CAL_MAX_ID - 1,
            "evaluation_identity_min": CAL_MAX_ID,
            "score_orientation": "lower = more adversarial; detect if score <= tau",
        }

        # ROC + score distribution
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for name, clean, adv in (
            (f"single sigma={best_sigma:g}", single_clean_eval, single_adv_eval),
            (f"multi {best_multi}", multi_clean_eval, multi_adv_eval),
        ):
            fpr, tpr, _ = roc_lower_is_adv(clean, adv)
            axes[0].plot(fpr, tpr, label=f"{name} AUC={auc_lower_is_adv(clean, adv):.4f}")
        axes[0].set_xlim(0, 0.2)
        axes[0].set_ylim(0.8, 1.005)
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title(f"{model} detection ROC (eval)")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        for c, color in (("clean", "tab:blue"), ("pgd", "tab:orange"), ("apgd", "tab:red")):
            arr = np.array([ms[c][i] for i in eval_idx])
            axes[1].hist(arr, bins=60, alpha=0.55, density=True, label=c, color=color)
        axes[1].set_xlabel(f"multi-scale consistency ({best_multi})")
        axes[1].set_ylabel("density")
        axes[1].set_title(f"{model} score distribution (eval)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"detection_{model}.png", dpi=160)
        plt.close(fig)

    save_csv(OUT / "single_scale_analysis.csv", sgm_rows)
    save_csv(OUT / "multiscale_analysis.csv", multi_rows)
    save_csv(OUT / "detection_scores.csv", score_rows)
    save_json(OUT / "thresholds_and_summary.json", thresholds)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
