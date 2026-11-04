"""Spectral-consistency detection metrics.

Score convention: **lower = more adversarial** (consistency drops under
low-pass filtering). Thresholds are calibrated on clean validation scores:
``detect if score <= tau``.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def threshold_at_fpr(clean_scores, target_fpr):
    """Choose tau on clean validation scores so that FPR <= target."""
    clean = np.asarray(clean_scores, dtype=np.float64)
    if clean.size == 0:
        raise ValueError("empty clean scores")
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError("target_fpr must be in [0,1]")
    if target_fpr == 0.0:
        return float(np.min(clean) - 1e-12)
    # Lower-tail quantile: at most floor(target*n) clean scores below tau.
    tau = float(np.quantile(clean, target_fpr, method="lower"))
    return tau


def rates_at_threshold(clean_scores, adv_scores, tau):
    clean = np.asarray(clean_scores, dtype=np.float64)
    adv = np.asarray(adv_scores, dtype=np.float64)
    tpr = float((adv <= tau).mean()) if adv.size else float("nan")
    fpr = float((clean <= tau).mean()) if clean.size else float("nan")
    return {
        "threshold": float(tau),
        "TPR": tpr,
        "FPR": fpr,
        "TNR": 1.0 - fpr,
        "FNR": 1.0 - tpr,
        "n_clean": int(clean.size),
        "n_adv": int(adv.size),
    }


def tpr_at_fpr(clean_scores, adv_scores, target_fpr):
    tau = threshold_at_fpr(clean_scores, target_fpr)
    return rates_at_threshold(clean_scores, adv_scores, tau)


def auc_lower_is_adv(clean_scores, adv_scores):
    clean = np.asarray(clean_scores, dtype=np.float64)
    adv = np.asarray(adv_scores, dtype=np.float64)
    y = np.r_[np.zeros(clean.size), np.ones(adv.size)]
    s = -np.r_[clean, adv]
    return float(roc_auc_score(y, s))


def roc_lower_is_adv(clean_scores, adv_scores):
    clean = np.asarray(clean_scores, dtype=np.float64)
    adv = np.asarray(adv_scores, dtype=np.float64)
    y = np.r_[np.zeros(clean.size), np.ones(adv.size)]
    s = -np.r_[clean, adv]
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def multiscale_score(consistency, method="mean"):
    """Combine an ``[N, S]`` consistency matrix into ``[N]``.

    mean: average consistency over scales;
    min: worst (lowest) consistency over scales.
    """
    arr = np.asarray(consistency, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("consistency must be [N, n_scales]")
    if method == "mean":
        return arr.mean(axis=1)
    if method == "min":
        return arr.min(axis=1)
    raise ValueError(f"unknown multiscale method {method!r}")
