"""Identity-paired summary statistics.

Palmprint images from the same identity are not independent experimental
units. For confidence intervals we first average within each identity, then
bootstrap over identities.
"""
import numpy as np


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def group_means(values, groups):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        out[g] = float(values[groups == g].mean())
    return out


def paired_summary(values, groups, n_boot=2000, seed=0):
    """Mean/std/CI over per-identity means."""
    per_group = group_means(values, groups)
    arr = np.array(list(per_group.values()), dtype=np.float64)
    mean, std = mean_std(arr)
    lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed)
    return {
        "n_identities": int(arr.size),
        "paired_mean": mean,
        "paired_std": std,
        "paired_ci_low": lo,
        "paired_ci_high": hi,
    }


def independent_summary(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    mean, std = mean_std(values)
    lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return {
        "n": int(values.size),
        "mean": mean,
        "std": std,
        "ci_low": lo,
        "ci_high": hi,
    }
