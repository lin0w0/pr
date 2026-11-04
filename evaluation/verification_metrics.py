"""1:1 verification metrics (EER, TAR@FAR, threshold calibration).

Score convention: **higher = more likely genuine**.

EER is computed from the *complete* ROC over every distinct observed score
(genuine union impostor), not only thresholds taken from genuine scores. At
equal scores the acceptance rule is ``score >= threshold``, matching
:func:`at_threshold` and the official ``getEER.py`` score convention.

The ROC traversal is streaming: only the sorted score arrays and the union of
distinct thresholds are materialised. This is required for the Tongji
validation/test sets, where a 3000 x 6000 comparison yields 18M impostor scores
and a fully materialised FPR/TPR array would need hundreds of MB.

Primary EER = interpolated FAR/FRR crossing. The closest discrete operating
point is also reported because the official ``getEER.py`` prints both a
brentq-interpolated EER and a discrete ``eer_1/2``.
"""
import numpy as np
import torch
import torch.nn.functional as F

_CHUNK = 1_000_000
_trapz = getattr(np, "trapezoid", np.trapz)


def pairwise_scores(features_a, features_b, labels_a, labels_b, chunk_size=512):
    """Return (genuine, impostor) cosine score arrays for all A x B pairs.

    Computation is chunked so 6000 x 6000 does not require one huge matrix.
    """
    fa = F.normalize(torch.as_tensor(features_a, dtype=torch.float32), dim=1)
    fb = F.normalize(torch.as_tensor(features_b, dtype=torch.float32), dim=1)
    la = np.asarray(labels_a, dtype=np.int64)
    lb = np.asarray(labels_b, dtype=np.int64)
    if len(fa) != len(la) or len(fb) != len(lb):
        raise ValueError("feature/label length mismatch")

    genuine, impostor = [], []
    for start in range(0, len(fa), chunk_size):
        sims = (fa[start:start + chunk_size] @ fb.T).numpy()
        same = la[start:start + chunk_size, None] == lb[None, :]
        genuine.append(sims[same].astype(np.float32))
        impostor.append(sims[~same].astype(np.float32))
    return np.concatenate(genuine), np.concatenate(impostor)


def _counts_ge(sorted_arr, thresholds):
    """Fraction of ``sorted_arr`` values >= each threshold (chunked)."""
    n = len(sorted_arr)
    out = np.empty(len(thresholds), dtype=np.float64)
    for start in range(0, len(thresholds), _CHUNK):
        chunk = thresholds[start:start + _CHUNK]
        c = np.searchsorted(sorted_arr, chunk, side="left")
        out[start:start + len(chunk)] = 1.0 - c.astype(np.float64) / n
    return out


def roc_curve(genuine, impostor):
    """Full ROC as arrays (for tests/plots; EER uses the streaming version).

    Returns ``(fpr, tpr, thresholds)`` in sklearn order: thresholds
    descending, FPR ascending. Sentinel points ``(+inf, 0, 0)`` and
    ``(-inf, 1, 1)`` are included.
    """
    g = np.sort(np.asarray(genuine, dtype=np.float64))
    i = np.sort(np.asarray(impostor, dtype=np.float64))
    if g.size == 0 or i.size == 0:
        raise ValueError("genuine and impostor score arrays must be non-empty")
    thresholds = np.union1d(g, i)
    tpr = _counts_ge(g, thresholds)
    fpr = _counts_ge(i, thresholds)
    thresholds_desc = np.concatenate(([np.inf], thresholds[::-1], [-np.inf]))
    fpr_desc = np.concatenate(([0.0], fpr[::-1], [1.0]))
    tpr_desc = np.concatenate(([0.0], tpr[::-1], [1.0]))
    return fpr_desc, tpr_desc, thresholds_desc


def _interpolate_crossing(fpr0, fnr0, thr0, fpr1, fnr1, thr1):
    denom = (fpr1 - fpr0) - (fnr1 - fnr0)
    u = 0.5 if abs(denom) < 1e-15 else (fnr0 - fpr0) / denom
    u = float(np.clip(u, 0.0, 1.0))
    fpr_x = fpr0 + u * (fpr1 - fpr0)
    fnr_x = fnr0 + u * (fnr1 - fnr0)
    if np.isfinite(thr0) and np.isfinite(thr1):
        thr_x = float(thr0 + u * (thr1 - thr0))
    else:
        thr_x = None
    return 0.5 * (fpr_x + fnr_x), thr_x


def _stream_roc(genuine, impostor, tol=1e-12):
    """Single-pass complete-ROC traversal.

    Returns EER/AUC/TAR@FAR without materialising per-threshold arrays.
    Thresholds are visited in descending order; FPR is non-decreasing and TPR
    non-increasing along that traversal.
    """
    g = np.sort(np.asarray(genuine, dtype=np.float32))
    i = np.sort(np.asarray(impostor, dtype=np.float32))
    if g.size == 0 or i.size == 0:
        raise ValueError("genuine and impostor score arrays must be non-empty")

    thresholds = np.union1d(g, i)  # ascending, unique
    prev_fpr, prev_tpr, prev_thr = 0.0, 0.0, np.inf
    auc = 0.0
    seen_negative = False
    eer = eer_thr = None
    mode = "closest"
    disc_d = np.inf
    disc_fpr = disc_fnr = 0.0
    disc_thr = None
    tar_1 = 0.0
    tar_01 = 0.0
    n_points = 0

    end = len(thresholds)
    while end > 0:
        start = max(0, end - _CHUNK)
        t = thresholds[start:end][::-1]  # descending
        tpr_chunk = _counts_ge(g, t)
        fpr_chunk = _counts_ge(i, t)
        fnr_chunk = 1.0 - tpr_chunk
        d_chunk = fpr_chunk - fnr_chunk

        n = len(t)
        prev_fprs = np.empty(n, dtype=np.float64)
        prev_tprs = np.empty(n, dtype=np.float64)
        prev_thrs = np.empty(n, dtype=np.float64)
        prev_fprs[0], prev_tprs[0], prev_thrs[0] = prev_fpr, prev_tpr, prev_thr
        if n > 1:
            prev_fprs[1:] = fpr_chunk[:-1]
            prev_tprs[1:] = tpr_chunk[:-1]
            prev_thrs[1:] = t[:-1]

        # AUC over this chunk (FPR is non-decreasing along descending thresholds)
        auc += float(np.sum(0.5 * (tpr_chunk + prev_tprs) * (fpr_chunk - prev_fprs)))

        # TAR@FAR
        mask1 = fpr_chunk <= 0.01
        if mask1.any():
            tar_1 = max(tar_1, float(tpr_chunk[mask1].max()))
        mask01 = fpr_chunk <= 0.001
        if mask01.any():
            tar_01 = max(tar_01, float(tpr_chunk[mask01].max()))

        # closest discrete operating point
        jd = int(np.argmin(np.abs(d_chunk)))
        if abs(d_chunk[jd]) < disc_d:
            disc_d = float(abs(d_chunk[jd]))
            disc_fpr, disc_fnr = float(fpr_chunk[jd]), float(fnr_chunk[jd])
            disc_thr = float(t[jd]) if np.isfinite(t[jd]) else None

        # EER crossing, first one in descending-threshold order wins
        if eer is None:
            zeros = np.flatnonzero(np.abs(d_chunk) <= tol)
            if zeros.size:
                j = int(zeros[0])
                eer = float(fpr_chunk[j])
                eer_thr = float(t[j]) if np.isfinite(t[j]) else None
                mode = "exact"
            else:
                neg = (np.cumsum(d_chunk < -tol) > 0)
                if seen_negative:
                    neg = np.ones(n, dtype=bool)
                cand = np.flatnonzero(neg & (d_chunk > tol))
                if cand.size:
                    j = int(cand[0])
                    eer, eer_thr = _interpolate_crossing(
                        float(prev_fprs[j]), float(1.0 - prev_tprs[j]), float(prev_thrs[j]),
                        float(fpr_chunk[j]), float(fnr_chunk[j]), float(t[j]),
                    )
                    mode = "interpolated"
                if np.any(d_chunk < -tol):
                    seen_negative = True

        prev_fpr = float(fpr_chunk[-1])
        prev_tpr = float(tpr_chunk[-1])
        prev_thr = float(t[-1])
        n_points += n
        end = start

    # sentinel (FPR=1, TPR=1)
    auc += 0.5 * (1.0 + prev_tpr) * (1.0 - prev_fpr)

    if eer is None:
        if n_points == 0:
            eer, mode = 0.5, "degenerate"
        else:
            eer = 0.5 * (disc_fpr + disc_fnr)

    return {
        "eer_pct": round(float(eer) * 100.0, 6),
        "eer_threshold": eer_thr,
        "eer_mode": mode,
        "eer_discrete_pct": round(0.5 * (disc_fpr + disc_fnr) * 100.0, 6),
        "eer_discrete_threshold": disc_thr,
        "auc": round(float(auc), 8),
        "tar_at_far_1pct": round(float(tar_1) * 100.0, 6),
        "tar_at_far_0.1pct": round(float(tar_01) * 100.0, 6),
        "n_thresholds": int(len(thresholds)),
    }


def equal_error_rate(genuine, impostor):
    """EER from the complete ROC (streaming, no per-threshold arrays)."""
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("genuine and impostor score arrays must be non-empty")
    out = _stream_roc(genuine, impostor)
    out.pop("tar_at_far_1pct")
    out.pop("tar_at_far_0.1pct")
    out.update({
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "score_min": float(min(genuine.min(), impostor.min())),
        "score_max": float(max(genuine.max(), impostor.max())),
    })
    return out


def verification_metrics(genuine, impostor):
    """Full verification report using one streaming ROC pass.

    ``tar_at_far_*`` are **test-set/oracle** values (threshold chosen from the
    same scores). For a validation-calibrated operating point use
    :func:`threshold_at_far` + :func:`evaluate_at_threshold`.
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("genuine and impostor score arrays must be non-empty")
    out = _stream_roc(genuine, impostor)
    out.update({
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "score_min": float(min(genuine.min(), impostor.min())),
        "score_max": float(max(genuine.max(), impostor.max())),
        "tar_at_far_1pct_is_test_oracle": True,
        "tar_at_far_0.1pct_is_test_oracle": True,
    })
    return out


def at_threshold(genuine, impostor, threshold):
    """FAR/FRR/TAR/TNR at an explicit cosine threshold (accept if score >= t)."""
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    tar = float((genuine >= threshold).mean())
    far = float((impostor >= threshold).mean())
    return {
        "threshold": float(threshold),
        "TAR": tar,
        "FRR": 1.0 - tar,
        "FAR": far,
        "TNR": 1.0 - far,
    }


def threshold_at_far(genuine, impostor, target_far):
    """Calibrate a threshold on **validation** scores for a target FAR.

    Chooses the smallest threshold whose validation FAR is <= ``target_far``,
    maximizing TAR subject to the FAR budget. ``score >= threshold`` counts as
    acceptance, so tied scores are handled consistently with the EER code.
    """
    if not (0.0 <= target_far <= 1.0):
        raise ValueError("target_far must be in [0, 1]")
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("genuine and impostor score arrays must be non-empty")

    m = int(impostor.size)
    k = int(np.floor(target_far * m))
    if k >= m:
        thr = -np.inf
    else:
        imp32 = impostor.astype(np.float32)
        u, counts = np.unique(np.sort(imp32), return_counts=True)
        n_ge = m - np.concatenate(([0], np.cumsum(counts)[:-1]))
        valid = np.flatnonzero(n_ge <= k)
        if valid.size:
            thr = float(u[int(valid[0])])
        else:
            thr = float(np.nextafter(np.max(impostor), np.inf))
    cal = at_threshold(genuine, impostor, thr)
    return {
        "threshold": thr,
        "target_far": float(target_far),
        "calibration_far": cal["FAR"],
        "calibration_tar": cal["TAR"],
        "calibration_frr": cal["FRR"],
        "calibration_tnr": cal["TNR"],
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
    }


def evaluate_at_threshold(genuine, impostor, threshold):
    """Apply a previously calibrated threshold to a held-out score set."""
    return at_threshold(genuine, impostor, threshold)
