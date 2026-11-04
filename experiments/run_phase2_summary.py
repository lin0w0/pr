"""Consolidate Phase 2 summary CSVs into results/phase2/final_summary.csv."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE2 = ROOT / "results" / "phase2"
MODELS = ("compnet", "ccnet")


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.is_file() else []


def add(out, experiment, model, setting, metric, value, lo=None, hi=None):
    out.append({
        "experiment": experiment,
        "model": model,
        "setting": setting,
        "metric": metric,
        "value": value,
        "ci_low": "" if lo is None else lo,
        "ci_high": "" if hi is None else hi,
    })


def main():
    out = []
    for m in MODELS:
        for r in rows(PHASE2 / "attenuation" / m / "summary.csv"):
            setting = f"{r['condition']}|sigma={r['sigma']}"
            add(out, "attenuation", m, setting, "consistency_paired_mean",
                r["paired_mean"], r["paired_ci_low"], r["paired_ci_high"])
        for r in rows(PHASE2 / "sgm" / m / "summary.csv"):
            setting = f"{r['condition']}|{r['loss']}|scale={r['scale']}"
            add(out, "sgm", m, setting, "grad_l1_mean", r["grad_l1_mean"])
            add(out, "sgm", m, setting, "near_zero_rel",
                r["near_zero_rel_lt_1pct_model_max"])
            for b in ("low", "middle", "high"):
                add(out, "sgm", m, setting, f"band_ratio_{b}",
                    r[f"band_ratio_{b}_aggregate"],
                    r.get(f"band_ratio_{b}_paired_ci_low"),
                    r.get(f"band_ratio_{b}_paired_ci_high"))
        for r in rows(PHASE2 / "band_sensitivity" / m / "summary.csv"):
            add(out, "band_sensitivity", m, r["band"], "feature_shift_mean",
                r["feature_shift_mean"], r["paired_ci_low"], r["paired_ci_high"])
            add(out, "band_sensitivity", m, r["band"], "rank1_pct",
                r["rank1_pct"], r.get("rank1_paired_ci_low_pct"),
                r.get("rank1_paired_ci_high_pct"))
            add(out, "band_sensitivity", m, r["band"], "identification_error_pct",
                r["identification_error_pct"])
        for r in rows(PHASE2 / "band_attacks" / m / "summary.csv"):
            setting = f"{r['band']}|{r['objective']}"
            add(out, "band_attacks", m, setting, "asr_clean_correct_pct",
                r["asr_clean_correct_pct"], r.get("asr_paired_ci_low_pct"),
                r.get("asr_paired_ci_high_pct"))
            add(out, "band_attacks", m, setting, "rank1_pct", r["rank1_pct"],
                r.get("rank1_paired_ci_low_pct"), r.get("rank1_paired_ci_high_pct"))
            add(out, "band_attacks", m, setting, "feature_shift_mean",
                r["feature_shift_mean"], r["paired_ci_low"], r["paired_ci_high"])

    path = PHASE2 / "final_summary.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
