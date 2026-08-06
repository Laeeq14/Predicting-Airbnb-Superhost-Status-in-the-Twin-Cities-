"""
monitoring/drift_report.py
==========================
Evidently AI batch data drift monitor for the Superhost Predictor.

Strategy
--------
The model's training distribution is captured in ``ml_pipeline/model_metadata.json``
(``feature_defaults``, ``superhost_avg``, ``non_superhost_avg``).  These medians are
the ground-truth reference for what "healthy" inference data looks like.

We reconstruct a synthetic **reference DataFrame** from those stored statistics
(N=200 rows — enough for Evidently's statistical tests to be stable) without
needing the original 80MB CSVs, making this CI-friendly and fully reproducible.

For local production use the function also accepts an optional *current_df* argument
so you can pass live inference logs (e.g. from a request-logging middleware) and
get a real drift signal.

Outputs
-------
- ``monitoring/drift_report.html``    — full interactive Evidently HTML report
- ``monitoring/drift_summary.json``   — machine-readable summary for the API

CLI
---
    python -m monitoring.drift_report
    python monitoring/drift_report.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent          # project root
META_PATH    = BASE_DIR / "ml_pipeline" / "model_metadata.json"
OUT_DIR      = Path(__file__).parent                 # monitoring/
REPORT_HTML  = OUT_DIR / "drift_report.html"
SUMMARY_JSON = OUT_DIR / "drift_summary.json"

# Features we monitor for drift (the 10 most important/interpretable ones)
MONITOR_FEATURES = [
    "review_scores_rating",
    "reviews_per_month",
    "host_response_rate",
    "host_acceptance_rate",
    "host_listings_count",
    "host_experience_years",
    "num_amenities",
    "number_of_reviews",
    "review_scores_cleanliness",
    "review_scores_communication",
]

N_REFERENCE = 200   # synthetic reference rows
N_CURRENT   = 100   # synthetic current rows (used when no live data is passed)


# ── Reference data builder ────────────────────────────────────────────────────

def _load_metadata() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"model_metadata.json not found at {META_PATH}. "
            "Run ml_pipeline/train_model.py first."
        )
    with open(META_PATH) as f:
        return json.load(f)


def _synthetic_dataframe(
    feature_defaults: dict,
    superhost_avg: dict,
    non_superhost_avg: dict,
    n: int,
    seed: int,
    noise_scale: float = 0.05,
) -> pd.DataFrame:
    """
    Build a synthetic DataFrame by sampling around the training medians.

    For each row we randomly pick superhost (40%) vs. non-superhost (60%)
    to match the approximate training base rate, then add small Gaussian
    noise so Evidently's statistical tests have real variance to work with.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        is_sh = rng.random() < 0.40
        avgs  = superhost_avg if is_sh else non_superhost_avg

        row: dict = {}
        for feat in MONITOR_FEATURES:
            center = avgs.get(feat, feature_defaults.get(feat, 0.0))
            if center is None:
                center = 0.0
            # Add proportional Gaussian noise (floored at 0 for rate/score cols)
            noise  = rng.normal(0, max(abs(center) * noise_scale, 0.01))
            row[feat] = max(0.0, float(center) + noise)

        row["host_is_superhost"] = int(is_sh)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Main report generator ─────────────────────────────────────────────────────

def generate_drift_report(
    current_df: pd.DataFrame | None = None,
    save_html: bool = True,
    save_json: bool = True,
) -> dict:
    """
    Generate an Evidently drift report.

    Parameters
    ----------
    current_df:
        Optional live inference DataFrame with the same columns as MONITOR_FEATURES.
        When None (e.g. in CI or first run), a synthetic current dataset is built
        with slight distribution shift to produce a meaningful demo report.
    save_html:
        Write the full interactive report to monitoring/drift_report.html.
    save_json:
        Write a machine-readable summary to monitoring/drift_summary.json.

    Returns
    -------
    dict — drift summary (mirrors drift_summary.json content)
    """
    # Lazy import — keeps startup time fast when monitoring isn't needed
    # Evidently 0.7.x moved the classic API under evidently.legacy.*
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.legacy.metrics.data_drift.dataset_drift_metric import DatasetDriftMetric
    from evidently.legacy.report import Report

    meta              = _load_metadata()
    feature_defaults  = meta["feature_defaults"]
    superhost_avg     = meta.get("superhost_avg", {})
    non_superhost_avg = meta.get("non_superhost_avg", {})

    logger.info("Building synthetic reference dataset (N=%d)…", N_REFERENCE)
    reference_df = _synthetic_dataframe(
        feature_defaults, superhost_avg, non_superhost_avg,
        n=N_REFERENCE, seed=42, noise_scale=0.05,
    )

    if current_df is None:
        logger.info(
            "No live current_df provided — building synthetic current dataset "
            "(N=%d, slightly shifted for demo)…", N_CURRENT
        )
        # Simulate mild drift: shift a few features by ~10% to produce a realistic demo
        current_df = _synthetic_dataframe(
            feature_defaults, superhost_avg, non_superhost_avg,
            n=N_CURRENT, seed=99, noise_scale=0.12,   # wider noise = mild drift
        )
        # Deliberately nudge review_scores_rating slightly downward to show drift
        current_df["review_scores_rating"] = (
            current_df["review_scores_rating"] * 0.97
        ).clip(lower=0)

    # Ensure both frames have the same monitored columns
    cols = MONITOR_FEATURES + ["host_is_superhost"]
    ref  = reference_df[cols].copy()
    cur  = current_df.reindex(columns=cols).copy()

    logger.info("Running Evidently report…")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=ref, current_data=cur)

    # ── Extract structured summary ────────────────────────────────────────────
    report_dict = report.as_dict()
    metrics     = report_dict.get("metrics", [])

    # Pull DatasetDriftMetric result (overall drift flag + counts)
    dataset_drift_metric = next(
        (m for m in metrics if m.get("metric") == "DatasetDriftMetric"), None
    )
    drift_result = dataset_drift_metric.get("result", {}) if dataset_drift_metric else {}

    # Pull per-feature drift from DataDriftTable (Evidently 0.7.x structure)
    feature_drift: list[dict] = []
    data_drift_table = next(
        (m for m in metrics if m.get("metric") == "DataDriftTable"), None
    )
    if data_drift_table:
        drift_by_columns = data_drift_table.get("result", {}).get("drift_by_columns", {})
        for col, col_result in drift_by_columns.items():
            if col not in MONITOR_FEATURES:
                continue
            feature_drift.append({
                "feature":        col,
                "drift_detected": bool(col_result.get("drift_detected", False)),
                "stattest":       col_result.get("stattest_name", ""),
                "p_value":        round(col_result.get("p_value", 1.0), 4)
                                  if col_result.get("p_value") is not None else None,
                "drift_score":    round(col_result.get("drift_score", 0.0), 4)
                                  if col_result.get("drift_score") is not None else None,
            })

    n_drifted = int(drift_result.get("number_of_drifted_columns", 0))
    n_total   = int(drift_result.get("number_of_columns", len(MONITOR_FEATURES)))
    share_drifted = round(n_drifted / n_total, 4) if n_total else 0.0

    summary = {
        "status":                  "drift_detected" if drift_result.get("dataset_drift") else "ok",
        "dataset_drift":           bool(drift_result.get("dataset_drift", False)),
        "share_of_drifted_columns": share_drifted,
        "number_of_drifted_columns": n_drifted,
        "number_of_columns":       n_total,
        "monitored_features":      MONITOR_FEATURES,
        "feature_drift":           feature_drift,
        "model_name":              meta.get("best_model_name", "unknown"),
        "training_date":           meta.get("training_date", "unknown"),
        "report_path":             str(REPORT_HTML) if save_html else None,
    }

    # ── Persist outputs ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if save_html:
        report.save_html(str(REPORT_HTML))
        logger.info("Drift report saved → %s", REPORT_HTML)

    if save_json:
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Drift summary saved → %s", SUMMARY_JSON)

    return summary


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("=" * 60)
    print("  Superhost Predictor — Evidently Drift Monitor")
    print("=" * 60)

    summary = generate_drift_report(save_html=True, save_json=True)

    print(f"\n  Status             : {summary['status'].upper()}")
    print(f"  Dataset drift      : {summary['dataset_drift']}")
    print(f"  Drifted columns    : {summary['number_of_drifted_columns']} / {summary['number_of_columns']}")
    print(f"  Share drifted      : {summary['share_of_drifted_columns']:.1%}")
    print("\n  Per-feature results:")
    for fd in summary["feature_drift"]:
        flag = "[DRIFT]" if fd["drift_detected"] else "[ok]   "
        pval = f"p={fd['p_value']:.4f}" if fd["p_value"] is not None else "p=N/A"
        print(f"    {flag}  {fd['feature']:<38} {pval}")

    print(f"\n  HTML report : {REPORT_HTML}")
    print(f"  JSON summary: {SUMMARY_JSON}")
    print()

    # Exit with code 1 if drift share exceeds 50% — mirrors the CI gate
    if summary["share_of_drifted_columns"] > 0.50:
        print("  [FAIL] Drift threshold exceeded (>50% of features drifted).")
        sys.exit(1)
    else:
        print("  [PASS] All drift checks passed.")
        sys.exit(0)
