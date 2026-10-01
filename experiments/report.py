"""
experiments/report.py
----------------------------------------------------------------
Full Experiment Report Generator
----------------------------------------------------------------

Runs the complete experiment pipeline in order:
  1. Power analysis   (pre-experiment sizing)
  2. A/B test         (treatment/control comparison)
  3. Bootstrap CI     (95% CI on Δmean, 10k resamples)
  4. Merge + write    experiment_report.json

The JSON report is the artefact consumed by the CI gate
(experiments_gate job in ci.yml).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone

from experiments.power_analysis import compute_sample_size
from experiments.ab_test import run_experiment
from experiments.bootstrap_ci import bootstrap_delta_mean, bootstrap_bca

import numpy as np


REPORT_PATH = Path(__file__).parent / "experiment_report.json"


def build_report() -> dict:
    """
    Run all three experiment stages and compile a single report dict.
    """

    # ── Stage 1: Power analysis ───────────────────────────────────────────────
    print("\n[1/3] Running pre-experiment power analysis…")
    power = compute_sample_size()
    print(f"      n_per_arm required: {power.n_per_arm:,}")

    # ── Stage 2: A/B test ────────────────────────────────────────────────────
    print("[2/3] Running A/B test…")
    ab_result = run_experiment()
    print(f"      n_treatment={ab_result.n_treatment:,}  n_control={ab_result.n_control:,}")
    print(f"      Δ mean = {ab_result.delta_mean:+.4f}  Cohen's d = {ab_result.cohens_d:.4f}")
    print(f"      Mann-Whitney p = {ab_result.mann_whitney.p_value:.6f}")

    # Check we met the required sample size
    sufficient = (
        ab_result.n_treatment >= power.n_per_arm
        and ab_result.n_control >= power.n_per_arm
    )
    print(f"      Power gate ({'✅ PASSED' if sufficient else '⚠ FAILED: below required n'})")

    # ── Stage 3: Bootstrap CI ─────────────────────────────────────────────────
    print("[3/3] Running bootstrap CI (10,000 resamples)…")
    from experiments.ab_test import (
        RESPONSE_RATE_THRESHOLD, PRIMARY_OUTCOME,
        _load_data, _clean_response_rate,
    )

    df = _load_data()
    df["response_rate_clean"] = _clean_response_rate(df["host_response_rate"])
    df_clean = df.dropna(subset=["response_rate_clean", PRIMARY_OUTCOME])
    treatment_mask = df_clean["response_rate_clean"] >= RESPONSE_RATE_THRESHOLD

    treatment = df_clean[treatment_mask][PRIMARY_OUTCOME].dropna().values.astype(float)
    control   = df_clean[~treatment_mask][PRIMARY_OUTCOME].dropna().values.astype(float)

    boot_result = bootstrap_delta_mean(
        treatment, control, outcome_metric=PRIMARY_OUTCOME
    )
    bca = bootstrap_bca(treatment, control)
    print(f"      95% CI: [{boot_result.ci_lower:+.4f}, {boot_result.ci_upper:+.4f}]")
    if bca:
        print(f"      BCa CI: [{bca[0]:+.4f}, {bca[1]:+.4f}]")

    # ── Compile report ────────────────────────────────────────────────────────
    boot_dict = asdict(boot_result)
    if bca:
        boot_dict["bca_ci_lower"] = bca[0]
        boot_dict["bca_ci_upper"] = bca[1]

    ab_dict = asdict(ab_result)
    # Convert nested dataclasses already handled by asdict

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_version": "1.0.0",
        "power_analysis": asdict(power),
        "ab_test": ab_dict,
        "bootstrap_ci": boot_dict,
        # ── CI gate summary ────────────────────────────────────────────────
        "ci_gate": {
            "power_requirement_met": sufficient,
            "n_required_per_arm": power.n_per_arm,
            "n_treatment_actual": ab_result.n_treatment,
            "n_control_actual": ab_result.n_control,
            "mann_whitney_significant": ab_result.mann_whitney.significant,
            "mann_whitney_p": ab_result.mann_whitney.p_value,
            "bootstrap_ci_excludes_zero": boot_result.ci_excludes_zero,
            "bootstrap_ci": [boot_result.ci_lower, boot_result.ci_upper],
            "cohens_d": ab_result.cohens_d,
            "delta_mean": ab_result.delta_mean,
        },
    }

    return report


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("-" * 65)
    print("  Experiment Report Generator")
    print("-" * 65)

    try:
        report = build_report()
    except FileNotFoundError as e:
        print(f"\n  ⚠  {e}")
        print("  Run from the project root: python -m experiments.report")
        return

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    gate = report["ci_gate"]
    print("\n━" * 65)
    print("  CI Gate Summary")
    print("-" * 65)
    print(f"  Power gate         : {'✅ PASSED' if gate['power_requirement_met'] else '❌ FAILED'}")
    print(f"  Mann-Whitney sig.  : {'✅ YES' if gate['mann_whitney_significant'] else '— NO'}  (p={gate['mann_whitney_p']:.6f})")
    print(f"  Bootstrap CI ≠ 0   : {'✅ YES' if gate['bootstrap_ci_excludes_zero'] else '— NO'}  {gate['bootstrap_ci']}")
    print(f"  Cohen's d          : {gate['cohens_d']:.4f}")
    print(f"  Δ mean             : {gate['delta_mean']:+.4f}")
    print(f"\n  ✅  Report saved → {REPORT_PATH}")
    print("-" * 65)


if __name__ == "__main__":
    main()
