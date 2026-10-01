"""
experiments/bootstrap_ci.py
----------------------------------------------------------------
Bootstrap 95 % Confidence Interval on Treatment Effect
----------------------------------------------------------------

Why bootstrap?
--------------
Parametric confidence intervals for Δmean assume normality.
review_scores_rating is bounded [1, 5] and heavily right-skewed.
The percentile bootstrap makes no distributional assumptions —
it directly resamples the empirical distribution 10,000 times
and reads off the 2.5th and 97.5th percentiles.

This is the same technique used in:
  - Netflix's experimentation platform (bootstrapped lift CIs)
  - Airbnb's own metric framework (documented in their eng blog)
  - The Diabetes project in this portfolio (bootstrapped AUC CIs)

Method: Percentile Bootstrap
-----------------------------
  1. Draw B=10,000 bootstrap resamples (with replacement) from
     treatment and control independently.
  2. Compute Δmean = mean(treatment*) − mean(control*) for each.
  3. The 95 % CI is the [2.5th, 97.5th] percentile of the
     bootstrap distribution of Δmean.

Also computed
-------------
  - Bootstrap SE (std of bootstrap Δmean distribution)
  - Bias estimate (mean of bootstrap Δmean − observed Δmean)
  - Bias-corrected and accelerated (BCa) CI when scipy is available
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_B    = 10_000   # bootstrap iterations
DEFAULT_SEED = 42       # reproducibility
DEFAULT_CI   = 0.95


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BootstrapCIResult:
    """Bootstrap CI result — serialisable to JSON."""
    outcome_metric: str
    observed_delta_mean: float       # treatment_mean − control_mean
    n_treatment: int
    n_control: int
    n_bootstrap: int
    seed: int
    ci_level: float
    ci_lower: float                  # percentile bootstrap lower bound
    ci_upper: float                  # percentile bootstrap upper bound
    bootstrap_se: float              # std of bootstrap distribution
    bias_estimate: float             # E[bootstrap Δ] − observed Δ
    ci_excludes_zero: bool           # True → significant at ci_level
    interpretation: str


# ── Core bootstrap function ───────────────────────────────────────────────────

def bootstrap_delta_mean(
    treatment: np.ndarray,
    control:   np.ndarray,
    B:         int   = DEFAULT_B,
    seed:      int   = DEFAULT_SEED,
    ci_level:  float = DEFAULT_CI,
    outcome_metric: str = "review_scores_rating",
) -> BootstrapCIResult:
    """
    Compute a percentile bootstrap CI for (mean_treatment − mean_control).

    Parameters
    ----------
    treatment      : 1-D array of outcome values for the treatment arm.
    control        : 1-D array of outcome values for the control arm.
    B              : Number of bootstrap resamples (default 10,000).
    seed           : Random seed for reproducibility (default 42).
    ci_level       : Confidence level, e.g. 0.95 for 95 % CI.
    outcome_metric : Name of the outcome (for labelling only).

    Returns
    -------
    BootstrapCIResult
    """
    rng = np.random.default_rng(seed)

    observed_delta = float(treatment.mean() - control.mean())
    nt, nc = len(treatment), len(control)

    # ── Bootstrap loop ────────────────────────────────────────────────────────
    bootstrap_deltas = np.empty(B, dtype=float)
    for i in range(B):
        t_boot = rng.choice(treatment, size=nt, replace=True)
        c_boot = rng.choice(control,  size=nc, replace=True)
        bootstrap_deltas[i] = t_boot.mean() - c_boot.mean()

    # ── Percentile CI ─────────────────────────────────────────────────────────
    alpha     = 1.0 - ci_level
    ci_lower  = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_upper  = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))
    boot_se   = float(bootstrap_deltas.std(ddof=1))
    bias      = float(bootstrap_deltas.mean() - observed_delta)

    ci_excludes_zero = not (ci_lower <= 0.0 <= ci_upper)

    # ── Interpretation ────────────────────────────────────────────────────────
    ci_pct   = int(ci_level * 100)
    direction = "above" if observed_delta > 0 else "below"
    sig_str   = (
        f"The {ci_pct}% CI [{ci_lower:+.4f}, {ci_upper:+.4f}] excludes zero, "
        f"confirming the effect is statistically detectable at the {1-ci_level:.0%} level."
        if ci_excludes_zero
        else
        f"The {ci_pct}% CI [{ci_lower:+.4f}, {ci_upper:+.4f}] includes zero, "
        f"so we cannot rule out a null effect at the {1-ci_level:.0%} level."
    )

    interpretation = (
        f"Treatment arm mean is {abs(observed_delta):.4f} points {direction} "
        f"the control arm (Δ = {observed_delta:+.4f}). "
        f"Bootstrap SE = {boot_se:.4f}, bias = {bias:+.4f}. "
        f"{sig_str}"
    )

    return BootstrapCIResult(
        outcome_metric=outcome_metric,
        observed_delta_mean=round(observed_delta, 6),
        n_treatment=nt,
        n_control=nc,
        n_bootstrap=B,
        seed=seed,
        ci_level=ci_level,
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        bootstrap_se=round(boot_se, 6),
        bias_estimate=round(bias, 6),
        ci_excludes_zero=ci_excludes_zero,
        interpretation=interpretation,
    )


# ── BCa bootstrap (when scipy is available) ───────────────────────────────────

def bootstrap_bca(
    treatment: np.ndarray,
    control:   np.ndarray,
    B:         int   = DEFAULT_B,
    seed:      int   = DEFAULT_SEED,
    ci_level:  float = DEFAULT_CI,
) -> Optional[tuple[float, float]]:
    """
    Bias-corrected and accelerated (BCa) bootstrap CI.
    Returns (lower, upper) or None if scipy is unavailable.

    BCa corrects for:
      - Bias: systematic shift of bootstrap distribution
      - Skewness: non-symmetric bootstrap distribution
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return None

    rng = np.random.default_rng(seed)

    def statistic_fn(t, c):
        return t.mean() - c.mean()

    # Scipy's bootstrap handles BCa natively
    res = sp_stats.bootstrap(
        (treatment, control),
        statistic=statistic_fn,
        n_resamples=B,
        confidence_level=ci_level,
        method="BCa",
        random_state=rng,
    )
    return (round(float(res.confidence_interval.low), 6),
            round(float(res.confidence_interval.high), 6))


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    """
    Standalone runner. Loads data and runs bootstrap CI from scratch.
    Intended to be run AFTER ab_test.py (which validates sample sizes).
    """
    from experiments.ab_test import (
        RESPONSE_RATE_THRESHOLD,
        PRIMARY_OUTCOME,
        SECONDARY_OUTCOME,
        _load_data,
        _clean_response_rate,
    )

    print("-" * 65)
    print("  Bootstrap 95% Confidence Interval on Treatment Effect")
    print("-" * 65)

    df = _load_data()
    df["response_rate_clean"] = _clean_response_rate(df["host_response_rate"])
    df_clean = df.dropna(subset=["response_rate_clean", PRIMARY_OUTCOME])

    treatment_mask = df_clean["response_rate_clean"] >= RESPONSE_RATE_THRESHOLD
    treatment = df_clean[treatment_mask][PRIMARY_OUTCOME].dropna().values.astype(float)
    control   = df_clean[~treatment_mask][PRIMARY_OUTCOME].dropna().values.astype(float)

    print(f"\n  Running {DEFAULT_B:,} bootstrap iterations (seed={DEFAULT_SEED})…")
    result = bootstrap_delta_mean(treatment, control, outcome_metric=PRIMARY_OUTCOME)

    print(f"\n  Outcome metric   : {result.outcome_metric}")
    print(f"  Observed Δ mean  : {result.observed_delta_mean:+.4f}")
    print(f"  Bootstrap SE     : {result.bootstrap_se:.4f}")
    print(f"  Bias estimate    : {result.bias_estimate:+.4f}")
    print(f"  95% CI (pct.)    : [{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]")
    print(f"  CI excludes 0?   : {'YES ✅' if result.ci_excludes_zero else 'NO —'}")
    print()
    print(f"  {result.interpretation}")

    # BCa if scipy available
    bca = bootstrap_bca(treatment, control)
    if bca:
        print(f"\n  BCa CI (scipy)   : [{bca[0]:+.4f}, {bca[1]:+.4f}]")
    else:
        print("\n  BCa CI           : scipy not installed — percentile CI used")

    # Persist
    out_path = Path(__file__).parent / "bootstrap_ci_result.json"
    data = asdict(result)
    if bca:
        data["bca_ci_lower"] = bca[0]
        data["bca_ci_upper"] = bca[1]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  ✅  Saved → {out_path}")
    print("-" * 65)


if __name__ == "__main__":
    main()
