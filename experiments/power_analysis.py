"""
experiments/power_analysis.py
----------------------------------------------------------------
Pre-Experiment Power Analysis
----------------------------------------------------------------

Purpose
-------
Before running any A/B test, calculate the **minimum sample size per arm**
required to detect a meaningful effect.  This prevents underpowered tests
and is the first step any honest experimenter documents.

Experiment framing
------------------
Policy change under study:
  "Does enforcing a ≥95 % response-rate threshold (the 'Superhost Badge'
   policy) produce a statistically detectable improvement in guest review
   scores compared with hosts below that threshold?"

Design choices (all justified below)
-------------------------------------
  alpha  = 0.05   → 5 % false-positive rate (industry standard)
  power  = 0.80   → 80 % chance of detecting a real effect (Cohen 1988)
  delta  = 0.05   → Minimum detectable effect: 0.05 points on the 1–5 rating
                     scale.  Airbnb's own research treats a 0.1-star improvement
                     as commercially meaningful; we are conservative at 0.05.
  sigma  = 0.30   → Estimated SD of review_scores_rating from training data
                     (empirical; see train_log.txt: σ ≈ 0.28–0.32).
  sides  = 2      → Two-tailed: we do NOT assume the direction of the effect
                     in advance (scientific honesty).

Output
------
  PowerAnalysisResult dataclass with n_per_arm, total_n, effect_size_d,
  alpha, power, delta, sigma — all serialisable to JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

# Justified in docstring above.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_DELTA = 0.05   # minimum detectable effect (rating points)
DEFAULT_SIGMA = 0.30   # estimated population SD of review_scores_rating


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class PowerAnalysisResult:
    """Immutable record of a pre-experiment power calculation."""
    alpha: float
    power: float
    delta: float
    sigma: float
    effect_size_d: float
    n_per_arm: int
    total_n: int
    notes: str


# ── Core calculation ─────────────────────────────────────────────────────────

def _z_score(p: float) -> float:
    """
    Inverse normal CDF via rational approximation (Abramowitz & Stegun 26.2.17).
    Accurate to ±4.5e-4; sufficient for sample-size planning.
    We avoid scipy here so this module has zero heavy dependencies.
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1); got {p}")

    # Rational approximation constants
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    # Work in the upper tail; apply symmetry at the end
    if p > 0.5:
        sign = 1.0
        q = 1.0 - p
    else:
        sign = -1.0
        q = p

    t = math.sqrt(-2.0 * math.log(q))
    numerator = c0 + c1 * t + c2 * t**2
    denominator = 1.0 + d1 * t + d2 * t**2 + d3 * t**3
    z = sign * (t - numerator / denominator)
    return z


def compute_sample_size(
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    delta: float = DEFAULT_DELTA,
    sigma: float = DEFAULT_SIGMA,
) -> PowerAnalysisResult:
    """
    Two-sample, two-tailed t-test sample size formula (equal group sizes):

        n = 2 * sigma^2 * (z_{1-alpha/2} + z_{power})^2 / delta^2

    Parameters
    ----------
    alpha   : Type-I error rate (default 0.05)
    power   : Desired statistical power (default 0.80)
    delta   : Minimum detectable effect in rating units (default 0.05)
    sigma   : Estimated population SD of the outcome (default 0.30)

    Returns
    -------
    PowerAnalysisResult
    """
    if delta <= 0:
        raise ValueError(f"delta must be > 0; got {delta}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0; got {sigma}")

    z_alpha = _z_score(1.0 - alpha / 2.0)   # critical value (two-tailed)
    z_beta  = _z_score(power)               # power z-score

    n_raw     = 2.0 * (sigma**2) * (z_alpha + z_beta)**2 / (delta**2)
    n_per_arm = math.ceil(n_raw)             # always round up
    total_n   = 2 * n_per_arm
    cohens_d  = delta / sigma

    notes = (
        f"To detect a {delta:.3f}-point improvement in review_scores_rating "
        f"(Cohen's d = {cohens_d:.3f}) with α={alpha} and {power*100:.0f}% power, "
        f"we need ≥{n_per_arm} hosts per arm ({total_n} total). "
        f"The Twin Cities dataset contains ~10,700 unique listings — "
        f"well above this threshold."
    )

    return PowerAnalysisResult(
        alpha=alpha,
        power=power,
        delta=delta,
        sigma=sigma,
        effect_size_d=round(cohens_d, 4),
        n_per_arm=n_per_arm,
        total_n=total_n,
        notes=notes,
    )


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    result = compute_sample_size()

    print("-" * 60)
    print("  Pre-Experiment Power Analysis")
    print("-" * 60)
    print(f"  Outcome metric  : review_scores_rating (1–5 scale)")
    print(f"  Treatment       : host_response_rate ≥ 95 % (Badge policy)")
    print(f"  Control         : host_response_rate < 95 %")
    print(f"  Alpha (α)       : {result.alpha}")
    print(f"  Power (1−β)     : {result.power}")
    print(f"  Min effect (δ)  : {result.delta} rating points")
    print(f"  Pop. SD (σ)     : {result.sigma}")
    print(f"  Cohen's d       : {result.effect_size_d}")
    print(f"  N per arm       : {result.n_per_arm:,}")
    print(f"  Total N needed  : {result.total_n:,}")
    print()
    print(f"  {result.notes}")
    print("-" * 60)

    # Persist for CI / downstream consumers
    out_path = Path(__file__).parent / "power_analysis_result.json"
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\n  ✅  Saved → {out_path}")


if __name__ == "__main__":
    main()
