"""
experiments/ab_test.py
----------------------------------------------------------------
A/B Experiment: Superhost Response-Rate Badge Policy
----------------------------------------------------------------

Research Question
-----------------
Does enforcing a high response-rate threshold (≥ 95 %) produce a
statistically detectable improvement in guest review_scores_rating,
compared with hosts below that threshold?

This is a **natural experiment / quasi-experiment** using the
observational Airbnb Twin Cities dataset.  We do NOT fabricate
treatment assignment — we use the real response-rate split that
already exists in the data, which is the honest way to frame an
A/B analysis on observational data.

Experiment Design
-----------------
  Unit of analysis : individual listing (host-listing pair)
  Treatment arm    : host_response_rate ≥ 95 %
  Control arm      : host_response_rate  < 95 %
  Outcome metric   : review_scores_rating (primary)
                     review_scores_communication (secondary — causal mechanism)
  Randomisation    : Not randomised (observational).  We document this as a
                     limitation and discuss confounders (host experience, price).
  Balance check    : We verify treatment/control are comparable on key covariates.

Statistical Tests
-----------------
  Primary          : Mann-Whitney U test (non-parametric, no normality assumption)
  Secondary        : Independent-samples t-test (for comparison)
  Effect size      : Cohen's d
  Significance     : α = 0.05 (pre-registered in power_analysis.py)
  Sample sizing    : n_per_arm established BEFORE seeing results (power_analysis.py)

Why Mann-Whitney U as primary?
  review_scores_rating is bounded [1, 5] and right-skewed (most ratings ≥ 4.5).
  Non-parametric tests make no distributional assumptions; t-test is secondary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "listings.csv"
FALLBACK_DATA_PATH = Path(__file__).parent.parent / "listings_detailed_june.csv"

# Pre-registered threshold (matches power_analysis.py design)
RESPONSE_RATE_THRESHOLD = 95.0  # %

# Pre-registered outcome metrics
PRIMARY_OUTCOME   = "review_scores_rating"
SECONDARY_OUTCOME = "review_scores_communication"

# Minimum group size to run the test (from power analysis: n_per_arm ≥ required)
MIN_GROUP_SIZE = 50   # conservative floor; actual requirement is ~139

ALPHA = 0.05


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GroupStats:
    name: str
    n: int
    mean: float
    median: float
    std: float
    q25: float
    q75: float


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float


@dataclass
class ABTestResult:
    """Complete A/B experiment result, serialisable to JSON via asdict()."""
    experiment_name: str
    treatment_definition: str
    control_definition: str
    response_rate_threshold: float
    primary_outcome: str
    secondary_outcome: str

    # Group sizes
    n_treatment: int
    n_control: int
    n_excluded: int   # listings with missing response_rate or outcome

    # Balance check
    covariate_balance: dict

    # Primary outcome
    treatment_stats: GroupStats
    control_stats: GroupStats
    delta_mean: float              # treatment_mean − control_mean
    pooled_std: float
    cohens_d: float                # effect size
    mann_whitney: TestResult       # primary test
    ttest: TestResult              # secondary test

    # Secondary outcome (communication score)
    secondary_treatment_mean: float
    secondary_control_mean: float
    secondary_mann_whitney: TestResult

    # Interpretation
    conclusion: str
    limitations: list[str]
    power_analysis_ref: str


# ── Helper: Mann-Whitney U (pure NumPy, no scipy dependency) ─────────────────

def _mann_whitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Two-sided Mann-Whitney U test using normal approximation.
    Valid when both groups have n > 20 (our groups are >> 20).

    Returns (U_statistic, p_value).
    """
    nx, ny = len(x), len(y)

    # Rank all values jointly
    combined = np.concatenate([x, y])
    # Use argsort to assign ranks (mid-ranks for ties)
    order = combined.argsort(kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)

    # Tie correction: average ranks for equal values
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[order[i]] == combined[order[j]]:
            j += 1
        if j > i + 1:
            avg_rank = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg_rank
        i = j

    # U statistic for x (treatment)
    R1 = ranks[:nx].sum()
    U1 = R1 - nx * (nx + 1) / 2.0
    U2 = nx * ny - U1

    # Normal approximation with tie correction
    n = nx + ny
    # Tie correction factor
    unique, counts = np.unique(combined, return_counts=True)
    tie_correction = np.sum(counts**3 - counts)
    sigma2 = (
        nx * ny / 12.0
        * ((n + 1) - tie_correction / (n * (n - 1)))
    )
    sigma = math.sqrt(max(sigma2, 1e-10))

    U = min(U1, U2)
    z = (U - nx * ny / 2.0) / sigma
    # Two-tailed p-value via error function
    p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(z)))

    return float(U1), float(p_value)


def _standard_normal_cdf(z: float) -> float:
    """Φ(z) via math.erf."""
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def _ttest_ind(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Welch's two-sample t-test (unequal variances).
    Returns (t_statistic, p_value) two-tailed.
    """
    nx, ny = len(x), len(y)
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y   = x.var(ddof=1), y.var(ddof=1)

    se = math.sqrt(var_x / nx + var_y / ny)
    if se == 0:
        return 0.0, 1.0

    t = (mean_x - mean_y) / se

    # Welch–Satterthwaite degrees of freedom
    num = (var_x / nx + var_y / ny) ** 2
    den = (var_x / nx) ** 2 / (nx - 1) + (var_y / ny) ** 2 / (ny - 1)
    df  = num / den if den > 0 else nx + ny - 2

    # Two-tailed p from Student's t distribution via regularised incomplete beta
    # Use a normal approximation when df > 30 (df will be >> 30 here)
    if df > 30:
        p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(t)))
    else:
        # Fallback: scipy if available, else normal approx
        try:
            from scipy import stats as _sp_stats
            p_value = float(_sp_stats.t.sf(abs(t), df) * 2)
        except ImportError:
            p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(t)))

    return float(t), float(p_value)


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-SD Cohen's d: (mean_x − mean_y) / s_pooled."""
    nx, ny = len(x), len(y)
    s_pooled = math.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    return float((x.mean() - y.mean()) / s_pooled) if s_pooled > 0 else 0.0


def _group_stats(name: str, arr: np.ndarray) -> GroupStats:
    return GroupStats(
        name=name,
        n=len(arr),
        mean=round(float(arr.mean()), 6),
        median=round(float(np.median(arr)), 6),
        std=round(float(arr.std(ddof=1)), 6),
        q25=round(float(np.percentile(arr, 25)), 6),
        q75=round(float(np.percentile(arr, 75)), 6),
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    """
    Load listing-level CSV.  Uses the largest available file.
    Raises FileNotFoundError with a clear message if neither exists.
    """
    for path in [DATA_PATH, FALLBACK_DATA_PATH]:
        if path.exists():
            # Load only columns we need — large CSVs, be selective
            usecols = [
                "host_response_rate",
                "review_scores_rating",
                "review_scores_communication",
                "host_experience_years",   # covariate balance check
                "price",                   # covariate balance check
                "number_of_reviews",       # covariate balance check
            ]
            try:
                df = pd.read_csv(path, usecols=usecols, low_memory=False)
                return df
            except ValueError:
                # Some columns may not exist in all CSV variants — load all
                df = pd.read_csv(path, low_memory=False)
                return df

    raise FileNotFoundError(
        f"No listing CSV found at:\n  {DATA_PATH}\n  {FALLBACK_DATA_PATH}\n"
        "Run the experiment from the project root directory."
    )


def _clean_response_rate(series: pd.Series) -> pd.Series:
    """Convert '95%' string format -> float 95.0 where needed."""
    # Check for object or any string-like dtype (pandas 2.x uses pd.StringDtype)
    dtype_name = str(series.dtype)
    is_string_like = (
        series.dtype == object
        or dtype_name.startswith("string")
        or dtype_name.startswith("str")
    )
    if is_string_like:
        return (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )
    return pd.to_numeric(series, errors="coerce")


# ── Main experiment function ──────────────────────────────────────────────────

def run_experiment(
    threshold: float = RESPONSE_RATE_THRESHOLD,
    min_group_size: int = MIN_GROUP_SIZE,
    alpha: float = ALPHA,
    df: Optional[pd.DataFrame] = None,
) -> ABTestResult:
    """
    Run the full A/B experiment and return an ABTestResult.

    Parameters
    ----------
    threshold       : response_rate cutoff for treatment arm (default 95 %)
    min_group_size  : minimum n per arm to proceed (default 50)
    alpha           : significance level (default 0.05)
    df              : optional pre-loaded DataFrame (used in tests)
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    if df is None:
        df = _load_data()

    n_raw = len(df)

    # ── 2. Clean response rate ────────────────────────────────────────────────
    df = df.copy()
    df["response_rate_clean"] = _clean_response_rate(df["host_response_rate"])

    # ── 3. Drop rows missing critical fields ──────────────────────────────────
    required = ["response_rate_clean", PRIMARY_OUTCOME]
    df_clean = df.dropna(subset=required)
    n_excluded = n_raw - len(df_clean)

    # ── 4. Split arms ─────────────────────────────────────────────────────────
    treatment_mask = df_clean["response_rate_clean"] >= threshold
    control_mask   = df_clean["response_rate_clean"] <  threshold

    treatment = df_clean[treatment_mask][PRIMARY_OUTCOME].dropna().values.astype(float)
    control   = df_clean[control_mask][PRIMARY_OUTCOME].dropna().values.astype(float)

    n_treatment = len(treatment)
    n_control   = len(control)

    if n_treatment < min_group_size or n_control < min_group_size:
        raise ValueError(
            f"Insufficient sample sizes: treatment n={n_treatment}, "
            f"control n={n_control}. Both must be ≥ {min_group_size}."
        )

    # ── 5. Covariate balance check ────────────────────────────────────────────
    balance: dict = {}
    for cov in ["host_experience_years", "price", "number_of_reviews"]:
        if cov in df_clean.columns:
            try:
                t_series = pd.to_numeric(
                    df_clean[treatment_mask][cov].astype(str)
                    .str.replace(r"[$,]", "", regex=True),
                    errors="coerce"
                ).dropna()
                c_series = pd.to_numeric(
                    df_clean[control_mask][cov].astype(str)
                    .str.replace(r"[$,]", "", regex=True),
                    errors="coerce"
                ).dropna()
                t_vals = t_series.values.astype(float)
                c_vals = c_series.values.astype(float)
                if len(t_vals) > 5 and len(c_vals) > 5:
                    _, p_bal = _ttest_ind(t_vals, c_vals)
                    balance[cov] = {
                        "treatment_mean": round(float(t_vals.mean()), 4),
                        "control_mean": round(float(c_vals.mean()), 4),
                        "p_value_balance": round(p_bal, 6),
                        "imbalanced": p_bal < 0.05,
                    }
            except Exception:
                pass   # skip covariates that can't be numerically compared

    # ── 6. Primary outcome: Mann-Whitney U ────────────────────────────────────
    U_stat, mw_p = _mann_whitney_u(treatment, control)
    mw_result = TestResult(
        test_name="Mann-Whitney U (two-tailed)",
        statistic=round(U_stat, 2),
        p_value=round(mw_p, 6),
        significant=mw_p < alpha,
        alpha=alpha,
    )

    # ── 7. Secondary test: Welch's t-test ────────────────────────────────────
    t_stat, t_p = _ttest_ind(treatment, control)
    ttest_result = TestResult(
        test_name="Welch's independent-samples t-test (two-tailed)",
        statistic=round(t_stat, 6),
        p_value=round(t_p, 6),
        significant=t_p < alpha,
        alpha=alpha,
    )

    # ── 8. Effect size ────────────────────────────────────────────────────────
    d = _cohens_d(treatment, control)
    delta_mean = float(treatment.mean() - control.mean())
    pooled_std  = math.sqrt(
        ((n_treatment - 1) * treatment.var(ddof=1) + (n_control - 1) * control.var(ddof=1))
        / (n_treatment + n_control - 2)
    )

    # ── 9. Secondary outcome: communication score ─────────────────────────────
    sec_treatment = df_clean[treatment_mask][SECONDARY_OUTCOME].dropna().values.astype(float)
    sec_control   = df_clean[control_mask][SECONDARY_OUTCOME].dropna().values.astype(float)

    sec_mw_stat, sec_mw_p = (
        _mann_whitney_u(sec_treatment, sec_control)
        if len(sec_treatment) > 5 and len(sec_control) > 5
        else (0.0, 1.0)
    )
    sec_mw_result = TestResult(
        test_name="Mann-Whitney U — review_scores_communication (two-tailed)",
        statistic=round(sec_mw_stat, 2),
        p_value=round(sec_mw_p, 6),
        significant=sec_mw_p < alpha,
        alpha=alpha,
    )

    # ── 10. Conclusion ────────────────────────────────────────────────────────
    direction  = "higher" if delta_mean > 0 else "lower"
    sig_word   = "statistically significant" if mw_result.significant else "not statistically significant"
    magnitude  = (
        "negligible (d < 0.2)"  if abs(d) < 0.2 else
        "small (0.2 ≤ d < 0.5)" if abs(d) < 0.5 else
        "medium (0.5 ≤ d < 0.8)" if abs(d) < 0.8 else
        "large (d ≥ 0.8)"
    )

    conclusion = (
        f"Hosts with response_rate ≥ {threshold}% have {direction} average "
        f"review_scores_rating by {abs(delta_mean):.4f} points "
        f"(Cohen's d = {d:.4f}, {magnitude}). "
        f"The Mann-Whitney U test (U={U_stat:.0f}, p={mw_p:.4f}) is {sig_word} "
        f"at α={alpha}. "
        f"Welch's t-test corroborates: t={t_stat:.4f}, p={t_p:.4f}."
    )

    limitations = [
        "Observational data — treatment assignment is NOT randomised. "
        "Hosts who respond quickly may differ systematically (selection bias).",
        "Confounders identified in balance check: "
        + ", ".join(k for k, v in balance.items() if v.get("imbalanced", False))
        + " (p < 0.05 imbalance). A propensity-score match or regression "
        "adjustment would strengthen causal claims."
        if any(v.get("imbalanced", False) for v in balance.values())
        else "No significant covariate imbalance detected for measured confounders.",
        "Outcome metric (review_scores_rating) may be subject to survivorship "
        "bias — hosts with very few reviews are included.",
        "Effect size is estimated from a single cross-sectional scrape; "
        "temporal confounding (seasonality) is not controlled.",
    ]

    return ABTestResult(
        experiment_name="Superhost Response-Rate Badge Policy — A/B Analysis",
        treatment_definition=f"host_response_rate ≥ {threshold}%",
        control_definition=f"host_response_rate < {threshold}%",
        response_rate_threshold=threshold,
        primary_outcome=PRIMARY_OUTCOME,
        secondary_outcome=SECONDARY_OUTCOME,
        n_treatment=n_treatment,
        n_control=n_control,
        n_excluded=n_excluded,
        covariate_balance=balance,
        treatment_stats=_group_stats("treatment", treatment),
        control_stats=_group_stats("control", control),
        delta_mean=round(delta_mean, 6),
        pooled_std=round(pooled_std, 6),
        cohens_d=round(d, 6),
        mann_whitney=mw_result,
        ttest=ttest_result,
        secondary_treatment_mean=round(float(sec_treatment.mean()), 6) if len(sec_treatment) > 0 else 0.0,
        secondary_control_mean=round(float(sec_control.mean()), 6) if len(sec_control) > 0 else 0.0,
        secondary_mann_whitney=sec_mw_result,
        conclusion=conclusion,
        limitations=limitations,
        power_analysis_ref="experiments/power_analysis.py — n_per_arm established BEFORE data was examined.",
    )


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("-" * 65)
    print("  A/B Experiment: Superhost Response-Rate Badge Policy")
    print("-" * 65)

    try:
        result = run_experiment()
    except FileNotFoundError as e:
        print(f"\n  ⚠  {e}")
        print("  Run from the project root: python -m experiments.ab_test")
        return

    print(f"\n  Treatment : {result.treatment_definition}")
    print(f"  Control   : {result.control_definition}")
    print(f"  Outcome   : {result.primary_outcome}")
    print()
    print(f"  Treatment n : {result.n_treatment:,}")
    print(f"  Control   n : {result.n_control:,}")
    print(f"  Excluded    : {result.n_excluded:,}")
    print()
    print("  ── Group Means ──────────────────────────────────────────")
    print(f"  Treatment mean : {result.treatment_stats.mean:.4f}")
    print(f"  Control   mean : {result.control_stats.mean:.4f}")
    print(f"  Δ mean         : {result.delta_mean:+.4f}")
    print(f"  Cohen's d      : {result.cohens_d:.4f}")
    print()
    print("  ── Statistical Tests ────────────────────────────────────")
    mw  = result.mann_whitney
    tt  = result.ttest
    sig = lambda r: "✅ SIGNIFICANT" if r.significant else "— not significant"
    print(f"  Mann-Whitney U : U={mw.statistic:.0f}, p={mw.p_value:.6f}  {sig(mw)}")
    print(f"  Welch's t-test : t={tt.statistic:.4f}, p={tt.p_value:.6f}  {sig(tt)}")
    print()
    print(f"  ── Conclusion ───────────────────────────────────────────")
    print(f"  {result.conclusion}")
    print()
    print("  ── Covariate Balance ────────────────────────────────────")
    for cov, stats in result.covariate_balance.items():
        flag = "⚠️  IMBALANCED" if stats["imbalanced"] else "✅  balanced"
        print(
            f"  {cov:<28} "
            f"T={stats['treatment_mean']:.3f}  C={stats['control_mean']:.3f}  "
            f"p={stats['p_value_balance']:.4f}  {flag}"
        )
    print("-" * 65)


if __name__ == "__main__":
    main()
