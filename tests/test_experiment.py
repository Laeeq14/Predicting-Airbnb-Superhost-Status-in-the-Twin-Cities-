"""
tests/test_experiment.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pytest suite for the experiments/ package
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These tests run in CI WITHOUT the large CSV files (gitignored).
Synthetic DataFrames are injected directly into the experiment
functions so that CI is fast and deterministic.

Test categories
---------------
  Power analysis   — sample size math is correct
  Mann-Whitney U   — statistic and p-value match known answers
  Welch's t-test   — same
  Bootstrap CI     — CI is reproducible and directionally correct
  ABTestResult     — end-to-end run on synthetic data succeeds
  Edge cases       — small groups raise ValueError
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd

# ── Imports under test ────────────────────────────────────────────────────────

from experiments.power_analysis import compute_sample_size, _z_score
from experiments.ab_test import (
    _mann_whitney_u,
    _ttest_ind,
    _cohens_d,
    run_experiment,
    MIN_GROUP_SIZE,
    RESPONSE_RATE_THRESHOLD,
)
from experiments.bootstrap_ci import bootstrap_delta_mean


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def treatment_arr(rng) -> np.ndarray:
    """Synthetic treatment arm: mean ≈ 4.85, higher than control."""
    return rng.normal(loc=4.85, scale=0.20, size=500).clip(1, 5)


@pytest.fixture()
def control_arr(rng) -> np.ndarray:
    """Synthetic control arm: mean ≈ 4.70."""
    return rng.normal(loc=4.70, scale=0.25, size=400).clip(1, 5)


@pytest.fixture()
def synthetic_df(rng) -> pd.DataFrame:
    """
    Minimal synthetic listings DataFrame for end-to-end experiment tests.
    Avoids any dependency on the real (gitignored) CSV files.
    """
    n = 1000
    response_rates = np.concatenate([
        rng.uniform(95, 100, size=600),   # treatment-eligible
        rng.uniform(50, 94.9, size=400),  # control
    ])
    rating = np.where(
        response_rates >= RESPONSE_RATE_THRESHOLD,
        rng.normal(4.85, 0.20, size=n).clip(1, 5),
        rng.normal(4.70, 0.25, size=n).clip(1, 5),
    )
    comm_score = np.where(
        response_rates >= RESPONSE_RATE_THRESHOLD,
        rng.normal(4.90, 0.15, size=n).clip(1, 5),
        rng.normal(4.75, 0.20, size=n).clip(1, 5),
    )
    return pd.DataFrame({
        "host_response_rate": response_rates,
        "review_scores_rating": rating,
        "review_scores_communication": comm_score,
        "host_experience_years": rng.uniform(0, 15, size=n),
        "price": rng.uniform(50, 300, size=n),
        "number_of_reviews": rng.integers(1, 200, size=n).astype(float),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Power Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPowerAnalysis:

    def test_z_score_standard_values(self):
        """z(0.975) ≈ 1.96 and z(0.84) ≈ 1.00 within tolerance."""
        assert abs(_z_score(0.975) - 1.96) < 0.01
        assert abs(_z_score(0.840) - 1.00) < 0.02

    def test_default_output_shape(self):
        result = compute_sample_size()
        assert result.n_per_arm > 0
        assert result.total_n == 2 * result.n_per_arm

    def test_larger_effect_requires_fewer_samples(self):
        """Smaller effect size → larger sample requirement."""
        small_effect  = compute_sample_size(delta=0.05)
        larger_effect = compute_sample_size(delta=0.15)
        assert small_effect.n_per_arm > larger_effect.n_per_arm

    def test_higher_power_requires_more_samples(self):
        low_power  = compute_sample_size(power=0.70)
        high_power = compute_sample_size(power=0.90)
        assert high_power.n_per_arm > low_power.n_per_arm

    def test_cohens_d_formula(self):
        result = compute_sample_size(delta=0.30, sigma=0.30)
        assert abs(result.effect_size_d - 1.0) < 1e-6

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            compute_sample_size(delta=0.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            compute_sample_size(sigma=0.0)

    def test_n_per_arm_is_ceiling(self):
        """n_per_arm must be an integer and must be rounded up."""
        result = compute_sample_size()
        assert isinstance(result.n_per_arm, int)
        # If we compute raw n and ceil it, it must match
        from experiments.power_analysis import DEFAULT_ALPHA, DEFAULT_POWER, DEFAULT_DELTA, DEFAULT_SIGMA, _z_score
        z_a = _z_score(1 - DEFAULT_ALPHA / 2)
        z_b = _z_score(DEFAULT_POWER)
        n_raw = 2 * DEFAULT_SIGMA**2 * (z_a + z_b)**2 / DEFAULT_DELTA**2
        assert result.n_per_arm == math.ceil(n_raw)


# ─────────────────────────────────────────────────────────────────────────────
# Mann-Whitney U Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMannWhitneyU:

    def test_identical_distributions_high_p(self, rng):
        """Identical distributions → p-value should NOT be significant."""
        x = rng.normal(5.0, 0.3, size=300)
        _, p = _mann_whitney_u(x, x.copy())
        # For truly identical arrays the U statistic should produce a large p
        # (not necessarily > 0.05 with exact tie correction, but > 0.3 is safe)
        assert p > 0.30

    def test_clearly_separated_distributions(self, rng):
        """Well-separated → p << 0.05."""
        x = rng.normal(5.0, 0.1, size=500)
        y = rng.normal(1.0, 0.1, size=500)
        _, p = _mann_whitney_u(x, y)
        assert p < 0.001

    def test_returns_two_floats(self, treatment_arr, control_arr):
        U, p = _mann_whitney_u(treatment_arr, control_arr)
        assert isinstance(U, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_treatment_higher_than_control(self, treatment_arr, control_arr):
        """With treatment mean > control mean, expect a significant result."""
        _, p = _mann_whitney_u(treatment_arr, control_arr)
        assert p < 0.05, (
            f"Expected significant effect with clearly separated arrays; got p={p:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Welch's t-test Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTTest:

    def test_same_group_p_near_one(self, rng):
        x = rng.normal(4.8, 0.2, size=200)
        # Duplicate the array — t should be ~0, p ~1
        t, p = _ttest_ind(x, x.copy())
        assert abs(t) < 0.1
        assert p > 0.90

    def test_separated_p_significant(self, rng):
        x = rng.normal(5.0, 0.2, size=300)
        y = rng.normal(4.5, 0.2, size=300)
        t, p = _ttest_ind(x, y)
        assert p < 0.001
        assert t > 0  # x > y

    def test_output_types(self, treatment_arr, control_arr):
        t, p = _ttest_ind(treatment_arr, control_arr)
        assert isinstance(t, float)
        assert isinstance(p, float)


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's d Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCohensD:

    def test_identical_means_zero_d(self, rng):
        x = rng.normal(4.8, 0.3, size=500)
        y = x.copy()
        d = _cohens_d(x, y)
        assert abs(d) < 1e-10

    def test_sign_direction(self, treatment_arr, control_arr):
        d = _cohens_d(treatment_arr, control_arr)
        assert d > 0  # treatment mean > control mean

    def test_magnitude_reasonable(self, treatment_arr, control_arr):
        d = _cohens_d(treatment_arr, control_arr)
        # With 0.15-point separation and σ≈0.22, expect d ≈ 0.15/0.22 ≈ 0.68
        assert 0.3 < abs(d) < 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:

    def test_reproducibility(self, treatment_arr, control_arr):
        """Same seed → identical CI bounds."""
        r1 = bootstrap_delta_mean(treatment_arr, control_arr, B=500, seed=99)
        r2 = bootstrap_delta_mean(treatment_arr, control_arr, B=500, seed=99)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_ci_contains_observed_delta(self, treatment_arr, control_arr):
        """The observed Δ should lie within the bootstrap CI."""
        r = bootstrap_delta_mean(treatment_arr, control_arr, B=1000, seed=42)
        assert r.ci_lower <= r.observed_delta_mean <= r.ci_upper

    def test_ci_excludes_zero_for_separated_arrays(self, treatment_arr, control_arr):
        """Clearly separated distributions → CI must exclude 0."""
        r = bootstrap_delta_mean(treatment_arr, control_arr, B=2000, seed=42)
        assert r.ci_excludes_zero, (
            f"Expected CI to exclude 0 for separated distributions; "
            f"got [{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
        )

    def test_null_effect_ci_contains_zero(self, rng):
        """Identical distributions → CI should NOT reliably exclude 0."""
        x = rng.normal(4.8, 0.25, size=400)
        y = rng.normal(4.8, 0.25, size=400)
        # Run many seeds and check most don't exclude 0
        exclusions = 0
        for seed in range(20):
            r = bootstrap_delta_mean(x, y, B=500, seed=seed)
            if r.ci_excludes_zero:
                exclusions += 1
        # At 95% CI level, expect ~5% false exclusion rate (≤ 3/20)
        assert exclusions <= 5, (
            f"Null effect CIs excluded zero too often: {exclusions}/20"
        )

    def test_n_stored_correctly(self, treatment_arr, control_arr):
        r = bootstrap_delta_mean(treatment_arr, control_arr, B=100)
        assert r.n_treatment == len(treatment_arr)
        assert r.n_control   == len(control_arr)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end ABTestResult Tests (synthetic data, no CSV required)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunExperiment:

    def test_runs_on_synthetic_data(self, synthetic_df):
        result = run_experiment(df=synthetic_df)
        assert result.n_treatment > 0
        assert result.n_control   > 0

    def test_group_sizes_add_up(self, synthetic_df):
        result = run_experiment(df=synthetic_df)
        total_clean = (
            result.n_treatment
            + result.n_control
            + result.n_excluded
        )
        # Should be ≤ original row count (some may be excluded)
        assert total_clean <= len(synthetic_df)

    def test_treatment_mean_above_control(self, synthetic_df):
        """Synthetic data was constructed so treatment > control."""
        result = run_experiment(df=synthetic_df)
        assert result.treatment_stats.mean > result.control_stats.mean

    def test_delta_mean_sign(self, synthetic_df):
        result = run_experiment(df=synthetic_df)
        assert result.delta_mean > 0

    def test_mann_whitney_significant(self, synthetic_df):
        """With 600 treatment and 400 control, the effect should be significant."""
        result = run_experiment(df=synthetic_df)
        assert result.mann_whitney.significant, (
            f"Expected significant Mann-Whitney; got p={result.mann_whitney.p_value:.4f}"
        )

    def test_conclusion_is_string(self, synthetic_df):
        result = run_experiment(df=synthetic_df)
        assert isinstance(result.conclusion, str)
        assert len(result.conclusion) > 50

    def test_limitations_present(self, synthetic_df):
        result = run_experiment(df=synthetic_df)
        assert len(result.limitations) >= 3

    def test_too_small_group_raises(self):
        """Tiny groups below MIN_GROUP_SIZE must raise ValueError."""
        df_tiny = pd.DataFrame({
            "host_response_rate": [95.0] * 10 + [80.0] * 10,
            "review_scores_rating": [4.8] * 10 + [4.5] * 10,
            "review_scores_communication": [4.9] * 10 + [4.6] * 10,
        })
        with pytest.raises(ValueError, match="Insufficient sample sizes"):
            run_experiment(df=df_tiny, min_group_size=50)

    def test_string_response_rate_parsed(self, rng):
        """response_rate stored as '95%' string should be handled."""
        n = 300
        rates = [f"{r:.0f}%" for r in np.concatenate([
            rng.uniform(95, 100, size=180),
            rng.uniform(50, 94, size=120),
        ])]
        df = pd.DataFrame({
            "host_response_rate": rates,
            "review_scores_rating": rng.normal(4.8, 0.2, size=n).clip(1, 5),
            "review_scores_communication": rng.normal(4.85, 0.15, size=n).clip(1, 5),
        })
        result = run_experiment(df=df, min_group_size=20)
        assert result.n_treatment > 0
        assert result.n_control   > 0
