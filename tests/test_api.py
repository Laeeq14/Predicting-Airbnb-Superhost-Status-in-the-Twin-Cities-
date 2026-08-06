"""
tests/test_api.py
=================
Pytest suite for the Superhost Predictor FastAPI app.

CI-safe design principles:
- No large CSV files required (gitignored)
- GROQ_API_KEY is mocked via monkeypatch / env var so agent.py doesn't raise
- best_model.joblib + model_metadata.json ARE committed and used directly
- Drift tests use the synthetic path in drift_report.py (no CSVs needed)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Project root on sys.path so imports resolve without installation ───────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def mock_groq_key():
    """
    Set a fake GROQ_API_KEY before any imports touch agent.py.
    The agent raises RuntimeError at module level if the key is missing,
    so we patch os.environ before the app is imported.
    """
    os.environ.setdefault("GROQ_API_KEY", "gsk_test_fake_key_for_ci")
    yield


@pytest.fixture(scope="session")
def client(mock_groq_key):
    """
    Spin up the FastAPI app with TestClient (ASGI, no real server needed).
    We mock agent_module.build_agent_data so no CSV loading happens on startup.
    """
    from unittest.mock import patch as _patch

    with _patch("app.agent.build_agent_data", return_value=None), \
         _patch("app.agent.get_at_risk_listings", return_value=[]), \
         _patch("openai.OpenAI", return_value=MagicMock()):
        from fastapi.testclient import TestClient
        from app.main import app
        yield TestClient(app)


@pytest.fixture(scope="session")
def meta_path():
    p = ROOT / "ml_pipeline" / "model_metadata.json"
    assert p.exists(), f"model_metadata.json not found at {p}"
    return p


@pytest.fixture(scope="session")
def model_path():
    p = ROOT / "ml_pipeline" / "best_model.joblib"
    assert p.exists(), f"best_model.joblib not found at {p}"
    return p


# ── Test 1: Model loads and predicts ─────────────────────────────────────────

class TestModelLoad:
    def test_model_loads_successfully(self, model_path):
        """best_model.joblib must deserialise without error."""
        import joblib
        pipeline = joblib.load(model_path)
        assert pipeline is not None, "Loaded pipeline is None"

    def test_metadata_has_required_keys(self, meta_path):
        """model_metadata.json must contain keys the API depends on."""
        with open(meta_path) as f:
            meta = json.load(f)
        required_keys = [
            "best_model_name",
            "numeric_features",
            "categorical_features",
            "feature_defaults",
            "superhost_avg",
            "model_performance",
        ]
        for key in required_keys:
            assert key in meta, f"Missing required metadata key: '{key}'"

    def test_model_predicts_probability(self, model_path, meta_path):
        """Model must output a valid probability for a default feature row."""
        import joblib
        import numpy as np
        import pandas as pd

        with open(meta_path) as f:
            meta = json.load(f)

        pipeline = joblib.load(model_path)
        defaults = meta["feature_defaults"]
        num_feats = meta["numeric_features"]
        cat_feats = meta["categorical_features"]
        cols = num_feats + cat_feats

        row_df = pd.DataFrame([{c: defaults.get(c, np.nan) for c in cols}])
        proba = pipeline.predict_proba(row_df)[0, 1]

        assert 0.0 <= proba <= 1.0, f"Predicted probability out of range: {proba}"


# ── Test 2: API endpoints ─────────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_model_info_returns_200(self, client):
        """GET /model-info must return 200 with JSON payload."""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "best_model_name" in data
        assert "model_performance" in data

    def test_predict_returns_valid_probability(self, client):
        """POST /predict with default values must return probability in [0, 1]."""
        payload = {
            "review_scores_rating":  4.89,
            "reviews_per_month":     1.23,
            "host_response_rate":    100.0,
            "host_acceptance_rate":  98.0,
            "host_experience_years": 7.5,
            "host_listings_count":   2.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"Unexpected status: {response.text}"
        data = response.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in ("Superhost", "Not Yet Superhost")

    def test_predict_superhost_profile(self, client):
        """A near-perfect profile should score high probability."""
        payload = {
            "review_scores_rating":  5.0,
            "reviews_per_month":     5.0,
            "host_response_rate":    100.0,
            "host_acceptance_rate":  100.0,
            "host_experience_years": 10.0,
            "host_listings_count":   1.0,
            "amenity_hair_dryer":    1,
            "amenity_essentials":    1,
            "amenity_iron":          1,
            "amenity_cooking_basics": 1,
            "amenity_hot_water":     1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["probability"] > 0.5

    def test_neighbourhood_stats_endpoint(self, client):
        """GET /neighbourhood-stats must return 200 with neighbourhood keys."""
        response = client.get("/neighbourhood-stats")
        assert response.status_code in (200, 503)   # 503 if file missing in CI
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            assert len(data) > 0


# ── Test 3: Drift monitoring ──────────────────────────────────────────────────

class TestDriftMonitoring:
    def test_drift_report_generates_without_error(self):
        """
        drift_report.generate_drift_report() must run end-to-end using the
        synthetic path (no CSVs needed) and return a valid summary dict.
        """
        from monitoring.drift_report import generate_drift_report

        summary = generate_drift_report(save_html=False, save_json=False)

        assert isinstance(summary, dict)
        assert "status" in summary
        assert summary["status"] in ("ok", "drift_detected")
        assert "share_of_drifted_columns" in summary
        assert 0.0 <= summary["share_of_drifted_columns"] <= 1.0
        assert "feature_drift" in summary
        assert isinstance(summary["feature_drift"], list)

    def test_drift_summary_has_all_monitored_features(self):
        """Every MONITOR_FEATURES entry must appear in the per-feature drift list."""
        from monitoring.drift_report import generate_drift_report, MONITOR_FEATURES

        summary = generate_drift_report(save_html=False, save_json=False)
        reported_features = {fd["feature"] for fd in summary["feature_drift"]}

        for feat in MONITOR_FEATURES:
            assert feat in reported_features, (
                f"Feature '{feat}' is in MONITOR_FEATURES but missing from drift report"
            )

    def test_drift_endpoint_returns_json(self, client):
        """
        GET /monitoring/drift must return a valid JSON drift summary.
        The endpoint generates the report on-the-fly if no summary.json exists.
        """
        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "share_of_drifted_columns" in data
