"""FastAPI backend — Superhost Predictor & Performance Simulator"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model_loader import get_model, get_metadata

BASE_DIR  = Path(__file__).parent.parent
GEO_FILE  = BASE_DIR / 'neighbourhoods.geojson'
NEIGH_FILE = BASE_DIR / 'ml_pipeline' / 'neighbourhood_stats.json'

app = FastAPI(title="Superhost Predictor API")

# ── Static files ────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ── Model info ──────────────────────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    meta = get_metadata()
    return JSONResponse(meta)


# ── Neighbourhood stats ─────────────────────────────────────────────────────
@app.get("/neighbourhood-stats")
def neighbourhood_stats():
    if not NEIGH_FILE.exists():
        raise HTTPException(503, "neighbourhood_stats.json not found. Run training first.")
    with open(NEIGH_FILE) as f:
        return JSONResponse(json.load(f))


# ── GeoJSON ──────────────────────────────────────────────────────────────────
@app.get("/geojson")
def geojson():
    if not GEO_FILE.exists():
        raise HTTPException(404, "GeoJSON not found")
    with open(GEO_FILE) as f:
        return JSONResponse(json.load(f))


# ── Prediction ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    review_scores_rating:  float = 4.5
    reviews_per_month:     float = 2.0
    host_response_rate:    float = 90.0
    host_acceptance_rate:  float = 85.0
    host_experience_years: float = 5.0
    host_listings_count:   float = 3.0
    num_amenities:         float = 30.0


def _build_row(slider_vals: dict[str, float], meta: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with all model features, using defaults for non-slider cols."""
    row = meta["feature_defaults"].copy()
    row.update(slider_vals)
    # Re-derive log1p features
    for base in ["host_listings_count", "number_of_reviews", "review_count", "avg_comment_length"]:
        lp = f"{base}_log1p"
        if lp in row and base in row:
            row[lp] = float(np.log1p(row[base]))
    num_feats = meta["numeric_features"]
    cat_feats = meta["categorical_features"]
    cols = num_feats + cat_feats
    return pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])


def _predict_prob(row_df: pd.DataFrame) -> float:
    pipeline = get_model()
    return float(pipeline.predict_proba(row_df)[0, 1])


def _generate_recommendations(slider_vals: dict, base_prob: float, meta: dict) -> list[dict]:
    """For each improvable slider feature, compute the probability delta at superhost median."""
    sh_avg = meta.get("superhost_avg", {})
    FEATURE_LABELS = {
        "review_scores_rating":  "Review Score Rating",
        "reviews_per_month":     "Reviews per Month",
        "host_response_rate":    "Response Rate",
        "host_acceptance_rate":  "Acceptance Rate",
        "num_amenities":         "Number of Amenities",
    }
    IMPROVEMENT_DIRECTION = {  # True = higher is better
        "review_scores_rating":  True,
        "reviews_per_month":     True,
        "host_response_rate":    True,
        "host_acceptance_rate":  True,
        "num_amenities":         True,
    }
    MESSAGE_TEMPLATES = {
        "review_scores_rating":  "Improving your **review score** from **{cur:.1f}** → **{tgt:.1f}** could boost your probability by **+{d:.0%}**. Focus on cleanliness and communication.",
        "reviews_per_month":     "Increasing **booking velocity** from **{cur:.1f}** → **{tgt:.1f}** reviews/month could add **+{d:.0%}**. Optimize your pricing and availability calendar.",
        "host_response_rate":    "Boosting your **response rate** from **{cur:.0f}%** → **{tgt:.0f}%** could gain **+{d:.0%}**. Enable Airbnb notifications and respond within 1 hour.",
        "host_acceptance_rate":  "Raising your **acceptance rate** from **{cur:.0f}%** → **{tgt:.0f}%** could add **+{d:.0%}**. Keep your calendar up-to-date to reduce declines.",
        "num_amenities":         "Adding **more amenities** (**{cur:.0f}** → **{tgt:.0f}** items) could boost by **+{d:.0%}**. Essentials like fast Wi-Fi and workspace matter most.",
    }

    recs = []
    for feat, label in FEATURE_LABELS.items():
        target = sh_avg.get(feat)
        current = slider_vals.get(feat)
        if target is None or current is None:
            continue
        higher_is_better = IMPROVEMENT_DIRECTION.get(feat, True)
        already_good = current >= target if higher_is_better else current <= target
        if already_good:
            continue
        # Compute delta probability
        new_vals = slider_vals.copy()
        new_vals[feat] = target
        meta2 = get_metadata()
        row_df = _build_row(new_vals, meta2)
        new_prob = _predict_prob(row_df)
        delta = new_prob - base_prob
        if delta < 0.005:
            continue
        msg = MESSAGE_TEMPLATES.get(feat, "Improving **{feat}** could help.").format(
            cur=current, tgt=target, d=delta
        )
        recs.append({
            "feature": feat,
            "label": label,
            "current": round(current, 2),
            "target": round(target, 2),
            "delta_probability": round(delta, 4),
            "message": msg,
        })
    recs.sort(key=lambda x: -x["delta_probability"])
    return recs[:3]


@app.post("/predict")
def predict(req: PredictRequest):
    meta = get_metadata()
    slider_vals = req.model_dump()
    row_df = _build_row(slider_vals, meta)
    probability = _predict_prob(row_df)
    recommendations = _generate_recommendations(slider_vals, probability, meta)
    return {
        "probability": round(probability, 4),
        "prediction": "Superhost" if probability >= 0.5 else "Not Yet Superhost",
        "recommendations": recommendations,
    }


# ── Scale simulation ─────────────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    review_scores_rating:  float = 4.5
    reviews_per_month:     float = 2.0
    host_response_rate:    float = 90.0
    host_acceptance_rate:  float = 85.0
    host_experience_years: float = 5.0
    num_amenities:         float = 30.0
    max_listings:          int   = 50


@app.post("/simulate")
def simulate(req: SimulateRequest):
    meta = get_metadata()
    base_vals = req.model_dump()
    max_l = base_vals.pop("max_listings", 50)
    curve = []
    for n in range(1, max_l + 1):
        vals = base_vals.copy()
        vals["host_listings_count"] = float(n)
        row_df = _build_row(vals, meta)
        prob = _predict_prob(row_df)
        curve.append({"listings": n, "probability": round(prob, 4)})
    # Find sweet spot (peak)
    peak = max(curve, key=lambda x: x["probability"])
    return {"curve": curve, "sweet_spot": peak["listings"]}
