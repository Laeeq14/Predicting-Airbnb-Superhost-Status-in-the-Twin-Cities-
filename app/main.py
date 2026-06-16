"""FastAPI backend — Superhost Predictor & Performance Simulator"""
from __future__ import annotations
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model_loader import get_model, get_metadata
from . import agent as agent_module

logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
GEO_FILE  = BASE_DIR / 'neighbourhoods.geojson'
NEIGH_FILE = BASE_DIR / 'ml_pipeline' / 'neighbourhood_stats.json'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run agent data build in a background thread so it doesn't block startup."""
    def _build():
        try:
            pipeline = get_model()
            meta     = get_metadata()
            agent_module.build_agent_data(pipeline, meta)
        except Exception as exc:
            logger.warning(f"Agent data build failed: {exc}")
    threading.Thread(target=_build, daemon=True).start()
    yield


app = FastAPI(title="Superhost Predictor API", lifespan=lifespan)

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
# ── Amenity flags (mirrors train_model.AMENITY_FLAGS) ──────────────────────────
AMENITY_FLAG_COLS = [
    'amenity_coffee', 'amenity_wine_glasses', 'amenity_baking_sheet',
    'amenity_extra_pillows_blankets', 'amenity_shower_gel', 'amenity_toaster',
    'amenity_hair_dryer', 'amenity_iron', 'amenity_cooking_basics',
    'amenity_dishes_silverware', 'amenity_long_term_stays', 'amenity_self_check_in',
    'amenity_dining_table', 'amenity_private_entrance', 'amenity_essentials',
    'amenity_hangers', 'amenity_room_darkening_shades', 'amenity_dishwasher',
    'amenity_dedicated_workspace', 'amenity_hot_water',
]


class PredictRequest(BaseModel):
    review_scores_rating:  float = 4.5
    reviews_per_month:     float = 2.0
    host_response_rate:    float = 90.0
    host_acceptance_rate:  float = 85.0
    host_experience_years: float = 5.0
    host_listings_count:   float = 3.0
    # 20 binary amenity flags (0 = absent, 1 = present)
    amenity_coffee:                  int = 0
    amenity_wine_glasses:            int = 0
    amenity_baking_sheet:            int = 0
    amenity_extra_pillows_blankets:  int = 0
    amenity_shower_gel:              int = 0
    amenity_toaster:                 int = 0
    amenity_hair_dryer:              int = 0
    amenity_iron:                    int = 0
    amenity_cooking_basics:          int = 0
    amenity_dishes_silverware:       int = 0
    amenity_long_term_stays:         int = 0
    amenity_self_check_in:           int = 0
    amenity_dining_table:            int = 0
    amenity_private_entrance:        int = 0
    amenity_essentials:              int = 0
    amenity_hangers:                 int = 0
    amenity_room_darkening_shades:   int = 0
    amenity_dishwasher:              int = 0
    amenity_dedicated_workspace:     int = 0
    amenity_hot_water:               int = 0


def _build_row(slider_vals: dict[str, float], meta: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with all model features, using defaults for non-slider cols."""
    row = meta["feature_defaults"].copy()
    row.update(slider_vals)
    # Derive num_amenities from the binary amenity flags (consistent with training)
    row['num_amenities'] = float(sum(slider_vals.get(col, 0) for col in AMENITY_FLAG_COLS))
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
    """For each improvable slider feature and each missing amenity, compute the probability delta."""
    sh_avg = meta.get("superhost_avg", {})
    amenity_labels = meta.get("amenity_flags", {})   # col_name -> human label

    FEATURE_LABELS = {
        "review_scores_rating":  "Review Score Rating",
        "reviews_per_month":     "Reviews per Month",
        "host_response_rate":    "Response Rate",
        "host_acceptance_rate":  "Acceptance Rate",
    }
    IMPROVEMENT_DIRECTION = {  # True = higher is better
        "review_scores_rating":  True,
        "reviews_per_month":     True,
        "host_response_rate":    True,
        "host_acceptance_rate":  True,
    }
    MESSAGE_TEMPLATES = {
        "review_scores_rating":  "Improving your **review score** from **{cur:.1f}** → **{tgt:.1f}** could boost your probability by **+{d:.0%}**. Focus on cleanliness and communication.",
        "reviews_per_month":     "Increasing **booking velocity** from **{cur:.1f}** → **{tgt:.1f}** reviews/month could add **+{d:.0%}**. Optimize your pricing and availability calendar.",
        "host_response_rate":    "Boosting your **response rate** from **{cur:.0f}%** → **{tgt:.0f}%** could gain **+{d:.0%}**. Enable Airbnb notifications and respond within 1 hour.",
        "host_acceptance_rate":  "Raising your **acceptance rate** from **{cur:.0f}%** → **{tgt:.0f}%** could add **+{d:.0%}**. Keep your calendar up-to-date to reduce declines.",
    }

    recs = []

    # ── Continuous slider features ────────────────────────────────────────────
    for feat, label in FEATURE_LABELS.items():
        target = sh_avg.get(feat)
        current = slider_vals.get(feat)
        if target is None or current is None:
            continue
        higher_is_better = IMPROVEMENT_DIRECTION.get(feat, True)
        already_good = current >= target if higher_is_better else current <= target
        if already_good:
            continue
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
            "type": "slider",
        })

    # ── Amenity what-if: flip each absent amenity to 1 and measure delta ──────
    for col in AMENITY_FLAG_COLS:
        if slider_vals.get(col, 0) == 1:
            continue  # already has this amenity
        label = amenity_labels.get(col, col.replace('amenity_', '').replace('_', ' ').title())
        new_vals = slider_vals.copy()
        new_vals[col] = 1
        meta2 = get_metadata()
        row_df = _build_row(new_vals, meta2)
        new_prob = _predict_prob(row_df)
        delta = new_prob - base_prob
        if delta < 0.005:
            continue
        recs.append({
            "feature": col,
            "label": label,
            "current": 0,
            "target": 1,
            "delta_probability": round(delta, 4),
            "message": f"Adding **{label}** could boost your Superhost probability by **+{delta:.0%}**. "
                        f"Superhosts offer this amenity significantly more often — it signals genuine hospitality.",
            "type": "amenity",
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
    max_listings:          int   = 50
    # 20 binary amenity flags
    amenity_coffee:                  int = 0
    amenity_wine_glasses:            int = 0
    amenity_baking_sheet:            int = 0
    amenity_extra_pillows_blankets:  int = 0
    amenity_shower_gel:              int = 0
    amenity_toaster:                 int = 0
    amenity_hair_dryer:              int = 0
    amenity_iron:                    int = 0
    amenity_cooking_basics:          int = 0
    amenity_dishes_silverware:       int = 0
    amenity_long_term_stays:         int = 0
    amenity_self_check_in:           int = 0
    amenity_dining_table:            int = 0
    amenity_private_entrance:        int = 0
    amenity_essentials:              int = 0
    amenity_hangers:                 int = 0
    amenity_room_darkening_shades:   int = 0
    amenity_dishwasher:              int = 0
    amenity_dedicated_workspace:     int = 0
    amenity_hot_water:               int = 0


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


# ── Agent 3 ──────────────────────────────────────────────────────────────────────
@app.get("/agent/at-risk")
def agent_at_risk(county: str | None = None):
    listings = agent_module.get_at_risk_listings(county=county)
    if not listings:
        status = "loading" if not agent_module._agent_cache.get("at_risk_all") else "ready"
        return JSONResponse({"status": status, "listings": []})
    return JSONResponse({"status": "ready", "listings": listings})


@app.get("/agent/counties")
def agent_counties():
    """Return all unique county names present in the full at-risk pool."""
    counties = agent_module.get_available_counties()
    return JSONResponse({"counties": counties})


@app.post("/agent/tickets/{listing_id}")
def agent_tickets(listing_id: int):
    try:
        result = agent_module.generate_tickets_for_listing(listing_id)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {e}")
