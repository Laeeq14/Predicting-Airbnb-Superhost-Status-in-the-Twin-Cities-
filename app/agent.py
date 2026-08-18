"""
Agent 3 — Smart Task Ticketing
Pipeline: LightGBM (Tuned) prediction → SHAP attribution → review retrieval → Groq LLM → structured tickets
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import shap
from openai import OpenAI
from pydantic import BaseModel

# Load .env from the project root (two levels up from this file)
load_dotenv(Path(__file__).parent.parent / ".env")

# Suppress expected SHAP + sklearn warnings that fill server logs
warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", message=".*shap values output has changed.*")

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
LISTINGS_CSV  = BASE_DIR / "listings_detailed_june.csv"
REVIEWS_CSV   = BASE_DIR / "reviews_detailed_june.csv"

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to a .env file at the project root: GROQ_API_KEY=your_key_here"
    )

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MIN_REVIEWS   = 5
AT_RISK_N     = 20   # number of listings to surface in the UI per view
REVIEW_N      = 5    # last N reviews sent to LLM

# Amenity flags (mirrors train_model.AMENITY_FLAGS) — used to parse raw CSV
AMENITY_FLAGS = {
    'amenity_coffee':                  'coffee',
    'amenity_wine_glasses':            'wine glasses',
    'amenity_baking_sheet':            'baking sheet',
    'amenity_extra_pillows_blankets':  'extra pillows and blankets',
    'amenity_shower_gel':              'shower gel',
    'amenity_toaster':                 'toaster',
    'amenity_hair_dryer':              'hair dryer',
    'amenity_iron':                    'iron',
    'amenity_cooking_basics':          'cooking basics',
    'amenity_dishes_silverware':       'dishes and silverware',
    'amenity_long_term_stays':         'long term stays allowed',
    'amenity_self_check_in':           'self check-in',
    'amenity_dining_table':            'dining table',
    'amenity_private_entrance':        'private entrance',
    'amenity_essentials':              'essentials',
    'amenity_hangers':                 'hangers',
    'amenity_room_darkening_shades':   'room-darkening shades',
    'amenity_dishwasher':              'dishwasher',
    'amenity_dedicated_workspace':     'dedicated workspace',
    'amenity_hot_water':               'hot water',
}
AMENITY_LABELS = {
    'amenity_coffee':                  'Coffee',
    'amenity_wine_glasses':            'Wine Glasses',
    'amenity_baking_sheet':            'Baking Sheet',
    'amenity_extra_pillows_blankets':  'Extra Pillows & Blankets',
    'amenity_shower_gel':              'Shower Gel',
    'amenity_toaster':                 'Toaster',
    'amenity_hair_dryer':              'Hair Dryer',
    'amenity_iron':                    'Iron',
    'amenity_cooking_basics':          'Cooking Basics',
    'amenity_dishes_silverware':       'Dishes & Silverware',
    'amenity_long_term_stays':         'Long-Term Stays Allowed',
    'amenity_self_check_in':           'Self Check-In',
    'amenity_dining_table':            'Dining Table',
    'amenity_private_entrance':        'Private Entrance',
    'amenity_essentials':              'Essentials',
    'amenity_hangers':                 'Hangers',
    'amenity_room_darkening_shades':   'Room-Darkening Shades',
    'amenity_dishwasher':              'Dishwasher',
    'amenity_dedicated_workspace':     'Dedicated Workspace',
    'amenity_hot_water':               'Hot Water',
}


# ── Pydantic schemas ─────────────────────────────────────────────────────────
class Ticket(BaseModel):
    category:           Literal["Maintenance", "Housekeeping", "Amenities", "Communication"]
    priority:           Literal["Low", "Medium", "High"]
    root_cause:         str
    recommended_action: str


class TicketList(BaseModel):
    listing_id:   int
    listing_name: str
    tickets:      list[Ticket]


# ── Module-level cache ───────────────────────────────────────────────────────
_agent_cache: dict[str, Any] = {}   # populated once by build_agent_data()
_groq_client: OpenAI | None = None


def _get_groq() -> OpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
    return _groq_client


# ── Data loading helpers ──────────────────────────────────────────────────────
_LISTING_COLS = [
    "id", "name", "host_name", "neighbourhood_cleansed",
    "review_scores_rating", "number_of_reviews", "reviews_per_month",
    "host_response_rate", "host_acceptance_rate", "host_listings_count",
    "host_total_listings_count", "host_identity_verified",
    "latitude", "longitude", "accommodates", "bathrooms", "bedrooms",
    "beds", "price", "minimum_nights", "maximum_nights",
    "number_of_reviews", "review_scores_cleanliness",
    "review_scores_communication", "review_scores_value",
    "property_type", "room_type",
    "amenities",   # needed to derive specific amenity flags
]


def _parse_price(s) -> float:
    """Strip dollar signs and commas, return float."""
    try:
        return float(str(s).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return float("nan")


def _parse_pct(s) -> float:
    """Strip percent sign, return float 0-100."""
    try:
        return float(str(s).replace("%", "").strip())
    except (ValueError, TypeError):
        return float("nan")


def _safe_read_listings() -> pd.DataFrame:
    needed = list(dict.fromkeys(_LISTING_COLS))   # deduplicated, order preserved
    df = pd.read_csv(LISTINGS_CSV, usecols=lambda c: c in needed, low_memory=False)
    if "price" in df.columns:
        df["price"] = df["price"].apply(_parse_price)
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_pct)
    for col in ["host_identity_verified"]:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "t": 1, "f": 0}).fillna(0).astype(float)
    return df


# ── SHAP helpers ──────────────────────────────────────────────────────────────
def _build_feature_row(listing_row: pd.Series, meta: dict) -> pd.DataFrame:
    """Turn a raw listing Series into the one-row DataFrame the pipeline expects."""
    defaults = meta["feature_defaults"].copy()

    mapping = {
        "host_response_rate":    "host_response_rate",
        "host_acceptance_rate":  "host_acceptance_rate",
        "host_listings_count":   "host_listings_count",
        "host_total_listings_count": "host_total_listings_count",
        "host_identity_verified": "host_identity_verified",
        "latitude":  "latitude",
        "longitude": "longitude",
        "accommodates": "accommodates",
        "bathrooms":    "bathrooms",
        "bedrooms":     "bedrooms",
        "beds":         "beds",
        "price":        "price",
        "minimum_nights": "minimum_nights",
        "maximum_nights": "maximum_nights",
        "number_of_reviews": "number_of_reviews",
        "review_scores_rating": "review_scores_rating",
        "review_scores_cleanliness":    "review_scores_cleanliness",
        "review_scores_communication":  "review_scores_communication",
        "review_scores_value":          "review_scores_value",
        "reviews_per_month":    "reviews_per_month",
        "neighbourhood_cleansed": "neighbourhood_cleansed",
        "property_type": "property_type",
        "room_type":     "room_type",
    }

    for src, dst in mapping.items():
        if src in listing_row.index and pd.notna(listing_row[src]):
            defaults[dst] = listing_row[src]

    # Derived fields
    defaults["host_experience_years"] = meta["feature_defaults"].get("host_experience_years", 5.0)
    # Parse amenity flags from raw amenities string (replaces generic num_amenities)
    amenity_str = str(listing_row.get('amenities', '')).lower()
    for col, match in AMENITY_FLAGS.items():
        defaults[col] = 1 if match in amenity_str else 0
    defaults["num_amenities"] = float(sum(defaults.get(c, 0) for c in AMENITY_FLAGS))
    defaults["review_count"]  = defaults.get("number_of_reviews", defaults.get("review_count", 20.0))
    defaults["avg_comment_length"] = meta["feature_defaults"].get("avg_comment_length", 200.0)

    # log1p features
    for base in ["host_listings_count", "number_of_reviews", "review_count", "avg_comment_length"]:
        lp = f"{base}_log1p"
        if lp in defaults and base in defaults:
            defaults[lp] = float(np.log1p(defaults[base]))

    num_feats = meta["numeric_features"]
    cat_feats = meta["categorical_features"]
    cols = num_feats + cat_feats
    return pd.DataFrame([{c: defaults.get(c, np.nan) for c in cols}])


def _compute_shap_for_listing(listing_row: pd.Series, pipeline, meta: dict) -> dict[str, float]:
    """
    Returns a dict mapping original feature names → SHAP value.
    Positive SHAP = pushes toward Superhost; Negative = hurts probability.
    """
    X = _build_feature_row(listing_row, meta)
    preprocessor = pipeline["preproc"]
    clf          = pipeline["clf"]

    # Build transformed feature name list
    num_names = meta["numeric_features"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    ohe = cat_transformer.named_steps["ohe"]
    cat_names = list(ohe.get_feature_names_out(meta["categorical_features"]))
    all_names = num_names + cat_names

    # Transform to numpy, then wrap back into DataFrame with feature names
    # so LightGBM doesn't raise feature-name warnings
    X_np = preprocessor.transform(X)
    X_transformed = pd.DataFrame(X_np, columns=all_names)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_transformed)

    # SHAP return shapes vary by model/version:
    #   list of 2 arrays  -> [neg_class, pos_class], each (n_samples, n_features)
    #   3D ndarray        -> (n_samples, n_features, n_classes)  [RF with newer SHAP]
    #   2D ndarray        -> (n_samples, n_features)              [already positive class]
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1]).flatten()          # positive class (Superhost)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]                        # 3D: sample 0, positive class
    else:
        sv = np.array(shap_vals).flatten()

    return dict(zip(all_names, sv))


# ── Main agent builder ────────────────────────────────────────────────────────
def build_agent_data(pipeline, meta: dict) -> None:
    """
    Called once at startup. Loads data, runs SHAP on at-risk listings,
    caches results in _agent_cache.
    """
    logger.info("Agent: loading listings CSV…")
    listings = _safe_read_listings()

    # Filter to at-risk: rating below 4.8, enough reviews
    at_risk_mask = (
        listings["review_scores_rating"].notna()
        & (listings["review_scores_rating"] < 4.8)
        & (listings["number_of_reviews"] >= MIN_REVIEWS)
    )
    at_risk = listings[at_risk_mask].copy()
    logger.info(f"Agent: {len(at_risk)} at-risk listings found")

    # Load reviews index (listing_id → count)
    logger.info("Agent: loading reviews CSV…")
    reviews_df = pd.read_csv(REVIEWS_CSV, usecols=["listing_id", "date", "comments"],
                             low_memory=False)
    reviews_df = reviews_df.dropna(subset=["comments"])
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")

    # Keep only at-risk listings that have reviews
    valid_ids = set(reviews_df["listing_id"].unique())
    at_risk   = at_risk[at_risk["id"].isin(valid_ids)].copy()

    # Run SHAP on a sample to find where rating is the dominant negative driver.
    # Limit to 100 listings — enough to surface a rich at-risk pool with fast load.
    sample = at_risk.sample(min(100, len(at_risk)), random_state=42)

    # Build SHAP explainer ONCE (expensive) — reuse for all listings
    preprocessor    = pipeline["preproc"]
    clf             = pipeline["clf"]
    num_names       = meta["numeric_features"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    ohe             = cat_transformer.named_steps["ohe"]
    cat_names       = list(ohe.get_feature_names_out(meta["categorical_features"]))
    all_feat_names  = num_names + cat_names
    explainer       = shap.TreeExplainer(clf)

    # ── Vectorised batch SHAP (single call, ~10x faster than per-row) ──
    # Preprocess all rows into a single matrix first, then call shap_values once.
    row_list   = []
    valid_rows = []
    for _, row in sample.iterrows():
        try:
            X    = _build_feature_row(row, meta)
            X_np = preprocessor.transform(X)
            row_list.append(X_np[0])
            valid_rows.append(row)
        except Exception as exc:
            logger.debug(f"Agent preproc skipped listing {row.get('id')}: {exc}")

    ranked_listings = []
    if not row_list:
        logger.warning("Agent: no listings could be preprocessed for SHAP.")
    else:
        X_batch = pd.DataFrame(row_list, columns=all_feat_names)

        # Single batch SHAP call
        try:
            sv_batch = explainer.shap_values(X_batch)
        except Exception as exc:
            logger.error(f"Agent: SHAP batch failed: {exc}")
            sv_batch = None

        if sv_batch is not None:
            # X_batch is already preprocessed — use clf directly (not pipeline)
            # to avoid re-running the OHE preprocessor on already-transformed data
            probs_batch = clf.predict_proba(X_batch.values)[:, 1]

            # Handle all SHAP return shapes:
            #   list of 2 arrays  -> [neg_class, pos_class], each (n, features)
            #   3D ndarray        -> (n, features, n_classes)  [RF with newer SHAP]
            #   2D ndarray        -> (n, features)              [already positive class]
            if isinstance(sv_batch, list):
                sv_pos_batch = np.array(sv_batch[1])       # positive class
            elif isinstance(sv_batch, np.ndarray) and sv_batch.ndim == 3:
                sv_pos_batch = sv_batch[:, :, 1]           # 3D: slice positive class
            else:
                sv_pos_batch = np.array(sv_batch)          # 2D: already positive class

            for i, row in enumerate(valid_rows):
                try:
                    sv_pos   = sv_pos_batch[i]
                    shap_map = dict(zip(all_feat_names, sv_pos))

                    rating_shap = shap_map.get("review_scores_rating", 0.0)
                    if rating_shap < -0.05:
                        prob = float(probs_batch[i])
                        # Use listing row directly to detect absent amenities
                        amenity_str = str(row.get("amenities", "")).lower()
                        missing_amenity_labels = [
                            AMENITY_LABELS[col]
                            for col, match in AMENITY_FLAGS.items()
                            if shap_map.get(col, 0) < -0.03 and match not in amenity_str
                        ]
                        ranked_listings.append({
                            "listing_id":        int(row["id"]),
                            "listing_name":      str(row.get("name", "Unknown"))[:60],
                            "host_name":         str(row.get("host_name", "Unknown")),
                            "county":            str(row.get("neighbourhood_cleansed", "—")),
                            "rating":            round(float(row["review_scores_rating"]), 2),
                            "review_count":      int(row["number_of_reviews"]),
                            "probability":       round(prob, 3),
                            "rating_shap":       round(rating_shap, 4),
                            "missing_amenities": missing_amenity_labels[:3],
                        })
                except Exception as exc:
                    logger.debug(f"Agent SHAP skipped listing {row.get('id')}: {exc}")

    # Sort by worst probability first
    # Filter out listing IDs > JS Number.MAX_SAFE_INTEGER (2^53-1)
    # to avoid silent precision loss when the ID is serialised to JSON in the browser.
    JS_MAX_SAFE = 9_007_199_254_740_991
    ranked_listings = [r for r in ranked_listings if r["listing_id"] <= JS_MAX_SAFE]
    ranked_listings.sort(key=lambda x: x["probability"])

    # Store ALL ranked listings so per-county slicing can draw top N from each county
    _agent_cache["at_risk_all"] = ranked_listings
    _agent_cache["at_risk"]     = ranked_listings[:AT_RISK_N]   # global top-20 (default)
    _agent_cache["reviews_df"]  = reviews_df
    logger.info(f"Agent: {len(ranked_listings)} listings ranked; top {AT_RISK_N} surfaced globally")


def get_at_risk_listings(county: str | None = None) -> list[dict]:
    """Return top AT_RISK_N listings, optionally filtered to a single county."""
    if county:
        all_listings = _agent_cache.get("at_risk_all", [])
        county_listings = [r for r in all_listings if r["county"].lower() == county.lower()]
        return county_listings[:AT_RISK_N]
    return _agent_cache.get("at_risk", [])


def get_available_counties() -> list[str]:
    """Return sorted list of unique county names across all ranked at-risk listings."""
    all_listings = _agent_cache.get("at_risk_all", [])
    return sorted({r["county"] for r in all_listings if r["county"] and r["county"] != "—"})


# ── Ticket generation ─────────────────────────────────────────────────────────
def generate_tickets_for_listing(listing_id: int) -> dict:
    """
    Retrieves last REVIEW_N review comments for listing_id, calls Groq,
    returns TicketList-compatible dict.
    """
    # Search the full ranked pool so county-filtered listings (outside the global
    # top-20 slice) can still have tickets generated.
    cache = _agent_cache.get("at_risk_all", _agent_cache.get("at_risk", []))
    listing_info = next((x for x in cache if x["listing_id"] == listing_id), None)
    if listing_info is None:
        raise ValueError(f"Listing {listing_id} not in at-risk cache")

    reviews_df: pd.DataFrame = _agent_cache["reviews_df"]
    listing_reviews = (
        reviews_df[reviews_df["listing_id"] == listing_id]
        .sort_values("date", ascending=False)
        .head(REVIEW_N)
    )

    if listing_reviews.empty:
        raise ValueError(f"No reviews found for listing {listing_id}")

    # Format review text
    review_text = "\n\n".join([
        f"Review {i+1} ({row['date'].strftime('%Y-%m') if pd.notna(row['date']) else 'Unknown'}):\n{row['comments']}"
        for i, (_, row) in enumerate(listing_reviews.iterrows())
    ])

    system_prompt = (
        "You are an Airbnb property operations analyst. "
        "Analyse the provided guest reviews and extract specific, actionable operational issues. "
        "Cluster related complaints into task tickets. "
        "Be concrete: root_cause should name the exact problem (e.g. 'No coffee maker in kitchen'), "
        "recommended_action should be a direct fix (e.g. 'Purchase and install a drip coffee maker'). "
        "Only generate tickets for real, specific issues mentioned in the reviews. "
        "Return a JSON object with this exact structure: "
        '{"listing_id": 0, "listing_name": "", "tickets": '
        '[{"category": "Maintenance or Housekeeping or Amenities or Communication", '
        '"priority": "Low or Medium or High", '
        '"root_cause": "specific problem from reviews", '
        '"recommended_action": "concrete fix action"}]}'
    )

    user_prompt = (
        f"Property: {listing_info['listing_name']} ({listing_info['county']} County)\n"
        f"Current rating: {listing_info['rating']}/5.0\n"
    )
    missing = listing_info.get('missing_amenities', [])
    if missing:
        user_prompt += f"Missing high-impact amenities: {', '.join(missing)}\n"
    user_prompt += f"\nRecent guest reviews:\n{review_text}\n\nExtract operational issues and generate task tickets."

    client = _get_groq()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Ensure listing metadata is injected (LLM may not know the IDs)
    parsed["listing_id"]   = listing_id
    parsed["listing_name"] = listing_info["listing_name"]
    return parsed
