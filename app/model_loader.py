"""Singleton model loader — loads best model & metadata once at startup."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import joblib

ML_DIR = Path(__file__).parent.parent / 'ml_pipeline'

# Prefer best_model.joblib (dynamic winner); fall back to xgb_model.joblib
# for backwards compatibility with existing trained artefacts.
_BEST_MODEL_PATH = ML_DIR / 'best_model.joblib'
_XGB_MODEL_PATH  = ML_DIR / 'xgb_model.joblib'
META_PATH        = ML_DIR / 'model_metadata.json'


def _resolve_model_path() -> Path:
    if _BEST_MODEL_PATH.exists():
        return _BEST_MODEL_PATH
    if _XGB_MODEL_PATH.exists():
        return _XGB_MODEL_PATH
    raise RuntimeError(
        f"No trained model found in {ML_DIR}. "
        "Run: python ml_pipeline/train_model.py"
    )


@lru_cache(maxsize=1)
def get_model():
    path = _resolve_model_path()
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    if not META_PATH.exists():
        raise RuntimeError(
            f"Metadata not found at {META_PATH}. "
            "Run: python ml_pipeline/train_model.py"
        )
    with open(META_PATH) as f:
        return json.load(f)
