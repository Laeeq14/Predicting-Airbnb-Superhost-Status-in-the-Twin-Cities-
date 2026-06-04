"""
patch_metadata.py  —  Recovery script
Regenerates model_metadata.json from already-saved joblib models.
Run this when train_model.py crashed AFTER saving the .joblib files
but BEFORE writing model_metadata.json.

Usage:
    py ml_pipeline/patch_metadata.py
"""
import json, joblib, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = Path(__file__).parent

JUNE_LISTINGS = BASE_DIR / 'listings_detailed_june.csv'
SEP_LISTINGS  = BASE_DIR / 'listings_new.csv'
JUNE_REVIEWS  = BASE_DIR / 'reviews_detailed_june.csv'
SEP_REVIEWS   = BASE_DIR / 'reviews.csv'

SCRAPE_DATE_JUNE = pd.to_datetime('2025-06-22')
SCRAPE_DATE_SEP  = pd.to_datetime('2025-09-24')

TARGET = 'host_is_superhost'

COLS_KEEP = [
    'id', 'host_id', 'host_since', 'host_response_rate', 'host_acceptance_rate',
    'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
    'host_identity_verified', 'neighbourhood_cleansed', 'latitude', 'longitude',
    'property_type', 'room_type', 'accommodates', 'bathrooms', 'bathrooms_text',
    'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating', 'review_scores_cleanliness',
    'review_scores_communication', 'review_scores_value', 'reviews_per_month'
]
DROP_FOR_MODEL = ['id', 'host_id', 'host_since', 'amenities', 'bathrooms_text', 'period', 'scrape_date']

SLIDER_FEATURES = [
    'review_scores_rating', 'reviews_per_month', 'host_response_rate',
    'host_acceptance_rate', 'host_experience_years', 'host_listings_count', 'num_amenities'
]
SLIDER_CONFIG = {
    'review_scores_rating':  {'label': 'Review Score Rating',    'min': 1.0,  'max': 5.0,  'step': 0.1, 'unit': '/ 5.0'},
    'reviews_per_month':     {'label': 'Reviews per Month',      'min': 0.0,  'max': 10.0, 'step': 0.1, 'unit': '/mo'},
    'host_response_rate':    {'label': 'Response Rate',          'min': 0.0,  'max': 100.0,'step': 1.0, 'unit': '%'},
    'host_acceptance_rate':  {'label': 'Acceptance Rate',        'min': 0.0,  'max': 100.0,'step': 1.0, 'unit': '%'},
    'host_experience_years': {'label': 'Host Experience',        'min': 0.0,  'max': 20.0, 'step': 0.5, 'unit': 'yrs'},
    'host_listings_count':   {'label': 'Number of Listings',     'min': 1.0,  'max': 50.0, 'step': 1.0, 'unit': ''},
    'num_amenities':         {'label': 'Number of Amenities',    'min': 0.0,  'max': 80.0, 'step': 1.0, 'unit': ''},
}

# ── Known results from the training run (from train_log.txt) ────────────────
# Precision/recall filled in below via model evaluation where possible,
# or carried over from prior run for models we can't re-evaluate here.
KNOWN_RESULTS = {
    'Logistic Regression':    {'accuracy': 0.7385, 'roc_auc': 0.8148, 'f1': 0.7576, 'precision': 0.7339, 'recall': 0.7828},
    'Decision Tree':          {'accuracy': 0.7636, 'roc_auc': 0.8102, 'f1': 0.7765, 'precision': 0.7665, 'recall': 0.7869},
    'Random Forest':          {'accuracy': 0.7872, 'roc_auc': 0.8661, 'f1': 0.7959, 'precision': None,   'recall': None},
    'XGBoost':                {'accuracy': 0.7749, 'roc_auc': 0.8541, 'f1': 0.7881, 'precision': 0.7745, 'recall': 0.8023},
    'LightGBM':               {'accuracy': 0.7813, 'roc_auc': 0.8561, 'f1': 0.7950, 'precision': None,   'recall': None},
    'CatBoost':               {'accuracy': 0.7850, 'roc_auc': 0.8608, 'f1': 0.7980, 'precision': None,   'recall': None},
    'Random Forest (Tuned)':  {'accuracy': 0.7845, 'roc_auc': 0.8618, 'f1': 0.7966, 'precision': None,   'recall': None},
    'XGBoost (Tuned)':        {'accuracy': 0.7765, 'roc_auc': 0.8570, 'f1': 0.7910, 'precision': None,   'recall': None},
    'LightGBM (Tuned)':       {'accuracy': 0.7845, 'roc_auc': 0.8675, 'f1': 0.7964, 'precision': None,   'recall': None},
    'CatBoost (Tuned)':       {'accuracy': 0.7717, 'roc_auc': 0.8563, 'f1': 0.7834, 'precision': None,   'recall': None},
    'Voting Ensemble':        {'accuracy': 0.7850, 'roc_auc': 0.8658, 'f1': 0.7978, 'precision': None,   'recall': None},
}


def load_listings(filepath, scrape_date, period_label):
    df = pd.read_csv(filepath, low_memory=False)
    cols = [c for c in COLS_KEEP if c in df.columns]
    df = df[cols].copy()
    df.drop_duplicates(subset='id', inplace=True)
    df = df[df[TARGET].notna()].copy()
    df['period'] = period_label
    df['scrape_date'] = scrape_date
    for col in [TARGET, 'host_identity_verified']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map({'t': 1, 'f': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = df[col].replace({'': np.nan}).astype(float)
    if 'price' in df.columns:
        df['price'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['price'] = df['price'].replace({'': np.nan}).astype(float)
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['host_experience_years'] = ((scrape_date - df['host_since']).dt.days / 365.25).clip(lower=0)
    df['host_experience_years'] = df['host_experience_years'].fillna(0)
    if 'bathrooms' not in df.columns or df['bathrooms'].isna().all():
        if 'bathrooms_text' in df.columns:
            def _extract_bath(x):
                m = re.search(r'(\d+\.?\d*)', str(x))
                return float(m.group(1)) if m else np.nan
            df['bathrooms'] = df['bathrooms_text'].apply(_extract_bath)
    if 'amenities' in df.columns:
        df['num_amenities'] = df['amenities'].apply(
            lambda x: len(re.findall(r'''["']([^"']+)["']''', str(x))))
    else:
        df['num_amenities'] = 0
    df['num_amenities'] = df['num_amenities'].fillna(0)
    return df


def main():
    print("=" * 60)
    print("  PATCH METADATA — regenerating model_metadata.json")
    print("=" * 60)

    # 1. Load + preprocess data (same logic as train_model.py)
    print("\n[1/5] Loading data...")
    df_june = load_listings(JUNE_LISTINGS, SCRAPE_DATE_JUNE, 'june_2025')
    df_sep  = load_listings(SEP_LISTINGS,  SCRAPE_DATE_SEP,  'sep_2025')
    combined = pd.concat([df_june, df_sep], ignore_index=True)

    rev_june = pd.read_csv(JUNE_REVIEWS, low_memory=False)
    rev_sep  = pd.read_csv(SEP_REVIEWS,  low_memory=False)
    all_reviews = pd.concat([rev_june, rev_sep], ignore_index=True)
    all_reviews = all_reviews.drop_duplicates(subset=['id'])
    all_reviews['comment_len'] = all_reviews['comments'].fillna('').astype(str).str.len()
    reviews_agg = (
        all_reviews.groupby('listing_id')
        .agg(review_count=('id', 'count'), avg_comment_length=('comment_len', 'mean'))
        .reset_index()
    )

    df = combined.merge(reviews_agg, left_on='id', right_on='listing_id', how='left')
    df.drop(columns=['listing_id'], inplace=True, errors='ignore')
    df['review_count']       = df['review_count'].fillna(0)
    df['avg_comment_length'] = df['avg_comment_length'].fillna(0)

    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].replace(['', 'nan', 'None', 'none', 'NaN'], np.nan)
    for c in [c for c in df.columns if c.startswith('review_scores_')]:
        df.loc[df['number_of_reviews'] == 0, c] = 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    for c in ['number_of_reviews', 'review_count', 'avg_comment_length', 'host_listings_count']:
        if c in df.columns:
            df[f'{c}_log1p'] = np.log1p(df[c])

    # 2. Train/test split (same seed as training)
    print("[2/5] Splitting data...")
    drop_cols = [c for c in DROP_FOR_MODEL if c in df.columns] + [TARGET]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[TARGET].astype(int)
    numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    unique_hosts = df['host_id'].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_hosts)
    n_train = int(len(unique_hosts) * 0.80)
    train_hosts = set(unique_hosts[:n_train])
    train_mask = df['host_id'].isin(train_hosts).values
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Load saved models and evaluate on test set for precision/recall
    print("[3/5] Loading saved models and evaluating...")
    best_pipeline   = joblib.load(OUT_DIR / 'best_model.joblib')
    tree_ref        = joblib.load(OUT_DIR / 'xgb_model.joblib')

    results = {}
    for name, vals in KNOWN_RESULTS.items():
        results[name] = {k: round(v, 4) for k, v in vals.items() if v is not None}

    # Evaluate best model (LightGBM Tuned) and tree ref (XGBoost Tuned) for full metrics
    for label, pipeline in [('LightGBM (Tuned)', best_pipeline), ('XGBoost (Tuned)', tree_ref)]:
        try:
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            results[label] = {
                'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
                'roc_auc':   round(float(roc_auc_score(y_test, y_prob)), 4),
                'f1':        round(float(f1_score(y_test, y_pred)), 4),
                'precision': round(float(precision_score(y_test, y_pred)), 4),
                'recall':    round(float(recall_score(y_test, y_pred)), 4),
            }
            print(f"  {label:<36} AUC={results[label]['roc_auc']:.4f}")
        except Exception as e:
            print(f"  [WARN] Could not evaluate {label}: {e}")

    best_name = max(results, key=lambda n: results[n]['roc_auc'])
    print(f"  Winner: {best_name}  (AUC={results[best_name]['roc_auc']:.4f})")

    # 4. Feature importance + scale curve from tree_ref (XGBoost Tuned)
    print("[4/5] Computing feature importance and scale curve...")
    ohe = tree_ref.named_steps['preproc'].named_transformers_['cat'].named_steps['ohe']
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feat_names = numeric_features + cat_names
    importances = tree_ref.named_steps['clf'].feature_importances_
    fi_sorted = sorted(zip(all_feat_names, importances.tolist()), key=lambda x: x[1], reverse=True)
    top_features = [{'feature': f, 'importance': round(i, 5)} for f, i in fi_sorted[:20]]

    feat_defaults = {}
    for col in numeric_features:
        feat_defaults[col] = float(X_train[col].median())
    for col in categorical_features:
        feat_defaults[col] = str(X_train[col].mode().iloc[0]) if not X_train[col].mode().empty else 'missing'

    sh_mask  = y_train == 1
    nsh_mask = y_train == 0
    superhost_avg, non_superhost_avg = {}, {}
    for f in SLIDER_FEATURES:
        if f in X_train.columns:
            superhost_avg[f]     = round(float(X_train[sh_mask][f].median()), 3)
            non_superhost_avg[f] = round(float(X_train[nsh_mask][f].median()), 3)

    scale_curve = []
    sample_row = feat_defaults.copy()
    for val in range(1, 51):
        sample_row['host_listings_count']       = float(val)
        sample_row['host_listings_count_log1p'] = float(np.log1p(val))
        row_df = pd.DataFrame([sample_row])[X_train.columns]
        prob   = float(tree_ref.predict_proba(row_df)[0, 1])
        scale_curve.append({'listings': val, 'probability': round(prob, 4)})

    # 5. Write metadata.json
    print("[5/5] Writing model_metadata.json...")
    metadata = {
        'best_model_name':      best_name,
        'numeric_features':     numeric_features,
        'categorical_features': categorical_features,
        'feature_defaults':     feat_defaults,
        'feature_importance':   top_features,
        'model_performance':    results,
        'superhost_rate':       round(float(y.mean()), 4),
        'superhost_avg':        superhost_avg,
        'non_superhost_avg':    non_superhost_avg,
        'slider_features':      SLIDER_FEATURES,
        'slider_config':        SLIDER_CONFIG,
        'scale_curve':          scale_curve,
        'total_listings':       int(len(df)),
        'training_date':        datetime.now().isoformat(),
    }
    with open(OUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  [OK] model_metadata.json written with all 11 models.")
    print(f"  Winner: {best_name}")
    print("\n  Run: py run_app.py\n")


if __name__ == '__main__':
    main()
