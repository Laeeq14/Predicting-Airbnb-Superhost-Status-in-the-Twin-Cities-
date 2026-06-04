"""
Superhost Predictor — ML Training Pipeline
Combines June 2025 + September 2025 Twin Cities datasets.
Uses host-aware train/test split to prevent temporal leakage.

Models compared:
  Logistic Regression, Decision Tree,
  Random Forest, Random Forest (Tuned),
  XGBoost, XGBoost (Tuned),
  LightGBM, LightGBM (Tuned),
  CatBoost, CatBoost (Tuned),
  Voting Ensemble (best RF + best XGB + best LGB + best CB)

Winner = highest ROC-AUC on held-out test set.
Saves: best_model.joblib, model_metadata.json, neighbourhood_stats.json
"""
import pandas as pd
import numpy as np
import re, json, joblib, warnings
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import xgboost as xgb

# Optional imports — gracefully skipped if not installed
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed — skipping LightGBM models. "
          "Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("[WARN] catboost not installed — skipping CatBoost models. "
          "Install with: pip install catboost")

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # project root
OUT_DIR    = Path(__file__).parent                 # ml_pipeline/

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

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
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
            lambda x: len(re.findall(r'''["']([^"']+)["']''', str(x)))
        )
    else:
        df['num_amenities'] = 0
    df['num_amenities'] = df['num_amenities'].fillna(0)

    return df


def aggregate_reviews(reviews_df):
    reviews_df = reviews_df.copy()
    reviews_df['comment_len'] = reviews_df['comments'].fillna('').astype(str).str.len()
    agg = (
        reviews_df.groupby('listing_id')
        .agg(review_count=('id', 'count'), avg_comment_length=('comment_len', 'mean'))
        .reset_index()
    )
    return agg


# ─────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────
def evaluate(pipeline, name, X_test, y_test, results):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    r = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'roc_auc':   round(float(roc_auc_score(y_test, y_prob)), 4),
        'f1':        round(float(f1_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred)), 4),
        'recall':    round(float(recall_score(y_test, y_pred)), 4),
    }
    results[name] = r
    print(f"  {name:<36} Acc={r['accuracy']:.4f}  AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")
    return pipeline


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SUPERHOST PREDICTOR — EXTENDED MODEL TRAINING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Load listings ────────────────────────────────────────────
    print("\n[1/9] Loading listings...")
    df_june = load_listings(JUNE_LISTINGS, SCRAPE_DATE_JUNE, 'june_2025')
    df_sep  = load_listings(SEP_LISTINGS,  SCRAPE_DATE_SEP,  'sep_2025')
    combined = pd.concat([df_june, df_sep], ignore_index=True)
    print(f"  June: {len(df_june):,} | Sep: {len(df_sep):,} | Combined: {len(combined):,}")

    # ── 2. Load & aggregate reviews ─────────────────────────────────
    print("\n[2/9] Loading & aggregating reviews...")
    rev_june = pd.read_csv(JUNE_REVIEWS, low_memory=False)
    rev_sep  = pd.read_csv(SEP_REVIEWS,  low_memory=False)
    all_reviews = pd.concat([rev_june, rev_sep], ignore_index=True)
    all_reviews = all_reviews.drop_duplicates(subset=['id'])
    reviews_agg = aggregate_reviews(all_reviews)
    print(f"  Unique reviews: {len(all_reviews):,} | Aggregated listings: {len(reviews_agg):,}")

    # ── 3. Merge reviews → listings ─────────────────────────────────
    df = combined.merge(reviews_agg, left_on='id', right_on='listing_id', how='left')
    df.drop(columns=['listing_id'], inplace=True, errors='ignore')
    df['review_count']       = df['review_count'].fillna(0)
    df['avg_comment_length'] = df['avg_comment_length'].fillna(0)

    # ── 4. Final cleaning ───────────────────────────────────────────
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].replace(['', 'nan', 'None', 'none', 'NaN'], np.nan)

    for c in [c for c in df.columns if c.startswith('review_scores_')]:
        df.loc[df['number_of_reviews'] == 0, c] = 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    for c in ['number_of_reviews', 'review_count', 'avg_comment_length', 'host_listings_count']:
        if c in df.columns:
            df[f'{c}_log1p'] = np.log1p(df[c])

    # ── 5. Neighbourhood stats (for map) ────────────────────────────
    print("\n[3/9] Computing neighbourhood stats...")
    neigh_stats = {}
    for neigh, grp in df.groupby('neighbourhood_cleansed'):
        neigh_stats[str(neigh)] = {
            'superhost_rate':          round(float(grp[TARGET].mean()), 4),
            'listing_count':           int(len(grp)),
            'median_price':            round(float(grp['price'].median()) if 'price' in grp else 0, 2),
            'median_review_score':     round(float(grp['review_scores_rating'].median()), 2),
            'median_response_rate':    round(float(grp['host_response_rate'].median()), 1),
            'median_reviews_pm':       round(float(grp['reviews_per_month'].median()), 2),
            'lat':                     round(float(grp['latitude'].mean()), 5),
            'lon':                     round(float(grp['longitude'].mean()), 5),
        }
    with open(OUT_DIR / 'neighbourhood_stats.json', 'w') as f:
        json.dump(neigh_stats, f, indent=2)
    print(f"  {len(neigh_stats)} neighbourhoods saved")

    # ── 6. Feature preparation ──────────────────────────────────────
    print("\n[4/9] Preparing feature matrix...")
    drop_cols = [c for c in DROP_FOR_MODEL if c in df.columns] + [TARGET]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[TARGET].astype(int)

    numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    print(f"  Numeric features : {len(numeric_features)}")
    print(f"  Categorical      : {categorical_features}")
    print(f"  Samples          : {len(X):,} | Superhost rate: {y.mean():.1%}")

    # ── 7. Host-aware train/test split ──────────────────────────────
    print("\n[5/9] Host-aware train/test split...")
    unique_hosts = df['host_id'].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_hosts)
    n_train = int(len(unique_hosts) * 0.80)
    train_hosts = set(unique_hosts[:n_train])

    train_mask = df['host_id'].isin(train_hosts).values
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train SH: {y_train.mean():.1%} | Test SH: {y_test.mean():.1%}")

    # ── 8. Build preprocessing pipeline ────────────────────────────
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features),
    ], remainder='drop')

    # ── 9. Cross-validation strategy ────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results   = {}   # {name: metrics_dict}
    trained   = {}   # {name: fitted_pipeline}
    N_ITER    = 100  # randomised search iterations — more thorough search

    # ── 10. Base models ──────────────────────────────────────────────
    print("\n[6/9] Training base models...")

    base_models = {
        'Logistic Regression': Pipeline([
            ('preproc', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('preproc', preprocessor),
            ('clf', DecisionTreeClassifier(max_depth=8, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('preproc', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        'XGBoost': Pipeline([
            ('preproc', preprocessor),
            ('clf', xgb.XGBClassifier(n_estimators=200, random_state=42,
                                       eval_metric='logloss', verbosity=0, n_jobs=-1))
        ]),
    }

    if HAS_LGB:
        base_models['LightGBM'] = Pipeline([
            ('preproc', preprocessor),
            ('clf', lgb.LGBMClassifier(n_estimators=200, random_state=42,
                                        verbosity=-1, n_jobs=-1))
        ])

    if HAS_CB:
        base_models['CatBoost'] = Pipeline([
            ('preproc', preprocessor),
            ('clf', CatBoostClassifier(iterations=200, random_seed=42,
                                        verbose=0, thread_count=-1))
        ])

    for name, model in base_models.items():
        model.fit(X_train, y_train)
        trained[name] = evaluate(model, name, X_test, y_test, results)

    # ── 11. Hyperparameter tuning ────────────────────────────────────
    print(f"\n[7/9] Hyperparameter tuning (n_iter={N_ITER} per model)...")

    # — Random Forest —
    rf_params = {
        'clf__n_estimators':      [100, 200, 400, 600],
        'clf__max_depth':         [None, 6, 10, 16, 24],
        'clf__max_features':      ['sqrt', 'log2', 0.2, 0.4, 0.6],
        'clf__min_samples_split': [2, 5, 10, 20],
        'clf__min_samples_leaf':  [1, 2, 4],
        'clf__class_weight':      [None, 'balanced'],
    }
    rf_search = RandomizedSearchCV(
        trained['Random Forest'], rf_params,
        n_iter=N_ITER, scoring='roc_auc', cv=cv,
        n_jobs=-1, verbose=0, random_state=42
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"  RF best params : {rf_search.best_params_}")
    trained['Random Forest (Tuned)'] = evaluate(best_rf, 'Random Forest (Tuned)', X_test, y_test, results)

    # — XGBoost —
    xgb_params = {
        'clf__n_estimators':    [100, 300, 500, 700],
        'clf__max_depth':       [3, 4, 5, 6, 8],
        'clf__learning_rate':   [0.005, 0.01, 0.05, 0.1, 0.2],
        'clf__subsample':       [0.5, 0.6, 0.7, 0.8, 1.0],
        'clf__colsample_bytree':[0.4, 0.6, 0.8, 1.0],
        'clf__reg_alpha':       [0, 0.01, 0.1, 1.0],
        'clf__reg_lambda':      [0.5, 1.0, 2.0, 5.0],
        'clf__gamma':           [0, 0.1, 0.5, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        trained['XGBoost'], xgb_params,
        n_iter=N_ITER, scoring='roc_auc', cv=cv,
        n_jobs=-1, verbose=0, random_state=42
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  XGB best params: {xgb_search.best_params_}")
    trained['XGBoost (Tuned)'] = evaluate(best_xgb, 'XGBoost (Tuned)', X_test, y_test, results)

    # — LightGBM —
    if HAS_LGB:
        lgb_params = {
            'clf__n_estimators':      [100, 300, 500, 700],
            'clf__max_depth':         [-1, 4, 6, 8, 12],
            'clf__learning_rate':     [0.005, 0.01, 0.05, 0.1, 0.2],
            'clf__num_leaves':        [15, 31, 63, 127],
            'clf__subsample':         [0.5, 0.7, 0.8, 1.0],
            'clf__colsample_bytree':  [0.4, 0.6, 0.8, 1.0],
            'clf__reg_alpha':         [0, 0.01, 0.1, 1.0],
            'clf__reg_lambda':        [0, 0.1, 1.0, 5.0],
            'clf__min_child_samples': [5, 10, 20, 50],
        }
        lgb_search = RandomizedSearchCV(
            trained['LightGBM'], lgb_params,
            n_iter=N_ITER, scoring='roc_auc', cv=cv,
            n_jobs=-1, verbose=0, random_state=42
        )
        lgb_search.fit(X_train, y_train)
        best_lgb = lgb_search.best_estimator_
        print(f"  LGB best params: {lgb_search.best_params_}")
        trained['LightGBM (Tuned)'] = evaluate(best_lgb, 'LightGBM (Tuned)', X_test, y_test, results)
    else:
        best_lgb = None

    # — CatBoost —
    if HAS_CB:
        cb_params = {
            'clf__iterations':        [100, 300, 500, 700],
            'clf__depth':             [4, 5, 6, 8, 10],
            'clf__learning_rate':     [0.01, 0.03, 0.05, 0.1, 0.2],
            'clf__l2_leaf_reg':       [1, 3, 5, 10, 20],
            'clf__bagging_temperature':[0.0, 0.5, 1.0, 2.0],
            'clf__border_count':      [32, 64, 128, 254],
        }
        cb_search = RandomizedSearchCV(
            trained['CatBoost'], cb_params,
            n_iter=N_ITER, scoring='roc_auc', cv=cv,
            n_jobs=1, verbose=0, random_state=42   # CatBoost uses internal threads
        )
        cb_search.fit(X_train, y_train)
        best_cb = cb_search.best_estimator_
        print(f"  CB best params : {cb_search.best_params_}")
        trained['CatBoost (Tuned)'] = evaluate(best_cb, 'CatBoost (Tuned)', X_test, y_test, results)
    else:
        best_cb = None

    # — Voting Ensemble (soft vote of the best tuned models) —
    print("\n  Building Voting Ensemble...")
    ensemble_estimators = [
        ('rf',  best_rf.named_steps['clf']),
        ('xgb', best_xgb.named_steps['clf']),
    ]
    if best_lgb is not None:
        ensemble_estimators.append(('lgb', best_lgb.named_steps['clf']))
    if best_cb is not None:
        ensemble_estimators.append(('cb', best_cb.named_steps['clf']))

    # The ensemble runs on already-preprocessed data, so wrap differently
    # Strategy: apply best_xgb's preprocessor then vote on clf outputs
    # Use Pipeline: shared preprocessor + VotingClassifier over raw clf objects
    voting_clf = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        n_jobs=-1
    )
    # Build a unified pipeline: preprocess once, then vote
    X_train_prep = best_xgb.named_steps['preproc'].transform(X_train)
    X_test_prep  = best_xgb.named_steps['preproc'].transform(X_test)
    voting_clf.fit(X_train_prep, y_train)

    # Wrap voting_clf for uniform evaluation: create a pseudo-pipeline
    class PrepVotingPipeline:
        """Thin wrapper so the ensemble behaves like a sklearn Pipeline.
        Supports joblib/pickle serialisation via __getstate__/__setstate__.
        """
        def __init__(self, preproc, voter):
            self.preproc = preproc
            self.voter   = voter
        def predict(self, X):
            return self.voter.predict(self.preproc.transform(X))
        def predict_proba(self, X):
            return self.voter.predict_proba(self.preproc.transform(X))
        def __getstate__(self):
            return self.__dict__
        def __setstate__(self, state):
            self.__dict__.update(state)

    ensemble_pipeline = PrepVotingPipeline(
        best_xgb.named_steps['preproc'],
        voting_clf
    )
    trained['Voting Ensemble'] = evaluate(ensemble_pipeline, 'Voting Ensemble', X_test, y_test, results)

    # ── 12. Select best model by ROC-AUC ────────────────────────────
    print("\n[8/9] Selecting best model...")
    best_name = max(results, key=lambda n: results[n]['roc_auc'])
    best_pipeline = trained[best_name]
    print(f"  Winner: {best_name}  (AUC={results[best_name]['roc_auc']:.4f})")

    # ── 13. Feature importance (from best available tuned tree model) ──
    # Priority: XGBoost Tuned → LightGBM Tuned → Random Forest Tuned
    # All have .feature_importances_ and use the same preprocessor.
    _tree_ref_name = None
    _tree_ref_pipeline = None
    for _candidate in ['XGBoost (Tuned)', 'LightGBM (Tuned)', 'Random Forest (Tuned)']:
        if _candidate in trained and hasattr(trained[_candidate], 'named_steps'):
            _tree_ref_name     = _candidate
            _tree_ref_pipeline = trained[_candidate]
            break
    if _tree_ref_pipeline is None:
        # Absolute fallback: any Pipeline in trained
        for _candidate in ['XGBoost', 'LightGBM', 'Random Forest']:
            if _candidate in trained and hasattr(trained[_candidate], 'named_steps'):
                _tree_ref_name     = _candidate
                _tree_ref_pipeline = trained[_candidate]
                break

    print(f"  Using '{_tree_ref_name}' for feature importance & scale curve.")

    ohe       = _tree_ref_pipeline.named_steps['preproc'].named_transformers_['cat'].named_steps['ohe']
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feat_names = numeric_features + cat_names
    importances    = _tree_ref_pipeline.named_steps['clf'].feature_importances_

    fi_sorted = sorted(zip(all_feat_names, importances.tolist()),
                       key=lambda x: x[1], reverse=True)
    top_features = [{'feature': f, 'importance': round(i, 5)} for f, i in fi_sorted[:20]]

    # ── 14. Feature defaults (median / mode) ─────────────────────────
    feat_defaults = {}
    for col in numeric_features:
        feat_defaults[col] = float(X_train[col].median())
    for col in categorical_features:
        feat_defaults[col] = str(X_train[col].mode().iloc[0]) if not X_train[col].mode().empty else 'missing'

    # ── 15. Superhost vs Non-Superhost medians ───────────────────────
    sh_mask  = y_train == 1
    nsh_mask = y_train == 0
    X_tr_sh  = X_train[sh_mask]
    X_tr_nsh = X_train[nsh_mask]

    superhost_avg = {}
    non_superhost_avg = {}
    for f in SLIDER_FEATURES:
        if f in X_tr_sh.columns:
            superhost_avg[f]     = round(float(X_tr_sh[f].median()), 3)
            non_superhost_avg[f] = round(float(X_tr_nsh[f].median()), 3)

    # ── 16. Scale curve (using best available tree model) ─────────────
    scale_curve = []
    sample_row = feat_defaults.copy()
    for val in range(1, 51):
        sample_row['host_listings_count']       = float(val)
        sample_row['host_listings_count_log1p'] = float(np.log1p(val))
        row_df = pd.DataFrame([sample_row])[X_train.columns]
        prob   = float(_tree_ref_pipeline.predict_proba(row_df)[0, 1])
        scale_curve.append({'listings': val, 'probability': round(prob, 4)})

    # ── 17. Save best model + metadata ──────────────────────────────
    print("\n  Saving model & metadata...")
    joblib.dump(best_pipeline, OUT_DIR / 'best_model.joblib')
    # Save best available tree model for feature importance / scale curve
    joblib.dump(_tree_ref_pipeline, OUT_DIR / 'xgb_model.joblib')
    print(f"  Reference tree model: '{_tree_ref_name}' -> xgb_model.joblib")

    metadata = {
        'best_model_name':     best_name,
        'numeric_features':    numeric_features,
        'categorical_features':categorical_features,
        'feature_defaults':    feat_defaults,
        'feature_importance':  top_features,
        'model_performance':   results,
        'superhost_rate':      round(float(y.mean()), 4),
        'superhost_avg':       superhost_avg,
        'non_superhost_avg':   non_superhost_avg,
        'slider_features':     SLIDER_FEATURES,
        'slider_config':       SLIDER_CONFIG,
        'scale_curve':         scale_curve,
        'total_listings':      int(len(df)),
        'training_date':       datetime.now().isoformat(),
    }
    with open(OUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("  [OK] best_model.joblib")
    print(f"  [OK] xgb_model.joblib  ({_tree_ref_name} — used for feature importance)")
    print("  [OK] model_metadata.json")

    # ── 18. Final comparison table ───────────────────────────────────
    DISPLAY_ORDER = [
        'Logistic Regression', 'Decision Tree',
        'Random Forest', 'Random Forest (Tuned)',
        'XGBoost', 'XGBoost (Tuned)',
        'LightGBM', 'LightGBM (Tuned)',
        'CatBoost', 'CatBoost (Tuned)',
        'Voting Ensemble',
    ]
    print(f"\n[9/9] FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<36} {'Accuracy':>9} {'ROC-AUC':>9} {'F1':>9}")
    print("  " + "-" * 66)
    for name in DISPLAY_ORDER:
        if name not in results:
            continue
        m = results[name]
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<36} {m['accuracy']:>9.4f} {m['roc_auc']:>9.4f} {m['f1']:>9.4f}{marker}")
    print("=" * 70)
    print(f"\n  Best model ({best_name}) saved to best_model.joblib")
    print("  Run: python run_app.py\n")


if __name__ == '__main__':
    main()
