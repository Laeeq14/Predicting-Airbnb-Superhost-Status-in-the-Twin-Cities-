"""
Quick patch: re-saves model_metadata.json with amenity_flags and amenity_superhost_rates
without retraining. Reads existing models + data to compute the missing keys.
"""
import sys, json, re, joblib, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = Path(__file__).parent

# ── Load the two CSVs (same logic as train_model.py) ────────────────
JUNE_LISTINGS = BASE_DIR / 'listings_detailed_june.csv'
SEP_LISTINGS  = BASE_DIR / 'listings_new.csv'
SCRAPE_DATE_JUNE = pd.to_datetime('2025-06-22')
SCRAPE_DATE_SEP  = pd.to_datetime('2025-09-24')

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

TARGET = 'host_is_superhost'

def load_amenities(filepath, scrape_date, period_label):
    df = pd.read_csv(filepath, usecols=['id', 'host_id', 'host_is_superhost', 'amenities'], low_memory=False)
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().map({'t': 1, 'f': 0})
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    df = df.dropna(subset=[TARGET])
    df['period'] = period_label

    if 'amenities' in df.columns:
        amenity_lower = df['amenities'].astype(str).str.lower()
        for col, match in AMENITY_FLAGS.items():
            df[col] = amenity_lower.str.contains(match, regex=False).astype(int)
    return df

print("Loading data for amenity rate calculation...")
df_june = load_amenities(JUNE_LISTINGS, SCRAPE_DATE_JUNE, 'june_2025')
df_sep  = load_amenities(SEP_LISTINGS,  SCRAPE_DATE_SEP,  'sep_2025')
combined = pd.concat([df_june, df_sep], ignore_index=True)

# Host-aware split (same seed as training)
unique_hosts = combined['host_id'].unique()
rng = np.random.default_rng(42)
rng.shuffle(unique_hosts)
n_train = int(len(unique_hosts) * 0.80)
train_hosts = set(unique_hosts[:n_train])
train_mask = combined['host_id'].isin(train_hosts).values

y_train = combined[TARGET].values[train_mask]
sh_mask  = y_train == 1
nsh_mask = y_train == 0

amenity_superhost_rates = {}
for col in AMENITY_FLAGS:
    sh_arr  = combined[col].values[train_mask][sh_mask]
    nsh_arr = combined[col].values[train_mask][nsh_mask]
    sh_rate  = round(float(sh_arr.mean()),  4)
    nsh_rate = round(float(nsh_arr.mean()), 4)
    amenity_superhost_rates[col] = {
        'sh_pct':  round(sh_rate  * 100, 1),
        'nsh_pct': round(nsh_rate * 100, 1),
        'diff':    round((sh_rate - nsh_rate) * 100, 1),
    }
    print(f"  {col:<40} SH={sh_rate*100:.1f}% NonSH={nsh_rate*100:.1f}%")

# ── Load existing metadata and patch ─────────────────────────────────
META_PATH = OUT_DIR / 'model_metadata.json'
with open(META_PATH) as f:
    metadata = json.load(f)

metadata['amenity_flags']           = AMENITY_LABELS
metadata['amenity_superhost_rates'] = amenity_superhost_rates

# Also update slider_features to include amenity flags (remove num_amenities)
SLIDER_FEATURES = [
    'review_scores_rating', 'reviews_per_month', 'host_response_rate',
    'host_acceptance_rate', 'host_experience_years', 'host_listings_count',
    *AMENITY_FLAGS.keys(),
]
SLIDER_CONFIG = {
    'review_scores_rating':  {'label': 'Review Score Rating',    'min': 1.0,  'max': 5.0,  'step': 0.1, 'unit': '/ 5.0'},
    'reviews_per_month':     {'label': 'Reviews per Month',      'min': 0.0,  'max': 10.0, 'step': 0.1, 'unit': '/mo'},
    'host_response_rate':    {'label': 'Response Rate',          'min': 0.0,  'max': 100.0,'step': 1.0, 'unit': '%'},
    'host_acceptance_rate':  {'label': 'Acceptance Rate',        'min': 0.0,  'max': 100.0,'step': 1.0, 'unit': '%'},
    'host_experience_years': {'label': 'Host Experience',        'min': 0.0,  'max': 20.0, 'step': 0.5, 'unit': 'yrs'},
    'host_listings_count':   {'label': 'Number of Listings',     'min': 1.0,  'max': 50.0, 'step': 1.0, 'unit': ''},
}
metadata['slider_features'] = SLIDER_FEATURES
metadata['slider_config']   = SLIDER_CONFIG

# Update superhost_avg / non_superhost_avg for amenity flags too
sh_mask_full  = combined[TARGET] == 1
nsh_mask_full = combined[TARGET] == 0
existing_sh  = metadata.get('superhost_avg', {})
existing_nsh = metadata.get('non_superhost_avg', {})
for col in AMENITY_FLAGS:
    existing_sh[col]  = round(float(combined[col][sh_mask_full].median()),  3)
    existing_nsh[col] = round(float(combined[col][nsh_mask_full].median()), 3)
metadata['superhost_avg']     = existing_sh
metadata['non_superhost_avg'] = existing_nsh

with open(META_PATH, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("\n[OK] model_metadata.json patched with amenity_flags and amenity_superhost_rates")
print(f"     amenity_flags keys: {list(AMENITY_LABELS.keys())[:3]} ...")
print(f"     amenity_superhost_rates keys: {list(amenity_superhost_rates.keys())[:3]} ...")
