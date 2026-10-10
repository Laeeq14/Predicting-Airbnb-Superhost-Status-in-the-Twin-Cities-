"""
Graph Embedding Experiment — Spectral Host Co-occurrence Embeddings
====================================================================
Hypothesis
----------
  Hosts in the same neighbourhood share guest expectations, operational norms,
  and market dynamics. A host <-> host co-occurrence graph (same neighbourhood)
  may encode spatial social signal not captured by the raw `neighbourhood_cleansed`
  one-hot feature or lat/lon alone.

Approach
--------
  1. Load the same June + Sep 2025 listing CSVs used by train_model.py.
  2. Build a host <-> host UNDIRECTED weighted graph:
       edge(host_A, host_B)  iff  both have >=1 listing in the same neighbourhood
       edge_weight = number of shared neighbourhoods
  3. Compute a 64-dimensional SPECTRAL GRAPH EMBEDDING via truncated SVD on
     the row-normalised adjacency matrix (equivalent to the leading eigenvectors
     of the graph's transition matrix). This is the linear algebraic counterpart
     of DeepWalk / Node2Vec without requiring a C++ compiler.
       - Isolated nodes receive the zero vector.
  4. Map each listing row to its host's embedding vector (64 new features).
  5. Augment the existing tabular feature matrix with the 64 embedding dims.
  6. Train Random Forest + XGBoost on:
       (a) Tabular only   -- baseline (replicates train_model.py conditions)
       (b) Tabular + Graph embeddings -- augmented
  7. Compare ROC-AUC, F1, Precision, Recall on the same host-aware test split.
  8. Save results to experiments/graph_experiment_results.json.
  9. Print an honest ablation summary.

Dependencies — ALL already in requirements.txt / base install:
  networkx>=3.3   (graph construction + stats)
  scipy           (sparse matrix + truncated SVD)
  scikit-learn    (preprocessing + models)
  numpy, pandas

Run:
  cd Predicting-Airbnb-Superhost-Status-in-the-Twin-Cities-
  python -m experiments.graph_node2vec
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent          # project root
EXP_DIR  = Path(__file__).parent                 # experiments/

JUNE_LISTINGS = BASE_DIR / "listings_detailed_june.csv"
SEP_LISTINGS  = BASE_DIR / "listings_new.csv"

SCRAPE_DATE_JUNE = pd.to_datetime("2025-06-22")
SCRAPE_DATE_SEP  = pd.to_datetime("2025-09-24")

TARGET = "host_is_superhost"

# ── Spectral embedding hyper-parameters ──────────────────────────────────────
# Truncated SVD on the row-normalised adjacency matrix — the linear algebraic
# counterpart of DeepWalk / Node2Vec. Requires only scipy + sklearn.
EMBEDDING_DIM = 64     # number of singular vectors (embedding dimension)
SEED          = 42

# ── Columns mirrored from train_model.py ─────────────────────────────────────
COLS_KEEP = [
    "id", "host_id", "host_since", "host_response_rate", "host_acceptance_rate",
    "host_is_superhost", "host_listings_count", "host_total_listings_count",
    "host_identity_verified", "neighbourhood_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms", "bathrooms_text",
    "bedrooms", "beds", "amenities", "price", "minimum_nights", "maximum_nights",
    "number_of_reviews", "review_scores_rating", "review_scores_cleanliness",
    "review_scores_communication", "review_scores_value", "reviews_per_month",
]

DROP_FOR_MODEL = [
    "id", "host_id", "host_since", "amenities",
    "bathrooms_text", "period", "scrape_date",
]

AMENITY_FLAGS = {
    "amenity_coffee":                 "coffee",
    "amenity_wine_glasses":           "wine glasses",
    "amenity_baking_sheet":           "baking sheet",
    "amenity_extra_pillows_blankets": "extra pillows and blankets",
    "amenity_shower_gel":             "shower gel",
    "amenity_toaster":                "toaster",
    "amenity_hair_dryer":             "hair dryer",
    "amenity_iron":                   "iron",
    "amenity_cooking_basics":         "cooking basics",
    "amenity_dishes_silverware":      "dishes and silverware",
    "amenity_long_term_stays":        "long term stays allowed",
    "amenity_self_check_in":          "self check-in",
    "amenity_dining_table":           "dining table",
    "amenity_private_entrance":       "private entrance",
    "amenity_essentials":             "essentials",
    "amenity_hangers":                "hangers",
    "amenity_room_darkening_shades":  "room-darkening shades",
    "amenity_dishwasher":             "dishwasher",
    "amenity_dedicated_workspace":    "dedicated workspace",
    "amenity_hot_water":              "hot water",
}


# =============================================================================
# 1. DATA LOADING  (mirrors train_model.py exactly)
# =============================================================================

def load_listings(filepath: Path, scrape_date: pd.Timestamp, period: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
    cols = [c for c in COLS_KEEP if c in df.columns]
    df = df[cols].copy()
    df.drop_duplicates(subset="id", inplace=True)
    df = df[df[TARGET].notna()].copy()
    df["period"]      = period
    df["scrape_date"] = scrape_date

    for col in [TARGET, "host_identity_verified"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map({"t": 1, "f": 0})
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = df[col].replace({"": np.nan}).astype(float)

    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace(r"[\$,]", "", regex=True)
        df["price"] = df["price"].replace({"": np.nan}).astype(float)

    df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
    df["host_experience_years"] = (
        (scrape_date - df["host_since"]).dt.days / 365.25
    ).clip(lower=0).fillna(0)

    if "bathrooms" not in df.columns or df["bathrooms"].isna().all():
        if "bathrooms_text" in df.columns:
            def _bath(x: str) -> float:
                m = re.search(r"(\d+\.?\d*)", str(x))
                return float(m.group(1)) if m else np.nan
            df["bathrooms"] = df["bathrooms_text"].apply(_bath)

    if "amenities" in df.columns:
        df["num_amenities"] = df["amenities"].apply(
            lambda x: len(re.findall(r"""["']([^"']+)["']""", str(x)))
        )
        amenity_lower = df["amenities"].astype(str).str.lower()
        for col, match in AMENITY_FLAGS.items():
            df[col] = amenity_lower.str.contains(match, regex=False).astype(int)
    else:
        df["num_amenities"] = 0
        for col in AMENITY_FLAGS:
            df[col] = 0
    df["num_amenities"] = df["num_amenities"].fillna(0)
    return df


def prepare_features(df: pd.DataFrame):
    """Clean, engineer log features, return (X, y, host_ids)."""
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].replace(["", "nan", "None", "none", "NaN"], np.nan)

    for c in [c for c in df.columns if c.startswith("review_scores_")]:
        df.loc[df["number_of_reviews"] == 0, c] = 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    for c in ["number_of_reviews", "review_count", "avg_comment_length",
              "host_listings_count"]:
        if c in df.columns:
            df[f"{c}_log1p"] = np.log1p(df[c])

    host_ids = df["host_id"].values.copy()
    drop_cols = [c for c in DROP_FOR_MODEL if c in df.columns] + [TARGET]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET].astype(int)
    return X, y, host_ids


# =============================================================================
# 2. GRAPH CONSTRUCTION
# =============================================================================

def build_host_graph(df_raw: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected weighted graph:
      Nodes  = unique host_ids
      Edges  = two hosts share >=1 listing in the same neighbourhood
      Weight = number of shared neighbourhoods
    """
    print("\n[Graph] Building host<->host co-occurrence graph...")

    neigh_hosts: dict[str, set] = {}
    for _, row in df_raw[["host_id", "neighbourhood_cleansed"]].drop_duplicates().iterrows():
        nb = str(row["neighbourhood_cleansed"])
        h  = int(row["host_id"])
        neigh_hosts.setdefault(nb, set()).add(h)

    G = nx.Graph()
    for h in df_raw["host_id"].unique():
        G.add_node(int(h))

    edge_weights: dict[tuple[int, int], int] = {}
    for _nb, hosts in neigh_hosts.items():
        host_list = sorted(hosts)
        for i in range(len(host_list)):
            for j in range(i + 1, len(host_list)):
                a, b = host_list[i], host_list[j]
                key  = (min(a, b), max(a, b))
                edge_weights[key] = edge_weights.get(key, 0) + 1

    for (a, b), w in edge_weights.items():
        G.add_edge(a, b, weight=w)

    degrees  = [d for _, d in G.degree()]
    isolated = sum(1 for d in degrees if d == 0)
    print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}  |  "
          f"Isolated: {isolated:,}  |  Avg degree: {np.mean(degrees):.2f}")
    return G


# =============================================================================
# 3. SPECTRAL GRAPH EMBEDDING  (scipy sparse SVD — no C++ required)
# =============================================================================

def learn_embeddings(G: nx.Graph) -> dict[int, np.ndarray]:
    """
    Compute a spectral graph embedding via truncated SVD on the
    row-normalised adjacency matrix A_norm = D^{-1} A.

    This is the linear algebraic equivalent of DeepWalk / Node2Vec:
    the leading singular vectors capture the principal directions of
    structural co-occurrence in the graph — nodes with similar
    neighbourhood patterns end up close in embedding space.

    Isolated nodes (degree 0) receive the zero vector.
    Requires only scipy + sklearn — no C++ compiler needed.
    """
    nodes       = sorted(G.nodes())          # stable ordering
    n           = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    print(f"\n[Graph] Building sparse adjacency matrix  ({n:,} x {n:,})...")

    # Build weighted COO sparse adjacency matrix
    rows, cols, data = [], [], []
    for u, v, attr in G.edges(data=True):
        w = float(attr.get("weight", 1.0))
        i, j = node_to_idx[u], node_to_idx[v]
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

    # Row-normalise: A_norm = D^{-1} A  (transition matrix)
    degree = np.asarray(A.sum(axis=1)).flatten()
    degree_inv = np.where(degree > 0, 1.0 / degree, 0.0)
    D_inv = sp.diags(degree_inv)
    A_norm = D_inv.dot(A)

    # Truncated SVD — k must be < min(n, n), cap at available rank
    k = min(EMBEDDING_DIM, n - 2)      # svds requires k < min(shape)
    print(f"[Graph] Running truncated SVD  k={k}  (effective embedding dim)...")
    U, S, _Vt = svds(A_norm, k=k, random_state=SEED)

    # Sort by descending singular value (svds returns ascending order)
    order = np.argsort(S)[::-1]
    U     = U[:, order]
    S     = S[order]

    # Scale: U * sqrt(S)  so that variance is proportional to singular value
    emb_matrix = U * np.sqrt(S)[np.newaxis, :]     # (n, k)

    # L2-normalise rows so all hosts live on the unit hypersphere
    emb_matrix = normalize(emb_matrix, norm="l2", axis=1)

    # Pad to EMBEDDING_DIM if actual k < requested dim (sparse graph edge case)
    if k < EMBEDDING_DIM:
        pad = np.zeros((n, EMBEDDING_DIM - k), dtype=np.float32)
        emb_matrix = np.hstack([emb_matrix, pad])

    # Build host_id -> vector dict; isolated nodes keep zero vector
    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    embeddings: dict[int, np.ndarray] = {}
    for node, idx in node_to_idx.items():
        vec = emb_matrix[idx]
        # Zero-out vectors for isolated nodes (degree == 0)
        embeddings[node] = vec if degree[idx] > 0 else zero.copy()

    covered  = sum(1 for v in embeddings.values() if v.any())
    isolated = sum(1 for v in embeddings.values() if not v.any())
    print(f"  Non-zero embeddings: {covered:,}  |  Isolated (zero): {isolated:,}")
    return embeddings


def attach_embeddings(
    X: pd.DataFrame,
    host_ids: np.ndarray,
    embeddings: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Append EMBEDDING_DIM graph columns to X."""
    zero = np.zeros(EMBEDDING_DIM)
    cols = [f"graph_emb_{i}" for i in range(EMBEDDING_DIM)]
    mat  = np.vstack([embeddings.get(int(h), zero) for h in host_ids])
    emb_df = pd.DataFrame(mat, columns=cols, index=X.index)
    return pd.concat([X.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)


# =============================================================================
# 4. MODEL TRAINING & EVALUATION
# =============================================================================

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric     = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X_train.select_dtypes(include="object").columns.tolist()
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")


def train_and_eval(
    label: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
    results: dict,
) -> None:
    """Train RF + XGBoost, evaluate, store results in-place."""
    import xgboost as xgb

    preprocessor = build_preprocessor(X_train)

    models = {
        f"RF  [{label}]": Pipeline([
            ("preproc", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_features="sqrt",
                random_state=SEED, n_jobs=-1,
            )),
        ]),
        f"XGB [{label}]": Pipeline([
            ("preproc", preprocessor),
            ("clf", xgb.XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", verbosity=0,
                random_state=SEED, n_jobs=-1,
            )),
        ]),
    }

    for mname, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        r = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "roc_auc":   round(float(roc_auc_score(y_test, y_prob)), 4),
            "f1":        round(float(f1_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall":    round(float(recall_score(y_test, y_pred)), 4),
        }
        results[mname] = r
        print(
            f"  {mname:<44}  "
            f"AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}"
        )


# =============================================================================
# 5. MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("  GRAPH EMBEDDING ABLATION — Node2Vec Host Co-occurrence")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load ---------------------------------------------------------
    print("\n[1/6] Loading listings...")
    df_june = load_listings(JUNE_LISTINGS, SCRAPE_DATE_JUNE, "june_2025")
    df_sep  = load_listings(SEP_LISTINGS,  SCRAPE_DATE_SEP,  "sep_2025")
    df_raw  = pd.concat([df_june, df_sep], ignore_index=True)
    print(f"  Combined: {len(df_raw):,} listings")

    # Graph (built on ALL hosts — Node2Vec is unsupervised, no label leakage)
    G = build_host_graph(df_raw)

    # Embeddings ---------------------------------------------------
    print("\n[2/6] Computing spectral graph embeddings (truncated SVD)...")
    embeddings = learn_embeddings(G)
    emb_out = {str(k): v.tolist() for k, v in embeddings.items()}
    with open(EXP_DIR / "graph_embeddings.json", "w") as fh:
        json.dump(emb_out, fh)
    print("  Saved -> experiments/graph_embeddings.json")

    # Features -----------------------------------------------------
    print("\n[3/6] Preparing tabular features...")
    df_feat = df_raw.copy()
    X_tab, y, host_ids = prepare_features(df_feat)
    print(f"  Samples: {len(X_tab):,}  |  Superhost rate: {y.mean():.1%}")

    print("\n[4/6] Attaching graph embeddings...")
    X_graph = attach_embeddings(X_tab, host_ids, embeddings)
    print(f"  Tabular cols: {X_tab.shape[1]}  |  Augmented cols: {X_graph.shape[1]}")

    # Train/test split (same seed and logic as train_model.py) -----
    print("\n[5/6] Host-aware 80/20 train/test split...")
    unique_hosts = df_raw["host_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_hosts)
    train_hosts = set(unique_hosts[: int(len(unique_hosts) * 0.80)])
    mask = df_raw["host_id"].isin(train_hosts).values

    X_tab_tr,   X_tab_te   = X_tab[mask],   X_tab[~mask]
    X_graph_tr, X_graph_te = X_graph[mask], X_graph[~mask]
    y_tr, y_te = y[mask], y[~mask]
    print(f"  Train: {len(X_tab_tr):,}  |  Test: {len(X_tab_te):,}")

    # Ablation -----------------------------------------------------
    print("\n[6/6] Ablation...")
    print("=" * 70)
    results: dict = {}

    print("\n  --- Tabular only (baseline) ---")
    train_and_eval("tabular_only",  X_tab_tr,   y_tr, X_tab_te,   y_te, results)

    print("\n  --- Tabular + spectral graph embeddings (augmented) ---")
    train_and_eval("tabular_graph", X_graph_tr, y_tr, X_graph_te, y_te, results)

    # Delta summary ------------------------------------------------
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)
    deltas: dict = {}
    # Keys must exactly match those produced by train_and_eval:
    #   RF  uses two spaces:  f"RF  [{label}]"
    #   XGB uses one space:   f"XGB [{label}]"
    key_map = {"RF": "RF  ", "XGB": "XGB "}
    for mt in ["RF", "XGB"]:
        prefix  = key_map[mt]
        tab_k   = f"{prefix}[tabular_only]"
        graph_k = f"{prefix}[tabular_graph]"
        if tab_k in results and graph_k in results:
            d_auc = results[graph_k]["roc_auc"] - results[tab_k]["roc_auc"]
            d_f1  = results[graph_k]["f1"]      - results[tab_k]["f1"]
            arrow = "UP" if d_auc > 0 else ("DOWN" if d_auc < 0 else "FLAT")
            print(f"  {mt:<5}  AUC {arrow} {d_auc:+.4f}  |  F1 {arrow} {d_f1:+.4f}")
            deltas[mt] = {"delta_auc": round(d_auc, 4), "delta_f1": round(d_f1, 4)}

    # Interpretation -----------------------------------------------
    print("\n  FINDING")
    print("  " + "-" * 66)
    any_positive = any(v["delta_auc"] > 0.002 for v in deltas.values())
    if any_positive:
        print("  Graph embeddings improved AUC beyond noise threshold (+0.002).")
        print("  Spatial peer co-occurrence encodes signal beyond the OHE")
        print("  neighbourhood feature and lat/lon — evidence of spatial social")
        print("  contagion in Superhost behaviour.")
    else:
        best = max((v["delta_auc"] for v in deltas.values()), default=0.0)
        if best > 0:
            print(f"  Marginal improvement (+{best:.4f} AUC). Likely noise-level.")
        else:
            print("  Graph embeddings did NOT improve over the tabular baseline.")
        print("  neighbourhood_cleansed OHE + lat/lon already capture the")
        print("  spatial signal available in this dataset. Tabular encoding is")
        print("  sufficient for this graph density and size.")
    print("=" * 70)

    # Persist results ----------------------------------------------
    degrees = [d for _, d in G.degree()]
    output  = {
        "experiment":    "node2vec_host_cooccurrence",
        "run_date":      datetime.now().isoformat(),
        "graph_stats": {
            "num_nodes":      G.number_of_nodes(),
            "num_edges":      G.number_of_edges(),
            "avg_degree":     round(float(np.mean(degrees)), 2),
            "median_degree":  round(float(np.median(degrees)), 2),
            "isolated_nodes": sum(1 for d in degrees if d == 0),
        },
        "hyperparameters": {
            "method":        "spectral_svd",
            "embedding_dim": EMBEDDING_DIM,
            "normalisation": "row (D^{-1} A)",
            "scaling":       "U * sqrt(S)",
            "row_l2_norm":   True,
        },
        "model_results": results,
        "deltas":        deltas,
    }
    out_path = EXP_DIR / "graph_experiment_results.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print("\\n  Results  -> experiments/graph_experiment_results.json")
    print(  "  Embeddings -> experiments/graph_embeddings.json")
    print(  "\n  Done.\n")


if __name__ == "__main__":
    main()
