# Superhost Predictor — Twin Cities 🏠

> **Predict Airbnb Superhost status and turn predictions into action.**
> Upload a host profile, get a probability, and receive SHAP-backed recommendations, a what-if simulator, and Groq-generated operational tickets from real guest reviews — all in one FastAPI-powered stack.

Report: [group23_final_report.pdf](https://github.com/user-attachments/files/24547128/group23_final_report.pdf)

---

## 📈 Benchmark Results — 11 Model Comparison

*Evaluated on a host-aware held-out test set (20% of unique host IDs). Both scrape periods (June + September 2025) combined.*

| Model | Accuracy | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.7439 | 0.8168 | 0.7611 | 0.7415 | 0.7818 |
| Decision Tree | 0.7615 | 0.8194 | 0.7792 | 0.7538 | 0.8064 |
| Random Forest | **0.7925** | **0.8703** | **0.8026** | 0.7970 | 0.8084 |
| XGBoost | 0.7647 | 0.8463 | 0.7757 | 0.7718 | 0.7797 |
| LightGBM | 0.7749 | 0.8540 | 0.7885 | 0.7734 | 0.8043 |
| CatBoost | 0.7802 | 0.8621 | 0.7932 | 0.7794 | 0.8074 |
| Random Forest (Tuned) | 0.7882 | 0.8663 | 0.7990 | 0.7918 | 0.8064 |
| XGBoost (Tuned) | 0.7749 | 0.8537 | 0.7904 | 0.7686 | 0.8135 |
| LightGBM (Tuned) | 0.7754 | 0.8550 | 0.7885 | 0.7752 | 0.8023 |
| CatBoost (Tuned) | 0.7797 | 0.8595 | 0.7948 | 0.7733 | 0.8176 |
| Voting Ensemble | 0.7813 | 0.8660 | 0.7952 | 0.7777 | 0.8135 |

**Winner: Random Forest — ROC-AUC 0.8703, F1 0.8026**

Saved as `ml_pipeline/best_model.joblib`. Winner is determined programmatically by highest ROC-AUC on the held-out test set — no manual selection.

---

## ⚡ What Makes This Different

| Capability | Detail |
|---|---|
| **11 models, 6 families** | LR, DT, RF, XGBoost, LightGBM, CatBoost — all tuned with `RandomizedSearchCV(n_iter=100)` + StratifiedKFold |
| **Host-aware train/test split** | Splits on unique `host_id`, not rows — prevents leakage when the same host appears in both scrape periods |
| **Dual scrape periods** | June 22, 2025 + September 24, 2025 Twin Cities data combined for temporal coverage |
| **20 discriminative amenity flags** | Binary features selected by ≥14pp Superhost-rate difference AND ≥15% listing coverage |
| **SHAP at-risk agent** | Batch SHAP on 100 at-risk listings at startup; surfaces hosts where low rating is the dominant negative driver |
| **Groq LLM ticket generation** | Last 5 guest reviews → Llama-3.3-70b → categorized, prioritized operational tickets |
| **What-if simulator** | Sweeps `host_listings_count` 1→50, returns the full probability curve and the peak listing count |
| **Actionable recommendations** | `/predict` computes probability deltas for each improvable feature and missing amenity; returns top 3 |
| **Evidently drift monitoring** | CI-safe drift gate using synthetic reference data from `model_metadata.json` — no 80MB CSVs needed |
| **CI/CD regression gate** | GitHub Actions: ruff lint + pytest + Evidently drift check with ≤50% threshold on every PR |

---

## 🏗️ Architecture

```
Two Airbnb scrape CSVs (June + Sep 2025)
           │
           ▼
┌──────────────────────────────────┐
│  ml_pipeline/train_model.py      │  Train + tune 11 models
│  → best_model.joblib             │  Host-aware split on host_id
│  → model_metadata.json           │  Performance, defaults, superhost avgs
└──────────────┬───────────────────┘
               │
        ┌──────▼──────────────────────────────────────────────┐
        │  app/main.py  (FastAPI)                              │
        │                                                      │
        │  POST /predict             → probability + recs      │
        │  POST /simulate            → listings-count curve    │
        │  GET  /neighbourhood-stats → GeoJSON choropleth      │
        │  GET  /agent/at-risk       → SHAP-ranked hosts       │
        │  GET  /agent/counties      → county filter           │
        │  POST /agent/tickets/{id}  → Groq LLM tickets        │
        │  GET  /monitoring/drift    → Evidently JSON          │
        │  GET  /monitoring/drift-report → HTML report         │
        └──────────────────────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │  app/static/        │
            │  index.html (UI)    │
            └─────────────────────┘
```

### SHAP At-Risk Agent (`app/agent.py`)

```
Startup (background thread at FastAPI lifespan)
  ↓
Load listings → filter: rating < 4.8, reviews ≥ 5
  ↓
Batch SHAP (100 listings, single vectorized call — ~10× faster than per-row)
  ↓
Filter: rating_shap < -0.05  (rating is the dominant negative driver)
  ↓
Detect missing high-impact amenities  (shap < -0.03 AND amenity absent)
  ↓
Rank by Superhost probability ascending (worst hosts first)
  ↓
Cache → /agent/at-risk  (optional ?county= filter)

On POST /agent/tickets/{listing_id}:
  Retrieve last 5 guest reviews
  → Groq Llama-3.3-70b (json_object mode)
  → Pydantic-validated TicketList
     category: Maintenance | Housekeeping | Amenities | Communication
     priority: Low | Medium | High
     root_cause: specific problem
     recommended_action: concrete fix
```

---

## 📊 Dataset

| Source | File | Contents |
|---|---|---|
| Inside Airbnb (Jun 2025) | `listings_detailed_june.csv` | 5,000+ listings — host, location, amenities, ratings |
| Inside Airbnb (Sep 2025) | `listings_new.csv` | Second scrape period, same schema |
| Reviews (Jun) | `reviews_detailed_june.csv` | Review text + timestamps |
| Reviews (Sep) | `reviews.csv` | Second scrape period reviews |
| Neighbourhoods | `neighbourhoods.geojson` | Twin Cities GeoJSON for choropleth |

**Target**: `host_is_superhost` (binary) · **Superhost rate**: 52.2% across combined dataset

---

## 🔧 Feature Engineering

| Feature | How computed |
|---|---|
| `host_experience_years` | `(scrape_date − host_since).days / 365.25` — per scrape period |
| `num_amenities` | Regex count of quoted items in raw amenities string |
| 20 binary amenity flags | `amenities.str.contains(match)` — substring match per amenity |
| `review_count` | Count of review rows per `listing_id` from reviews CSV |
| `avg_comment_length` | Mean character length of review comments per listing |
| `*_log1p` | `np.log1p()` on `host_listings_count`, `number_of_reviews`, `review_count`, `avg_comment_length` |

**Amenity flags** (selected by ≥14pp Superhost-rate difference AND ≥15% coverage):
`coffee` · `wine glasses` · `baking sheet` · `extra pillows & blankets` · `shower gel` · `toaster` · `hair dryer` · `iron` · `cooking basics` · `dishes & silverware` · `long-term stays` · `self check-in` · `dining table` · `private entrance` · `essentials` · `hangers` · `room-darkening shades` · `dishwasher` · `dedicated workspace` · `hot water`

**Superhost vs. Non-Superhost averages** (from `model_metadata.json`):

| Feature | Superhost | Non-Superhost |
|---|---|---|
| `review_scores_rating` | **4.93** | 4.75 |
| `host_acceptance_rate` | **99%** | 98% |
| `host_experience_years` | **8.6 yrs** | 6.6 yrs |
| `reviews_per_month` | 1.53 | 1.20 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (for the at-risk agent ticket generation)

### 1. Clone

```bash
git clone https://github.com/Laeeq14/Predicting-Airbnb-Superhost-Status-in-the-Twin-Cities-.git
cd Predicting-Airbnb-Superhost-Status-in-the-Twin-Cities-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_xxxx
```

### 4. (Optional) Re-train the model

> Skip if you just want to run the app — `best_model.joblib` and `model_metadata.json` are already committed.

Download the CSVs from [Inside Airbnb](https://insideairbnb.com/get-the-data/), place them in the project root, then:

```bash
python ml_pipeline/train_model.py
```

This trains all 11 models, prints the benchmark table, and saves `best_model.joblib` + `model_metadata.json`.

### 5. Run the app

```bash
python run_app.py
```

Or directly:

```bash
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | HTML frontend |
| `GET` | `/model-info` | Model name, metrics, feature list |
| `POST` | `/predict` | Superhost probability + top 3 actionable recommendations |
| `POST` | `/simulate` | Probability curve across listing counts (1–50) |
| `GET` | `/neighbourhood-stats` | Per-neighbourhood superhost rate, median price, review score |
| `GET` | `/geojson` | Twin Cities neighbourhood GeoJSON |
| `GET` | `/agent/at-risk` | SHAP-ranked at-risk listings (optional `?county=` filter) |
| `GET` | `/agent/counties` | County names in the at-risk pool |
| `POST` | `/agent/tickets/{listing_id}` | Groq LLM task tickets from guest reviews |
| `GET` | `/monitoring/drift` | Evidently drift summary JSON |
| `GET` | `/monitoring/drift-report` | Full interactive Evidently HTML report |

### Example — predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "review_scores_rating": 4.7,
    "reviews_per_month": 1.2,
    "host_response_rate": 85.0,
    "host_acceptance_rate": 90.0,
    "host_experience_years": 3.0,
    "host_listings_count": 2.0,
    "amenity_hair_dryer": 1,
    "amenity_essentials": 1
  }'
```

---

## 🧪 Test Suite

```bash
pytest tests/ -v
pytest tests/ -v --cov=app --cov=monitoring --cov-report=term-missing
```

| Test class | Coverage |
|---|---|
| `TestModelLoad` | `best_model.joblib` deserializes; `model_metadata.json` has all required keys; model outputs probability in [0, 1] |
| `TestAPIEndpoints` | `/model-info`, `/predict` (default + ideal Superhost profile → probability > 0.5), `/neighbourhood-stats` |
| `TestDriftMonitoring` | Drift report runs end-to-end (synthetic path, no CSVs); all monitored features appear in output; `/monitoring/drift` returns valid JSON |

CI design: `GROQ_API_KEY` monkeypatched with a fake key; `build_agent_data` mocked so no CSV loading occurs during tests; only committed artifacts used directly.

---

## 📡 CI/CD Pipeline

Two jobs on every push and PR to `main`:

**Job 1 — Lint & Test** (hard gate)

```yaml
ruff check app/ monitoring/ tests/
pytest tests/ -v --cov=app --cov=monitoring
```

Coverage report uploaded as artifact. Must pass before drift check runs.

**Job 2 — Evidently Drift Gate** (informational, `continue-on-error: true`)

```yaml
python -m monitoring.drift_report
```

- Runs against synthetic reference data — no large CSVs required
- Fails if more than **50%** of monitored features show statistical drift
- Drift summary posted as a comment on every PR
- Promote to a hard gate by removing `continue-on-error` when wired to real inference logs

---

## 📁 Project Structure

```
Predicting-Airbnb-Superhost-Status-in-the-Twin-Cities-/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # Ruff + pytest + Evidently drift gate
│
├── ml_pipeline/
│   ├── train_model.py              # 11-model training pipeline, host-aware split
│   ├── best_model.joblib           # Winning model (Random Forest, ROC-AUC 0.8703)
│   ├── model_metadata.json         # Performance table, feature lists, superhost avgs
│   └── neighbourhood_stats.json    # Per-neighbourhood stats for choropleth
│
├── app/
│   ├── main.py                     # FastAPI — predict, simulate, agent, drift
│   ├── agent.py                    # SHAP at-risk agent + Groq LLM ticket generation
│   ├── model_loader.py             # Lazy-loads model + metadata on first request
│   └── static/
│       └── index.html              # Vanilla HTML/JS frontend
│
├── monitoring/
│   ├── drift_report.py             # Evidently AI drift monitor (synthetic reference)
│   ├── drift_report.html           # Full interactive HTML report
│   └── drift_summary.json          # Machine-readable drift summary
│
├── tests/
│   └── test_api.py                 # pytest — model, API, drift (3 test classes)
│
├── classification_modeling.ipynb   # Original EDA + notebook-phase experiments
├── run_app.py                      # Convenience launcher
└── requirements.txt                # FastAPI, sklearn, SHAP, Evidently, Groq, ruff
```

---

## 🤖 Key Design Decisions

**Host-aware train/test split** — The dataset combines two scrape periods. The same host appears in both. A naive `train_test_split()` puts the same host on both sides of the split, letting the model memorize host-specific patterns and inflating test metrics. Splitting on `host_id` ensures no host's data appears in both train and test — equivalent to group K-fold validation.

**20 specific amenity flags** — `num_amenities` treats all amenities equally. The 20 flags were selected by two criteria: ≥14pp Superhost-rate difference AND ≥15% listing coverage. This makes each flag individually actionable in the at-risk agent ("add a baking sheet" vs. "add more amenities").

**Batch SHAP** — Running `TreeExplainer.shap_values()` per row on 100 listings took ~30 seconds at startup. Restructured to preprocess all rows into a single matrix and call `shap_values()` once — ~10× faster.

**Synthetic Evidently reference data** — Training CSVs are ~80MB each, unsuitable for git or CI. Feature medians and class-conditional averages are stored in `model_metadata.json` at training time. A 200-row synthetic reference DataFrame is reconstructed from those statistics at drift-check time — statistically sufficient, zero large-file dependencies.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML** | scikit-learn, XGBoost, LightGBM, SHAP |
| **API** | FastAPI, Uvicorn, Pydantic |
| **LLM** | Groq (`llama-3.3-70b-versatile`) via OpenAI-compatible client |
| **Monitoring** | Evidently AI |
| **CI** | GitHub Actions, ruff, pytest, pytest-cov |
| **Frontend** | Vanilla HTML/JS |

---

## 📄 License

MIT
