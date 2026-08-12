# Restaurants4U: Personalized Restaurant Recommendation Engine
### Phase 1: Core MLOps Pipeline &nbsp;·&nbsp; Phase 2: Serving, Deployment & Monitoring

---

## Team Members & Roles
* **Chipemba Bwacha** – Project Lead (Deployment Lifecycle & Systems Orchestration)
* **Shini Kim** – ML Lead (Feature Engineering, Core Algorithmic Scripts, & Model Optimization)
* **Xihai Luo** – Engineering Lead (MLOps Pipeline Infrastructure, DVC Architecture, & Codebase Versioning)

## Project Overview
An automated, reproducible MLOps pipeline for a **Two-Stage Content-Based Restaurant Recommendation Engine**. A Random Forest model predicts each restaurant's probability of being highly rated from its metadata; that probability is combined with distance from the user via exponential decay to rank real candidates in real time.

---

## Quickstart

```bash
pip install -r requirements-dev.txt

# Run the pipeline (or `dvc repro` to do the same via DVC)
python src/validate_data.py
python src/prepare.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py

# Test
pytest tests/ -v
ruff check .

# Serve — API + web app on http://localhost:8000
uvicorn app.main:app --reload
```

Open `http://localhost:8000/` for the web app, `/docs` for interactive API docs. Full setup (Docker, AWS, CI/CD secrets) is in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Repository Structure
```text
restaurants4u/
├── .github/workflows/
│   ├── ci-cd.yml            # Lint → test → build → push to ECR → deploy to ECS
│   └── monitoring.yml       # Scheduled drift check + auto-retrain PR
├── .dvc/                    # DVC configuration
├── app/                     # FastAPI serving layer
│   ├── main.py              # /predict, /recommend, /nearby, /health
│   ├── model_service.py     # Model loading + inference
│   ├── data_service.py      # Real-time candidate retrieval by location
│   └── schemas.py           # Request/response models
├── frontend/
│   └── index.html           # Minimal web app (served at "/")
├── src/
│   ├── validate_data.py     # Raw data QA + cleaning
│   ├── prepare.py           # Label engineering + train/val/test split
│   ├── feature_engineering.py
│   ├── features.py          # Shared feature transform (train + serve)
│   ├── train.py              # Model fitting with MLflow tracking
│   └── evaluate.py           # Test metrics + plots
├── data/
│   ├── raw/                 # Tracked by DVC
│   ├── validated/ · processed/ · features/   # Pipeline outputs
├── models/                  # Serialized production model (.pkl)
├── monitoring/
│   ├── drift_detection.py   # EvidentlyAI data drift check
│   └── retrain.py           # dvc repro + push, on drift
├── scripts/
│   └── get_real_recommendation.py   # CLI demo against a running API
├── tests/                   # pytest — feature logic + API
├── Dockerfile
├── dvc.yaml / dvc.lock / params.yaml
├── metrics.json
└── docs/DEPLOYMENT.md       # Full setup, secrets, and deployment guide
```

---

## Part 1: Dataset & Pipeline

**Dataset:** "380,000 Restaurants (Mostly USA Based)" from Kaggle (~332MB). Target label `is_highly_rated = Rating >= 4.0`.

| Feature | Type | Transform |
|---|---|---|
| `Latitude` / `Longitude` | Geospatial | Validated to bounding box, used for distance decay |
| `Website` / `Phone` / `Images` | Sparse | Binary presence flag (`has_website`, etc.) — the actual URL is preserved separately for display, not fed to the model |
| `Title` / `Address` | Text | Length used as a structural feature |
| `Category` | Nominal | Frequency-capped to top 20 + `"Other"`, one-hot encoded |

**Pipeline (5 DVC stages):** `validate_data` → `prepare` → `feature_engineering` → `train` → `evaluate`. Config lives in `params.yaml`; the DAG in `dvc.yaml`. `feature_engineering.py` and `app/model_service.py` both import the same transform from `src/features.py`, so training-time and serving-time encoding can't drift apart.

**Storage:** raw data tracked via `dvc add` + S3; pipeline outputs tracked via `dvc.lock`. `dvc pull` / `dvc push` sync with the S3 remote (see Part 1.4 of the original spec for team credential setup).

**Experiment tracking:** `mlflow ui` → `http://127.0.0.1:5000`, experiment `Restaurant_Metadata_Ranking_Classifier`.

---

## Part 2: Serving, Deployment & Monitoring

**API** (`app/`, FastAPI):
- `POST /predict` — score one restaurant
- `POST /recommend` — rank a supplied list of candidates by `P(highly_rated) × e^(−λ·distance)`
- `GET /nearby?lat=&lon=&category=` — retrieves real nearby candidates from the validated dataset and ranks them; no candidate list needed from the caller
- `GET /health` — model/data load status

**Web app** (`frontend/index.html`): a location + category search that calls `/nearby` and shows ranked results, served directly at `/` by the API — no separate frontend server.

**Containerization:** `Dockerfile` builds a slim image with the model, feature report, validated dataset, and app code; healthcheck on `/health`.

**CI/CD** (`.github/workflows/ci-cd.yml`): every push/PR runs `ruff` + `pytest`; merges to `main` additionally build the image, push to ECR, and deploy to ECS Fargate.

**Monitoring** (`.github/workflows/monitoring.yml`): a daily job runs `monitoring/drift_detection.py` (EvidentlyAI `DataDriftPreset`) comparing recent production requests against training data. If drift exceeds threshold, `monitoring/retrain.py` runs `dvc repro` and opens a PR with the refreshed model for review — it does not auto-deploy.

**Known limitation:** candidate retrieval in `/nearby` is a linear haversine scan (`app/data_service.py`), not a spatial index — adequate at this dataset's size, but the KDTree-based approach originally scoped for Phase 2 is the natural next step before production traffic.

Full setup steps, required AWS/GitHub secrets, and every gotcha we hit along the way live in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).
