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

# Appendix A
## A.1 Supplementary

---

## Part 1: Dataset Selection and Documentation

### 1.1 Dataset Specification

The dataset we are using is the **380,000 Restaurants (Mostly USA Based)** tabular open-source dataset from Kaggle (~332 MB). This datasets contains a compilation of 380,000 restaurants. While the majority of the restaurants are in the United States, there are also a couple from other countries.

The system frames recommendations as a **Context-Aware Supervised Classification and Ranking Problem**. In simple terms, we want our model to output a list of restaurant rankings near them based on their current location. A use case is when the user wnats to find nearby restaurants that have a high rating.

The target variable ($y$) is a custom-engineered binary label:

```python
Rating >= 4.0 ? is_highly_rated = 1 : is_highly_rated = 0
```

This threshold functions as our quality boundary metric, modeling the statistical probability that an establishment will satisfy a given consumer request.

#### Feature Schema Metadata

| Explicit Input Vector | Variable Classification | Engineering Transformation Strategy |
| --- | --- | --- |
| `Rating` | Continuous Ordinal | Parsed via `pd.to_numeric` to enforce structural evaluation typing; binned to construct target vector $y$. |
| `Latitude` / `Longitude` | Continuous Geospatial | Validated against global bounding intervals ($[-90, 90]$ / $[-180, 180]$); acts as spatial routing indices. |
| `Website` / `Phone` / `Images` | Sparse Structural | Mapped to binary ($0$ for null, $1$ for populated). |
| `Category` | High-Cardinality Nominal | Grouped via frequency-capping at the 20th percentile; group sparse categories into `"Other"` before One-Hot encoding. |

### 1.2 Data Quality Assessment & Production Mitigation Strategy

Tabular entries obtained from open scraping ecosystems natively expose high rates of corruption. The data ingestion engine incorporates three strict structural checkpoints within `src/prepare.py` to prevent downstream pipeline crashes:

1. **Type Discrepancy Sanitization:**
Ensure data types are consistent throughout the dataset. Use explicit Pandas numeric coercions: `pd.to_numeric(..., errors='coerce')` to force consistency. Any anonomalies such as "None" strings will turn into nulls (`NaN`).
2. **Null Values:** Rows containing null values (`Rating`, `Latitude`, or `Longitude`) are eliminated using listwise deletion (`dropna`). This keeps missing values from corrupting the distance arrays in our spatial index trees.
3. **Duplication:** Identical string entries sharing matching duplicate values across the `Title` and `Address` parameters are pruned. This avoids artificial performance inflation during model evaluation.

### 1.3 Validation Partition Strategy

To accurately calculate model generalization error without data leakage, the system uses a **Stratified Random Split** on the engineered `is_highly_rated` classification label. This ensures that the class balance is perfectly preserved across partitions:

* **Training Set (80%):** Used to fit the model and train the model.
* **Test Set (20%):** A complete held-out evaluation block used to calculate final performance metrics (Accuracy, Precision, Recall).

### 1.4 Data Version Control (DVC) with AWS S3 Remote Storage

Our actual dataset is stored in AWS S3 and tracked via DVC. DVC tracks files by generating tiny, text-based pointer assets (`.dvc`) that contain unique MD5 file hashes. This allows us to version control large files without bloating the Git repository. Use ``dvc pull`` to retrieve the actual dataset from the S3 remote storage.

## Part 2: Architecture Design

### 2.1 Production Inference Serving Strategy

The complete recommendation engine handles live requests using an efficient **Two-Stage Retrieval & Ranking Inference Framework**:

For Phase 1, we focused on setting up the pipeline with DVC, MLFlow, and AWS S3 remote storage. And also training and evaluating a simple Random Forest model with two experiments - representing a subset of the **Ranking Stage**.

<img width="502" height="650" alt="Architectural Diagram" src="https://github.com/user-attachments/assets/c655032f-5d23-416e-9176-3aa696e6f4c4" />

To prioritize close, highly rated options, predictions are combined with an **Exponential Distance Decay Function**:


$$\text{Recommendation Score} = P(\text{Highly Rated}) \times e^{-\lambda \cdot \text{distance}}$$


Where $P(\text{Highly Rated})$ is the continuous probability output from the model, and $\lambda$ represents our distance penalty hyperparameter. This ensures relevant suggestions are delivered within milliseconds.

### 2.2 Model Training & Evaluation Strategy
<img width="549" height="953" alt="Model Strategy" src="https://github.com/user-attachments/assets/c5bdba53-e398-4f80-a1d1-4beb1d6c27f5" />
## Part 3: DVC Pipeline Implementation

### 3.1 Central Configuration Management (`params.yaml`)

Pipeline variables are isolated inside a centralized parameters file. This allows team members to run new experiments without modifying the underlying Python code:

```yaml
prepare:
  raw_input_path: "data/raw/200k_Restaurants_Mostly_US.csv"
  processed_dir: "data/processed"
  test_size: 0.2
  seed: 42

train:
  n_estimators: 120
  max_depth: 10
  model_path: "models/restaurant_classifier.pkl"

evaluate:
  metrics_path: "metrics.json"

```

### 3.2 Automated Workflow Pipeline Execution Configuration (`dvc.yaml`)

The pipeline structure connects scripts, parameters, and outputs into a transparent Directed Acyclic Graph (DAG):

```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/200k_Restaurants_Mostly_US.csv
    params:
      - prepare.raw_input_path
      - prepare.test_size
      - prepare.seed
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.parquet
    params:
      - train.n_estimators
      - train.max_depth
      - train.model_path
    outs:
      - models/restaurant_classifier.pkl

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/restaurant_classifier.pkl
      - data/processed/test.parquet
    metrics:
      - metrics.json:
          cache: false

```

### 3.3 Cloud Remote Storage Infrastructure Link

To collaborate, `dvc push`, each team member must be assigned an access key, e.g., access id and secret key, on AWS IAM. Then the team member must add the access key to their local development environment either through the use of environmental variables or store the credentials in a `.dvc/config.local` file. For pulling, `dvc pull`, no credentials are needed.

```bash
# .dvc/config.local option
dvc remote modify --local myremote access_key_id "THEIR_AWS_ACCESS_KEY_ID"

dvc remote modify --local myremote secret_access_key "THEIR_AWS_SECRET_ACCESS_KEY"
```

## Part 4: Experiment Tracking

### 4.1 Parameter Experimentation Matrix

To verify the stability of our tracking pipeline infrastructure, we configured and evaluated two separate training runs by updating the variables inside `params.yaml`:

1. **Experiment 1 (Shallow Baseline Profile):**
* `n_estimators`: 50
* `max_depth`: 5
* *Execution Execution Key:* `dvc repro`


2. **Experiment 2 (Deep Complex Capacity Profile):**
* `n_estimators`: 100
* `max_depth`: 10
* *Execution Execution Key:* `dvc repro`



### 4.2 Pipeline Caching & Reproducibility Verification

When executing the parameter variance loop, running `dvc repro` tests the integrity of your pipeline's caching system.

When Experiment 2 is triggered, DVC hashes the states of your files and recognizes that the raw data hasn't changed. It automatically skips the execution of `src/prepare.py`, pulling the processed Parquet files directly from the local cache. It only reruns `src/train.py` and `src/evaluate.py` to log the new run details to MLflow and update `metrics.json`.

If the pipeline is executed again without any modifications, the cache intercepts all operations, outputting the confirmation:

```text
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.

```

### 4.3 Evaluation Verification & UI Management

1. **Tracking Dashboard Access:** Run `mlflow ui` inside an active terminal workspace window to start MLFlow server running on local.
2. **Experiment Validation:** Open `http://127.0.0.1:5000` to access the MLflow tracking UI. Navigating to the `Restaurant_Metadata_Ranking_Classifier` panel allows you to view the parameters, training run histories, and metrics side by side.
3. **Visual Metric Analytics:** By selecting both experiment run entries and clicking **Compare**, you can render the parallel coordinates chart. This fulfills the assignment's visual verification criteria, mapping hyperparameter tuning changes directly to model evaluation scores.
