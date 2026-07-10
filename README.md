# Restaurants4U: Scalable Context-Aware Restaurant Recommendation Engine
### Phase 1 Deliverable: Core MLOps Production Pipeline & System Architecture

---

## Team Members & Roles
* **Chipemba Bwacha** – Project Lead (Deployment Lifecycle & Systems Orchestration)
* **Shini Kim** – ML Lead (Feature Engineering, Core Algorithmic Scripts, & Model Optimization)
* **Xihai Luo** – Engineering Lead (MLOps Pipeline Infrastructure, DVC Architecture, & Codebase Versioning)

---

## Project Overview
This project builds an automated, robust, reproducible MLOps pipeline for a **Two-Stage Content-Based Restaurant Recommendation Engine**. Leveraging metadata from hundreds of thousands of food establishments, the system filters out geospatial profiles and uses a Random Forest classifier to predict high-quality dining selections based on user location constraints and baseline characteristics.

---

## Repository Structure
```text
restaurants4u/
├── .dvc/                   # DVC configuration parameters
├── data/
│   ├── raw/                # Tracked by DVC (Contains raw data pointer files)
│   └── processed/          # Generated pipeline outputs (Parquet formats)
├── models/                 # Serialized production models (.pkl)
├── src/                    # Modularized execution scripts
│   ├── prepare.py          # Data ingestion, casting, and stratified splitting
│   ├── train.py            # Model fitting with MLflow experiment tracking
│   └── evaluate.py         # Test validation and JSON metrics generation
├── dvc.yaml                # Pipeline workflow stage definitions
├── dvc.lock                # Cryptographic reproducibility lockfile
├── params.yaml             # Centralized hyperparameter configuration file
├── metrics.json            # Automated output performance metrics
└── README.md               # System documentation
```

## Part 1: Dataset Selection and Documentation

### 1.1 Dataset Specification & Provenance

The underlying intelligence of the recommendation framework relies on the **380,000 Restaurants (Mostly USA Based)** tabular open-source dataset, ingested via the local tracking binary `200k_Restaurants_Mostly_US.csv` (~332 MB). This asset provides a comprehensive geospatial mapping of food service configurations across domestic markets, optimizing resource utilization compared to heavy, unstructured text reviews.

The system frames recommendations as a **Context-Aware Supervised Classification and Ranking Problem**. The target variable ($y$) is a custom-engineered binary label:


**is highly rated = 1, if rating >= 4.0. 0 otherwise**


This threshold functions as our quality boundary metric, modeling the statistical probability that an establishment will satisfy a given consumer request.

#### Feature Schema Metadata

| Explicit Input Vector | Variable Classification | Engineering Transformation Strategy |
| --- | --- | --- |
| `Rating` | Continuous Ordinal | Parsed via `pd.to_numeric` to enforce structural evaluation typing; binned to construct target vector $y$. |
| `Latitude` / `Longitude` | Continuous Geospatial | Validated against global bounding intervals ($[-90, 90]$ / $[-180, 180]$); acts as spatial routing indices. |
| `Website` / `Phone` / `Images` | Sparse Structural | Missing properties are mapped to binary completeness signals ($0$ for null, $1$ for populated) to evaluate corporate infrastructure density. |
| `Category` | High-Cardinality Nominal | Grouped via frequency-capping at the 20th percentile; remaining sparse identifiers fold into an `"Other"` token before One-Hot encoding. |

### 1.2 Data Quality Assessment & Production Mitigation Strategy

Tabular entries obtained from open scraping ecosystems natively expose high rates of corruption. The data ingestion engine incorporates three strict structural checkpoints within `src/prepare.py` to prevent downstream pipeline crashes:

1. **Type Discrepancy Sanitization:** Mid-file header repetitions and mismatched object string tokens (e.g., text artifacts within numeric coordinate properties) are captured via explicit Pandas numeric coercions: `pd.to_numeric(..., errors='coerce')`. Anomalous text blocks degrade gracefully into nulls (`NaN`).
2. **Cascading Null Erasure:** Rows containing corrupted metadata anchors (`Rating`, `Latitude`, or `Longitude`) are eliminated using listwise deletion (`dropna`). This keeps missing values from corrupting the distance arrays in our spatial index trees.
3. **Entity Deduplication:** Identical string entries sharing matching duplicate values across the `Title` and `Address` parameters are pruned. This avoids artificial performance inflation during model evaluation.

### 1.3 Validation Partition Strategy

To accurately calculate model generalization error without data leakage, the system uses a **Stratified Random Split** on the engineered `is_highly_rated` classification label. This ensures that the class balance is perfectly preserved across partitions:

* **Training Set (80%):** The primary optimization space used by the learning algorithm to adjust ensemble decision boundaries.
* **Test Set (20%):** A complete held-out evaluation block used to calculate final performance metrics (Accuracy, Precision, Recall).

### 1.4 Data Version Control (DVC) Local Linear Tracking

Large physical tracking files are completely isolated from Git's version control index to prevent repository history corruption. DVC tracks files by generating tiny, text-based pointer assets (`.dvc`) that contain unique MD5 file hashes:

```bash
# Initialize local MLOps directory config trackers
dvc init

# Track the specific local raw tabular input data file path
dvc add data/raw/200k_Restaurants_Mostly_US.csv

# Bind tracking pointers and automatic file ignore filters to Git
git add data/raw/200k_Restaurants_Mostly_US.csv.dvc data/raw/.gitignore
git commit -m "track: anchor raw 200k restaurant tabular source data pointer"

```

---

## Part 2: Architecture Design

### 2.1 Technology Stack Selection Justifications

* **Storage & Ingestion Security (`PyArrow` Parquet Framework):** Replaces basic CSV outputs with compressed columnar binary `Parquet` frames. This ensures faster disk read/write cycles and preserves strict data schema types across execution steps.
* **Pipeline Lineage Management (`DVC`):** Decouples large model files and tabular data frames from source code tracking repositories. It matches project dependencies with code execution states to provide verifiable reproducibility (`dvc repro`).
* **Experiment Management (`MLflow`):** Simplifies hyperparameter grid exploration, metrics logging, and asset tracking, allowing side-by-side run evaluations through an intuitive dashboard interface.
* **Core Predictive Execution Array (`Scikit-Learn` & `Scipy`):** Provides optimized tree structures (`RandomForestClassifier`) and rapid coordinate spatial trees (`KDTree`) needed to evaluate multidimensional arrays under low latency.
* **Serving Endpoint Web Gateway (`FastAPI`):** Selected for its native asynchronous code execution, automatic typing enforcement via Pydantic, and low-latency request lifecycle operations.

### 2.2 Architectural Optimization Pivot: Feature Store Streamlining

Our initial system specification included a centralized feature store engine (Feast). However, we adapted our system architecture to optimize operational overhead for this project phase.

Because the restaurant dataset represents highly stable business data profiles rather than high-velocity streaming user click streams, running a live distributed key-value caching engine (like Redis) adds unnecessary infrastructure complexity. Feature mappings are engineered deterministically in our DVC preparation block. This completely removes training-serving skew without introducing the operational overhead of a live database cluster.

### 2.3 Production Inference Serving Strategy

The engine handles live requests using an efficient **Two-Stage Retrieval & Ranking Inference Framework**:

```
[Live Inference Payload: User Coordinates + Cuisine Filter]
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ 1. RETRIEVAL STAGE (Spatial Filtering Engine)           │
│    - FastAPI routes query to an internal spatial tree. │
│    - Haversine bounds filter coordinates dynamically.  │
│    - Cuts search space from 380k to 200 local options. │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ 2. RANKING STAGE (Supervised Classification Layer)     │
│    - Candidate parameters are fed to the Forest model. │
│    - Evaluates scores via model.predict_proba()        │
│    - Blends outputs via an exponential decay function. │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
[Sort Matrix and Stream Top 10 Best Recommendations to User]

```

To prioritize close, highly rated options, predictions are combined with an **Exponential Distance Decay Function**:


$$\text{Recommendation Score} = P(\text{Highly Rated}) \times e^{-\lambda \cdot \text{distance}}$$


Where $P(\text{Highly Rated})$ is the continuous probability output from the Random Forest model, and $\lambda$ represents our distance penalty hyperparameter. This ensures relevant suggestions are delivered within milliseconds.

---

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

To collaborate across independent development nodes, the project configuration connects directly to an AWS S3 data lake storage remote:

```bash
# Link the workspace to the group s3 target bucket
dvc remote add -d s3_remote s3:/ *******TODO***********

# Push physical data frames and assets to the cloud remote
dvc push

```

---

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

1. **Tracking Dashboard Access:** Run `mlflow ui --port 5000` inside an active terminal workspace window to initialize the server logging daemon.
2. **Experiment Validation:** Open `http://localhost:5000` to access the MLflow tracking UI. Navigating to the `Restaurant_Metadata_Ranking_Classifier` panel allows you to view the parameters, training run histories, and accuracy scores side by side.
3. **Visual Metric Analytics:** By selecting both experiment run entries and clicking **Compare**, you can render the parallel coordinates chart. This fulfills the assignment's visual verification criteria, mapping hyperparameter tuning changes directly to model evaluation scores.
