# Restaurants4U: Personalized Restaurant Recommendation Engine
### Phase 1 Deliverable: Core MLOps Production Pipeline & System Architecture

---

## Team Members & Roles
* **Chipemba Bwacha** – Project Lead (Deployment Lifecycle & Systems Orchestration)
* **Shini Kim** – ML Lead (Feature Engineering, Core Algorithmic Scripts, & Model Optimization)
* **Xihai Luo** – Engineering Lead (MLOps Pipeline Infrastructure, DVC Architecture, & Codebase Versioning)


## Project Overview
This project builds an automated, robust, reproducible MLOps pipeline for a **Two-Stage Content-Based Restaurant Recommendation Engine**. Leveraging metadata from hundreds of thousands of food establishments, the system filters out geospatial profiles and uses Machine Learning to predict high-quality dining selections based on user location constraints and baseline characteristics.

---

## High-level Repository Structure
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

### 2.1 Technology Stack Selection Justifications

* **Storage & Ingestion Security (`PyArrow` Parquet Framework):** Replaces basic CSV outputs with compressed columnar binary `Parquet` frames. This ensures faster disk read/write cycles and preserves strict data schema types across execution steps.
* **Pipeline Lineage Management (`DVC`):** Decouples large model files and tabular data frames from source code tracking repositories. It matches project dependencies with code execution states to provide verifiable reproducibility (`dvc repro`).
* **Experiment Management (`MLflow`):** Simplifies hyperparameter grid exploration, metrics logging, and asset tracking, allowing side-by-side run evaluations through an intuitive dashboard interface.
* **Model Training & Evaluation (`Scikit-Learn`):** Provides Machine Learning Models (`RandomForestClassifier`) used for training and evaluation.
* **Deployment Strategy (`Streaming`):** The system is designed to handle live requests in a streaming fashion, where the user sends their coordinates and cuisine filter, and then the system returns the top 10 recommendations in real-time.

Please note that these two following components will be implemented in the next phase of the project.

* **Spatial Filtering Engine (`Scipy`):** Rapid coordinate spatial trees (`KDTree`) needed to evaluate multidimensional arrays under low latency.
* **Serving Endpoint Web Gateway (`FastAPI`):** Selected for its native asynchronous code execution, automatic typing enforcement via Pydantic, and low-latency request lifecycle operations.

### 2.2 Production Inference Serving Strategy

The complete recommendation engine handles live requests using an efficient **Two-Stage Retrieval & Ranking Inference Framework**:

For Phase 1, we focused on setting up the pipeline with DVC, MLFlow, and AWS S3 remote storage. And also training and evaluating a simple Random Forest model with two experiments - representing a subset of the **Ranking Stage**.

![Complete architecture strategy](https://cdn.discordapp.com/attachments/932417023290531902/1525219686634229861/Untitled_Diagram.drawio_2.png?ex=6a529706&is=6a514586&hm=fa06eef376a2ad70100202b4290ff43ce533dfb1ea495a63f71a6b80dd84a9cb&)

To prioritize close, highly rated options, predictions are combined with an **Exponential Distance Decay Function**:


$$\text{Recommendation Score} = P(\text{Highly Rated}) \times e^{-\lambda \cdot \text{distance}}$$


Where $P(\text{Highly Rated})$ is the continuous probability output from the model, and $\lambda$ represents our distance penalty hyperparameter. This ensures relevant suggestions are delivered within milliseconds.

### 2.3 Model Training & Evaluation Strategy
![Model Architecture Strategy](https://cdn.discordapp.com/attachments/932417023290531902/1525217906705563748/Model_Strategy.png?ex=6a52955e&is=6a5143de&hm=42f37f2dc5a14923a955e5c20f01f1afbd4fa96f0c850847e4f5ddefaaff522e&)

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
