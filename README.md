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
