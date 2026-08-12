# Deployment, CI/CD & Monitoring Guide

This adds three things to the existing DVC/MLflow pipeline: a FastAPI
serving layer (Dockerized, deployed to AWS ECS Fargate), a GitHub
Actions CI/CD pipeline, and an EvidentlyAI drift-monitoring +
auto-retrain workflow.

## 1. What changed after reading the real pipeline code

The first draft of this guide assumed `train.py` persisted a full
`sklearn.Pipeline`. After seeing `prepare.py`, `feature_engineering.py`,
`train.py`, `evaluate.py`, and `validate_data.py`, the actual picture is:

- `models/restaurant_classifier.pkl` is the **bare `RandomForestClassifier`**
  — `train.py` calls `joblib.dump(clf, ...)` directly on it.
- All feature engineering (presence flags, `title_length`/`address_length`,
  category capping + one-hot) lives in a separate `feature_engineering.py`
  step that runs between `prepare.py` and `train.py`.
- The fitted category vocabulary (`top_categories`) and final training
  column order aren't baked into the model — they only exist as a
  side-effect: `feature_engineering.py` writes them to
  `reports/feature_summary.json`.

**Fix applied:** `engineer_features()` was pulled out of
`feature_engineering.py` into a new shared module, **`src/features.py`**,
imported by both `feature_engineering.py` (training) and
`app/model_service.py` (serving). This makes the two paths structurally
impossible to drift apart — there's one implementation, not two kept in
sync by hand. Both files live in `src/`, matching the rest of your
pipeline scripts (`src/prepare.py`, `src/train.py`, `src/evaluate.py`,
`src/validate_data.py`) — `dvc.yaml` has been updated accordingly.
`feature_engineering.py` in this delivery is a patched version of yours
with the duplicated function removed and replaced by
`from features import engineer_features` (a plain sibling import, not
`src.features`, since when DVC runs `python src/feature_engineering.py`
directly, Python puts `src/` itself on the import path — not the repo
root); everything else in the file is unchanged.

At serving time, `app/model_service.py` loads `top_categories` and the
training column order from `reports/feature_summary.json` at startup
(alongside the model pickle), then calls
`src.features.encode_for_inference()` to turn each incoming request into
the exact same numeric layout the model was trained on.

### Consequence for the API shape

Because `title_length`/`address_length` are real model features, the
`/predict` and `/recommend` request schemas now require `title` and
`address` text, not just a restaurant ID — see the updated
`app/schemas.py`. `has_website`/`has_phone`/`has_images` are accepted as
booleans (the model only ever saw `.notna()` on those columns, never the
actual URL/phone value, so a boolean round-trips through the same logic
correctly).

### `dvc.yaml` was out of date

The version in the README only defined 3 stages (`prepare`, `train`,
`evaluate`) and referenced `data/raw/...csv` directly as `prepare`'s
input. The real pipeline has 5 stages, since `validate_data.py` and
`feature_engineering.py` sit in between. I've included a corrected
`dvc.yaml` reflecting the actual script dependencies, param keys, and
outputs (including declaring `reports/feature_summary.json` as a
`cache: false` metric — same treatment as `metrics.json` — since the
API needs it and it's small enough to commit to git directly rather
than push through the S3 remote). Diff this against your current
`dvc.yaml` before overwriting it, in case there's drift I'm not aware of.

## 2. Remaining assumptions to verify

0. **DVC pull targets: two different mechanisms, two different syntaxes.**
   `data/raw/*.csv` is tracked via a standalone `dvc add` (its own `.dvc`
   pointer file) since it's a manually-added input, not something any
   pipeline stage produces. Pipeline-stage outputs — `data/processed/*.parquet`,
   `data/features/*.parquet`, `models/restaurant_classifier.pkl` — are
   tracked in `dvc.lock` instead, and DVC resolves those from the output
   path directly, with **no** `.dvc` suffix. Mixing the two up is exactly
   what caused the `'data/raw' does not exist as ... a stage name`
   error — `dvc pull data/raw` needs either `dvc pull` (no target, pulls
   everything) or `dvc pull data/raw/200k_Restaurants_Mostly_US.csv.dvc`
   (the actual pointer file), while `dvc pull models/restaurant_classifier.pkl.dvc`
   would fail the same way in reverse, since that file has no `.dvc`
   sibling — it only exists in `dvc.lock`.

1. **Positive class label.** Code assumes `is_highly_rated == 1` is the
   positive class in `model.classes_`, matching
   `(Rating >= threshold).astype(int)` in `prepare.py`. This should be
   correct given what you shared, but worth a sanity check against a
   real loaded model (`clf.classes_`).
2. **`pd.get_dummies` column naming.** `cat_<category>` column names are
   assumed stable between training runs for the same `top_category_count`.
   If a training run ever has zero examples of "Other" (all restaurants
   fall in the top N), `cat_Other` won't exist in `feature_columns.json`,
   and `reindex(..., fill_value=0)` in `encode_for_inference` will drop
   any unseen category to all-zero rather than erroring — worth a log
   line if you want visibility into that at serving time.
3. **Production request logging for drift detection.** `monitoring/drift_detection.py`
   expects a `data/production_logs/*.parquet` file of recent inference
   requests to compare against training data. This repo doesn't yet
   have that logging wired up — you'll want to add either:
   - request logging middleware in `app/main.py` that batches and
     uploads feature payloads to S3, or
   - a periodic export from wherever request logs actually land (e.g.
     ECS/CloudWatch logs, an application DB).

## 3. Required GitHub repository secrets

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | CI/CD auth for ECR push + ECS deploy + DVC S3 pull |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | ECR repo name for the API image |
| `ECS_CLUSTER` / `ECS_SERVICE` | Target ECS Fargate cluster/service |

These should ideally be a **dedicated CI IAM user** (or OIDC role) with
scoped permissions: `ecr:*` on the one repo, `ecs:UpdateService` /
`ecs:DescribeServices` on the one service, and read access to the DVC
S3 bucket — not the same broad credentials individual team members use
for `dvc push`.

## 4. One-time AWS setup (not automated by this PR)

- Create an ECR repository matching `ECR_REPOSITORY`.
- Create an ECS Fargate cluster + service, an ALB/target group in front
  of it on port 8000, and an `ecsTaskExecutionRole`. Update
  `task-definition.json` with the real execution role ARN and region.
- Create the CloudWatch log group `/ecs/restaurants4u-api`.

If you'd rather deploy to Cloud Run or Azure Container Apps instead of
ECS, the `Dockerfile` and app code are cloud-agnostic — only the
`build-and-push` / `deploy` jobs in `.github/workflows/ci-cd.yml` would
need swapping out. Happy to build that variant instead if AWS ECS
isn't the right target.

## 5. Local development

```bash
pip install -r requirements-dev.txt
dvc pull  # pulls all tracked data + the model artifact
# reports/feature_summary.json is committed to git directly (cache: false
# in dvc.yaml, same as metrics.json), so a normal git pull covers it —
# `dvc pull` alone won't fetch it since it was never pushed to the remote.
uvicorn app.main:app --reload

# in another terminal
curl localhost:8000/health
```

```bash
docker build -t restaurants4u-api .
docker run -p 8000:8000 restaurants4u-api
```

## 6. Running tests / lint locally

```bash
ruff check .
pytest tests/ -v
```

## 7. Monitoring & retraining flow

1. `.github/workflows/monitoring.yml` runs daily (cron) or on demand.
2. `monitoring/drift_detection.py` runs Evidently's `DataDriftPreset`
   comparing `data/processed/train.parquet` (reference) against recent
   production requests (current), and writes `monitoring/drift_summary.json`
   plus an HTML report as a workflow artifact.
3. If `share_of_drifted_columns >= threshold` (default 0.5, tune this),
   the `retrain` job runs `dvc repro` against current raw data and
   opens a PR with the updated `metrics.json` / `dvc.lock` — it does
   **not** auto-merge or auto-deploy. A human reviews the metrics
   before merging to `main`, at which point the normal CI/CD pipeline
   builds and deploys the new model.
