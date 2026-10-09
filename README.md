# Olist Late Delivery Prediction — MLOps

An end-to-end MLOps project for predicting late e-commerce deliveries using the Brazilian Olist dataset.

This repository demonstrates the transition from exploratory machine learning notebooks to a containerized inference system with reproducible dependencies, model registration, automated testing, CI/CD, and production monitoring.

## Architecture

The application uses the following components:

- **FastAPI:** REST endpoints for single and batch predictions.
- **scikit-learn:** Persisted preprocessing and Logistic Regression model.
- **MLflow:** Model registration and production model loading.
- **PostgreSQL:** Persistent prediction monitoring records.
- **Docker Compose:** Orchestration of application services.
- **DVC:** Tracking of production model artifacts.
- **Great Expectations:** Data validation.
- **Prometheus client:** API request and prediction metrics.
- **GitHub Actions:** Automated quality checks, tests, Docker image builds, and publishing to GitHub Container Registry.

### Service flow

```text
                    Client
                      |
                      v
                FastAPI :8000
                      |
           +----------+----------+
           |                     |
           v                     v
     MLflow Registry        PostgreSQL
        :5000              Prediction Logs
           ^                     |
           |                     v
       model-init          Monitoring Worker
                                 |
                                 v
                         Rolling Drift Checks
                         and Warning Logs
```

## Requirements

For containerized deployment:

- Git
- Docker Desktop with Docker Compose
- Internet connectivity during the initial image build to download the production model bundle

For local development:

- Python 3.13
- uv package manager

## Quick Start — Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/osamasawalha01/olist-delivery-prediction-mlops.git
cd olist-delivery-prediction-mlops
```

### 2. Configure environment variables

Copy `.env.example` to `.env`.

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Update `.env` with a secure PostgreSQL password and the corresponding `DATABASE_URL`.

The database hostname is `db` because the application connects to PostgreSQL through the Docker Compose network.

Do not commit `.env` or production credentials.

### 3. Build and start

```bash
docker compose up -d --build
```

The Compose stack starts PostgreSQL and MLflow, registers the model, then starts the prediction API. A separate monitoring worker performs periodic drift checks.

### 4. Check services

```bash
docker compose ps
```

Open:

- FastAPI interactive documentation: http://localhost:8000/docs
- API health check: http://localhost:8000/health
- Model information: http://localhost:8000/model
- Prometheus metrics: http://localhost:8000/metrics
- MLflow UI: http://localhost:5000

### 5. Stop services

```bash
docker compose down
```

Named Docker volumes preserve PostgreSQL and MLflow data across ordinary shutdowns.

**Warning:** `docker compose down -v` deletes named volumes and can remove persisted database and model registry data.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Basic API health response |
| GET | `/model` | Active model name and version |
| POST | `/predict` | Predict delivery status for one order |
| POST | `/predict/batch` | Predict delivery status for multiple orders |
| GET | `/metrics` | Prometheus-compatible application metrics |

The `/predict` and `/predict/batch` endpoints accept structured order features. Consult the automatically generated OpenAPI documentation at `/docs` for the request schemas.

Prediction responses include the predicted class, label, probability, model name, and model version.

The `/health` endpoint reports application health but is not a comprehensive dependency readiness check.

## Model and Inference

The initial production model is a scikit-learn Logistic Regression classifier trained on historical Olist order data.

The inference pipeline reuses persisted preprocessing and model artifacts rather than fitting new transformations for each request.

The model is registered in MLflow as:

```text
olist-logistic-regression
```

The production alias is:

```text
production
```

The Docker Compose `model-init` service initializes the registry before the API starts.

The application supports loading the production model from MLflow and a local-artifact loading mode used by automated tests.

### Model artifact bootstrap

During the Docker image build, `scripts/bootstrap_model.py` retrieves the versioned model bundle from a GitHub Release and verifies its SHA-256 checksum.

This makes the Docker build independent of the developer's local DVC cache, provided the release asset remains accessible.

## Data and Artifact Versioning

DVC tracks the production model artifact directory using:

```text
models/production.dvc
```

The tracked production bundle contains:

- `05_feature_list.csv`
- `05_preprocessor.joblib`
- `06_logistic_regression.joblib`
- `06_results_summary.csv`

**Current limitation:** The configured DVC remote uses a local filesystem directory (`../.dvc-storage`). A fresh clone cannot use `dvc pull` unless that storage is separately available.

Docker image builds instead use the versioned GitHub Release bootstrap mechanism.

## Experiment Tracking

MLflow provides model metadata, artifact storage, and model registry functionality.

The tracking server runs at:

```text
http://localhost:5000
```

Its SQLite backend and artifacts are stored in a persistent Docker volume.

The production model is registered automatically by the `model-init` service.

## Monitoring

The application collects Prometheus-compatible metrics including request counts, request latency, prediction counts, and prediction errors.

Prediction records are stored in PostgreSQL.

A separate monitoring worker checks prediction distribution drift every 300 seconds by default.

The current drift detector:

- Uses the most recent 100 recorded predictions.
- Requires at least 50 predictions before evaluation.
- Compares the production predicted-late rate with a configured baseline.
- Uses an absolute difference threshold of 0.15.
- Writes warning logs when drift is detected.

Inspect worker logs using:

```bash
docker compose logs monitoring-worker
```

**Monitoring limitations:** The implemented drift check measures changes in the predicted label distribution. It does not independently establish input feature drift or actual model performance degradation. Alerts currently appear in logs; external notification delivery is not configured.

## Local Development

Install Python 3.13 and uv, then install locked dependencies:

```bash
uv sync --frozen
```

Run quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run tests:

```bash
uv run python -m pytest -q
```

Some tests require local production model artifacts. The model bootstrap script can retrieve the versioned bundle:

```bash
uv run python scripts/bootstrap_model.py
```

For local-artifact inference, set `OLIST_MODEL_SOURCE=local` in the environment where the tests run.

## CI/CD

GitHub Actions runs automated workflows on pushes and pull requests targeting `main`.

The pipeline performs:

1. Dependency installation using uv.
2. Production model artifact bootstrap.
3. Ruff lint checks.
4. Ruff formatting checks.
5. Pytest execution using the local model.
6. Docker image build.
7. Docker image publication to GitHub Container Registry on eligible non-pull-request runs.

Container image:

```text
ghcr.io/osamasawalha01/olist-delivery-prediction-mlops:latest
```

A successful pipeline validates automated checks and image publishing. It does not, by itself, prove that a clean-machine deployment has been tested.

## Project Status

Implemented:

- Notebook-based data preparation and model training
- Persisted preprocessing and model inference
- FastAPI single and batch prediction endpoints
- MLflow model registry integration
- PostgreSQL prediction logging
- Docker Compose deployment
- DVC artifact tracking
- Great Expectations data validation
- Automated tests and Ruff quality checks
- GitHub Actions CI/CD
- Prometheus API metrics
- Automated rolling-window prediction drift checks

Remaining verification or improvements:

- Validate deployment from a completely fresh clone.
- Configure a portable DVC remote if cross-machine `dvc pull` is required.
- Add external drift alert notifications if required.
- Expand monitoring to include feature drift and actual model performance when ground-truth labels become available.

## License

No license is declared in this README. Refer to the repository for any separately published license information.
