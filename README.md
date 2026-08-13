# Cats vs Dogs MLOps Pipeline

End-to-end MLOps pipeline — binary image classification (Cats vs Dogs).

## Project Structure

```
cats-dogs-mlops/
├── .dvc/                    # DVC configuration
├── .github/workflows/       # CI (ci.yml) and CD (cd.yml) pipelines
├── api/                     # FastAPI inference service
│   ├── main.py              # Endpoints: /health /predict /stats /metrics
│   └── predictor.py         # Model loading & inference
├── k8s/                     # Kubernetes manifests
├── monitoring/              # Prometheus configuration
├── scripts/                 # Utility scripts
│   ├── download_data.py     # Kaggle dataset download
│   ├── preprocess_data.py   # DVC stage helper
│   ├── smoke_test.py        # Post-deploy smoke tests
│   ├── batch_evaluate.py    # M5 performance tracking
│   ├── create_submission.py # Build submission zip
│   ├── generate_report.py   # Build DOCX submission report
│   └── setup.bat            # Windows one-click setup
├── src/
│   ├── data/preprocess.py   # Image loading, normalisation, split logic
│   └── models/
│       ├── cnn_model.py     # 4-block CNN architecture
│       └── train.py         # Training with MLflow tracking
├── tests/                   # pytest unit tests
├── Dockerfile
├── docker-compose.yml       # API + Prometheus
├── dvc.yaml                 # DVC pipeline stages
├── params.yaml              # Hyperparameters
└── requirements.txt
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ (system install preferred) |
| Docker + Compose | Latest |
| Git | Any |

---

## Quick Start

> **Important:** All commands must be run from inside the `cats-dogs-mlops` folder.

```bash
cd cats-dogs-mlops
```

### For Evaluators — No Kaggle Account Needed

The zip includes **60 sample images** (`sample_data/Cat/` and `sample_data/Dog/`) so the full pipeline can be run immediately without any credentials.

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Preprocess sample data
python scripts/preprocess_data.py --raw-dir sample_data --processed-dir data/processed

# 3. Train (uses sample data, logs to MLflow)
set MLFLOW_ALLOW_FILE_STORE=true
python src/models/train.py --data-dir data/processed --artifacts-dir artifacts

# 4. View results in MLflow UI
set MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000

# 5. Run tests
pytest tests/ -v

# 6. Build and run Docker API
docker build -t cats-dogs-api:latest .
docker compose up -d
curl http://localhost:8000/health
```

> **To use the full Kaggle dataset instead:**
> Download from https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset,
> extract `Cat\` and `Dog\` folders into `data\raw\`, then run
> `python scripts/preprocess_data.py` (no arguments) before training.

---

### 1 — Install dependencies

```bash
# System Python (preferred — no venv needed)
pip install -r requirements.txt

# Optional: virtual environment (create it INSIDE cats-dogs-mlops)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Windows shortcut:** double-click `scripts/setup.bat` (already sets the correct directory)

---

### 2 — Initialise Git & DVC

```bash
git init
git add .
git commit -m "Initial commit"

dvc init
dvc remote add -d local_remote ../dvc_store
git add .dvc .dvcignore
git commit -m "Initialise DVC"
```

---

### 3 — Download dataset (M1)

```bash
# Manual download (recommended):
# 1. Go to https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
# 2. Click the Download button (downloads dog-and-cat-classification-dataset.zip)
# 3. Extract the zip — you should get Cat\ and Dog\ folders
# 4. Place them under data\raw\ so the structure is:
#      data\raw\Cat\*.jpg
#      data\raw\Dog\*.jpg
# 5. Skip the download script below — go directly to Step 4 (preprocess)

# Only needed if using Kaggle API instead of manual download:
# python scripts/download_data.py --output-dir data/raw

dvc add data/raw
git add data/raw.dvc
git commit -m "Track raw data with DVC"
```

---

### 4 — Run the full DVC pipeline (preprocess + train)

```bash
dvc repro
```

Or run stages manually:

```bash
# Preprocess
python scripts/preprocess_data.py

# Train (with MLflow tracking)
python src/models/train.py --data-dir data/processed --artifacts-dir artifacts

# Dry-run (no dataset needed — synthetic data)
python src/models/train.py --dry-run
```

---

### 5 — View experiment results (MLflow UI)

```bash
mlflow ui --host 127.0.0.1 --port 5000
# Open http://localhost:5000
```

---

### 6 — Run unit tests (M3)

```bash
pytest tests/ -v
```

---

### 7 — Build Docker image (M2)

```bash
docker build -t cats-dogs-api:latest .
```

---

### 8 — Deploy with Docker Compose (M4)

```bash
# Start API + Prometheus
docker compose up -d

# Check status
docker compose ps
docker compose logs -f api
```

---

### 9 — Test the API (M2)

```bash
# Health check
curl http://localhost:8000/health

# Predict (replace with real image path)
curl -X POST http://localhost:8000/predict -F "file=@cat.jpg"

# API stats
curl http://localhost:8000/stats
```

---

### 10 — Run smoke tests (M4)

```bash
python scripts/smoke_test.py --base-url http://localhost:8000
```

---

### 11 — Batch evaluation & monitoring (M5)

```bash
# Collect post-deploy metrics
python scripts/batch_evaluate.py --base-url http://localhost:8000

# Prometheus UI
# http://localhost:9090
```

---

### 12 — Kubernetes deployment (optional, M4)

```bash
# Update image name in k8s/deployment.yaml first
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/cats-dogs-api
```

---

### 13 — Generate submission files

```bash
# DOCX report
python scripts/generate_report.py --output submission_report.docx

# Zip (excludes data/artifacts to stay < 10 MB)
python scripts/create_submission.py --output submission.zip
```

---

## CI/CD — GitHub Actions

| Workflow | Trigger | Actions |
|---|---|---|
| `ci.yml` | push / PR | run tests → build Docker → push to Docker Hub (main only) |
| `cd.yml` | CI passes on main | pull image → compose up → smoke tests |

**Required GitHub Secrets:**

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status, model loaded flag |
| POST | `/predict` | Upload image → `{prediction, confidence, probabilities, latency_ms}` |
| GET | `/stats` | Request count and latency stats |
| GET | `/metrics` | Prometheus metrics |
