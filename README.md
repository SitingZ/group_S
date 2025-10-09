# MAIO — MLOps (Dockerized Local Setup)

This repo lets students run the **entire MLOps exercise locally** with Docker. You will:

- explore & train in **JupyterLab**  
- track experiments and register models in **MLflow**  
- serve predictions from a **containerized API**

> Based on lightweight [MLOps Zoomcamp](https://github.com/alexeygrigorev/lightweight-mlops-zoomcamp) (tracking → registry → serving) tailored for classroom use.

## Architecture

```
Your Browser
 ├─ JupyterLab (container :8888) ─ logs runs/artifacts ─┐
 └─ MLflow UI (container :5000)  ◀──────────────────────┤
                                                        │
 Model API (container :9696) ─ loads model from Registry┘
 Shared Docker volume: /mlflow  (mlflow.db + artifacts/)
```

---

## Prerequisites

- **Docker Desktop** (Windows/macOS/Linux).
- Git (optional if you download ZIP).

---

## Repository layout

```
maio-mlops/
├─ train/           # notebooks & training code
├─ serve/           # Flask/Gunicorn app (API)
├─ docker-compose.yml
├─ Dockerfile.train
├─ Dockerfile.serve
└─ Dockerfile.mlflow-ui
```

---

## Quick start

```bash
# 1) Start MLflow + Jupyter
docker compose up -d mlflow notebook

# 2) Open UIs
# MLflow:   http://localhost:5000
# Jupyter:  http://localhost:8888
```

Then:

1. In **Jupyter**, open a notebook in `train/`.  
2. Set the tracking URI and experiment:
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://mlflow:5000")
   mlflow.set_experiment("trip_duration_baseline")
   ```
3. Train, **log params/metrics/artifacts**, and **log a model** (include signature & input example).  
4. In **MLflow UI**, register the best run as a model (e.g., `trip_duration`) and set stage to **Staging**.

Serve the model:

```bash
docker compose up -d serve
# test
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID":100,"DOLocationID":102,"trip_distance":30}'
```

PowerShell alternative:
```powershell
$body = @{ PULocationID=100; DOLocationID=102; trip_distance=30 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:9696/predict -Method POST -Body $body -ContentType 'application/json'
```

---

## Services & ports

- **MLflow UI**: `http://localhost:5000`
- **JupyterLab**: `http://localhost:8888`
- **Model API**: `http://localhost:9696`

> If a port is busy, change the **left** side of the mapping in `docker-compose.yml` (e.g., `5001:5000`) and open the new host port.

---

## Typical workflow

1) **Train + log** in Jupyter (set tracking URI `http://mlflow:5000`).  
2) In **MLflow UI**: register best run → `trip_duration` → stage = *Staging*.  
3) **Serve** the staged model at `http://localhost:9696/predict`.  
4) When ready, promote to **Production** and (optionally) switch `MODEL_STAGE`.

---

## Clean up

```bash
# stop containers
docker compose down

# remove containers + volumes (deletes mlflow DB & artifacts!)
docker compose down -v
```

---

## TL;DR

```bash
docker compose up -d mlflow notebook
# open http://localhost:5000 and http://localhost:8888
# train & log; register best model as 'trip_duration' (stage: Staging)

docker compose up -d serve
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID":100,"DOLocationID":102,"trip_distance":30}'
```
