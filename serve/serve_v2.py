import os
import json
import time
from datetime import datetime
from pathlib import Path

import mlflow
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient

# ----------------------------- MLflow config -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# You can either pass MODEL_URI directly, or (recommended) use MODEL_NAME + MODEL_STAGE
MODEL_URI = os.getenv("MODEL_URI")  # e.g. "models:/trip_duration/Production"
MODEL_NAME = os.getenv("MODEL_NAME", "trip_duration")
MODEL_STAGE = os.getenv("MODEL_STAGE", "staging")

# For logging/debug: try to resolve the concrete model version if using name+stage
resolved_version = None
if not MODEL_URI:
    try:
        client = MlflowClient()
        latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if latest:
            resolved_version = latest[0].version
    except Exception:
        # if MLflow isn't reachable during startup, we'll still try to load by stage
        pass
    MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Load the model once at startup
model = mlflow.pyfunc.load_model(MODEL_URI)

# ----------------------------- App & helpers -----------------------------
app = Flask("duration-prediction")

def prepare_features(ride: dict) -> dict:
    """
    Convert incoming ride JSON to the exact feature dict expected by the
    original workshop model (DictVectorizer over categorical IDs + numeric distance).
    """
    return {
        "PULocationID": str(ride.get("PULocationID")),
        "DOLocationID": str(ride.get("DOLocationID")),
        "trip_distance": float(ride.get("trip_distance", 0.0)),
    }

def predict(features: dict) -> float:
    """
    DictVectorizer expects a *list of dicts*. Keep one-record batch: [features].
    """
    preds = model.predict([features])
    return float(preds[0])

def log_result(result: dict) -> None:
    """
    Simple file logging for the workshop (not for production).
    Writes to LOG_DIR/YYYY-MM-DD/UNIXTIME.json
    """
    base = Path(os.getenv("LOG_DIR", "logs"))
    now = datetime.now()
    path = base / now.strftime("%Y-%m-%d") / f"{int(time.mktime(now.timetuple()))}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f_out:
        json.dump(result, f_out, ensure_ascii=False)

# ----------------------------- Endpoints -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MODEL_URI,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_version": resolved_version,
    }

@app.post("/predict")
def predict_endpoint():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Expected JSON object with keys: ride, ride_id"), 400

    # Accept the workshop payload shape: { "ride": {...}, "ride_id": "..." }
    ride = body.get("ride")
    ride_id = body.get("ride_id")

    if not isinstance(ride, dict):
        return jsonify(error="Missing or invalid 'ride' object"), 400
    if ride_id is None:
        return jsonify(error="Missing 'ride_id'"), 400

    try:
        features = prepare_features(ride)
        duration = predict(features)
    except Exception as e:
        # Surface helpful debugging info instead of a blank 500
        return jsonify(
            error=type(e).__name__,
            detail=str(e),
            features=features if "features" in locals() else None
        ), 500

    result = {
        "prediction": {"duration": duration},
        "ride_id": ride_id,
        "model": {
            "name": MODEL_NAME,
            "stage": MODEL_STAGE,
            "version": resolved_version,
            "uri": MODEL_URI,
        },
    }

    # Workshop-only file log
    try:
        log_result(result)
    except Exception:
        # Don't fail the request if logging fails; just continue.
        pass

    return jsonify(result)

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    # For local debugging; in Docker we use gunicorn
    app.run(host="0.0.0.0", port=9696, debug=True)
