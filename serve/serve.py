# serve/serve.py  —— Diabetes progression serving (Flask)
import os, json
import joblib
import numpy as np
from flask import Flask, request, jsonify

# --------- load artifacts ----------
MODEL_PATH   = os.getenv("MODEL_PATH", "model/model.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "model/metrics.json")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

model_version = "unknown"
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH) as f:
            model_version = json.load(f).get("model_version", "unknown")
    except Exception:
        pass

# Diabetes features from sklearn.load_diabetes
FEATURES = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_version}

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Expected a JSON object"), 400

    # check required fields
    missing = [k for k in FEATURES if k not in payload]
    if missing:
        return jsonify(error="missing_fields", fields=missing), 400

    try:
        row = np.array([[float(payload[k]) for k in FEATURES]], dtype=float)
        y = float(model.predict(row)[0])
        return jsonify({"prediction": y})
    except Exception as e:
        return jsonify(error=type(e).__name__, detail=str(e)), 400

if __name__ == "__main__":
    # default port 9696 to match compose
    app.run(host="0.0.0.0", port=9696)