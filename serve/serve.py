import os
import pandas as pd
import mlflow
from flask import Flask, request, jsonify

# Point to the MLflow server (docker-compose service name)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
MODEL_NAME = os.getenv("MODEL_NAME", "trip_duration")
MODEL_STAGE = os.getenv("MODEL_STAGE", "staging")

# Load once at startup
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok", "model": f"{MODEL_NAME}@{MODEL_STAGE}"}

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    df = pd.DataFrame([payload])
    # ensure string IDs if you trained them as strings
    for col in ("PULocationID", "DOLocationID"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    y = model.predict(df)
    return jsonify({"duration": float(y[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
