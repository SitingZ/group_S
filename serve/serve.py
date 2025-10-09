# serve/serve.py
import os, json
import mlflow
from flask import Flask, request, jsonify

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
MODEL_NAME  = os.getenv("MODEL_NAME", "trip_duration")
MODEL_STAGE = os.getenv("MODEL_STAGE", "staging")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok", "model": f"{MODEL_NAME}@{MODEL_STAGE}"}

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Expected a JSON object"), 400

    # Build exactly one dict (types matching how you trained — usually str,str,float)
    rec = {
        "PULocationID": str(payload.get("PULocationID")),
        "DOLocationID": str(payload.get("DOLocationID")),
        "trip_distance": float(payload.get("trip_distance", 0)),
    }

    try:
        # DictVectorizer expects a list of dicts
        y = model.predict([rec])
        return jsonify({"duration": float(y[0])})
    except Exception as e:
        return jsonify(error=type(e).__name__, detail=str(e), rec=rec), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
