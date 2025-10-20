# train/diabetes_train.py
import os, json, joblib, numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import mlflow, mlflow.sklearn

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#  http://mlflow:5500
# export MLFLOW_TRACKING_URI=http://localhost:5500
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("diabetes_progression")

def train(version: str, reg):
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]  # progression index (higher = worse)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
    model_name = type(reg).__name__

    with mlflow.start_run(run_name=version):
        # params
        mlflow.log_param("model", model_name)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("random_seed", RANDOM_SEED)
        if isinstance(reg, Ridge):
            mlflow.log_param("alpha", reg.alpha)

        # fit + eval
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        rmse = mean_squared_error(yte, preds, squared=False)
        mlflow.log_metric("rmse", float(rmse))

        # save to MLflow artifacts
        mlflow.sklearn.log_model(pipe, "model")

        # save to local project for serving
        os.makedirs("model", exist_ok=True)
        joblib.dump(pipe, "model/model.pkl")
        with open("model/metrics.json", "w") as f:
            json.dump(
                {
                    "rmse": float(rmse),
                    "model_version": version,
                    "n_train": int(len(Xtr)),
                    "n_test": int(len(Xte)),
                    "random_seed": RANDOM_SEED,
                },
                f,
                indent=2,
            )

        print(f"{version} | {model_name} RMSE: {rmse:.4f}")

if __name__ == "__main__":
    # v0.1: baseline
    train(version="v0.1", reg=LinearRegression())
    # v0.2: improvement (Ridge)
    train(version="v0.2", reg=Ridge(alpha=1.0, random_state=RANDOM_SEED))
