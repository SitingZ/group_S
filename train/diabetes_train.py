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

# 关键：在容器里用服务名访问 mlflow；在宿主机上调试可改为 http://localhost:5000 或 5500
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("diabetes_progression")

def train(version="v0.1", use_ridge=False):
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]  # progression index

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model_name = "Ridge" if use_ridge else "LinearRegression"
    reg = Ridge(alpha=1.0, random_state=RANDOM_SEED) if use_ridge else LinearRegression()

    pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])

    with mlflow.start_run(run_name=version):
        mlflow.log_param("model", model_name)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("random_seed", RANDOM_SEED)

        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        rmse = mean_squared_error(yte, preds, squared=False)
        mlflow.log_metric("rmse", float(rmse))

        # 保存到 artifacts（MLflow）和项目本地 model/（给服务加载）
        mlflow.sklearn.log_model(pipe, "model")

        os.makedirs("model", exist_ok=True)
        joblib.dump(pipe, "model/model.pkl")
        with open("model/metrics.json", "w") as f:
            json.dump({
                "rmse": float(rmse),
                "model_version": version,
                "n_train": int(len(Xtr)),
                "n_test": int(len(Xte)),
                "random_seed": RANDOM_SEED
            }, f, indent=2)

        print(f"{version} | {model_name} RMSE: {rmse:.4f}")

if __name__ == "__main__":
    # v0.1: baseline
    train(version="v0.1", use_ridge=False)
    # v0.2: 改进（Ridge）
    # 想马上跑出对比就再来一遍；不想现在跑就先注释掉
    # train(version="v0.2", use_ridge=True)