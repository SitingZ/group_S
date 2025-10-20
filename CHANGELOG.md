# CHANGELOG

## v0.1
- Model: **LinearRegression** + **StandardScaler**
- RMSE: **53.85**
- Baseline established for diabetes progression
- Purpose: initial benchmark to evaluate future model improvements

---

## v0.2
- Model: **Ridge(alpha=1.0)** + **StandardScaler**
- RMSE: **53.77**
- Improvement: Ridge regression adds **L2 regularization**, reducing overfitting and improving model stability.
- Prediction example:
  ```json
  {"prediction": 226.91314681729256}

