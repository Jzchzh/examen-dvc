# src/models/evaluate.py
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

IN_DIR = Path("data/processed")
MODEL_PKL = Path("models/model.pkl")
SCALER_PKL = Path("models/scaler.pkl")
PRED_PATH = Path("data/predictions.csv")
SCORES_JSON = Path("metrics/scores.json")

def main():
    X_test = pd.read_csv(IN_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(IN_DIR / "y_test.csv").squeeze()

   
    scaler_info = joblib.load(SCALER_PKL)
    num_cols = scaler_info.get("num_cols", X_test.columns.tolist())
    X_test = X_test[num_cols]

    model = joblib.load(MODEL_PKL)
    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    r2  = float(r2_score(y_test, y_pred))

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(PRED_PATH, index=False)

    SCORES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_JSON, "w") as f:
        json.dump({"mse": mse, "r2": r2}, f, indent=2)

if __name__ == "__main__":
    main()

