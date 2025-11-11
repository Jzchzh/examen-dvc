# src/models/train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

IN_DIR = Path("data/processed")
BEST_PARAMS_PKL = Path("models/best_params.pkl")
SCALER_PKL = Path("models/scaler.pkl")
MODEL_PKL = Path("models/model.pkl")

def main():
    X_train = pd.read_csv(IN_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(IN_DIR / "y_train.csv").squeeze()

    
    scaler_info = joblib.load(SCALER_PKL)
    num_cols = scaler_info.get("num_cols", X_train.columns.tolist())
    X_train = X_train[num_cols]

    params = joblib.load(BEST_PARAMS_PKL)
    
    if isinstance(params, dict) and "best_params" in params:
        best_params = params["best_params"]
    else:
        best_params = params

    model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    model.fit(X_train, y_train)

    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)

if __name__ == "__main__":
    main()
