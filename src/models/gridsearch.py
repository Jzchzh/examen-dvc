# src/models/gridsearch.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import joblib

IN_DIR = Path("data/processed")
OUT_PKL = Path("models/best_params.pkl")

def main():
    X_train = pd.read_csv(IN_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(IN_DIR / "y_train.csv").squeeze()

   
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    X_train_num = X_train[num_cols]

    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    gs = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_train_num, y_train)

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_params_, OUT_PKL)

if __name__ == "__main__":
    main()
