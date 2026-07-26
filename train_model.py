"""
train_model.py
---------------
Loads dataset/air_quality.csv, cleans it, trains several regression models,
compares them, and saves the best one (plus the fitted scaler) to
models/aqi_model.pkl and models/aqi_scaler.pkl.

Run:
    python train_model.py
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils import FEATURE_COLUMNS, TARGET_COLUMN, clean_dataframe

warnings.filterwarnings("ignore")

DATA_PATH = "dataset/air_quality.csv"
MODEL_PATH = "models/aqi_model.pkl"
SCALER_PATH = "models/aqi_scaler.pkl"
META_PATH = "models/model_metrics.pkl"

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)

    # Simple outlier clipping (IQR method) on features, keeps training stable
    for col in FEATURE_COLUMNS:
        q1, q3 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q1, q3)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def build_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(kernel="rbf", C=50, epsilon=1.0),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42
        )
    return models


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    print("Loading and preparing data...")
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    results = {}

    print("\nTraining & evaluating models:\n" + "-" * 50)
    for name, model in models.items():
        # Tree-based models don't need scaling, but scaling doesn't hurt them;
        # linear/SVR benefit from it.
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = {"model": model, "metrics": metrics}
        print(
            f"{name:<20} MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.4f}"
        )

    best_name = max(results, key=lambda n: results[n]["metrics"]["R2"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]

    print("-" * 50)
    print(f"Best model: {best_name}  (R2={best_metrics['R2']:.4f})")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(
        {
            "best_model_name": best_name,
            "all_results": {n: r["metrics"] for n, r in results.items()},
            "feature_columns": FEATURE_COLUMNS,
        },
        META_PATH,
    )
    print(f"\nSaved best model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved metrics/metadata to {META_PATH}")


if __name__ == "__main__":
    main()
