"""
pipelines/train_pipeline.py

1. Fetches (features, targets) from the Hopsworks feature store.
2. Trains + evaluates multiple models for each forecast horizon (24h/48h/72h).
3. Picks the best model per horizon (by RMSE) and registers it in the
   Hopsworks Model Registry.

Models compared: Ridge Regression (baseline), Random Forest, XGBoost.
Metrics: RMSE, MAE, R².
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ xgboost not installed — skipping XGBoost models (pip install xgboost to enable)")

import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "aqi_base_lahore_fg"
FEATURE_GROUP_VERSION = 1

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

HORIZONS = ["target_pm25_24h", "target_pm25_48h", "target_pm25_72h"]

FEATURE_COLUMNS = [
    "pm25",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "hour",
    "day_of_week",
    "month",
    "pm25_lag_24h",
    "pm25_roll_3h",
    "pm25_change_rate_3h",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_features():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Reading feature group...")
    try:
        df = fg.read()
    except Exception as e:
        print(f"⚠️ Arrow Flight read failed ({e}), retrying with Hive fallback...")
        df = fg.read(read_options={"use_hive": True})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {df.shape[0]} rows")
    return df, project


# ---------------------------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------------------------

def chronological_split(df, test_frac=0.15):
    """
    Splits by TIME, not randomly — train on the earlier ~85%, test on the
    most recent ~15%. This mimics real deployment (predicting the future
    from the past) and avoids leaking near-identical adjacent hours between
    train and test, which a random split would do.
    """
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models():
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1
        )
    return models


# ---------------------------------------------------------------------------
# Train + evaluate for one horizon
# ---------------------------------------------------------------------------

def train_and_evaluate_horizon(df, target_col, test_frac=0.15):
    print(f"\n{'='*70}")
    print(f"HORIZON: {target_col}")
    print(f"{'='*70}")

    # each horizon may have a slightly different set of valid rows
    # (48h/72h targets can be NaN in cases 24h isn't — drop separately per horizon)
    horizon_df = df.dropna(subset=[target_col] + FEATURE_COLUMNS).copy()
    print(f"Usable rows for this horizon: {len(horizon_df)}")

    train_df, test_df = chronological_split(horizon_df, test_frac)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]

    results = {}
    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2}
        print(f"  {name:15s} RMSE={rmse:7.2f}  MAE={mae:7.2f}  R²={r2:.3f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"  → Best model for {target_col}: {best_name} (RMSE={results[best_name]['rmse']:.2f})")

    return results, best_name, X_train, X_test, y_test


# ---------------------------------------------------------------------------
# Register best model to Hopsworks Model Registry
# ---------------------------------------------------------------------------

def register_model(project, model, model_name, metrics, X_sample):
    mr = project.get_model_registry()

    local_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(model, local_path)

    hw_model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        input_example=X_sample.iloc[[0]],
        description=f"Lahore PM2.5 forecast model — {model_name}",
    )
    hw_model.save(str(local_path))
    print(f"  ✅ Registered '{model_name}' to Model Registry (metrics: {metrics})")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_training(register=True):
    df, project = load_features()

    summary = []

    for target_col in HORIZONS:
        results, best_name, X_train, X_test, y_test = train_and_evaluate_horizon(df, target_col)
        best = results[best_name]

        horizon_label = target_col.replace("target_pm25_", "").replace("h", "h")
        model_name = f"aqi_lahore_{horizon_label}_{best_name}"

        summary.append({
            "horizon": target_col,
            "best_model": best_name,
            "rmse": best["rmse"],
            "mae": best["mae"],
            "r2": best["r2"],
        })

        if register:
            register_model(
                project, best["model"], model_name,
                metrics={"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]},
                X_sample=X_train,
            )

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(pd.DataFrame(summary).to_string(index=False))

    return summary


if __name__ == "__main__":
    run_training(register=True)