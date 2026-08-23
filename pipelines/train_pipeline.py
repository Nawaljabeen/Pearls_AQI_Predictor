"""
pipelines/train_pipeline.py

1. Fetches (features, targets) from the Hopsworks feature store.
2. Performs TimeSeriesSplit cross-validation + Optuna hyperparameter tuning 
   for each model across 24h, 48h, and 72h horizons.
3. Evaluates best estimators on a held-out chronological test set.
4. Registers the best model per horizon (by RMSE) to the Hopsworks Model Registry.

Models tuned: Ridge Regression, Random Forest, XGBoost.
Metrics: RMSE, MAE, R².
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

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
FEATURE_GROUP_VERSION = 3

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
    # Cyclical time encodings
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    # Derived & lag features
    "pm25_lag_24h",
    "pm25_roll_3h",
    "pm25_change_rate_3h",
    "pm25_deviation",
    "rh_high_flag",
    "pm25_rh_interaction",
]




# ---------------------------------------------------------------------------
# Tuning Configuration
# ---------------------------------------------------------------------------
OPTUNA_TRIALS = 30  # Adjust higher (e.g., 50-100) if you have more compute time
TS_CV_SPLITS = 3


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
    df = df[df["source"] == "live"].sort_values("timestamp").reset_index(drop=True)

    # Derived feature
    df["pm25_deviation"] = df["pm25"] - df["pm25_roll_3h"]

    # --- Cyclical Time Encoding ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    print(f"Loaded {df.shape[0]} rows (Clarity Live Only with Cyclical Encodings)")
    return df, project


# ---------------------------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------------------------
def chronological_split(df, test_frac=0.15):
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Optuna Objective Function
# ---------------------------------------------------------------------------
def optimize_hyperparameters(trial, model_name, X, y):
    tscv = TimeSeriesSplit(n_splits=TS_CV_SPLITS)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_name == "ridge":
            alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
            model = Ridge(alpha=alpha)
            
        elif model_name == "random_forest":
            n_estimators = trial.suggest_int("n_estimators", 100, 400, step=50)
            max_depth = trial.suggest_int("max_depth", 5, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42, 
                n_jobs=-1
            )
            
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1
            }
            model = XGBRegressor(**params)
            
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        
    return np.mean(scores)


# ---------------------------------------------------------------------------
# Train + evaluate for one horizon
# ---------------------------------------------------------------------------
def train_and_evaluate_horizon(df, target_col, test_frac=0.15):
    print(f"\n{'='*70}")
    print(f"HORIZON: {target_col}")
    print(f"{'='*70}")

    horizon_df = df.dropna(subset=[target_col] + FEATURE_COLUMNS).copy()
    print(f"Usable rows for this horizon: {len(horizon_df)}")

    train_df, test_df = chronological_split(horizon_df, test_frac)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]

    model_architectures = ["ridge", "random_forest"]
    if HAS_XGBOOST:
        model_architectures.append("xgboost")

    results = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for name in model_architectures:
        print(f"\n--- Tuning {name} ---")
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: optimize_hyperparameters(trial, name, X_train, y_train), n_trials=OPTUNA_TRIALS)
        
        print(f"Best CV RMSE for {name}: {study.best_value:.2f}")
        print(f"Best Params: {study.best_params}")
        
        if name == "ridge":
            model = Ridge(**study.best_params)
        elif name == "random_forest":
            model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
        elif name == "xgboost":
            model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2}
        print(f"[Test Set] RMSE={rmse:7.2f}  MAE={mae:7.2f}  R²={r2:.3f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n→ Best overall model for {target_col}: {best_name} (Test RMSE={results[best_name]['rmse']:.2f})")

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

        horizon_label = target_col.replace("target_pm25_", "")
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