"""
pipelines/train_offset_pipeline.py

1. Fetches base city features and sector features from Hopsworks.
2. Merges datasets on timestamp and calculates target offset:
   target_offset = sector_pm25 - base_pm25
3. Prepares weather, temporal, baseline AQI, and one-hot sector features.
4. Runs unconstrained Optuna Bayesian optimization to honestly evaluate the data.
5. Evaluates the final model with an aggregate AND per-sector metric breakdown.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks

# Silence Optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

BASE_FG_NAME = "aqi_base_lahore_fg"
BASE_FG_VERSION = 3

SECTOR_FG_NAME = "aqi_sector_features_fg"
SECTOR_FG_VERSION = 1  # Set to your active version

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "aqi_lahore_sector_offset_model"

def load_and_prep_dataset():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    print(f"Reading base feature group ({BASE_FG_NAME} v{BASE_FG_VERSION})...")
    base_fg = fs.get_feature_group(BASE_FG_NAME, version=BASE_FG_VERSION)
    try:
        base_df = base_fg.read()
    except Exception:
        base_df = base_fg.read(read_options={"use_hive": True})

    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    base_df = base_df[base_df["source"] == "live"].sort_values("timestamp").reset_index(drop=True)
    base_df = base_df.rename(columns={"pm25": "base_pm25"})

    print(f"Reading sector feature group ({SECTOR_FG_NAME} v{SECTOR_FG_VERSION})...")
    sector_fg = fs.get_feature_group(SECTOR_FG_NAME, version=SECTOR_FG_VERSION)
    try:
        sector_df = sector_fg.read()
    except Exception:
        sector_df = sector_fg.read(read_options={"use_hive": True})

    sector_df["timestamp"] = pd.to_datetime(sector_df["timestamp"])

    print("Merging datasets...")
    merged = pd.merge(
        sector_df,
        base_df[[
            "timestamp", "base_pm25", "temperature_2m", "relative_humidity_2m",
            "surface_pressure", "wind_speed_10m", "precipitation"
        ]],
        on="timestamp",
        how="inner"
    )

    merged["target_offset"] = merged["sector_pm25"] - merged["base_pm25"]
    merged = merged.dropna(subset=["target_offset", "wind_speed_10m"]).copy()

    merged["hour_sin"] = np.sin(2 * np.pi * merged["hour"] / 24.0)
    merged["hour_cos"] = np.cos(2 * np.pi * merged["hour"] / 24.0)
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12.0)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12.0)
    merged["dow_sin"] = np.sin(2 * np.pi * merged["day_of_week"] / 7.0)
    merged["dow_cos"] = np.cos(2 * np.pi * merged["day_of_week"] / 7.0)

    # We keep the raw sector_name for evaluation mapping, while generating dummy columns
    merged_encoded = pd.get_dummies(merged, columns=["sector_name"], prefix="sector", dtype=float)
    merged_encoded["raw_sector_name"] = merged["sector_name"]

    print(f"✅ Prepared dataset: {len(merged_encoded)} rows across sector stations.")
    return merged_encoded, project

def train_offset_model():
    df, project = load_and_prep_dataset()

    base_feature_cols = [
        "base_pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "precipitation", "hour_sin", "hour_cos",
        "month_sin", "month_cos", "dow_sin", "dow_cos"
    ]
    sector_dummy_cols = [c for c in df.columns if c.startswith("sector_")]
    feature_cols = base_feature_cols + sector_dummy_cols

    df = df.sort_values("timestamp").reset_index(drop=True)
    train_idx = int(len(df) * 0.70)
    valid_idx = int(len(df) * 0.85)

    X_train_full = df.iloc[:valid_idx][feature_cols]
    y_train_full = df.iloc[:valid_idx]["target_offset"]
    
    # We keep the raw names for the per-sector breakdown later
    X_test = df.iloc[valid_idx:][feature_cols]
    y_test = df.iloc[valid_idx:]["target_offset"]
    test_sectors = df.iloc[valid_idx:]["raw_sector_name"]

    print(f"\nOptimization Split -> Train: {train_idx} | Valid: {valid_idx - train_idx} | Test: {len(X_test)}")

    def objective(trial):
        X_opt_train = df.iloc[:train_idx][feature_cols]
        y_opt_train = df.iloc[:train_idx]["target_offset"]
        X_opt_valid = df.iloc[train_idx:valid_idx][feature_cols]
        y_opt_valid = df.iloc[train_idx:valid_idx]["target_offset"]

        # STRICT ANTI-MEMORIZATION BOUNDS
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 4),               # Capped at 4 to prevent lookup tables
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.7),         # Heavy row subsampling
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.7), # Heavy feature subsampling
            "reg_alpha": trial.suggest_float("reg_alpha", 5.0, 20.0),        # Strong L1 penalty
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 20.0),      # Strong L2 penalty
            "random_state": 42,
            "n_jobs": -1
        }

        model = XGBRegressor(**params)
        model.fit(X_opt_train, y_opt_train)
        preds = model.predict(X_opt_valid)
        return np.sqrt(mean_squared_error(y_opt_valid, preds))

    print("\nStarting Honest, Unconstrained Optuna Search (20 Trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\n" + "="*60)
    print("BEST OPTUNA PARAMETERS FOUND:")
    print(study.best_params)
    print("="*60)

    final_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train_full, y_train_full)

    test_preds = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    print(f"\nAGGREGATE TEST METRICS:")
    print(f"Test RMSE  : {test_rmse:7.2f} µg/m³")
    print(f"Test MAE   : {test_mae:7.2f} µg/m³")
    print(f"Test R²    : {test_r2:7.3f}")

    print("\n" + "="*60)
    print("PER-SECTOR EVALUATION BREAKDOWN")
    print("="*60)
    print(f"{'Sector Name':<35} | {'RMSE':<8} | {'R²':<8}")
    print("-" * 60)
    
    unique_sectors = test_sectors.unique()
    for sector in sorted(unique_sectors):
        mask = (test_sectors == sector)
        if mask.sum() > 0:
            y_test_sec = y_test[mask]
            y_pred_sec = test_preds[mask]
            
            sec_rmse = np.sqrt(mean_squared_error(y_test_sec, y_pred_sec))
            # Handle edge case where a sector might have flat variance causing R2 to break
            if len(y_test_sec) > 1 and np.var(y_test_sec) > 0:
                sec_r2 = r2_score(y_test_sec, y_pred_sec)
            else:
                sec_r2 = float('nan')
                
            print(f"{sector:<35} | {sec_rmse:<8.2f} | {sec_r2:<8.3f}")

    local_path = MODEL_DIR / f"{MODEL_NAME}.pkl"
    joblib.dump(final_model, local_path)

    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name=MODEL_NAME,
        metrics={"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
        input_example=X_train_full.iloc[[0]],
        description="Lahore Sector Spatial Offset Model — XGBoost"
    )
    hw_model.save(str(local_path))
    print(f"\n✅ Registered honest '{MODEL_NAME}' to Hopsworks Model Registry!")

if __name__ == "__main__":
    train_offset_model()