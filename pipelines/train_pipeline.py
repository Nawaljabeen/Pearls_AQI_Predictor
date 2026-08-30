"""
pipelines/train_pipeline.py

Fetches backfilled feature data from Hopsworks.
Uses TimeSeriesSplit (5-Fold CV) for robust hyperparameter tuning via Optuna.
Evaluates models across multiple seasonal folds to prevent variance traps.
Trains final deployment models on 100% of the data using best parameters.
Pushes models and averaged CV metrics to the Hopsworks Model Registry.
"""

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "aqi_base_lahore_fg"
FEATURE_GROUP_VERSION = 5  

FEATURE_COLUMNS = [
    "pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_dir_sin", "wind_dir_cos", "boundary_layer_height",
    "precipitation", "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_smog_season",
    "pm25_lag_1h", "pm25_lag_6h", "pm25_lag_24h", "pm25_lag_48h", "pm25_lag_168h",
    "pm25_roll_3h", "pm25_roll_24h_mean", "pm25_roll_24h_std",
    "pm25_change_rate_3h", "pm25_deviation",
    "rh_high_flag", "pm25_rh_interaction",
]

HORIZONS = {
    "24h": "target_pm25_24h",
    "48h": "target_pm25_48h",
    "72h": "target_pm25_72h"
}

def train_and_register():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print(f"Fetching features from {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}...")
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

   
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").copy()

    df["pm25_deviation"] = df["pm25"] - df["pm25_roll_3h"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    for label, target_col in HORIZONS.items():
        print(f"\n{'='*70}")
        print(f"HORIZON: {target_col} (Time Series CV)")
        print(f"{'='*70}")

        
        horizon_df = df.dropna(subset=FEATURE_COLUMNS + [target_col]).copy()

        X = horizon_df[FEATURE_COLUMNS].reset_index(drop=True)
        y = horizon_df[target_col].reset_index(drop=True)

        print(f"Total usable rows: {len(X)}")

       
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42
            }

            tscv = TimeSeriesSplit(n_splits=5)
            fold_rmses = []

            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(X_train_fold, y_train_fold, verbose=False)
                preds = model.predict(X_val_fold)

                rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
                fold_rmses.append(rmse)

            return np.mean(fold_rmses)

        #Optuna tuning
        print("Running Optuna tuning (5-Fold TimeSeriesSplit)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15)  # Kept at 15 for speed, adjust if needed

        best_params = study.best_params
        print(f"Best CV RMSE: {study.best_value:.2f}")
        print(f"Best Params: {best_params}")

        
        tscv = TimeSeriesSplit(n_splits=5)
        rmses, maes, r2s = [], [], []

        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            cv_model = xgb.XGBRegressor(**best_params, random_state=42)
            cv_model.fit(X_train_fold, y_train_fold)
            preds = cv_model.predict(X_val_fold)

            rmses.append(np.sqrt(mean_squared_error(y_val_fold, preds)))
            maes.append(mean_absolute_error(y_val_fold, preds))
            r2s.append(r2_score(y_val_fold, preds))

        avg_metrics = {
            "cv_rmse": np.mean(rmses),
            "cv_mae": np.mean(maes),
            "cv_r2": np.mean(r2s)
        }

        print(f"\n[Averaged CV Metrics] RMSE= {avg_metrics['cv_rmse']:.2f}  MAE= {avg_metrics['cv_mae']:.2f}  R²= {avg_metrics['cv_r2']:.3f}")

        
        print(f"Training final deployment model on 100% of data ({len(X)} rows)...")
        final_model = xgb.XGBRegressor(**best_params, random_state=42)
        final_model.fit(X, y)

        # 6. Save and Register to Hopsworks
        model_name = f"aqi_lahore_{label}_xgboost"
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.pkl"
        joblib.dump(final_model, model_path)

        
        input_example = X.head(1)

        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        input_schema = Schema(X)
        output_schema = Schema(y)
        model_schema = ModelSchema(input_schema, output_schema)

        hw_model = mr.python.create_model(
            name=model_name,
            metrics=avg_metrics,
            model_schema=model_schema,
            input_example=input_example,
            description=f"XGBoost predicting {label} future PM2.5 (Trained with 5-Fold TSCV on 100% data)"
        )
        hw_model.save(model_path)
        print(f"Registered '{model_name}' to Hopsworks Registry.")

if __name__ == "__main__":
    train_and_register()