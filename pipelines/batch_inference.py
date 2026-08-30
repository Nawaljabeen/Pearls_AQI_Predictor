"""
pipelines/batch_inference.py
1. Fetches base city features and sector features from Hopsworks.
2. Predicts city-wide base PM2.5 for 24h, 48h, and 72h horizons.
3. Computes historical monthly mean offsets and applies them to all 19 sectors.
4. Calculates standard US EPA AQI scores for every prediction.
5. Pushes predictions directly back to the Hopsworks Feature Store.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "aqi_base_lahore_fg"
FEATURE_GROUP_VERSION = 5 
SECTOR_FG_NAME = "aqi_sector_features_fg"
SECTOR_FG_VERSION = 1
PRED_FG_NAME = "aqi_sector_predictions_fg"
PRED_FG_VERSION = 1


BASE_FEATURE_COLUMNS = [
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

def pm25_to_aqi(pm25):
    """Converts PM2.5 concentration to US EPA AQI value."""
    if pd.isna(pm25):
        return 0
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    if pm25 > 500.4:
        return 500
    return 0


def get_latest_model_version(mr, model_name):
   
    models = mr.get_models(model_name)
    if not models:
        raise ValueError(f"No registered models found with name '{model_name}'")
    return max(m.version for m in models)


def run_inference():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    
    print(f"Fetching latest features from {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}...")
    base_fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    try:
        base_df = base_fg.read()
    except Exception:
        base_df = base_fg.read(read_options={"use_hive": True})

    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    latest_data = base_df[base_df["city"] == "Lahore"].sort_values("timestamp").tail(1).copy()

    if latest_data.empty:
        print("❌ No base data found in feature store.")
        return

    latest_data["pm25_deviation"] = latest_data["pm25"] - latest_data["pm25_roll_3h"]
    current_time = latest_data["timestamp"].iloc[0]

    missing_cols = [c for c in BASE_FEATURE_COLUMNS if c not in latest_data.columns]
    if missing_cols:
        print(f"❌ Feature group v{FEATURE_GROUP_VERSION} is missing expected columns: {missing_cols}")
        return

    X_base = latest_data[BASE_FEATURE_COLUMNS]

    # 2. Compute Historical Monthly Mean Offsets
    sector_fg = fs.get_feature_group(SECTOR_FG_NAME, version=SECTOR_FG_VERSION)
    try:
        sector_df = sector_fg.read()
    except Exception:
        sector_df = sector_fg.read(read_options={"use_hive": True})

    sector_df["timestamp"] = pd.to_datetime(sector_df["timestamp"])

    merged_hist = pd.merge(
        sector_df,
        base_df[["timestamp", "pm25"]].rename(columns={"pm25": "base_pm25"}),
        on="timestamp",
        how="inner"
    )
    merged_hist["historical_offset"] = merged_hist["sector_pm25"] - merged_hist["base_pm25"]
    offset_lookup = merged_hist.groupby(["sector_name", "month"])["historical_offset"].mean().to_dict()
    unique_sectors = sector_df["sector_name"].unique()

    #predict across all horizons (24h, 48h, 72h)
    horizons = {"24h": 24, "48h": 48, "72h": 72}
    all_predictions = []

    for label, hours_ahead in horizons.items():
        target_time = current_time + timedelta(hours=hours_ahead)
        target_month = target_time.month

        model_name = f"aqi_lahore_{label}_xgboost"
        print(f"Processing {label} Horizon...")

        model_version = get_latest_model_version(mr, model_name)
        print(f"  using {model_name} v{model_version}")

        hw_model = mr.get_model(model_name, version=model_version)
        model_dir = hw_model.download()
        base_model = joblib.load(model_dir + f"/{model_name}.pkl")

        base_pred = max(0.0, float(base_model.predict(X_base)[0]))
        base_aqi = pm25_to_aqi(base_pred)

        # Base Station row
        all_predictions.append({
            "target_time": target_time,
            "horizon": label,
            "sector_name": "Clarity Base (City Average)",
            "predicted_pm25": base_pred,
            "predicted_aqi": base_aqi,
            "is_base": True
        })

        # Sector rows
        for sector in unique_sectors:
            sector_offset = offset_lookup.get((sector, target_month), 0.0)
            final_sector_pm25 = max(0.0, base_pred + sector_offset)
            final_sector_aqi = pm25_to_aqi(final_sector_pm25)

            all_predictions.append({
                "target_time": target_time,
                "horizon": label,
                "sector_name": sector,
                "predicted_pm25": final_sector_pm25,
                "predicted_aqi": final_sector_aqi,
                "is_base": False
            })

    preds_df = pd.DataFrame(all_predictions)

  
    print("Uploading predictions to Hopsworks Feature Store...")
    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["target_time", "horizon", "sector_name"],
        event_time="target_time",
        description="Multi-horizon AQI predictions for Lahore sectors.",
        time_travel_format="HUDI",
    )
    pred_fg.insert(preds_df, write_options={"use_spark": False})
    print(" Multi-horizon predictions uploaded to Hopsworks Feature Group successfully!")

if __name__ == "__main__":
    run_inference()