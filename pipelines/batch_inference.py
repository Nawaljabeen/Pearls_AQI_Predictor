"""
pipelines/batch_inference.py

1. Connects to Hopsworks and retrieves the latest engineered feature row.
2. Downloads the latest registered XGBoost models for 24h, 48h, and 72h horizons.
3. Generates PM2.5 predictions.
4. Pushes the predictions to a new Hopsworks Feature Group for the Streamlit frontend.
"""

import os
import joblib
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "aqi_base_lahore_fg"
FEATURE_GROUP_VERSION = 3  # Matches your latest backfill

# The exact columns the XGBoost models expect
FEATURE_COLUMNS = [
    "pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "precipitation", "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "pm25_lag_24h", "pm25_roll_3h", "pm25_change_rate_3h", "pm25_deviation",
    "rh_high_flag", "pm25_rh_interaction",
]

def run_inference():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 1. Fetch the latest feature row
    print("Fetching latest features...")
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
    # We only need the last few rows to ensure we get the absolute latest timestamp
    df = fg.read(read_options={"use_hive": True})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Get the single most recent row for Lahore
    latest_data = df[df["city"] == "Lahore"].sort_values("timestamp").tail(1).copy()
    
    if latest_data.empty:
        print("❌ No data found in feature store. Run feature pipeline first.")
        return
    latest_data["pm25_deviation"] = latest_data["pm25"] - latest_data["pm25_roll_3h"]
    current_time = latest_data["timestamp"].iloc[0]
    X_predict = latest_data[FEATURE_COLUMNS]
    
    print(f"Generating forecast based on current conditions at: {current_time}")

    # 2. Load Models and Predict
    horizons = {"24h": 24, "48h": 48, "72h": 72}
    predictions = []

    for label, hours_ahead in horizons.items():
        model_name = f"aqi_lahore_{label}_xgboost"
        print(f"Downloading model: {model_name}...")
        
        # Get the latest version of the model
        hw_model = mr.get_model(model_name, version = 2)
        model_dir = hw_model.download()
        model = joblib.load(model_dir + f"/{model_name}.pkl")

        # Predict
        pred_pm25 = model.predict(X_predict)[0]
        
        # Prevent negative PM2.5 predictions (models occasionally predict slightly below 0 in extreme clean air)
        pred_pm25 = max(0.0, pred_pm25)

        predictions.append({
            "city": "Lahore",
            "prediction_time": current_time,
            "target_time": current_time + timedelta(hours=hours_ahead),
            "horizon": label,
            "predicted_pm25": pred_pm25
        })

    # 3. Format Output
    preds_df = pd.DataFrame(predictions)
    
    print("\n" + "="*50)
    print("FORECAST RESULTS")
    print("="*50)
    print(preds_df[["target_time", "horizon", "predicted_pm25"]].to_string(index=False))
    
    # 4. Push to Predictions Feature Group for Frontend
    print("\nPushing predictions to Hopsworks...")
    pred_fg = fs.get_or_create_feature_group(
        name="aqi_predictions_lahore_fg",
        version=1,
        primary_key=["city", "target_time"],
        event_time="target_time",
        description="Daily 3-day PM2.5 forecasts for Lahore",
    )
    
    pred_fg.insert(preds_df, write_options={"use_spark": False})
    print("✅ Inference complete and published!")

if __name__ == "__main__":
    run_inference() 