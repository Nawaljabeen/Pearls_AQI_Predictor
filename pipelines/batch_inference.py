"""
pipelines/batch_inference.py

1. Connects to Hopsworks and retrieves the latest engineered feature row.
2. Downloads registered XGBoost models for 24h, 48h, and 72h horizons (Version 2).
3. Generates PM2.5 predictions.
4. Calculates local SHAP values to extract key drivers (positive & negative impacts).
5. Saves predictions and explainability attributes to Hopsworks Datasets (Resources/latest_predictions.csv).
"""

import os
import joblib
import pandas as pd
import shap
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

def explain_prediction_with_shap(model, X_single):
    """
    Calculates TreeSHAP values for a single inference row and 
    extracts the top positive and negative feature drivers.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_single)
    
    # Extract values for the single forecast row
    values = shap_values.values[0]
    feature_names = X_single.columns
    
    # Sort features by impact magnitude
    impacts = pd.Series(values, index=feature_names).sort_values(ascending=False)
    
    # Top 2 positive drivers (increasing predicted PM2.5)
    top_positive = impacts.head(2)
    pos_str = "; ".join([f"{col} (+{val:.1f})" for col, val in top_positive.items() if val > 0])
    
    # Top 2 negative drivers (decreasing predicted PM2.5)
    top_negative = impacts.tail(2)
    neg_str = "; ".join([f"{col} ({val:.1f})" for col, val in top_negative.items() if val < 0])
    
    return pos_str if pos_str else "None", neg_str if neg_str else "None"

def run_inference():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 1. Fetch the latest feature row
    print("Fetching latest features...")
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    
    df = fg.read(read_options={"use_hive": True})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Get the single most recent row for Lahore
    latest_data = df[df["city"] == "Lahore"].sort_values("timestamp").tail(1).copy()
    
    if latest_data.empty:
        print("❌ No data found in feature store. Run feature pipeline first.")
        return

    # Calculate missing deviation feature on the fly
    latest_data["pm25_deviation"] = latest_data["pm25"] - latest_data["pm25_roll_3h"]

    current_time = latest_data["timestamp"].iloc[0]
    X_predict = latest_data[FEATURE_COLUMNS]
    
    print(f"Generating forecast based on current conditions at: {current_time}")

    # 2. Load Models, Predict, and Calculate SHAP Values
    horizons = {"24h": 24, "48h": 48, "72h": 72}
    predictions = []

    for label, hours_ahead in horizons.items():
        model_name = f"aqi_lahore_{label}_xgboost"
        print(f"Downloading model: {model_name} (Version 2)...")
        
        hw_model = mr.get_model(model_name, version=3)
        model_dir = hw_model.download()
        model = joblib.load(model_dir + f"/{model_name}.pkl")

        # Predict
        pred_pm25 = model.predict(X_predict)[0]
        pred_pm25 = max(0.0, float(pred_pm25))

        # Calculate SHAP explainability
        pos_drivers, neg_drivers = explain_prediction_with_shap(model, X_predict)

        predictions.append({
            "city": "Lahore",
            "prediction_time": current_time,
            "target_time": current_time + timedelta(hours=hours_ahead),
            "horizon": label,
            "predicted_pm25": pred_pm25,
            "top_increasing_factors": pos_drivers,
            "top_decreasing_factors": neg_drivers
        })

    # 3. Format Output
    preds_df = pd.DataFrame(predictions)
    
    print("\n" + "="*75)
    print("FORECAST RESULTS WITH LOCAL SHAP EXPLANATIONS")
    print("="*75)
    print(preds_df[["target_time", "horizon", "predicted_pm25", "top_increasing_factors"]].to_string(index=False))
    
    # 4. Save to Hopsworks Datasets (REST API)
    print("\nSaving predictions and SHAP explanations to Hopsworks Datasets...")
    dataset_api = project.get_dataset_api()
    
    local_path = "latest_predictions.csv"
    preds_df.to_csv(local_path, index=False)
    
    # Upload to the 'Resources' folder in Hopsworks (overwrites previous daily file)
    dataset_api.upload(local_path, "Resources", overwrite=True)
    print("✅ Inference complete and published to Hopsworks Datasets!")

if __name__ == "__main__":
    run_inference()