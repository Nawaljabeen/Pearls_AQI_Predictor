"""
pipelines/explainability.py

Downloads trained XGBoost models from Hopsworks, generates
SHAP and LIME interpretability plots, and saves them to a 'reports/' folder.
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "aqi_base_lahore_fg"
FEATURE_GROUP_VERSION = 5  

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


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


def get_latest_model_version(mr, model_name):
   
    models = mr.get_models(model_name)
    if not models:
        raise ValueError(f"No registered models found with name '{model_name}'")
    return max(m.version for m in models)


def run_explainability(horizon_label="24h"):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("Fetching feature data for explainability...")
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

    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    X = df[FEATURE_COLUMNS].tail(500)

    model_name = f"aqi_lahore_{horizon_label}_xgboost"
    model_version = get_latest_model_version(mr, model_name)
    print(f"Downloading model '{model_name}' v{model_version}...")
    hw_model = mr.get_model(model_name, version=model_version)
    model_dir = hw_model.download()
    model = joblib.load(model_dir + f"/{model_name}.pkl")

    # 1. SHAP Visualizations
    print("Generating SHAP summary plots...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Feature Impact — {horizon_label} Horizon", fontsize=12, fontweight='bold')
    plt.tight_layout()
    shap_path = REPORTS_DIR / f"shap_summary_{horizon_label}.png"
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f" Saved SHAP plot to: {shap_path}")

    # 2. LIME Visualizations
    print("Generating LIME local explanation plot...")
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=FEATURE_COLUMNS,
        class_names=["pm25_target"],
        mode="regression"
    )

    exp = lime_explainer.explain_instance(
        data_row=X.iloc[-1],
        predict_fn=model.predict,
        num_features=5
    )

    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Local Drivers — Latest Prediction ({horizon_label})", fontsize=11, fontweight='bold')
    plt.tight_layout()
    lime_path = REPORTS_DIR / f"lime_explanation_{horizon_label}.png"
    plt.savefig(lime_path, dpi=300)
    plt.close()
    print(f" Saved LIME plot to: {lime_path}")


if __name__ == "__main__":
    run_explainability("24h")
    