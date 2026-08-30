"""
eda/eda_lahore.py

Exploratory Data Analysis on the Lahore base AQI feature group.
Run this locally (needs your .env with HOPSWORKS_API_KEY set).

Produces:
  - Summary stats (row counts, missingness)
  - Full PM2.5 time series plot (colored by interpolated vs measured)
  - Monthly average PM2.5 (seasonality check)
  - Hour-of-day and day-of-week average PM2.5
  - PM2.5 distribution histogram
  - Correlation heatmap (weather + engineered features vs pm25)
  - Lag feature sanity check (pm25_lag_24h vs actual pm25)
  - Target horizon comparison (24h/48h/72h)

All plots saved to eda/output/ as PNGs, plus a printed text summary.

Note: the dead/StateAir station and its "source" column were removed from
the pipeline (Clarity is the only feed now), so the old live-vs-dead
source breakdown and sensor-bias comparison no longer apply and have been
dropped from this script.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "aqi_base_lahore_fg"
FEATURE_GROUP_VERSION = 5  # match whatever version your backfill actually landed in

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# loading data from hopsworks

def load_data():
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
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# Summary stats, missing vals, row counts


def print_summary(df):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Date range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"Total rows: {len(df)}")

    print(f"\n% interpolated: {100 * df['pm25_interpolated'].mean():.1f}%")

    print("\n-- pm25 stats --")
    print(df["pm25"].describe())

    print("\n-- missing values per column --")
    print(df.isna().sum()[df.isna().sum() > 0])


# Plots

def plot_full_timeseries(df):
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = {False: "#2ca02c", True: "#ff7f0e"}
    labels = {False: "measured", True: "interpolated/filled"}
    for is_interp, group in df.groupby("pm25_interpolated"):
        ax.scatter(group["timestamp"], group["pm25"], s=2, label=labels.get(is_interp, str(is_interp)),
                   color=colors.get(is_interp, "#999999"), alpha=0.5)
    ax.set_title("Lahore PM2.5 — Full Time Series (colored by measured vs interpolated)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(markerscale=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_full_timeseries.png", dpi=120)
    plt.close()
    print("Saved 01_full_timeseries.png")

# monthly seasonality to see when pm2.5 spikes
def plot_monthly_seasonality(df):
    monthly = df.groupby(df["timestamp"].dt.month)["pm25"].agg(["mean", "median", "std"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly.index, monthly["mean"], yerr=monthly["std"], capsize=3, color="#d62728", alpha=0.7)
    ax.set_title("Average PM2.5 by Month (all years combined)")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_monthly_seasonality.png", dpi=120)
    plt.close()
    print("Saved 02_monthly_seasonality.png")

# plot to check hourly n weekly pattern
def plot_hour_and_dow(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hourly = df.groupby("hour")["pm25"].mean()
    axes[0].plot(hourly.index, hourly.values, marker="o", color="#9467bd")
    axes[0].set_title("Average PM2.5 by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("PM2.5 (µg/m³)")

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df.groupby("day_of_week")["pm25"].mean()
    axes[1].bar(dow.index, dow.values, color="#8c564b", alpha=0.8)
    axes[1].set_title("Average PM2.5 by Day of Week")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_labels)
    axes[1].set_ylabel("PM2.5 (µg/m³)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_hour_dow_patterns.png", dpi=120)
    plt.close()
    print("Saved 03_hour_dow_patterns.png")

# histogram to see which vars contribute to pm2.5 so i can feature select
def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["pm25"].dropna(), bins=80, color="#17becf", edgecolor="white")
    ax.set_title("PM2.5 Distribution")
    ax.set_xlabel("PM2.5 (µg/m³)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_pm25_distribution.png", dpi=120)
    plt.close()
    print("Saved 04_pm25_distribution.png")

# the heatmap to check all vars relating to pm2.5 for feature selection
def plot_correlation(df):
    cols = [
        "pm25", "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "wind_dir_sin", "wind_dir_cos", "boundary_layer_height",
        "precipitation", "is_smog_season",
        "pm25_lag_1h", "pm25_lag_6h", "pm25_lag_24h", "pm25_lag_48h", "pm25_lag_168h",
        "pm25_roll_3h", "pm25_roll_24h_mean", "pm25_roll_24h_std",
        "pm25_change_rate_3h",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_correlation_heatmap.png", dpi=120)
    plt.close()
    print("Saved 05_correlation_heatmap.png")


def plot_lag_sanity_check(df):
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sample["pm25_lag_24h"], sample["pm25"], s=4, alpha=0.3, color="#e377c2")
    ax.plot([0, sample["pm25"].max()], [0, sample["pm25"].max()], "k--", linewidth=1)
    ax.set_title("pm25 vs pm25_lag_24h")
    ax.set_xlabel("PM2.5, 24h ago")
    ax.set_ylabel("PM2.5, now")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_lag_sanity_check.png", dpi=120)
    plt.close()
    print("Saved 06_lag_sanity_check.png")

# plot to compare 24h prediction -> 72h
def plot_target_horizons(df):
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    horizons = ["target_pm25_24h", "target_pm25_48h", "target_pm25_72h"]
    titles = ["24h ahead", "48h ahead", "72h ahead"]

    for ax, col, title in zip(axes, horizons, titles):
        ax.scatter(sample["pm25"], sample[col], s=4, alpha=0.3, color="#bcbd22")
        max_val = max(sample["pm25"].max(), sample[col].max())
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1)
        ax.set_title(f"Current PM2.5 vs {title}")
        ax.set_xlabel("PM2.5 now")

    axes[0].set_ylabel("Target PM2.5")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_target_horizons.png", dpi=120)
    plt.close()
    print("Saved 07_target_horizons.png")


def run_eda():
    df = load_data()
    print_summary(df)

    print("\nGenerating plots...")
    plot_full_timeseries(df)
    plot_monthly_seasonality(df)
    plot_hour_and_dow(df)
    plot_distribution(df)
    plot_correlation(df)
    plot_lag_sanity_check(df)
    plot_target_horizons(df)
    print(f"\n✅ EDA complete. Plots saved to: {OUTPUT_DIR}")
    return df


if __name__ == "__main__":
    run_eda()