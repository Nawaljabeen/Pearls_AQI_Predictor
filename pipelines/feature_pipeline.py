"""
pipelines/feature_pipeline.py

Takes raw (aqi_df, weather_df) from fetch_data.py, merges them, engineers
features/targets, and pushes the result to the Hopsworks feature store.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks
import numpy as np

from fetch_data import fetch_all_raw

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "aqi_base_lahore_fg"
FEATURE_GROUP_VERSION = 3 # bump on real schema changes


def fill_aqi_gaps(aqi_df, max_gap_hours=12):

    filled_frames = []

    for city, group in aqi_df.groupby("city"):
        group = group.sort_values("timestamp").drop_duplicates(subset="timestamp")
        full_range = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq="h")

        g = group.set_index("timestamp").reindex(full_range)
        g.index.name = "timestamp"

        g["pm25_interpolated"] = g["pm25"].isna()
        g["pm25"] = g["pm25"].interpolate(method="linear", limit=max_gap_hours, limit_area="inside")

        
        g["source"] = g["source"].where(~g["pm25_interpolated"], "interpolated")
        g["source"] = g["source"].fillna("unknown")  # safety net for any leftover NaNs

        g["city"] = city
        filled_frames.append(g.reset_index())

    result = pd.concat(filled_frames, ignore_index=True)

    n_interp = result["pm25_interpolated"].sum() - result["pm25"].isna().sum()
    n_still_missing = result["pm25"].isna().sum()
    print(f"  Gap fill: {n_interp} hours interpolated, {n_still_missing} hours "
          f"still missing (gap > {max_gap_hours}h, left as NaN)")

    return result



#Merge


def merge_aqi_weather(aqi_df, weather_df):
    if aqi_df.empty or weather_df.empty:
        return pd.DataFrame()
    merged = pd.merge(aqi_df, weather_df, on=["city", "timestamp"], how="left")
    return merged



#Feature engineering


def _shift_by_time(df, hours, col="pm25"):
    """
    Time-aware shift: for each row, look up the value of `col` at exactly
    `hours` away by TIMESTAMP, not by row position.

    This matters because positional .shift() breaks across data gaps — after
    dropping rows we can't interpolate (e.g. the 2020-2023 StateAir desert),
    rows that are actually years apart end up sitting on adjacent lines in
    the dataframe, and a positional shift would silently pair them together
    as if they were really N hours apart. Shifting by real timestamp instead
    correctly returns NaN when the target hour simply isn't in the data.
    """
    lookup = df.set_index("timestamp")[col]
    shifted_index = df["timestamp"] + pd.Timedelta(hours=hours)
    return shifted_index.map(lookup)


def engineer_features(df):
   
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    df = df[df["source"] == "live"].copy()
    df = df.dropna(subset=["pm25"]).copy()

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Cyclical time encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    
    

    # Optical humidity interaction features to fix sensor inflation
    df["rh_high_flag"] = (df["relative_humidity_2m"] > 70).astype(int)
    df["pm25_rh_interaction"] = df["pm25"] * (df["relative_humidity_2m"] / 100.0)

    all_frames = []
    for city, group in df.groupby("city"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        group["pm25_lag_24h"] = _shift_by_time(group, 24, "pm25")
        group["pm25_lag_3h"] = _shift_by_time(group, 3, "pm25")
        group["pm25_change_rate_3h"] = group["pm25"] - group["pm25_lag_3h"]

        roll = group["pm25"].rolling(window=3, min_periods=1).mean()
        group["pm25_roll_3h"] = roll.where(group["pm25_lag_3h"].notna() | (group.index < 3))

        group["target_pm25_24h"] = _shift_by_time(group, -24, "pm25")
        group["target_pm25_48h"] = _shift_by_time(group, -48, "pm25")
        group["target_pm25_72h"] = _shift_by_time(group, -72, "pm25")

        group["city"] = city
        all_frames.append(group)

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop(columns=["pm25_lag_3h"])

    df = df.dropna(subset=[ "pm25_lag_24h"]).copy()

    # Updated modern pandas bfill/ffill syntax
    narrow_fill_cols = ["pm25_roll_3h", "pm25_change_rate_3h"]
    df[narrow_fill_cols] = df.groupby("city")[narrow_fill_cols].transform(
        lambda s: s.bfill(limit=2).ffill(limit=2)
    )
    df = df.dropna(subset=narrow_fill_cols).copy()

    return df



# Hopsworks push


def push_to_hopsworks(df):
    print("\n Connecting to Hopsworks  ")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="City-level base AQI (PM2.5) + weather features for Lahore, 3-day forecast targets.",
        time_travel_format="HUDI",
    )

    print("Uploading to Hopsworks")
    fg.insert(df, write_options={"use_spark": False})
    print(":> Upload complete.")



def run_feature_pipeline(start_dt, end_dt, include_dead_station=False, push=True):
    aqi_df, weather_df = fetch_all_raw(start_dt, end_dt, include_dead_station)

    if aqi_df.empty:
        print(" No AQI data returned , check thr fetch step.")
        return pd.DataFrame()

    aqi_df = fill_aqi_gaps(aqi_df, max_gap_hours=12)

    merged = merge_aqi_weather(aqi_df, weather_df)
    if merged.empty:
        print(" No data after merge , check the fetch step.")
        return pd.DataFrame()

    print(f"\n Merged: {merged.shape[0]} rows, {merged.shape[1]} columns")

    features = engineer_features(merged)
    print(f" After feature engineering: {features.shape[0]} rows, {features.shape[1]} columns")
    print(features["city"].value_counts())

    if push:
        push_to_hopsworks(features)

    return features


if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=7)  # short test window first
    run_feature_pipeline(start_dt, end_dt, push=True)  # push is False for local testing