"""
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
FEATURE_GROUP_VERSION = 5  #


def fill_aqi_gaps(aqi_df, end_dt, max_gap_hours=24):
    filled_frames = []

    end_dt_utc = pd.to_datetime(end_dt)
    if end_dt_utc.tzinfo is None:
        end_dt_utc = end_dt_utc.tz_localize("UTC")
    else:
        end_dt_utc = end_dt_utc.tz_convert("UTC")
    end_dt_hour = end_dt_utc.floor("h")

    for city, group in aqi_df.groupby("city"):
        group = group.sort_values("timestamp").drop_duplicates(subset="timestamp")

        group["timestamp"] = pd.to_datetime(group["timestamp"])
        if group["timestamp"].dt.tz is None:
            group["timestamp"] = group["timestamp"].dt.tz_localize("UTC")
        else:
            group["timestamp"] = group["timestamp"].dt.tz_convert("UTC")

        min_ts = group["timestamp"].min()

        full_range = pd.date_range(min_ts, end_dt_hour, freq="h")

        g = group.set_index("timestamp").reindex(full_range)
        g.index.name = "timestamp"

        g["pm25_interpolated"] = g["pm25"].isna()
        g["pm25"] = g["pm25"].interpolate(method="linear", limit=max_gap_hours, limit_area="inside")
        g["pm25"] = g["pm25"].ffill(limit=max_gap_hours)

        g["city"] = city
        filled_frames.append(g.reset_index())

    result = pd.concat(filled_frames, ignore_index=True)

    n_interp = result["pm25_interpolated"].sum()
    n_still_missing = result["pm25"].isna().sum()
    print(f"  Gap fill: {n_interp} hours interpolated/forward-filled, {n_still_missing} hours still missing.")

    return result


# Merge


def merge_aqi_weather(aqi_df, weather_df):
    if aqi_df.empty or weather_df.empty:
        return pd.DataFrame()

    aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"])
    if aqi_df["timestamp"].dt.tz is None:
        aqi_df["timestamp"] = aqi_df["timestamp"].dt.tz_localize("UTC")
    else:
        aqi_df["timestamp"] = aqi_df["timestamp"].dt.tz_convert("UTC")

    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    if weather_df["timestamp"].dt.tz is None:
        weather_df["timestamp"] = weather_df["timestamp"].dt.tz_localize("UTC")
    else:
        weather_df["timestamp"] = weather_df["timestamp"].dt.tz_convert("UTC")

    merged = pd.merge(aqi_df, weather_df, on=["city", "timestamp"], how="left")
    return merged


# Feature engineering


def _shift_by_time(df, hours, col="pm25"):
    
    lookup = df.set_index("timestamp")[col]
    shifted_index = df["timestamp"] + pd.Timedelta(hours=hours)
    return shifted_index.map(lookup)


def engineer_features(df):
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    df = df.dropna(subset=["pm25"]).copy()

    # Time based features
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

    #humidity interaction features to fix sensor inflation
    df["rh_high_flag"] = (df["relative_humidity_2m"] > 70).astype(int)
    df["pm25_rh_interaction"] = df["pm25"] * (df["relative_humidity_2m"] / 100.0)

    # Wind direction — cyclical encoding (degrees, 0-360) 
    df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360.0)
    df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360.0)

    # Smog season flag — Lahore's PM2.5 is strongly bimodal (Oct-Feb crop
    # burning + winter inversions vs. cleaner monsoon months)
    df["is_smog_season"] = df["month"].isin([10, 11, 12, 1, 2]).astype(int)

    all_frames = []
    for city, group in df.groupby("city"):
        group = group.sort_values("timestamp").reset_index(drop=True)

      
        group["pm25_lag_1h"] = _shift_by_time(group, -1, "pm25")
        group["pm25_lag_6h"] = _shift_by_time(group, -6, "pm25")
        group["pm25_lag_24h"] = _shift_by_time(group, -24, "pm25")
        group["pm25_lag_48h"] = _shift_by_time(group, -48, "pm25")
        group["pm25_lag_168h"] = _shift_by_time(group, -168, "pm25")  # same hour, 1 week ago
        group["pm25_lag_3h"] = _shift_by_time(group, -3, "pm25")
        group["pm25_change_rate_3h"] = group["pm25"] - group["pm25_lag_3h"]

        roll = group["pm25"].rolling(window=3, min_periods=1).mean()
        group["pm25_roll_3h"] = roll.where(group["pm25_lag_3h"].notna() | (group.index < 3))

     
        group["pm25_roll_24h_mean"] = group["pm25"].rolling(window=24, min_periods=24).mean()
        group["pm25_roll_24h_std"] = group["pm25"].rolling(window=24, min_periods=24).std()

       
        group["target_pm25_24h"] = _shift_by_time(group, 24, "pm25")
        group["target_pm25_48h"] = _shift_by_time(group, 48, "pm25")
        group["target_pm25_72h"] = _shift_by_time(group, 72, "pm25")

        group["city"] = city
        all_frames.append(group)

  
    df = pd.concat(all_frames, ignore_index=True)

    _check_df = df.dropna(subset=["target_pm25_24h"])
    if len(_check_df) > 0:
        sample = _check_df.sample(min(20, len(_check_df)), random_state=0)
        lookup = df.set_index("timestamp")["pm25"]
        for _, row in sample.iterrows():
            expected = lookup.get(row["timestamp"] + pd.Timedelta(hours=24))
            if expected is not None and not pd.isna(expected):
                assert abs(row["target_pm25_24h"] - expected) < 1e-6, (
                    f"target_pm25_24h sanity check failed at {row['timestamp']}: "
                    f"got {row['target_pm25_24h']}, expected {expected} "
                    f"(the real pm25 value 24h later). Check the sign in "
                    f"_shift_by_time(group, 24, 'pm25')."
                )

    df = df.drop(columns=["pm25_lag_3h"])
  
    df = df.dropna(subset=["pm25_lag_24h", "pm25_lag_168h", "pm25_roll_24h_mean"]).copy()

    narrow_fill_cols = ["pm25_roll_3h", "pm25_change_rate_3h"]
    df[narrow_fill_cols] = df.groupby("city")[narrow_fill_cols].transform(
        lambda s: s.bfill(limit=2).ffill(limit=2)
    )
    df = df.dropna(subset=narrow_fill_cols).copy()

    return df




def push_to_hopsworks(df):
    print("\n Connecting to Hopsworks  ")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="City-level base AQI (PM2.5, Clarity only) + weather features for Lahore, 3-day forecast targets.",
        time_travel_format="HUDI",
    )

    print("Uploading to Hopsworks")
    fg.insert(df, write_options={"use_spark": False})
    print(":> Upload complete.")


def run_feature_pipeline(start_dt, end_dt, push=True):
    aqi_df, weather_df = fetch_all_raw(start_dt, end_dt)

    if aqi_df.empty:
        print(" No AQI data returned , check thr fetch step.")
        return pd.DataFrame()

    aqi_df = fill_aqi_gaps(aqi_df, end_dt=end_dt, max_gap_hours=24)

    merged = merge_aqi_weather(aqi_df, weather_df)
    if merged.empty:
        print(" No data after merge , check the fetch step.")
        return pd.DataFrame()

    print(f"\n Merged: {merged.shape[0]} rows, {merged.shape[1]} columns")

    features = engineer_features(merged)
    print(f" After feature engineering: {features.shape[0]} rows, {features.shape[1]} columns")
    print(features["city"].value_counts())
    print("Feature pipeline about to push, max timestamp:", features["timestamp"].max())
    if push:
        push_to_hopsworks(features)

    return features


if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=21)  
    run_feature_pipeline(start_dt, end_dt, push=True)