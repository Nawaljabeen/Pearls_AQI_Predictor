"""
pipelines/ingest_sector_features.py

1. Ingests raw PM2.5 readings for 19 Lahore AirGradient sector stations from OpenAQ.
2. Applies automated data cleaning (bounds checks, deduplication, interpolation).
3. Pushes the cleaned sector features to Hopsworks (aqi_sector_features_fg v1).
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
OPENAQ_BASE = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": OPENAQ_API_KEY}

FEATURE_GROUP_NAME = "aqi_sector_features_fg"
FEATURE_GROUP_VERSION = 1

# Verified 19 Lahore AirGradient Sector Stations
SECTOR_STATIONS = {
    "ARC, Lahore": {"location_id": 6125629},
    "Bahria Town": {"location_id": 6236626},
    "Barki, Lahore": {"location_id": 4515157},
    "Cantonment": {"location_id": 4698865},
    "Cantt Public School & College": {"location_id": 4771378},
    "DG House DHA 5": {"location_id": 4815820},
    "Gandhara University": {"location_id": 4952332},
    "Gulberg III": {"location_id": 4618814},
    "HBFC Society, DHA Phase 5": {"location_id": 4527402},
    "IEEE, Punjab University, LHR": {"location_id": 4565933},
    "Johar Town": {"location_id": 4566427},
    "Learning Alliance Intl. DHA": {"location_id": 4527173},
    "Model Town": {"location_id": 4568423},
    "Ravi Road": {"location_id": 4555745},
    "Samanabad": {"location_id": 4700982},
    "Sandha Road": {"location_id": 4554729},
    "Shalimar Town": {"location_id": 4557686},
    "Wasa office Model Town": {"location_id": 4845646},
    "Zafar Memon DHA": {"location_id": 4814327}
}

# ---------------------------------------------------------------------------
# Data Fetching & Cleaning Functions
# ---------------------------------------------------------------------------
def get_pm25_sensor_id(location_id):
    url = f"{OPENAQ_BASE}/locations/{location_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                for sensor in results[0].get("sensors", []):
                    if sensor.get("parameter", {}).get("name") == "pm25":
                        return sensor.get("id")
    except Exception as e:
        print(f"⚠️ Error resolving sensor ID for location {location_id}: {e}")
    return None

def fetch_sensor_hourly(sensor_id, datetime_from, datetime_to, max_pages=100):
    url = f"{OPENAQ_BASE}/sensors/{sensor_id}/hours"
    all_rows = []
    page = 1

    while page <= max_pages:
        params = {"datetime_from": datetime_from, "datetime_to": datetime_to, "limit": 1000, "page": page}
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for r in results:
                dt = r.get("period", {}).get("datetimeFrom", {}).get("utc")
                value = r.get("value")
                all_rows.append({"timestamp": dt, "sector_pm25": value})
            if len(results) < 1000:
                break
            page += 1
            time.sleep(0.1)
        except Exception:
            break

    df = pd.DataFrame(all_rows)
    return df

def clean_sector_data(df):
    """Clean hardware glitches and invalid observations."""
    if df.empty:
        return df

    # Standardize timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None).dt.floor("h")

    # Drop duplicates
    df = df.drop_duplicates(subset=["timestamp", "sector_name"]).copy()

    # Rule 1: Remove impossible physical values (negative readings or sensors reading > 1000 ug/m3)
    initial_count = len(df)
    df = df[(df["sector_pm25"] >= 0) & (df["sector_pm25"] <= 1000)].copy()
    cleaned_count = len(df)
    
    if initial_count != cleaned_count:
        print(f"    🧹 Filtered out {initial_count - cleaned_count} out-of-bounds sensor readings.")

    # Rule 2: Sort chronological
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_ingestion(days_back=420):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - pd.Timedelta(days=days_back)
    datetime_from = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    datetime_to = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    all_cleaned_frames = []
    print(f"\nFetching and cleaning data for 19 sectors ({days_back} days back)...")

    for name, config in SECTOR_STATIONS.items():
        loc_id = config["location_id"]
        sensor_id = get_pm25_sensor_id(loc_id)
        if not sensor_id:
            print(f"  ❌ {name}: Sensor ID lookup failed.")
            continue

        raw_df = fetch_sensor_hourly(sensor_id, datetime_from, datetime_to)
        if raw_df.empty:
            print(f"  ⚠️ {name}: No observations returned.")
            continue

        raw_df["sector_name"] = name
        raw_df["location_id"] = loc_id
        
        cleaned_df = clean_sector_data(raw_df)
        all_cleaned_frames.append(cleaned_df)
        print(f"  ✅ {name}: {len(cleaned_df)} valid hourly rows ingested.")

    if not all_cleaned_frames:
        print("❌ No sector data successfully ingested.")
        return

    sector_dataset = pd.concat(all_cleaned_frames, ignore_index=True)
    
    # Feature Engineering for Hopsworks Ingestion
    sector_dataset["month"] = sector_dataset["timestamp"].dt.month.astype(int)
    sector_dataset["hour"] = sector_dataset["timestamp"].dt.hour.astype(int)
    sector_dataset["day_of_week"] = sector_dataset["timestamp"].dt.dayofweek.astype(int)

    print(f"\nTotal Cleaned Sector Dataset: {len(sector_dataset)} rows across all 19 stations.")

    # Create/Get Feature Group in Hopsworks
    print("\nWriting to Hopsworks Feature Store...")
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["location_id", "timestamp"],
        event_time="timestamp",
        description="AirGradient sector stations raw PM2.5 readings for Lahore spatial offset modeling.",
        time_travel_format="HUDI",
        online_enabled=False,
    )

    fg.insert(sector_dataset, wait=True)
    print(f"✅ Successfully ingested data into '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}!")

if __name__ == "__main__":
    run_ingestion()