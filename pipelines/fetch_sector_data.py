"""
pipelines/fetch_sector_data.py

Fetches historical data for AirGradient sector stations in Lahore,
merges it with the Clarity base station, calculates the spatial offset,
and generates exploratory visualizations.
"""

import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
OPENAQ_BASE = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": OPENAQ_API_KEY}

# Clarity Base Station
BASE_STATION_ID = 1894641

# TODO: Plug in your actual AirGradient location IDs from your CSV here
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
# OpenAQ Fetching Logic (Reused from your base pipeline)
# ---------------------------------------------------------------------------
def get_pm25_sensor_id(location_id):
    url = f"{OPENAQ_BASE}/locations/{location_id}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return None
    results = resp.json().get("results", [])
    if not results:
        return None
    for sensor in results[0].get("sensors", []):
        if sensor.get("parameter", {}).get("name") == "pm25":
            return sensor.get("id")
    return None

def fetch_sensor_hourly(sensor_id, datetime_from, datetime_to, max_pages=100):
    url = f"{OPENAQ_BASE}/sensors/{sensor_id}/hours"
    all_rows = []
    page = 1

    while page <= max_pages:
        params = {"datetime_from": datetime_from, "datetime_to": datetime_to, "limit": 1000, "page": page}
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        except requests.exceptions.RequestException:
            break
            
        if resp.status_code != 200:
            break

        results = resp.json().get("results", [])
        if not results:
            break

        for r in results:
            dt = r.get("period", {}).get("datetimeFrom", {}).get("utc")
            value = r.get("value")
            all_rows.append({"timestamp": dt, "pm25": value})

        if len(results) < 1000:
            break
        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None).dt.floor("h")
    return df

# ---------------------------------------------------------------------------
# Data Processing & Visualization
# ---------------------------------------------------------------------------
def analyze_sector_offsets(start_dt, end_dt):
    datetime_from = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    datetime_to = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print("Fetching Clarity Base Station...")
    base_sensor = get_pm25_sensor_id(BASE_STATION_ID)
    base_df = fetch_sensor_hourly(base_sensor, datetime_from, datetime_to)
    base_df = base_df.rename(columns={"pm25": "base_pm25"})

    sector_frames = []
    
    print("\nFetching AirGradient Sector Stations...")
    for name, config in SECTOR_STATIONS.items():
        loc_id = config["location_id"]
        sensor_id = get_pm25_sensor_id(loc_id)
        if not sensor_id:
            print(f"  ❌ {name}: No PM2.5 sensor found.")
            continue
            
        df = fetch_sensor_hourly(sensor_id, datetime_from, datetime_to)
        if df.empty:
            print(f"  ⚠️ {name}: No data in this time range.")
            continue
            
        df["sector_name"] = name
        df = df.rename(columns={"pm25": "sector_pm25"})
        sector_frames.append(df)
        print(f"  ✅ {name}: Fetched {len(df)} rows.")

    if not sector_frames:
        print("\nNo sector data retrieved. Exiting.")
        return

    # Merge and calculate offsets
    all_sectors_df = pd.concat(sector_frames, ignore_index=True)
    merged_df = pd.merge(all_sectors_df, base_df, on="timestamp", how="inner")
    
    # Mathematical definition of the offset
    merged_df["pm25_offset"] = merged_df["sector_pm25"] - merged_df["base_pm25"]
    
    print(f"\nMerged Dataset: {len(merged_df)} overlapping hourly observations across all sectors.")

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # 1. Boxplot of offsets per sector
    sns.boxplot(x="sector_name", y="pm25_offset", data=merged_df, ax=axes[0], palette="viridis")
    axes[0].set_title("Distribution of PM2.5 Offsets by Sector (Sector - Base)")
    axes[0].set_ylabel("PM2.5 Offset (µg/m³)")
    axes[0].set_xlabel("")
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2) # 0 line = identical to Clarity

    # 2. Time-series snippet (Last 14 days)
    recent_df = merged_df[merged_df["timestamp"] >= (merged_df["timestamp"].max() - pd.Timedelta(days=14))]
    sns.lineplot(x="timestamp", y="sector_pm25", hue="sector_name", data=recent_df, ax=axes[1], alpha=0.6)
    sns.lineplot(x="timestamp", y="base_pm25", data=recent_df, ax=axes[1], color="black", linewidth=3, label="Clarity Base")
    axes[1].set_title("Recent 14-Day Overlap (Sectors vs Base)")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig("sector_offset_analysis.png")
    print("\nVisualizations saved to 'sector_offset_analysis.png'.")

if __name__ == "__main__":
    # Roughly 14 months of overlap history
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - pd.Timedelta(days=420) 
    
    analyze_sector_offsets(start_dt, end_dt)