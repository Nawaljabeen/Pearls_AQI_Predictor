"""

Scope: Lahore
Base AQI signal:
  - Live:  Clarity        (2023-11 -> live)
  - Dead:  StateAir Lahore (2019-05 -> 2025-03)

Sector-level AirGradient stations are handled separately in fetch_sector_data.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
OPENAQ_BASE = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": OPENAQ_API_KEY}


# Station config — which we verified in CSV 


BASE_STATIONS = {
    "Lahore": {
        "live":    {"location_id": 1894641, "name": "Lahore (Clarity)"},
        "dead":    {"location_id": 8664,    "name": "US Diplomatic Post: Lahore (StateAir)"},
    },
}

CITY_CENTERS = {
    "Lahore":    {"lat": 31.5204, "lng": 74.3587},
}



#get sensor IDs for a given location


def get_pm25_sensor_id(location_id):
    """Looks up the pm25 sensor's ID for a given OpenAQ location."""
    url = f"{OPENAQ_BASE}/locations/{location_id}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"  could not fetch location {location_id}: {resp.text[:150]}")
        return None

    results = resp.json().get("results", [])
    if not results:
        return None

    for sensor in results[0].get("sensors", []):
        if sensor.get("parameter", {}).get("name") == "pm25":
            return sensor.get("id")

    return None


def fetch_sensor_hourly(sensors_id, datetime_from, datetime_to, max_pages=100):
    """Pulls hourly-averaged pm25 for one sensor over a date range. Paginated."""
    url = f"{OPENAQ_BASE}/sensors/{sensors_id}/hours"
    all_rows = []
    page = 1

    while page <= max_pages:
        params = {
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "limit": 1000,
            "page": page,
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        except requests.exceptions.RequestException as e:
            print(f"    !! request error for sensor {sensors_id}: {e}")
            break

        if resp.status_code != 200:
            print(f"   !! sensor {sensors_id} error {resp.status_code}: {resp.text[:150]}")
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


def fetch_city_base_aqi(city_name, datetime_from, datetime_to, include_dead_station=True):
    """
    Builds one continuous PM2.5 series for a city by combining its live station
    with its dead/historical station.
"""
    stations = BASE_STATIONS[city_name]
    frames = []

    live_id = stations["live"]["location_id"]
    live_sensor = get_pm25_sensor_id(live_id)
    if live_sensor:
        df_live = fetch_sensor_hourly(live_sensor, datetime_from, datetime_to)
        if not df_live.empty:
            df_live["source"] = "live"
            frames.append(df_live)
            print(f"  {city_name} LIVE station ({stations['live']['name']}): {len(df_live)} rows")

    if include_dead_station:
        dead_id = stations["dead"]["location_id"]
        dead_sensor = get_pm25_sensor_id(dead_id)
        if dead_sensor:
            df_dead = fetch_sensor_hourly(dead_sensor, datetime_from, datetime_to)
            if not df_dead.empty:
                df_dead["source"] = "dead"
                frames.append(df_dead)
                print(f"  {city_name} DEAD station ({stations['dead']['name']}): {len(df_dead)} rows")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # if both sources have a value for the same hour then give priority to live
    combined["source_priority"] = combined["source"].map({"live": 0, "dead": 1})
    combined = combined.sort_values(["timestamp", "source_priority"])
    combined = combined.drop_duplicates(subset="timestamp", keep="first")
    combined = combined.drop(columns=["source_priority"]).sort_values("timestamp").reset_index(drop=True)

    combined["city"] = city_name
    return combined[["city", "timestamp", "pm25", "source"]]


# Open-Meteo weather 


def fetch_openmeteo_history(city_name, lat, lng, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "precipitation",
        ],
    }

    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        print(f"  open-meteo error for {city_name}: {resp.text[:150]}")
        return pd.DataFrame()

    hourly = resp.json().get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None).dt.floor("h")
    df.drop(columns=["time"], inplace=True)
    df["city"] = city_name
    return df


# Gap analysis 


def analyze_gaps(aqi_df, city_name="Lahore"):

    df = aqi_df[aqi_df["city"] == city_name].copy()
    if df.empty:
        print(f"No data for {city_name} — nothing to analyze.")
        return

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    actual_hours = set(df["timestamp"])

    full_range = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="h")
    missing_hours = sorted(set(full_range) - actual_hours)

    total_expected = len(full_range)
    total_missing = len(missing_hours)
    pct_missing = 100 * total_missing / total_expected if total_expected else 0

    print(f"\n{'='*70}")
    print(f"GAP ANALYSIS — {city_name}")
    print(f"{'='*70}")
    print(f"Date range:        {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"Expected hours:    {total_expected}")
    print(f"Actual hours:      {len(actual_hours)}")
    print(f"Missing hours:     {total_missing}  ({pct_missing:.1f}%)")

    # finding contiguous missing blocks
    gaps = []
    if missing_hours:
        block_start = missing_hours[0]
        prev = missing_hours[0]
        for ts in missing_hours[1:]:
            if ts - prev > pd.Timedelta(hours=1):
                gaps.append((block_start, prev))
                block_start = ts
            prev = ts
        gaps.append((block_start, prev))

    gaps_sorted = sorted(gaps, key=lambda g: (g[1] - g[0]), reverse=True)

    print(f"\nNumber of distinct gap blocks: {len(gaps_sorted)}")
    if gaps_sorted:
        longest = gaps_sorted[0]
        longest_hours = (longest[1] - longest[0]).total_seconds() / 3600 + 1
        print(f"Largest gap: {longest[0]} -> {longest[1]}  ({longest_hours:.0f} hours, "
              f"~{longest_hours/24:.1f} days)")

    print(f"\nTop 10 largest gaps:")
    for start, end in gaps_sorted[:10]:
        hours = (end - start).total_seconds() / 3600 + 1
        print(f"  {start}  ->  {end}   ({hours:.0f}h, ~{hours/24:.1f} days)")

    # % missing by month
    df["month"] = df["timestamp"].dt.to_period("M")
    full_range_df = pd.DataFrame({"timestamp": full_range})
    full_range_df["month"] = full_range_df["timestamp"].dt.to_period("M")
    full_range_df["is_present"] = full_range_df["timestamp"].isin(actual_hours)

    monthly = full_range_df.groupby("month")["is_present"].agg(["sum", "count"])
    monthly["pct_missing"] = 100 * (1 - monthly["sum"] / monthly["count"])

    print(f"\n% missing by month:")
    print(monthly[["pct_missing"]].round(1).to_string())

    return {
        "pct_missing_overall": pct_missing,
        "gaps": gaps_sorted,
        "monthly": monthly,
    }


# Orchestrator: fetch base station


def fetch_all_raw(start_dt, end_dt, include_dead_station=True):
    """
    Returns (aqi_df, weather_df) , both long format, city column included,
    not merged the merging + feature engineering is happening in feature_pipeline.py
    """
    datetime_from = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    datetime_to = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    aqi_frames = []
    weather_frames = []

    for city_name in BASE_STATIONS.keys():
        print(f"\n! {city_name}")
        aqi_df = fetch_city_base_aqi(city_name, datetime_from, datetime_to, include_dead_station)
        if not aqi_df.empty:
            aqi_frames.append(aqi_df)

        center = CITY_CENTERS[city_name]
        weather_df = fetch_openmeteo_history(city_name, center["lat"], center["lng"], start_date_str, end_date_str)
        if not weather_df.empty:
            weather_frames.append(weather_df)

    aqi_all = pd.concat(aqi_frames, ignore_index=True) if aqi_frames else pd.DataFrame()
    weather_all = pd.concat(weather_frames, ignore_index=True) if weather_frames else pd.DataFrame()

    return aqi_all, weather_all


if __name__ == "__main__":
   
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - pd.Timedelta(days=7)
    aqi_df, weather_df = fetch_all_raw(start_dt, end_dt)
    print("\nAQI sample:\n", aqi_df.head())
    print("\nWeather sample:\n", weather_df.head())


    analyze_gaps(aqi_df, city_name="Lahore")