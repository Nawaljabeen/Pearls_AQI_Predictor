import os 
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import hopsworks

load_dotenv()

google_aqi_key = os.getenv("GOOGLE_AQI_API_KEY")
hopsworks_key = os.getenv("HOPSWORKS_API_KEY")

main_pak_hubs = {
    "Islamabad": {"lat":33.6844, "lng":73.0479},
    "Lahore" :{"lat": 31.5204, "lng": 74.3587},
    "Karachi": {"lat": 24.8607, "lng": 67.0011},
    "Peshawar": {"lat": 34.0151, "lng": 71.5249},
    "Quetta": {"lat": 30.1798, "lng": 66.9750},
}


def fetch_google_aqi_history(lat, lng, hours= 720):
    url = f"https://airquality.googleapis.com/v1/history:lookup?key={google_aqi_key}" 
    payload = {
        "location": {"latitude": lat, "longitude": lng},
        "hours" : hours,
        "extracomps": [
            "dom_pol_conc",
            "pol_conc",
            "local_aqi",
        ],
    }
    
    try:    
        response = requests.post(url, json = payload, timeout = 10)
        if response.status_code != 200:
            print(f"google aqi api error for ({lat}, {lng}):{ response.text}")
            return pd.DataFrame()
        
        data = response.json
        records = []
        
        for item in data.get("hoursInfo", []):
            dt = item.get("dateTime")
            indexes = item.get("indexes", [])
            
            uaqi = None
            dominant_pollutant = " unknown "
            for idx in indexes:
                if idx.get("code") == "uaqi":
                    uaqi = idx.get("aqi")
                    dominant_pollutant = idx.get("dominantPollutant", "unknown")
                    break
                pollutants = {
                    p.get("code", "").lower(): p.get("concentration", {}).get("value", 0.0)
                    for p in item.get("pollutants", [])
                }
                records.append({
                    "timestamp": dt,
                    "uaqi": uaqi,
                    "dominant_pollutant" : dominant_pollutant,
                    "pm25": pollutants.get("pm25", 0.0),
                    "pm10": pollutants.get("pm10", 0.0),
                    "no2": pollutants.get("no2", 0.0),
                    "co": pollutants.get("co", 0.0),
                    "so2": pollutants.get("so2", 0.0),
                    "o3": pollutants.get("o3", 0.0),
                })
                df = pd.DataFrame(records)
                if not df.empty:
                    df["timestamp"] = (
                        pd.to_datetime(df["timestamp"], utc = True).dt.tz_convert(None).dt.floor("h")
                        
                    )
                return df
        
    except Exception as e:
        print(f"Exception when fetching google aqi history : {e}")
        return pd.DataFrame()
    
def fetch_openmeteo_history(lat, lng, start_date, end_date):
    url =  "https://archive-api.open-meteo.com/v1/archive"   
    params = {
        "latitude" : lat,
        "longitude" : lng,
        "start_date" : start_date,
        "end_date" : end_date,
        "timezone" : "UTC",
        "hourly" : [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "precipitation",
        ],
    }
    
    try:
        response = requests.get(url, params = params, timeout = 10)
        if response.status_code != 200:
            print(f"open meteo api error: {response.text}")
            return pd.DataFrame()
        hourly = response.json().get("hourly", {})
        df= pd.DataFrame(hourly)
        if not df.empty:
            df["timestamp"] = (
                pd.to_datetime(df["time"], utc = True)
                .dt.tz_convert(None)
                .dt.floor("h")
            )
            df.drop(columns=["time"], inplace = True)
            return df
    except Exception as e:
        print(f"error fetching open meteo historical data: {e}")            
        return pd.DataFrame()
    
