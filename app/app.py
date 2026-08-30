"""
app.py — Lahore Air Quality Spatial Forecast Dashboard
Design Theme: Irasutoya Soft Pastel Aesthetic (League Spartan Typography)
"""

import os
from pathlib import Path
import base64
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

if (APP_DIR / "assets").exists():
    ASSETS_DIR = APP_DIR / "assets"
else:
    ASSETS_DIR = ROOT_DIR / "assets"

# ---------------------------------------------------------------------------
# Page Configuration & Global CSS Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lahore Spatial AQI Forecast",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=League+Spartan:wght@400;500;600;700;800&family=Londrina+Solid:wght@400;700;900&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span, button, input, select {
        font-family: 'Londrina Solid', 'League Spartan', sans-serif !important;
    }
    
    .stApp {
        background-color: #FAF8F5;
    }
    
    div[data-testid="stRadio"] {
        text-align: center;
    }
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: row;
        justify-content: center;
        gap: 28px;
    }
    div[data-testid="stRadio"] label {
        color: #2D3748 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span {
        color: #2D3748 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        opacity: 1 !important;
        background-color: transparent !important;
    }
    div[data-testid="stRadio"] label > div:first-child,
    div[data-testid="stRadio"] label [data-baseweb="radio"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stRadio"] label > div:first-child > div,
    div[data-testid="stRadio"] label [data-baseweb="radio"] > div {
        background-color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) > div:first-child > div,
    div[data-testid="stRadio"] label:has(input:checked) [data-baseweb="radio"] > div,
    div[data-testid="stRadio"] label[aria-checked="true"] > div:first-child > div,
    div[data-testid="stRadio"] label[aria-checked="true"] [data-baseweb="radio"] > div {
        background-color: #FF6F91 !important;
        box-shadow: none !important;
    }
    div[data-testid="stRadio"] label div:first-child,
    div[data-testid="stRadio"] label div:first-child *,
    div[data-testid="stRadio"] label {
        outline: none !important;
        background-color: transparent !important;
    }
    
    .stSelectbox [data-baseweb="select"] div,
    [data-baseweb="popover"] div,
    div[role="listbox"] div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #FFD1E0 !important;
        color: #2D3748 !important;
        border-radius: 999px !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.10) !important;
    }
    [data-baseweb="popover"],
    div[role="listbox"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12) !important;
    }
    [data-baseweb="select"] svg {
        fill: #2D3748 !important;
    }
    
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] * {
        background-color: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #4A5568 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    
    .dashboard-title {
        text-align: center;
        color: #2D3748;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 2px;
    }
    
    .dashboard-subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    
    .spotlight-card {
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    
    .guidance-bubble {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #4A5568;
    }
    .color-blob {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# AQI Metadata & Helpers
# ---------------------------------------------------------------------------
def get_aqi_details(aqi):
    if aqi <= 50:
        return ("Good", "#2E7D32", "#E8F5E9", "#A5D6A7", "aqi_good.png", "Air quality is clean. Perfect time for a run, a picnic, or opening up the windows!")
    elif aqi <= 100:
        return ("Moderate", "#F57F17", "#FFFDE7", "#FFE082", "aqi_moderate.png", "Air quality is acceptable. Great for most people, though extra sensitive lungs might notice a tiny tickle.")
    elif aqi <= 150:
        return ("Unhealthy for Sensitive Groups", "#E65100", "#FFF3E0", "#FFCC80", "aqi_sensitive.png", "Couch potatoes win today! If you get asthma or allergies, maybe skip the outdoor workout or wear a mask!")
    elif aqi <= 200:
        return ("Unhealthy", "#C62828", "#FFEBEE", "#EF9A9A", "aqi_unhealthy.png", "You will probably feel this in your throat. Wear an N95 mask outdoors and close your windows!")
    elif aqi <= 300:
        return ("Very Unhealthy", "#6A1B9A", "#F3E5F5", "#CE93D8", "aqi_very_unhealthy.png", "Health alert. Avoid outdoor activities, run air purifiers and chill inside!")
    else:
        return ("Hazardous", "#4E342E", "#EFEBE9", "#BCAAA4", "aqi_hazardous.png", "Emergency warnings. Remain strictly indoors with air filtration! It looks like a sci fi movie out there!")

def pm25_to_aqi(pm25):
    if pd.isna(pm25):
        return 0
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    if pm25 > 500.4:
        return 500
    return 0

def format_pkt_time(t_str):
    try:
        dt = pd.to_datetime(t_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC').tz_convert('Asia/Karachi')
        else:
            dt = dt.tz_convert('Asia/Karachi')
        return dt.strftime('%A, %b %d, %Y — %H:%M PKT')
    except Exception:
        return str(t_str)

SECTOR_COORDS = {
    "Clarity Base (City Average)": (31.5204, 74.3587),
    "ARC, Lahore": (31.5497, 74.3436),
    "Bahria Town": (31.3526, 74.1794),
    "Barki, Lahore": (31.5580, 74.4710),
    "Cantonment": (31.5038, 74.3854),
    "Cantt Public School & College": (31.5100, 74.3900),
    "DG House DHA 5": (31.4756, 74.4045),
    "Gandhara University": (31.4500, 74.3000),
    "Gulberg III": (31.5050, 74.3486),
    "HBFC Society, DHA Phase 5": (31.4680, 74.4120),
    "IEEE, Punjab University, LHR": (31.4987, 74.2931),
    "Johar Town": (31.4697, 74.2728),
    "Learning Alliance Intl. DHA": (31.4720, 74.3980),
    "Model Town": (31.4826, 74.3274),
    "Ravi Road": (31.6014, 74.3094),
    "Samanabad": (31.5375, 74.3031),
    "Sandha Road": (31.5500, 74.2900),
    "Shalimar Town": (31.5833, 74.3833),
    "Wasa office Model Town": (31.4800, 74.3300),
    "Zafar Memon DHA": (31.4600, 74.4200)
}

# ---------------------------------------------------------------------------
# Data Loading Directly from Hopsworks Feature Groups (No CSVs)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_predictions():
    if not HOPSWORKS_API_KEY:
        st.error("❌ HOPSWORKS_API_KEY is missing from environment variables or secrets.")
        return pd.DataFrame()
        
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        
        # 1. Read predictions directly from the Feature Group
        pred_fg = fs.get_feature_group("aqi_sector_predictions_fg", version=1)
        try:
            df = pred_fg.read()
        except Exception:
            df = pred_fg.read(read_options={"use_hive": True})
            
        df["sector_name"] = df["sector_name"].astype(str).str.strip()
        
        # 2. Append "Current (Live)" horizon rows dynamically from the base feature store
        base_fg = fs.get_feature_group("aqi_base_lahore_fg", version=5)
        try:
            base_df = base_fg.read()
        except Exception:
            base_df = base_fg.read(read_options={"use_hive": True})
            
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
        latest_base = base_df[base_df["city"] == "Lahore"].sort_values("timestamp").tail(1)
        
        if not latest_base.empty:
            curr_time = latest_base["timestamp"].iloc[0]
            curr_pm25 = float(latest_base["pm25"].iloc[0])
            curr_aqi = pm25_to_aqi(curr_pm25)
            
            current_rows = [{
                "target_time": str(curr_time),
                "horizon": "Current (Live)",
                "sector_name": "Clarity Base (City Average)",
                "predicted_pm25": curr_pm25,
                "predicted_aqi": curr_aqi,
                "is_base": True
            }]
            
            sector_fg = fs.get_feature_group("aqi_sector_features_fg", version=1)
            try:
                sector_df = sector_fg.read()
            except Exception:
                sector_df = sector_fg.read(read_options={"use_hive": True})
                
            sector_df["timestamp"] = pd.to_datetime(sector_df["timestamp"])
            
            merged_hist = pd.merge(
                sector_df,
                base_df[["timestamp", "pm25"]].rename(columns={"pm25": "base_pm25"}),
                on="timestamp",
                how="inner"
            )
            merged_hist["historical_offset"] = merged_hist["sector_pm25"] - merged_hist["base_pm25"]
            curr_month = curr_time.month
            offset_lookup = merged_hist.groupby(["sector_name", "month"])["historical_offset"].mean().to_dict()
            unique_sectors = sector_df["sector_name"].unique()
            
            for sector in unique_sectors:
                sector_offset = offset_lookup.get((sector, curr_month), 0.0)
                final_pm25 = max(0.0, curr_pm25 + sector_offset)
                final_aqi = pm25_to_aqi(final_pm25)
                current_rows.append({
                    "target_time": str(curr_time),
                    "horizon": "Current (Live)",
                    "sector_name": sector,
                    "predicted_pm25": final_pm25,
                    "predicted_aqi": final_aqi,
                    "is_base": False
                })
            
            df_curr = pd.DataFrame(current_rows)
            df = pd.concat([df, df_curr], ignore_index=True)
            
        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch predictions from Hopsworks Feature Store: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# Header & Horizon Selector
# ---------------------------------------------------------------------------
st.markdown("<div class='dashboard-title'>Lahore's City Wide 3 day AQI Forecaster based on realtime sensor data</div>", unsafe_allow_html=True)
st.markdown("<div class='dashboard-subtitle'>Real time air quality forecasting across 19 locations in Lahore </div>", unsafe_allow_html=True)

df_all = fetch_predictions()

if df_all.empty:
    st.error("Prediction feature group unavailable or empty in Hopsworks.")
    st.stop()

# Horizon Selector Radio Buttons (Default to Current Live)
c_sp1, c_rad, c_sp2 = st.columns([1, 2.5, 1])
with c_rad:
    selected_horizon = st.radio(
        "Select Prediction Horizon", 
        ["Current (Live)", "24h", "48h", "72h"], 
        horizontal=True,
        index=0,
        key="horizon_selector"
    )
st.write("")

df = df_all[df_all["horizon"] == selected_horizon].copy()

# ---------------------------------------------------------------------------
# Top Section: AQI Card (Top Left), Guidance Bubble with Asset (Top Right)
# ---------------------------------------------------------------------------
top_col1, top_col2 = st.columns([1, 1])

# Sorted sector list to maintain consistent alphabetical order across horizons
sector_list = sorted(list(df["sector_name"].unique()))

with top_col1:
    st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: transparent; margin-bottom: 6px;'>&nbsp;</div>", unsafe_allow_html=True)

    spotlight_placeholder = st.empty()

    # Stable selectbox with a fixed key so user selection is preserved across horizon changes
    selected_sector = st.selectbox(
        "Select Location Neighborhood:", 
        sector_list, 
        index=0, 
        key="neighborhood_selector"
    )

    selected_row = df[df["sector_name"] == selected_sector].iloc[0]
    category, accent_color, bg_pastel, border_color, img_name, health_advice = get_aqi_details(selected_row["predicted_aqi"])

    target_time_str = selected_row.get('target_time', '')
    formatted_target = format_pkt_time(target_time_str)

    spotlight_placeholder.markdown(f"""
    <div class="spotlight-card" style="background-color: {bg_pastel}; text-align: center; margin-top: 5px; margin-bottom: 18px;">
        <span style="background-color: {accent_color}; color: #FFFFFF; padding: 3px 12px; border-radius: 10px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">
            {category}
        </span>
        <div style="color: #2D3748; margin: 10px 0 2px 0; font-size: 2.2rem; font-weight: 800; text-align: center;">
            AQI {selected_row['predicted_aqi']}
        </div>
        <div style="color: #4A5568; font-weight: 600; font-size: 0.9rem; text-align: center;">
            {selected_sector} ({selected_row['predicted_pm25']:.1f} ug/m3)
        </div>
        <div style="color: #718096; font-size: 0.8rem; font-weight: 600; margin-top: 6px;">
            Timestamp: {formatted_target}
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #4A5568; margin-bottom: 6px;'>Localized Health Guidance</div>", unsafe_allow_html=True)
    
    img_path = ASSETS_DIR / img_name
    img_html = ""
    if img_path.exists():
        with open(img_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode()
        img_html = f"<div style='text-align: center; margin-top: 14px;'><img src='data:image/png;base64,{encoded_img}' style='width: 90px; height: auto;'/></div>"

    st.markdown(f"""
    <div class="guidance-bubble">
        <div style="color: #2D3748; font-size: 0.95rem; line-height: 1.4; font-weight: 500; text-align: left;">
            {health_advice}
        </div>
        {img_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Map Panel Section (Map on Right, Legend/Controls on Left)
# ---------------------------------------------------------------------------
panel_col1, panel_col2 = st.columns([1, 1.8])

with panel_col1:
    st.markdown("<div style='font-weight: 700; font-size: 1.1rem; color: #2D3748; margin-bottom: 10px;'>Map Legend & Guide</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.9rem; color: #718096; margin-bottom: 14px;'>Click any marker on the map or inspect the color bands below to evaluate sector pollution levels.</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="legend-item"><span class="color-blob" style="background-color: #2E7D32;"></span>Good (0 - 50)</div>
    <div class="legend-item"><span class="color-blob" style="background-color: #F57F17;"></span>Moderate (51 - 100)</div>
    <div class="legend-item"><span class="color-blob" style="background-color: #E65100;"></span>Sensitive Groups (101 - 150)</div>
    <div class="legend-item"><span class="color-blob" style="background-color: #C62828;"></span>Unhealthy (151 - 200)</div>
    <div class="legend-item"><span class="color-blob" style="background-color: #6A1B9A;"></span>Very Unhealthy (201 - 300)</div>
    <div class="legend-item"><span class="color-blob" style="background-color: #4E342E;"></span>Hazardous (301+)</div>
    """, unsafe_allow_html=True)

with panel_col2:
    m = folium.Map(location=[31.5204, 74.3587], zoom_start=11, tiles="OpenStreetMap")
    
    for _, row in df.iterrows():
        sec_name = row["sector_name"]
        aqi_val = row["predicted_aqi"]
        pm25_val = row["predicted_pm25"]
        
        cat_name, color_hex, _, _, _, _ = get_aqi_details(aqi_val)
        lat, lon = SECTOR_COORDS.get(sec_name, (31.5204, 74.3587))
        
        popup_content = f"""
        <div style="font-family: 'League Spartan', sans-serif; width: 170px;">
            <b style="font-size: 1.1rem; color: #2D3748;">{sec_name}</b><br>
            <span style="color: {color_hex}; font-weight: 700;">{cat_name}</span><br>
            <hr style="margin: 6px 0; border: none; border-top: 1px solid #E2E8F0;">
            <b>AQI:</b> {aqi_val}<br>
            <b>PM2.5:</b> {pm25_val:.1f} ug/m3<br>
            <span style="color: #718096; font-size: 0.85rem;">View: {selected_horizon}</span>
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            popup=folium.Popup(popup_content, max_width=220),
            tooltip=f"{sec_name}: AQI {aqi_val}",
            color="#2D3748",
            weight=2,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.85
        ).add_to(m)
        
    st_folium(m, width=650, height=420)

# ---------------------------------------------------------------------------
# Below Map: Cleanest & Worst Neighborhood Summary Cards
# ---------------------------------------------------------------------------
sectors_only = df[df["is_base"] == False]
cleanest_row = sectors_only.sort_values("predicted_pm25").iloc[0] if not sectors_only.empty else df.iloc[0]
polluted_row = sectors_only.sort_values("predicted_pm25", ascending=False).iloc[0] if not sectors_only.empty else df.iloc[0]
base_row = df[df["is_base"] == True].iloc[0] if not df[df["is_base"] == True].empty else df.iloc[0]

c_m1, c_m2, c_m3 = st.columns(3)
with c_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #718096; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">CITY BASE AVERAGE</div>
        <div style="color: #2D3748; font-size: 1.8rem; font-weight: 800; margin: 4px 0;">{base_row['predicted_aqi']} <span style="font-size: 0.8rem; color: #A0AEC0;">AQI</span></div>
        <div style="color: #718096; font-size: 0.85rem; font-weight: 600;">PM2.5: {base_row['predicted_pm25']:.1f} ug/m3</div>
    </div>
    """, unsafe_allow_html=True)

with c_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #2E7D32; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">CLEANEST NEIGHBORHOOD</div>
        <div style="color: #2D3748; font-size: 1.2rem; font-weight: 800; margin: 6px 0;">{cleanest_row['sector_name']}</div>
        <div style="color: #2E7D32; font-size: 0.9rem; font-weight: 700;">AQI {cleanest_row['predicted_aqi']} ({cleanest_row['predicted_pm25']:.1f} ug/m3)</div>
    </div>
    """, unsafe_allow_html=True)

with c_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #C62828; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">HIGHEST POLLUTION NODE</div>
        <div style="color: #2D3748; font-size: 1.2rem; font-weight: 800; margin: 6px 0;">{polluted_row['sector_name']}</div>
        <div style="color: #C62828; font-size: 0.9rem; font-weight: 700;">AQI {polluted_row['predicted_aqi']} ({polluted_row['predicted_pm25']:.1f} ug/m3)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bottom Section: Summary Table with PKT Timezones
# ---------------------------------------------------------------------------
st.markdown("<div style='font-weight: 800; font-size: 1.3rem; color: #2D3748; margin-bottom: 10px;'>Complete Multi-Horizon Sector Forecast Table</div>", unsafe_allow_html=True)

table_df = df_all.pivot_table(
    index="sector_name",
    columns="horizon",
    values="predicted_aqi"
).reset_index()

target_time_map = {}
for h in ["Current (Live)", "24h", "48h", "72h"]:
    subset = df_all[df_all["horizon"] == h]
    if not subset.empty:
        t_str = subset["target_time"].iloc[0]
        try:
            dt = pd.to_datetime(t_str)
            if dt.tzinfo is None:
                dt = dt.tz_localize('UTC').tz_convert('Asia/Karachi')
            else:
                dt = dt.tz_convert('Asia/Karachi')
            target_time_map[h] = dt.strftime('%a, %b %d — %H:%M PKT')
        except Exception:
            target_time_map[h] = t_str

cols_order = ["sector_name"] + [h for h in ["Current (Live)", "24h", "48h", "72h"] if h in table_df.columns]
table_df = table_df[cols_order]

rename_dict = {"sector_name": "Neighborhood Sector"}
for h in ["Current (Live)", "24h", "48h", "72h"]:
    if h in target_time_map:
        rename_dict[h] = f"{h} ({target_time_map[h]})"
    else:
        rename_dict[h] = h

table_df = table_df.rename(columns=rename_dict)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)