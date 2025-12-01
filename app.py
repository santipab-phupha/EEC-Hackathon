import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import geopandas as gpd
import lightgbm as lgb
from shapely.geometry import Point
import os
import google.generativeai as genai
from ultralytics import YOLO
import cv2

# === ตั้งค่า UI ===
st.set_page_config(page_title="🔥 Wildfire Hotspot Prediction", layout="wide")

# === โหลดโมเดลและข้อมูลพยากรณ์ ===
booster = lgb.Booster(model_file="model.txt")
to_pred = pd.read_csv("to_pred.csv")

features = [
    "lag1","lag3","lag7","lag14",
    "rollsum_3","rollsum_7","rollsum_14",
    "sin_doy","cos_doy","month","dow"
]

proba = booster.predict(to_pred[features], num_iteration=booster.best_iteration)
to_pred["proba_next_day"] = proba

# === โหลด polygon ประเทศไทย ===
url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
world = gpd.read_file(url)
th = world[world["name"]=="Thailand"].to_crs(epsg=4326)

gdf_pred = gpd.GeoDataFrame(
    to_pred,
    geometry=[Point(xy) for xy in zip(to_pred["lon_c"], to_pred["lat_c"])],
    crs="EPSG:4326"
)
gdf_pred = gdf_pred[gdf_pred.within(th.iloc[0].geometry)].copy()

# === UI Header ===
st.markdown(
    """
    <div style='border: 3px solid #FF4B4B; padding: 15px; border-radius: 10px; text-align: center;'>
        <h1 style='color: #FF4B4B;'>🔥 ระบบพยากรณ์ไฟป่าในประเทศไทย 🔥</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("")
st.sidebar.markdown(
    """
    <div style='border: 3px solid green; padding: 10px; border-radius: 8px; text-align: center;'>
        <h2 style='color:green;'>🌱 Py PHAR</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ ตั้งค่าการแสดงผล")

# ปุ่มเปิด/ปิด Forecasting
use_forecast = st.sidebar.checkbox("เปิดระบบ Forecasting", value=True)
# ปุ่มเปิด/ปิด Historical
show_hist = st.sidebar.checkbox("แสดงจุดไฟป่าที่เคยเกิดขึ้น (Historical)", value=False)

# === Folium Map ===
center = [th.geometry.iloc[0].centroid.y, th.geometry.iloc[0].centroid.x]
m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
folium.GeoJson(th.__geo_interface__, name="Thailand").add_to(m)

if use_forecast:
    radius = st.sidebar.slider("ขนาดรัศมี (radius)", 5, 30, 12)
    blur = st.sidebar.slider("Blur", 1, 30, 8)
    min_opacity = st.sidebar.slider("ความเข้มขั้นต่ำ (min_opacity)", 0.0, 1.0, 0.4, 0.05)
    threshold = st.sidebar.slider("แสดงเฉพาะความเสี่ยง ≥ ", 0.0, 1.0, 0.3, 0.05)

    heat_data = gdf_pred[gdf_pred["proba_next_day"] >= threshold][
        ["lat_c","lon_c","proba_next_day"]
    ].values.tolist()

    HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        min_opacity=min_opacity,
        max_val=1.0,
        name="Prediction HeatMap"
    ).add_to(m)

# === Historical Hotspots ===
if show_hist:
    hist_path = os.path.join(".", "viirs-jpss1_2024_Thailand.csv")
    if os.path.exists(hist_path):
        hist_df = pd.read_csv(hist_path)

        def pick_col(df, keys):
            for c in df.columns:
                if any(k in c.lower() for k in keys):
                    return c
            return None

        lat_col = pick_col(hist_df, ["lat", "latitude"])
        lon_col = pick_col(hist_df, ["lon", "longitude"])

        if lat_col and lon_col:
            hist_df = hist_df.rename(columns={lat_col: "lat", lon_col: "lon"})
            gdf_hist = gpd.GeoDataFrame(
                hist_df,
                geometry=[Point(xy) for xy in zip(hist_df["lon"], hist_df["lat"])],
                crs="EPSG:4326"
            )
            gdf_hist = gdf_hist[gdf_hist.within(th.iloc[0].geometry)].copy()
            if not gdf_hist.empty:
                marker_cluster = MarkerCluster(name="Historical Hotspots").add_to(m)
                for _, row in gdf_hist.head(int(len(gdf_hist) * 0.01)).iterrows():
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=2,
                        color="blue",
                        fill=True,
                        fill_opacity=0.6
                    ).add_to(marker_cluster)

folium.LayerControl().add_to(m)

# === Layout ครึ่งจอ ===
col1, col2 = st.columns([1,1])

with col1:
    map_data = st_folium(m, use_container_width=True, height=500)

with col2:
    st.markdown("### 🧾 เลือกชื่อดามเทียม")
    names = ["THEOS-2", "Sentinel-2", "NOAA-20"]
    selected_name = st.selectbox("เลือกชื่อ:", names, key="prosthetic_name")
    st.success(f"คุณเลือก: {selected_name}")

    model = YOLO("best.pt")

    example_map = {
        "เชียงใหม่ – ดอยอินทนนท์": "18_5880-98_4870.jpg",
        "อุบลราชธานี – ป่าดงใหญ่": "15_3000-104_8500.jpg",
        "นราธิวาส – ริมทะเล": "6_1000-101_7000.jpg"
    }

    colA, colB = st.columns([1,2])

    with colA:
        choice = st.radio("Example:", list(example_map.keys()))

    with colB:
        if choice:
            file = example_map[choice]
            if os.path.exists(file):
                results = model.predict(source=file, conf=0.25, save=False, verbose=False)

                # วาด bounding box
                result_img = results[0].plot()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                st.image(result_img, use_container_width=True)

                # === แปลงชื่อไฟล์เป็น lat/lon ===
                basename = os.path.splitext(file)[0]  # ตัด .jpg ออก
                lat_str, lon_str = basename.split("-")
                lat = float(lat_str.replace("_", "."))
                lon = float(lon_str.replace("_", "."))
# === Gemini Integration ===
api_key = "AIzaSyCNGmO0X87UdOnkk6FNkn-2mZLe0ysmW10"  # <<== ใส่คีย์ตรงนี้
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
# ✅ ตรวจว่ามี source ไหนบ้าง (Example หรือ Map)
lat, lon = None, None

if choice:  # กรณีเลือก Example
    file = example_map[choice]
    basename = os.path.splitext(file)[0]
    lat_str, lon_str = basename.split("-")
    lat = float(lat_str.replace("_", "."))
    lon = float(lon_str.replace("_", "."))
    st.info(f"📍 พิกัดจาก Example: {lat:.4f}, {lon:.4f}")

elif map_data and map_data["last_clicked"]:  # กรณีคลิกจากแผนที่
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.info(f"📍 พิกัดจาก Map: {lat:.4f}, {lon:.4f}")

# ✅ ถ้ามีพิกัดจากที่ใดที่หนึ่ง
if lat and lon:
    if st.button("🪴 วิธีการดำเนินการฟื้นฟู ⚡", use_container_width=True):
        if api_key:
            prompt = f"""
                แนะนำวิธีการฟื้นฟูพื้นที่ป่าไม้ในประเทศไทยที่พิกัด โดยในรายงานระบุจังหวัดแทน ไม่ต้องใส่ Latitude และ Longitude ซ้ำ:
                Latitude: {lat}, Longitude: {lon}.
                กรุณาอธิบายว่าควรใช้พันธุ์ไม้พื้นถิ่นชนิดใด
                และวิธีการปลูก/การจัดการที่เหมาะสม
                โดยอิงกับภูมิภาคประเทศไทย
                """
            try:
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")


