import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import folium
from streamlit_folium import st_folium

# ตั้งค่า Page และแสดงแผนที่เต็มหน้าจอ
st.set_page_config(page_title="Map View - Surin & Rattanaburi", layout="wide")
st.title("Agri-Burn Advisor: แผนที่พื้นที่สุรินทร์ และ รัตนบุรี แบบเต็มหน้าจอ")

# 1. สร้าง Dummy Polygons สำหรับ Surin และ รัตนบุรี
polygons = [
    Polygon([(103.48, 14.88), (103.50, 14.88), (103.50, 14.90), (103.48, 14.90)]),  # Surin
    Polygon([(103.82, 14.75), (103.84, 14.75), (103.84, 14.77), (103.82, 14.77)])   # Rattanaburi
]
regions = ['Surin', 'Rattanaburi']
gdf = gpd.GeoDataFrame(
    {'region': regions},
    geometry=gpd.GeoSeries(polygons),
    crs="EPSG:4326"
)

# 2. สร้าง Dummy Recommended Points หลายจุดต่อแต่ละพื้นที่
# กำหนด offsets
offsets = [(0.005, 0.005), (-0.005, 0.005), (0.005, -0.005), (-0.005, -0.005), (0, 0)]
points = []
for region, poly in zip(regions, polygons):
    cent = poly.centroid
    for dx, dy in offsets:
        x, y = cent.x + dx, cent.y + dy
        points.append({'region': region, 'geometry': Point(x, y)})

# สร้าง GeoDataFrame ของจุด
gdf_points = gpd.GeoDataFrame(points, crs="EPSG:4326")

# 3. สร้าง Dummy Weather & AQI Data
weather = {'date': '2025-07-25', 'rainfall_mm': 0.1, 'wind_speed_m_s': 3.2, 'humidity_pct': 60}
aqi = {'pm25': 40}
st.markdown(
    f"**สภาพอากาศ ณ {weather['date']}:** ☔ {weather['rainfall_mm']} mm, 💨 {weather['wind_speed_m_s']} m/s, 💧 {weather['humidity_pct']}%  \
     **AQI (PM2.5):** 🌫️ {aqi['pm25']} µg/m³"
)

# 4. สร้างแผนที่ Folium
center_lat = (14.88 + 14.75) / 2
center_lon = (103.49 + 103.83) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB positron')

# 5. แสดงพื้นที่ด้วยสีเขียวและ tooltip ชื่อภูมิภาค
for _, row in gdf.iterrows():
    folium.GeoJson(
        row.geometry,
        style_function=lambda f: {'fillColor': 'green', 'color': 'darkgreen', 'weight': 2, 'fillOpacity': 0.4},
        tooltip=row['region']
    ).add_to(m)

# 6. แสดงจุดแนะนำการเผา หลายจุด พร้อมไอคอน
for _, row in gdf_points.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        icon=folium.DivIcon(html="<div style='font-size:20px;color:red;'>🔥</div>"),
        tooltip=f"Burn: {row['region']}"
    ).add_to(m)

# 7. เพิ่ม Marker เมืองสำคัญ
folium.Marker(location=[14.88, 103.49], popup='เมืองสุรินทร์', icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
folium.Marker(location=[14.75, 103.83], popup='อำเภอรัตนบุรี', icon=folium.Icon(color='purple', icon='info-sign')).add_to(m)

# 8. แสดง wind speed เป็นวงกลมที่ตำแหน่งกลางระหว่างสองเขต
folium.Circle(
    location=[center_lat, center_lon],
    radius=weather['wind_speed_m_s'] * 100,
    color='blue', fill=True, fill_opacity=0.2,
    tooltip=f"Wind: {weather['wind_speed_m_s']} m/s"
).add_to(m)

# 9. แสดงแผนที่เต็มหน้าจอ
st.subheader("Map: Surin & Rattanaburi with Multiple Burn Points")
st_folium(m, width="100%", height=800)