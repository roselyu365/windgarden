import requests
import pandas as pd
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
print("All libraries imported successfully!")
#get the dataframe of weather data
def fetch_weather(lat=31.23, lon=121.47, days=7):
    start = date(2025, 9, 11)
    end = start + timedelta(days=days - 1)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "Asia/Shanghai"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(data["time"]),
        "temp": data["temperature_2m"],
        "wind_speed": data["windspeed_10m"],
        "wind_dir": data["winddirection_10m"]
    })
    return df
df = fetch_weather()
print("Weather data fetched successfully!")
# 特征增强
def petals_from_dir(d):
    if d < 60: return 6
    if d < 120: return 7
    if d < 180: return 8
    if d < 240: return 9
    if d < 300: return 10
    return 12

def enrich_features(df):
    df = df.copy()
    df["day_index"] = (df["time"].dt.date - df["time"].dt.date.min()).apply(lambda x: x.days)
    df["hour"] = df["time"].dt.hour
    df["petal_count"] = df["wind_dir"].apply(petals_from_dir)
    df["petal_len"] = 10 + (df["wind_speed"].clip(0, 15) / 15) * 40
    df["temp_norm"] = ((df["temp"] - 0) / (35 - 0)).clip(0, 1)
    return df
df = enrich_features(df)
print(df.head(24))
#成功获取数据