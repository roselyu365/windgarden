
# Required packages: python-dotenv, requests, lxml, matplotlib, drawsvg
# Please install them using pip in your terminal before running this script.
import dotenv
import requests
from lxml import html
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
import drawsvg
from scraping_util2s import get_url, parse
print("All libraries imported successfully!")
#get the dataframe of weather data

# 配置
lat, lon = 31.23, 121.47
start = date(2025, 9, 11)
end = start + timedelta(days=6)  # 共 7 天
url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": "temperature_2m,windspeed_10m,winddirection_10m",
    "start_date": start.isoformat(),
    "end_date": end.isoformat(),
    "timezone": "Asia/Shanghai"
}

r = requests.get(url, params=params)
data = r.json()["hourly"]

df = pd.DataFrame({
    "time": pd.to_datetime(data["time"]),
    "temp": data["temperature_2m"],
    "wind_speed": data["windspeed_10m"],
    "wind_dir": data["winddirection_10m"]
})

# 添加辅助列
# ...existing code...
df["day_index"] = (df["time"].dt.date - df["time"].dt.date.min()).apply(lambda x: x.days)
# ...existing code...
df["hour"] = df["time"].dt.hour

def petals_from_dir(d):
    if d < 60: return 6
    if d < 120: return 7
    if d < 180: return 8
    if d < 240: return 9
    if d < 300: return 10
    return 12

df["petal_count"] = df["wind_dir"].apply(petals_from_dir)

# 花瓣长度归一化（基于 0-15 m/s）
df["petal_len"] = 10 + (df["wind_speed"].clip(0, 15) / 15) * 40  # 像素或单位

# 温度颜色归一 (0-35℃)
Tmin, Tmax = 0, 35
df["temp_norm"] = ((df["temp"] - Tmin) / (Tmax - Tmin)).clip(0, 1)

print(df.head(24))
#成功获取数据
#核心：用极坐标生成花瓣多边形再平移到 (x,y) 位置。
import matplotlib.pyplot as plt
import numpy as np

def temp_to_color(norm):
    # 简单三段线性 (blue -> purple -> orange)
    # 你也可以用 matplotlib Colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tempmap", ["#2E6FEF", "#8A5AD8", "#F9A43A"])
    return cmap(norm)

def draw_flower(ax, center, petal_count, petal_len, color):
    n = petal_count
    theta = np.linspace(0, 2*np.pi, 400)
    r = petal_len * (0.6 + 0.4 * np.sin(n * theta))
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    ax.fill(x, y, color=color, alpha=0.85, edgecolor="none")

# 取前 3 天 * 24h
subset = df[df.day_index < 3]

fig, ax = plt.subplots(figsize=(14, 4))
for _, row in subset.iterrows():
    x = row["hour"] + row["day_index"] * 25  # 每天之间留一点空
    y = 0
    c = temp_to_color(row["temp_norm"])
    draw_flower(ax, (x, y), int(row["petal_count"]), row["petal_len"], c)

ax.set_xlim(-1, subset["hour"].max() + 75)
ax.set_ylim(-60, 60)
ax.axis("off")
plt.title("Wind Garden - Linear Timeline Prototype")
plt.show()
#画日内圆环
import plotly.graph_objects as go
import numpy as np

day0 = df[df.day_index==0]

fig = go.Figure()

for _, row in day0.iterrows():
    n = int(row.petal_count)
    L = row.petal_len
    # 极坐标花瓣
    theta = np.linspace(0, 2*np.pi, 300)
    r = L * (0.6 + 0.4 * np.sin(n * theta))
    # 该朵在圆环上的主角度
    phi = 2 * np.pi * (row.hour / 24)
    # 把整朵花整体旋转 + 平移到半径 R
    R = 150
    x = (R * np.cos(phi)) + r * np.cos(theta)
    y = (R * np.sin(phi)) + r * np.sin(theta)

    color = f"rgba({int(46 + (row.temp_norm)*200)}, {int(111 - row.temp_norm*30)}, {int(239 - row.temp_norm*200)},0.85)"

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        fill="toself",
        line=dict(width=0.5, color=color),
        hoverinfo="text",
        text=f"Hour {row.hour}<br>Temp {row.temp}°C<br>Wind {row.wind_speed} m/s<br>Dir {row.wind_dir}°"
    ))

fig.update_layout(
    title="Wind Garden - Daily Circular Layout",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    width=600,
    height=600
)
fig.show()
