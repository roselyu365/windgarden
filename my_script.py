
# Required packages: python-dotenv, requests, lxml, matplotlib, drawsvg
# Please install them using pip in your terminal before running this script.
import requests
import pandas as pd
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
print("All libraries imported successfully!")
from data import df
#核心：用极坐标生成花瓣多边形再平移到 (x,y) 位置。
def color_from_temp(norm: float) -> str:
    norm = max(0.0, min(1.0, float(norm)))
    r = int(46 + norm * 200)
    g = int(111 - norm * 30)
    b = int(239 - norm * 200)
    return f"rgba({r},{g},{b},0.85)"

# （可选）给 matplotlib 用的 RGBA tuple
def color_tuple_from_temp(norm: float):
    rgba = color_from_temp(norm)
    nums = rgba.replace("rgba(", "").replace(")", "").split(",")
    r, g, b, a = [float(x) for x in nums]
    return (r/255, g/255, b/255, a)

# 抽象花朵几何
def flower_polar_points(petal_count: int, petal_len: float, theta_steps: int = 300):
    theta = np.linspace(0, 2*np.pi, theta_steps)
    r = petal_len * (0.6 + 0.4 * np.sin(petal_count * theta))
    return theta, r

# 在 Matplotlib 轴上画一朵花
def draw_flower(ax, center, petal_count, petal_len, color, theta_steps=300):
    cx, cy = center
    theta, r = flower_polar_points(petal_count, petal_len, theta_steps)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    ax.fill(x, y, color=color, linewidth=0.3, edgecolor=color, alpha=0.9)

def plot_static_timeline(df, max_days=3, hour_spacing=1.0, day_gap=1.0, theta_steps=300):
    """
    线性时间轴风之花园

    参数:
        df: DataFrame (需要列 day_index, hour, petal_count, petal_len, temp_norm)
        max_days: 取前多少天
        hour_spacing: 小时之间的水平间距
        day_gap: 天与天之间在“24 小时”基础上的额外间隔
        theta_steps: 花瓣分辨率
    """
    subset = df[df["day_index"] < max_days].copy()
    if subset.empty:
        raise ValueError("没有匹配到任何数据")
    fig_timeline, ax = plt.subplots(figsize=(14, 4))

    daily_offset = 24 * hour_spacing + day_gap  # 用来替换“25”

    for _, row in subset.iterrows():
        x_center = row["hour"] * hour_spacing + row["day_index"] * daily_offset
        c = color_tuple_from_temp(row["temp_norm"])
        draw_flower(ax, (x_center, 0), int(row["petal_count"]), float(row["petal_len"]), c, theta_steps)

    ax.set_xlim(-1, subset["hour"].max() * hour_spacing + max_days * daily_offset)
    ax.set_ylim(-60, 60)
    ax.axis("off")
    ax.set_title("Wind Garden - Linear Timeline (Static)")
    return fig_timeline

def plot_ring_circle(df, target_day=0, ring_radius=150, theta_steps=300,
                     title="Wind Garden - Daily Circular Layout"):
    day_df = df[df["day_index"] == target_day].copy()
    if day_df.empty:
        raise ValueError(f"没有找到 target_day={target_day} 的数据")

    fig_ring = go.Figure()
    theta_base = np.linspace(0, 2*np.pi, theta_steps, endpoint=True)

    for _, row in day_df.iterrows():
        n = int(row["petal_count"])
        L = float(row["petal_len"])

        r = L * (0.6 + 0.4 * np.sin(n * theta_base))
        phi = 2 * np.pi * (row["hour"] / 24)

        x = (ring_radius * np.cos(phi)) + r * np.cos(theta_base)
        y = (ring_radius * np.sin(phi)) + r * np.sin(theta_base)

        color = color_from_temp(row["temp_norm"])

        fig_ring.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            fill="toself",
            line=dict(width=0.5, color=color),
            hoverinfo="text",
            text=(
                f"Day {row['day_index']} Hour {row['hour']}<br>"
                f"Temp {row['temp']:.1f}°C<br>"
                f"Wind {row['wind_speed']:.1f} m/s<br>"
                f"Dir {row['wind_dir']}°"
            )
        ))

    fig_ring.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        width=600,
        height=600,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig_ring
def add_reference_layers(fig, R=150):
    # 中心圆
    circle = go.Scatter(
        x=[R*np.cos(t) for t in np.linspace(0,2*np.pi,200)],
        y=[R*np.sin(t) for t in np.linspace(0,2*np.pi,200)],
        mode="lines",
        line=dict(color="rgba(180,180,180,0.4)", width=1),
        hoverinfo="skip",
        showlegend=False
    )
    fig.add_trace(circle)

    # 小时刻度
    for h in range(0,24,3):
        phi = 2*np.pi*h/24
        fig.add_trace(go.Scatter(
            x=[(R+5)*np.cos(phi)],
            y=[(R+5)*np.sin(phi)],
            mode="text",
            text=[str(h)],
            textfont=dict(size=10, color="#555"),
            hoverinfo="skip",
            showlegend=False
        ))
# 线性
fig_t = plot_static_timeline(df, max_days=3)
plt.show()

# 圆环
fig_r = plot_ring_circle(df, target_day=0)
add_reference_layers(fig_r)
fig_r.show()
