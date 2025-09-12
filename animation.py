import requests
import pandas as pd
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
print("All libraries imported successfully!")
from data import df

# animation.py
_GEOM_CACHE = {}  # (petals, L, res) -> (x,y)

def _flower_geometry(petals: int, L: float, resolution: int = 360):
    key = (petals, round(L,2), resolution)
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    theta = np.linspace(0, 2*np.pi, resolution, endpoint=True)
    r = L * (0.6 + 0.4 * np.sin(petals * theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    _GEOM_CACHE[key] = (x, y)
    return x, y

def prepare_geometry_cache(petals_list, lengths, resolution=360):
    for p in petals_list:
        for L in lengths:
            _flower_geometry(p, float(L), resolution=resolution)
    print(f"[animation] Cached {len(_GEOM_CACHE)} geometries.")

def _temp_color(norm):
    # 与 my_script 保持一致（如果要统一可抽象出去）
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tempmap", ["#2E6FEF", "#8A5AD8", "#F9A43A"])
    return cmap(norm)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# 假设已有的函数：
# _flower_geometry(petals, L, resolution) -> (x_full, y_full)
# _temp_color(norm_val)  -> RGBA 或 hex

def animate_timeline_multi(df,
                           days=3,
                           outfile="timeline_parallel.mp4",
                           fps=24,
                           grow_frames=12,
                           start_gap=3,
                           tail_frames=24,
                           resolution=360,
                           start_mode="uniform",   # "uniform" | "by_time"
                           max_active=None,        # 限制同时处于“生长中”的花朵数量
                           easing=True,            # 是否使用缓动
                           save=True,
                           interval_ms=40,
                           mode_color="temp"       # 预留：未来可扩展不同配色模式
                           ):
    """
    多朵花同时生长的线性时间轴动画。

    参数说明：
    - days: 取 df 中 day_index < days 的子集。
    - grow_frames: 单朵花从 0 → 完整长度的帧数。
    - start_gap: 相邻两朵花启动生长的帧间隔。小 => 并行多。
                 并行度近似 grow_frames / start_gap。
    - tail_frames: 全部完成后附加的静止帧（观感缓冲）。
    - start_mode:
        * "uniform": 按记录原顺序 index * start_gap。
        * "by_time": 按 cx（时间轴位置）排序再分配。
    - max_active: 若给定（整数），限制同一时刻正在生长的花朵数量。
                  若超出则顺延开始时间。
    - easing: 使用简单缓动（phase^0.8）让生长前期更快接近体积。
    - mode_color: 颜色模式占位（目前仅 "temp"）。
    """
    if grow_frames < 2:
        raise ValueError("grow_frames 必须 >= 2（否则 phase 计算会出现除零或无法形成动画）")

    sub = df[df.day_index < days].reset_index(drop=True)
    if sub.empty:
        raise ValueError("选定天数范围内没有数据")

    # 预处理记录：cx 为横向位置（与原 sequential 逻辑一致）
    records_raw = []
    for _, row in sub.iterrows():
        petals = int(row.petal_count)
        L = float(row.petal_len)
        cx = row.hour*3 + row.day_index * 60
        if mode_color == "temp":
            color = _temp_color(row.temp_norm)
        else:
            color = "#888888"
        records_raw.append({
            "petals": petals,
            "L": L,
            "cx": cx,
            "color": color,
            "temp_norm": getattr(row, "temp_norm", np.nan)
        })

    # 根据 start_mode 排序或保持顺序
    if start_mode == "by_time":
        # 按时间位置排序后再赋启动顺序
        ordered = sorted(records_raw, key=lambda r: r["cx"])
    else:
        ordered = records_raw

    # 分配 start_frame
    # 基础：start_frame = idx * start_gap
    # 如果 max_active 限制，则确保同时 active 数 <= max_active
    active_intervals = []  # 存 (start, end) 追踪进行中
    for idx, rec in enumerate(ordered):
        tentative_start = idx * start_gap
        if max_active is not None:
            start = tentative_start
            while True:
                # 计算该 start 时刻已有多少 active
                current_active = 0
                for (s, e) in active_intervals:
                    if s <= start < e:  # start 处于别的花生长区间
                        current_active += 1
                if current_active < max_active:
                    break
                start += 1  # 往后推迟一帧继续测试
            rec["start_frame"] = start
        else:
            rec["start_frame"] = tentative_start
        rec["end_frame"] = rec["start_frame"] + grow_frames
        active_intervals.append((rec["start_frame"], rec["end_frame"]))

    # 还原到原始顺序显示（不是必须，但保持逻辑一致）
    # 我们真正绘制时并不依赖顺序，只遍历所有
    records = ordered

    last_end = max(r["end_frame"] for r in records)
    total_frames = last_end + tail_frames

    # 预计算几何，避免帧中重复
    for rec in records:
        x_full, y_full = _flower_geometry(rec["petals"], rec["L"], resolution)
        rec["x_full"] = x_full
        rec["y_full"] = y_full
        rec["patch_ref"] = None     # 存放静态完成后的 patch
        rec["growing_ref"] = None   # 当前帧的临时生长 patch

    # 画布
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    # X 范围：略加余量
    max_cx = max(r["cx"] for r in records) if records else 0
    max_L = max(r["L"] for r in records) if records else 0
    ax.set_xlim(-1, max_cx + max_L * 1.5)
    ax.set_ylim(-60, 60)
    ax.set_title("Wind Garden - Parallel Growth Timeline")

    # 动画更新函数
    def update(frame):
        artists = []

        for rec in records:
            start_f = rec["start_frame"]
            end_f = rec["end_frame"]

            # 已完成且已经固定
            if frame >= end_f:
                if rec["growing_ref"] is not None:
                    # 清理残留临时对象
                    try:
                        rec["growing_ref"].remove()
                    except Exception:
                        pass
                    rec["growing_ref"] = None
                if rec["patch_ref"] is None:
                    # 第一次进入完成状态：创建最终固定形状
                    poly = ax.fill(rec["x_full"] + rec["cx"],
                                   rec["y_full"],
                                   color=rec["color"],
                                   edgecolor="none",
                                   alpha=0.85)[0]
                    rec["patch_ref"] = poly
                if rec["patch_ref"] is not None:
                    artists.append(rec["patch_ref"])
                continue

            # 尚未开始
            if frame < start_f:
                continue

            # 生长中的区间
            phase = (frame - start_f) / (grow_frames - 1)
            phase = min(max(phase, 0.0), 1.0)
            if easing:
                phase_draw = phase ** 0.8  # 简单缓动：前快后慢
            else:
                phase_draw = phase

            x = rec["x_full"] * phase_draw + rec["cx"]
            y = rec["y_full"] * phase_draw

            # 移除上一帧的 growing patch
            if rec["growing_ref"] is not None:
                try:
                    rec["growing_ref"].remove()
                except Exception:
                    pass
                rec["growing_ref"] = None

            poly = ax.fill(x, y, color=rec["color"], edgecolor="none", alpha=0.85)[0]
            rec["growing_ref"] = poly
            artists.append(poly)

        return artists

    anim = FuncAnimation(fig,
                         update,
                         frames=total_frames,
                         interval=interval_ms,
                         blit=False,
                         repeat=False)

    if save:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=3000)
            anim.save(outfile, writer=writer)
            print(f"[animation] Saved animation -> {outfile}")
        except Exception as e:
            print("[animation] Failed to save MP4 (maybe ffmpeg missing):", e)

    return anim


# -------------------- 使用示例（末尾调用） --------------------
if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(42)
    demo_rows = []
    for d in range(2):
        for h in range(0, 24, 3):
            demo_rows.append({
                "day_index": d,
                "hour": h,
                "petal_count": rng.integers(6, 11),
                "petal_len": rng.uniform(10, 35),
                "temp_norm": float(rng.uniform())
            })
    df_demo = pd.DataFrame(demo_rows)

    # 生成并行生长动画
    anim = animate_timeline_multi(
        df_demo,
        days=2,
        outfile="timeline_parallel.mp4",
        grow_frames=11,
        start_gap=10,          # 越小越密集同时生长
        tail_frames=40,
        start_mode="by_time", # 按时间位置先后
        max_active=3,         # 限制最多 3 朵同时生长（可去掉）
        resolution=360,
        easing=True,
        save=False            # 先预览，不保存
    )

    plt.show()
