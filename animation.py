import requests
import pandas as pd
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
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
    cmap = LinearSegmentedColormap.from_list("tempmap", ["#B3C7ED", "#C2ACE7", "#E3C198"])
    return cmap(norm)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# 假设已有的函数：
# _flower_geometry(petals, L, resolution) -> (x_full, y_full)
# _temp_color(norm_val)  -> RGBA 或 hex

import numpy as np

def _hex_to_rgb01(h):
    h = h.strip().lstrip("#")
    if len(h)==3:
        h = "".join([c*2 for c in h])
    r = int(h[0:2],16)/255
    g = int(h[2:4],16)/255
    b = int(h[4:6],16)/255
    return np.array([r,g,b], dtype=float)

# 修复后的渐变函数
def _make_vertical_gradient(h, w, top_color, bottom_color, dtype=float):
    """
    返回形状 (h, w, 3) 的浮点数组，top_color 在上（y 最大或最小取决于 extent，但通常逻辑：索引0行是顶部）。
    """
    c1 = _hex_to_rgb01(top_color)  # (3,)
    c2 = _hex_to_rgb01(bottom_color)
    # 垂直插值：行 0 -> 顶部；行 h-1 -> 底部
    t = np.linspace(0, 1, h, dtype=dtype)[:, None]  # (h,1)
    col = (1 - t) * c1[None, :] + t * c2[None, :]   # (h,3)
    # 扩展到宽度
    arr = np.repeat(col[:, None, :], w, axis=1)     # (h,w,3)
    return arr.astype(dtype, copy=False)

def _make_radial_gradient(h, w, inner_color, outer_color, power=1.35, dtype=float):
    """
    返回形状 (h, w, 3) 的径向渐变，中心亮，外圈暗。
    """
    c1 = _hex_to_rgb01(inner_color)
    c2 = _hex_to_rgb01(outer_color)
    y = np.linspace(-1, 1, h, dtype=dtype)[:, None]
    x = np.linspace(-1, 1, w, dtype=dtype)[None, :]
    r = np.sqrt(x * x + y * y)
    r = np.clip(r, 0, 1) ** power
    arr = (1 - r)[..., None] * c1[None, None, :] + r[..., None] * c2[None, None, :]
    return arr.astype(dtype, copy=False)


def _apply_vignette(arr, strength=0.5, power=2.2):
    if strength <= 0:
        return arr
    h,w = arr.shape[:2]
    y = np.linspace(-1,1,h)[:,None]
    x = np.linspace(-1,1,w)[None,:]
    r = np.sqrt(x*x + y*y)
    mask = 1 - strength * (r**power)
    mask = np.clip(mask, 0, 1)
    return arr * mask[:,:,None]

def _simple_noise(h,w, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((h,w))
    return base

def _blur_mean(img, k=3):
    if k <= 1:
        return img
    if k%2==0: k += 1
    pad = k//2
    # 3D array
    img_pad = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.empty_like(img)
    area = k*k
    for i in range(img.shape[0]):
        sl_y = slice(i, i+k)
        for j in range(img.shape[1]):
            sl_x = slice(j, j+k)
            block = img_pad[sl_y, sl_x, :]
            out[i,j,:] = block.sum(axis=(0,1))/area
    return out

def _noise_colormap(noise, contrast=1.3, base_color="#223344"):
    # 提升对比
    n = noise.astype(float)
    n = n - n.min()
    if n.ptp() > 1e-9:
        n /= n.ptp()
    n = n**contrast
    # 混合入色相（用 base_color 作为深色，叠加一个浅色)
    dark = _hex_to_rgb01(base_color)
    light = np.array([0.95, 0.95, 1.0])
    arr = dark[None,None,:]*(0.55 + 0.45*n[:,:,None]) + light[None,None,:]*(0.15*n[:,:,None])
    return np.clip(arr, 0, 1)

def _compose_layered_vertical_noise(h, w,
                                    top_color, bottom_color,
                                    noise_seed=0,
                                    noise_contrast=1.3,
                                    vignette=True,
                                    vignette_strength=0.5,
                                    blur_kernel=3):
    grad = _make_vertical_gradient(h,w, top_color, bottom_color)
    noise = _simple_noise(h,w, seed=noise_seed)
    # 平滑
    if blur_kernel>1:
        noise_s = _blur_mean(noise[:,:,None].repeat(3,axis=2), k=blur_kernel)[:,:,0]
    else:
        noise_s = noise
    noise_col = _noise_colormap(noise_s, contrast=noise_contrast, base_color=top_color)
    mix = grad*0.65 + noise_col*0.35
    if vignette:
        mix = _apply_vignette(mix, strength=vignette_strength)
    return mix

def _day_night_gradient(h, w, frame, total_cycle_frames,
                        dawn_color="#203050",
                        noon_color="#89c6ff",
                        dusk_color="#d98a48",
                        night_color="#05070b"):
    # 设一个 0~1 周期参数
    if total_cycle_frames is None or total_cycle_frames <= 0:
        t = 0
    else:
        t = (frame % total_cycle_frames)/total_cycle_frames
    # 四段（夜->黎明->白天->黄昏->夜）使用分段或样条，这里线性拼
    # 区间： 0.00 夜 0.20 黎明 0.45 正午 0.70 黄昏 1.00 夜
    stops = [0.0, 0.20, 0.45, 0.70, 1.0]
    cols  = [night_color, dawn_color, noon_color, dusk_color, night_color]
    # 找到区间
    for i in range(len(stops)-1):
        if stops[i] <= t <= stops[i+1]:
            u = (t - stops[i]) / (stops[i+1]-stops[i])
            c1 = _hex_to_rgb01(cols[i])
            c2 = _hex_to_rgb01(cols[i+1])
            top = c1*(1-u)+c2*u
            bottom = top*0.55  # 底部再暗一些
            break
    arr = _make_vertical_gradient(h,w,
                                  top_color = "#{:02x}{:02x}{:02x}".format(*(top*255).astype(int)),
                                  bottom_color = "#{:02x}{:02x}{:02x}".format(*(bottom*255).astype(int)))
    arr = _apply_vignette(arr, strength=0.45)
    return arr


def animate_timeline_multi(df,
                           days=3,
                           outfile="timeline_parallel.mp4",
                           fps=24,
                           grow_frames=12,
                           start_gap=3,
                           tail_frames=24,
                           resolution=360,
                           start_mode="uniform",
                           max_active=None,
                           easing=True,
                           save=True,
                           interval_ms=40,
                           mode_color="temp",
                           # === 新增垂直交错相关参数 ===
                           vertical_stagger=True,          # 是否启用垂直交错
                           stagger_strategy="alternate",   # alternate | by_time | day_layer | sine | random
                           vertical_step=22.0,             # 交错基本步长
                           sine_amplitude=28.0,            # 当 strategy='sine'
                           sine_freq=0.05,
                           sine_phase=0.0,
                           random_seed=123,
                           center_mode="zero",             # zero | middle (针对 day_layer)
                           # 如果你已有 label 参数，保留即可…
                           show_labels=True,
                           label_fmt="{day}日{hour:02d}h  花瓣:{petals} 长:{L:.1f} 温:{temp_norm:.2f}",
                           label_phase_mode="final",
                           label_y_mode="above",
                           label_y_offset=4.0,
                           label_color="auto",
                           label_fade=True,
                           stagger_labels=False,           # 若已有旧的标签交错逻辑，可先关掉
                           add_colorbar=True,
                               background_mode="vertical_gradient",  # none|solid|vertical_gradient|radial_gradient|noise|image|day_night|custom
    bg_color="#0d121a",                   # solid 或基础底色
    bg_color_top="#1f3152",
    bg_color_bottom="#070b11",
    bg_radial_inner="#233a59",
    bg_radial_outer="#050608",
    background_image=None,                # "assets/bg.jpg"
    background_alpha=1.0,
    bg_noise_seed=123,
    bg_noise_resolution=(380, 900),       # (H,W)
    bg_noise_scale=0.004,                 # 用于频率或平滑
    bg_noise_contrast=1.35,
    bg_vignette=True,
    bg_vignette_strength=0.55,            # 0~1 之间
    bg_blur_kernel=3,                     # 简单均值模糊半径(奇数)，0=不模糊
    background_animate=False,
    day_night_cycle_frames=None,          # 若为 int，表示一个昼夜周期帧数
    custom_bg_func=None,                  # callable(xlim,ylim,params)->np.ndarray(H,W,3)
    bg_cache_static=True,                 # 静态背景缓存
    bg_zorder=-50,

                           ):
    # ---- 原有安全检查 ----
    if grow_frames < 2:
        raise ValueError("grow_frames 必须 >= 2")

    sub = df[df.day_index < days].reset_index(drop=True)
    if sub.empty:
        raise ValueError("选定天数范围内没有数据")

    # ==== 构建记录 ====
    records_raw = []
    for ridx, (_, row) in enumerate(sub.iterrows()):
        petals = int(row.petal_count)
        L = float(row.petal_len)
        cx = row.hour * 3 + row.day_index * 60
        if mode_color == "temp":
            color = _temp_color(row.temp_norm)
        else:
            color = "#888888"
        rec = {
            "idx": ridx,
            "day": int(row.day_index),
            "hour": int(row.hour),
            "petals": petals,
            "L": L,
            "cx": cx,
            "color": color,
            "temp_norm": float(getattr(row, "temp_norm", float("nan")))
        }
        records_raw.append(rec)

    # 时间排序（若需要）
    if start_mode == "by_time":
        ordered = sorted(records_raw, key=lambda r: r["cx"])
    else:
        ordered = records_raw

    # ==== 分配 start_frame ====
    active_intervals = []
    for idx, rec in enumerate(ordered):
        tentative_start = idx * start_gap
        if max_active is not None:
            start = tentative_start
            while True:
                cur_active = sum(1 for (s, e) in active_intervals if s <= start < e)
                if cur_active < max_active:
                    break
                start += 1
            rec["start_frame"] = start
        else:
            rec["start_frame"] = tentative_start
        rec["end_frame"] = rec["start_frame"] + grow_frames
        active_intervals.append((rec["start_frame"], rec["end_frame"]))

    records = ordered

    # ==== 计算 y_base（垂直交错核心） ====
    rng = np.random.default_rng(random_seed)
    if vertical_stagger:
        if stagger_strategy == "alternate":
            # 按 records 的当前顺序交替
            for i, r in enumerate(records):
                sign = 1 if (i % 2 == 0) else -1
                r["y_base"] = sign * vertical_step
        elif stagger_strategy == "by_time":
            # 按 cx 排序再交替
            tmp = sorted(records, key=lambda x: x["cx"])
            for i, r in enumerate(tmp):
                sign = 1 if (i % 2 == 0) else -1
                r["y_base"] = sign * vertical_step
        elif stagger_strategy == "day_layer":
            # 每个 day 一个层；可让 day 居中展开
            days_present = sorted(set(r["day"] for r in records))
            if center_mode == "middle":
                # 让中间 day=0 居零，其它上下扩散
                # 映射成 -k ... 0 ... +k
                mid = (len(days_present) - 1) / 2
                mapping = {d: (d - mid) for d in days_present}
            else:
                # 直接 day * 1
                mapping = {d: d for d in days_present}
            for r in records:
                r["y_base"] = mapping[r["day"]] * vertical_step
        elif stagger_strategy == "sine":
            # 用 cx 构造波形
            for r in records:
                r["y_base"] = sine_amplitude * np.sin(r["cx"] * sine_freq + sine_phase)
        elif stagger_strategy == "random":
            for r in records:
                sign = rng.choice([-1, 1])
                mag = 1 + rng.uniform(0, 1)  # 可做不同幅度
                r["y_base"] = sign * vertical_step * mag
        else:
            # 未知策略，退回 0
            for r in records:
                r["y_base"] = 0.0
    else:
        for r in records:
            r["y_base"] = 0.0

    # ==== 预计算几何 ====
    for rec in records:
        x_full, y_full = _flower_geometry(rec["petals"], rec["L"], resolution)
        rec["x_full"] = x_full
        rec["y_full"] = y_full
        rec["patch_ref"] = None
        rec["growing_ref"] = None
        rec["text_ref"] = None
        rec["growing_text_ref"] = None
        if show_labels:
            rec["label_text"] = label_fmt.format(
                day=rec["day"],
                hour=rec["hour"],
                petals=rec["petals"],
                L=rec["L"],
                temp_norm=rec["temp_norm"],
                cx=rec["cx"],
                idx=rec["idx"]
            )
        else:
            rec["label_text"] = ""

    last_end = max(r["end_frame"] for r in records)
    total_frames = last_end + tail_frames

    # ==== 画布范围 ====
    max_cx = max(r["cx"] for r in records) if records else 0
    max_L = max(r["L"] for r in records) if records else 0
    min_y_base = min(r["y_base"] for r in records) if records else 0
    max_y_base = max(r["y_base"] for r in records) if records else 0

    # 纵向范围估算：让最大花半径与偏移都包含
    y_margin = max_L * 0.6 + 8
    y_min = min_y_base - y_margin
    y_max = max_y_base + y_margin

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_xlim(-1, max_cx + max_L * 1.5)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Wind Garden - Vertical Staggered Timeline", fontsize=14)

    # 颜色条（可选）
    if add_colorbar and mode_color == "temp" and len(records) > 0:
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib as mpl
        cmap = LinearSegmentedColormap.from_list("tempmap", ["#B3C7ED", "#C2ACE7", "#E3C198"])
        valid = [r["temp_norm"] for r in records if not np.isnan(r["temp_norm"])]
        if valid:
            norm = mpl.colors.Normalize(vmin=min(valid), vmax=max(valid))
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.035)
            cbar.ax.set_ylabel("Temp Norm", rotation=270, labelpad=10, fontsize=9)

    # ==== 标签位置函数（注意加 y_base） ====
    def _compute_label_position(rec, phase_draw):
        cx = rec["cx"]
        if label_y_mode == "center":
            base_y = rec["y_base"]
        else:
            base_y = rec["y_base"] + rec["L"] * 0.55 + label_y_offset
        return cx, base_y
    
        # ========== 背景准备 ==========
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 决定背景栅格分辨率 (越大越细腻；注意性能)
    bg_h, bg_w = bg_noise_resolution if background_mode != "image" else (480, 960)

    bg_img_static = None
    bg_im_artist = None

    def _build_static_bg():
        if background_mode in ("none", None):
            return None
        if background_mode == "solid":
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            return None
        if background_mode == "vertical_gradient":
            arr = _make_vertical_gradient(bg_h, bg_w, bg_color_top, bg_color_bottom)
        elif background_mode == "radial_gradient":
            arr = _make_radial_gradient(bg_h, bg_w, bg_radial_inner, bg_radial_outer)
            if bg_vignette:
                arr = _apply_vignette(arr, strength=bg_vignette_strength)
        elif background_mode == "noise":
            arr = _compose_layered_vertical_noise(
                bg_h, bg_w,
                top_color=bg_color_top,
                bottom_color=bg_color_bottom,
                noise_seed=bg_noise_seed,
                noise_contrast=bg_noise_contrast,
                vignette=bg_vignette,
                vignette_strength=bg_vignette_strength,
                blur_kernel=bg_blur_kernel
            )
        elif background_mode == "image":
            if background_image and os.path.isfile(background_image):
                arr = plt.imread(background_image)
                # 归一化
                if arr.dtype != np.float32 and arr.dtype != np.float64:
                    arr = arr.astype(float)/255.0
            else:
                raise FileNotFoundError(f"背景图不存在: {background_image}")
        elif background_mode == "day_night":
            # 初次 0 帧
            arr = _day_night_gradient(bg_h, bg_w, 0,
                                      total_cycle_frames=day_night_cycle_frames)
        elif background_mode == "custom":
            if callable(custom_bg_func):
                arr = custom_bg_func(xlim, ylim, dict(width=bg_w,height=bg_h))
            else:
                raise ValueError("custom_bg_func 未提供或不可调用")
        else:
            return None
        return np.clip(arr,0,1)

    if not background_animate:
        bg_img_static = _build_static_bg()
        if bg_img_static is not None:
            bg_im_artist = ax.imshow(bg_img_static,
                                     extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                                     origin="lower",
                                     zorder=bg_zorder,
                                     alpha=background_alpha,
                                     interpolation="bilinear")
    else:
        # 动态背景：比如 day_night 模式
        if background_mode == "day_night":
            # 初始化一张假图，后面 update 替换
            init_arr = _day_night_gradient(bg_h, bg_w, 0, day_night_cycle_frames)
            bg_im_artist = ax.imshow(init_arr,
                                     extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                                     origin="lower",
                                     zorder=bg_zorder,
                                     alpha=background_alpha,
                                     interpolation="bilinear")
        else:
            # 其他模式动态: 可以后续扩展
            pass

    # ==== 动画更新 ====
    def update(frame):
        artists = []
        for rec in records:
            start_f = rec["start_frame"]
            end_f = rec["end_frame"]

            # 完成
            if frame >= end_f:
                if rec["growing_ref"] is not None:
                    try: rec["growing_ref"].remove()
                    except Exception: pass
                    rec["growing_ref"] = None
                if rec["patch_ref"] is None:
                    poly = ax.fill(rec["x_full"] + rec["cx"],
                                   rec["y_full"] + rec["y_base"],
                                   color=rec["color"],
                                   edgecolor="none",
                                   alpha=0.85)[0]
                    rec["patch_ref"] = poly
                artists.append(rec["patch_ref"])

                if show_labels and label_phase_mode == "final":
                    if rec["growing_text_ref"] is not None:
                        try: rec["growing_text_ref"].remove()
                        except Exception: pass
                        rec["growing_text_ref"] = None
                    if rec["text_ref"] is None:
                        x_label, y_label = _compute_label_position(rec, 1.0)
                        txt_color = rec["color"] if label_color == "auto" else label_color
                        rec["text_ref"] = ax.text(
                            x_label, y_label, rec["label_text"],
                            ha="center", va="center",
                            fontsize=8, color=txt_color,
                            alpha=0.95 if label_fade else 1.0,
                            fontweight="medium"
                        )
                    artists.append(rec["text_ref"])
                continue

            # 未开始
            if frame < start_f:
                continue

            # 生长中
            phase = (frame - start_f) / (grow_frames - 1)
            phase = max(0.0, min(1.0, phase))
            phase_draw = phase ** 0.8 if easing else phase

            x = rec["x_full"] * phase_draw + rec["cx"]
            y = rec["y_full"] * phase_draw + rec["y_base"]

            if rec["growing_ref"] is not None:
                try: rec["growing_ref"].remove()
                except Exception: pass
                rec["growing_ref"] = None

            poly = ax.fill(x, y,
                           color=rec["color"],
                           edgecolor="none",
                           alpha=0.85)[0]
            rec["growing_ref"] = poly
            artists.append(poly)

            if show_labels and label_phase_mode == "grow":
                if rec["growing_text_ref"] is not None:
                    try: rec["growing_text_ref"].remove()
                    except Exception: pass
                    rec["growing_text_ref"] = None
                x_label, y_label = _compute_label_position(rec, phase_draw)
                txt_color = rec["color"] if label_color == "auto" else label_color
                rec["growing_text_ref"] = ax.text(
                    x_label, y_label, rec["label_text"],
                    ha="center", va="center",
                    fontsize=8,
                    color=txt_color,
                    alpha=(phase_draw if label_fade else 1.0),
                    fontweight="medium"
                )
                artists.append(rec["growing_text_ref"])
        return artists

    from matplotlib.animation import FuncAnimation, FFMpegWriter
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
            print(f"[animation] Saved ->", outfile)
        except Exception as e:
            print("[animation] Save failed:", e)

    return anim




# -------------------- 使用示例（末尾调用） --------------------
if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(7)
    rows = []
    for d in range(6):
        for h in range(0, 24, 3):
            rows.append({
                "day_index": d,
                "hour": h,
                "petal_count": rng.integers(6, 11),
                "petal_len": rng.uniform(12, 32),
                "temp_norm": float(rng.uniform())
            })
    df_demo = pd.DataFrame(rows)

    anim = animate_timeline_multi(
        df_demo,
        days=6,
        start_mode="by_time",
        vertical_stagger=True,
        stagger_strategy="sine",   # 改成 "sine" 看波浪; "day_layer" 看分天
        vertical_step=24,
        show_labels=True, 
        stagger_labels=True,
        background_mode="day_night" ,  # "vertical_gradient" | "day_night"
        bg_color_top="#2b5186",
        bg_color_bottom="#0a2c68",
        bg_vignette=False,              # 先看布局，标签可以后开
        save=False
    )
    plt.show()
