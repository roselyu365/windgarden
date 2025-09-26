# Windgarden
An animated timeline visualization built with Matplotlib that uses “flower” geometries to represent 7-day hourly weather data (temperature, wind speed, wind direction) for a given latitude/longitude from the Open-Meteo API. Each hour is represented as a stylized “flower” whose:
- Petal count encodes wind direction
- Petal length encodes wind speed
- Color encodes temperature


## Repository Structure (Table Form)

| File / Dir | Description |
|-----------|-------------|
| `data.py` | data scarching and a structured dataframe output |
| `my_script.py` | static visulization of weather data|
| `animation.py` | animation visulization of weather data|
| `scraping_util*.py` | Optional scraping / preprocessing utilities |
| `requirements.txt` | Python dependencies |
| `Figure_1-line...png` | Sample line-based output |
| `newplot.png` | Sample circular output |
| `README.md` | Project documentation |
| `.gitignore` | Git ignore rules |
| `.DS_Store` | macOS system file (ignored) |



##  Code Logic (Wind Garden Visualization)

### Overview
This script fetches 7-day hourly weather data (temperature, wind speed, wind direction) for a given latitude/longitude from the Open-Meteo API, transforms it into a structured DataFrame, and renders two experimental visualizations:
1. Linear timeline flower field ("Wind Garden - Linear Timeline Prototype")
2. Circular daily layout ("Wind Garden - Daily Circular Layout")

Each hour is represented as a stylized “flower” whose:
- Petal count encodes wind direction
- Petal length encodes wind speed
- Color encodes temperature

---

### 1. Data Acquisition
- API: https://api.open-meteo.com/v1/forecast
- Parameters: latitude, longitude, start_date, end_date, timezone, hourly variables
- Returned JSON: hourly arrays for time, temperature_2m, windspeed_10m, winddirection_10m
- Parsed into a pandas DataFrame with columns:
  - time (converted to datetime)
  - temp
  - wind_speed
  - wind_dir
  - day_index (0-based day offset)
  - hour (0–23)
  
Example (first few rows):
```
time                temp  wind_speed  wind_dir  day_index  hour
2025-09-11 00:00    24.3       2.1       140         0       0
...
```

---

### 2. Feature Engineering

#### 2.1 Petal Count Mapping (Wind Direction → Discrete Categories)
A simple stepped mapping:
- 0°–60° → 6 petals
- 60°–120° → 7 petals
- 120°–180° → 8 petals
- 180°–240° → 9 petals
- 240°–300° → 10 petals
- 300°–360° → 12 petals

Function:
```
if d < 60: 6
elif d < 120: 7
...
else: 12
```

(You could later replace this with dynamic binning or a continuous function.)

#### 2.2 Petal Length Normalization (Wind Speed)
Wind speed is clipped to [0, 15] m/s and mapped to a length interval [10, 50]:
\[ petal\_len = 10 + \frac{\text{clip}(wind\_speed, 0, 15)}{15} \times 40 \]

#### 2.3 Temperature Normalization (Color)
Assumed meaningful temperature range: 0–35℃.
\[ temp\_norm = \text{clip}\left(\frac{temp - 0}{35 - 0}, 0, 1\right) \]
This normalized value feeds into a custom 3-stop colormap (blue → purple → orange).


### 3. Flower Geometry

Each “flower” uses a sinusoidally modulated polar curve:
\[ r(\theta) = L \times (0.6 + 0.4 \times \sin(n \theta)) \]
Where:
- \[ L \] = petal length (scaled by wind speed)
- \[ n \] = petal_count (mapped from wind direction)
- \[ \theta \] ∈ \[0, 2\pi\]

Converted to Cartesian and optionally translated:
\[ x = x_c + r(\theta)\cos(\theta) \]
\[ y = y_c + r(\theta)\sin(\theta) \]

This produces soft petal lobes with radial symmetry. Higher \[ n \] creates more oscillations → more petals.

---

### 4. Visualization A: Linear Timeline ("Wind Garden - Linear Timeline Prototype")

Logic:
- Filter first 3 days: \[ day\_index < 3 \]
- Horizontal positioning:
  \[ x = hour + day\_index \times 25 \]
  (25 instead of 24 introduces a visual gap between days)
- Vertical center fixed at \[ y = 0 \]
- Flowers are drawn with Matplotlib’s `fill` (semi-transparent, no stroke)

Interpretation Example:
- A cluster of large, warm-colored (orange) flowers around hours 13–16 on Day 1 ⇒ Warm midday with stronger winds.
- A sequence of small, bluish, low-petal flowers late at night ⇒ Cool, low wind periods.

---

### 5. Visualization B: Circular Daily Layout

Transforms hourly sequence into a 24-hour ring:
- For each hour h:
  \[ \phi = 2\pi \times \frac{h}{24} \] (hour angle)
- Base ring radius: \[ R = 150 \]
- Entire flower is offset to center at \[ (R\cos\phi, R\sin\phi) \]
- Each flower’s internal shape still uses:
  \[ r(\theta) = L(0.6 + 0.4\sin(n\theta)) \]
- Implemented with Plotly `Scatter` traces (filled closed curves), enabling hover tooltips:
  Hour, Temp, Wind Speed, Wind Direction.

Interpretation Example:
- Opposite points correspond to 12-hour separations (e.g., 00:00 vs 12:00).
- Denser warm shapes on the right hemisphere might indicate warmer afternoon conditions.

### 6. Animation algorithm 
The animation transforms a multivariate, time-indexed dataset into a temporal “garden” of growing flowers. Each record (e.g., a day-hour tuple with attributes) becomes a flower whose geometry, position, color, and label encode different dimensions. The animation unfolds over frames: flowers appear, grow with easing, and then remain, collectively forming a horizontally ordered timeline with vertical staggering.

#### Input Data Semantics

Time axes:
day_index: coarse time (day-level)
hour: fine time (hour-level)
Shape parameters:
petal_count: number of lobes in the flower
petal_len: radial scale controlling flower size
Color parameter:
temp_norm in [0, 1]: normalized variable mapped to color (temperature in the demo)

#### Core Mappings

Time → Horizontal position: For each record, the x-center cx is computed as a weighted combination of hour and day_index (cx = hour * 3 + day_index * 60). This creates a left-to-right timeline where later times are further right.
Day/sequence → Vertical offset (stagger): To avoid overlap and add rhythm, each record receives a y_base offset determined by a chosen strategy (alternate, by_time, day_layer, sine, random). This yields layered or wave-like bands across the timeline.
Attributes → Flower geometry: The flower is defined in polar-like form using a sinusoidal radius:
r(θ) = L · [0.6 + 0.4 · sin(petals · θ)]
x(θ) = r(θ) cos θ, y(θ) = r(θ) sin θ This produces a multi-lobed “flower” whose petal_count controls lobe number and L controls overall radius. Geometry for identical (petals, L, resolution) tuples is cached to avoid recomputation.
Variable → Color: A custom colormap maps temp_norm to a perceptual gradient (cool → warm), reinforcing the sense of environmental variation.

#### Temporal Orchestration

Start scheduling: Records are assigned start_frame values. Two modes exist:
uniform: start frames increase uniformly with a fixed gap.
by_time: records are ordered by cx before scheduling, so earlier times grow earlier. Optionally, a max_active cap ensures only a limited number of flowers are “growing” simultaneously.
Growth phase: Each flower evolves over grow_frames. A normalized phase ∈ [0, 1] is computed per frame and optionally eased (phase^0.8) for smoother acceleration. The geometry is linearly scaled by phase during growth:
x_draw = cx + phase · x_full
y_draw = y_base + phase · y_full After growth completes, the full shape remains visible.
Label dynamics: Labels can appear only after growth (final) or fade in during growth (grow). Positioning is tied to cx and either the shape’s center or a point above the flower tip, offset by petal_len and a fixed margin.

#### Background Compositing

Static and dynamic backdrops provide context and depth without distracting from the foreground shapes:
vertical_gradient and radial_gradient: smooth tone transitions matching a night-sky aesthetic.
noise: layered noise blended with gradients for texture and atmosphere; optional vignette emphasizes the center.
day_night: a time-parameterized gradient that cycles through night → dawn → noon → dusk → night within a specified number of frames, enabling ambient temporal cues parallel to the event timeline.
image/custom: user-provided imagery or a generator function producing an RGB array.
The background is composited beneath all flowers (low z-order) and may be cached if static, avoiding per-frame recomputation.

#### Performance Principles

Geometry cache: Precomputes and stores (petals, L, resolution) shapes so repeated configurations render instantly.
Artist management: During growth, transient “growing” artists are replaced by static patches once complete to keep the scene efficient.
Parameter controls: resolution, grow_frames, start_gap, and label toggles balance visual fidelity with runtime speed.

#### Colorbar and Semantics

When temp_norm is used, a colorbar maps on-screen colors back to the underlying normalized values, preserving interpretability of the encoded variable.






