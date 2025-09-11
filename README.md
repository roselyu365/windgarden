# Project Name (Windgarden)
Generate static geometric visuals (e.g., horizontal line patterns and circular/ring graphics) with configurable parameters; optionally extensible for data scraping or asset preparation.

## Features
- Supports generation of:
  - Horizontal line / line-based pattern images
  - Circular or ring structures
- Sample output images (see below)
- Optional utility module(s) for scraping / preprocessing (scraping_util*)
- Clean dependency management via requirements.txt

##  Repository Structure 
.
├── my_script.py               # Main entry script (generation logic / dispatcher)
├── scraping_util*.py          # (Optional) scraping / preprocessing utilities
├── requirements.txt           # Python dependencies
├── Figure_1-line...png        # Sample line-based output
├── newplot.png                # Sample ring / circular output
├── README.md                  # Project documentation
├── .gitignore
└── .DS_Store                  # macOS system file (ignored)
```
(Modify/remove lines if your structure differs.)

##  Preview
| Type | Example |
|------|---------|
| Line / horizontal pattern | ![Line Example](Figure_1-line...png) |
| Ring / circular form | ![Ring Example](newplot.png) |

## 🚀 Quick Start

### 1. Requirements
- Python >= 2.32.3


```
# Install dependencies
pip install -r requirements.txt
```

# Generate a horizontal line graphic
python my_script.py 

# Generate a ring visualization
python my_script.py 

# List all options
python my_script.py --help
```

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

---

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

---

### 6. Color Encoding (Temperature)
Two variants:
1. Matplotlib colormap (blue → purple → orange)
2. Plotly RGBA manual mapping:
   - Red channel increases with temperature
   - Blue channel decreases
   - Subtle green adjustment

You could unify this for consistency by defining one color utility reused across backends.
```
### 7. Example Data-to-Visual Flow (Narrative)
1. Raw API: wind_dir = 135°, wind_speed = 5.2 m/s, temp = 21℃
2. Direction 135° ⇒ 8 petals
3. Wind speed 5.2 ⇒ \[ L ≈ 10 + (5.2/15)*40 ≈ 23.9 \]
4. Temp 21℃ ⇒ \[ temp\_norm ≈ 0.6 \] ⇒ mid purple-orange mix
5. In timeline (Day 0, Hour 3) ⇒ x ≈ 3
6. In circular layout ⇒ angle ≈ \[ 2\pi * 3/24 = \pi/4 \]



