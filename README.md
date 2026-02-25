# ⚡ Vibration Analysis — Machine Health Monitoring System

A complete vibration signal processing pipeline that reads acceleration data from Excel,
performs time-domain and frequency-domain analysis, detects machine faults,
and presents results in an interactive Streamlit dashboard.

---

## 📸 Dashboard Preview

> Upload your Excel file → instant analysis, interactive charts, downloadable report

![Dashboard Preview](https://youtu.be/UBn5qiGRgbg)

https://youtu.be/UBn5qiGRgbg

---

## 🗂️ Repository Structure

```
vibration-analysis/
│
├── Dashboard_App.py          # Interactive web dashboard (main deliverable)
├── main.py                   # Core analysis pipeline (CLI version, schedulable)
├── vibration_analysis.ipynb  # Step-by-step notebook with explanations
│
├── vibration_data.xlsx       # Sample data — use this to test the app
├── requirements.txt          # All Python dependencies
│
└── outputs/
    ├── exmple_vibration_analysis_report.html # Sample generated HTML report
    └── example_dashboard.png # Pre-generated dashboard screenshot
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/vibration-analysis.git
cd vibration-analysis
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv_vibration_analysis
```

Activate the virtual environment:

On Windows:
```bash
venv_vibration_analysis\Scripts\activate
```

On macOS/Linux:
```bash
source venv_vibration_analysis/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard
```bash
streamlit run Dashboard_App.py
```
Then open **http://localhost:8501** in your browser and upload `vibration_data.xlsx`.

### 5. Or run the CLI script directly
```bash
python main.py
```
Output saved to `outputs/vibration_dashboard_<timestamp>.png`

---

## 🔬 What the System Does

### Full Pipeline

```
RAW EXCEL DATA
    ↓
1. LOAD & CLEAN         → remove sensor errors (999g), NaN, convert types
    ↓
2. TIME DOMAIN          → RMS, Peak, Crest Factor, Kurtosis
    ↓
3. FREQUENCY DOMAIN     → FFT + Welch PSD (which frequencies are vibrating?)
    ↓
4. DIAGNOSE             → rule-based engine with ISO 10816 thresholds
    ↓
5. VISUALIZE            → interactive Plotly charts + downloadable HTML report
```

### Step 1 — Data Cleaning
- Converts non-numeric values (`SENSOR_ERROR` strings) via `pd.to_numeric(errors='coerce')`
- Removes impossible sensor readings (`|accel| ≥ 100g` — known DAQ error code)
- Sorts by timestamp, resets index

### Step 2 — Sampling Rate
```python
dt = df['timestamp_s'].diff().median()   # robust against irregular gaps
fs = 1.0 / dt                            # e.g. 999.8 Hz
```
Nyquist theorem → max detectable frequency = `fs / 2`

### Step 3 — Time Domain Statistics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMS** | `√(mean(x²))` | Overall vibration energy. ISO 10816 primary indicator |
| **Peak** | `max(\|x\|)` | Maximum single acceleration event |
| **Crest Factor** | `Peak / RMS` | <3 normal · >3 monitor · >5 bearing damage |
| **Kurtosis** | `E[(x-μ)⁴] / σ⁴` | ≈3 normal · >4 non-Gaussian · >6 bearing fault |

### Step 4 — Frequency Domain (FFT)

```
X[k] = Σ x[n] · e^(-j·2π·k·n/N)    for k = 0, 1, …, N/2

Where:
  x[n]  = acceleration signal
  X[k]  = complex amplitude at frequency bin k
  N     = total sample count
  fs    = sampling frequency
```

- **Raw FFT** — exact amplitude at each frequency bin
- **Welch PSD** — Hann-windowed segments averaged → smoother, less noisy, units: g²/Hz

### Step 5 — Diagnostic Engine

Rule-based engine with configurable thresholds (ISO 10816 defaults, tunable in sidebar):

| Metric | Warning | Critical | Source |
|--------|---------|----------|--------|
| RMS | > 0.10 g | > 0.20 g | ISO 10816 |
| Crest Factor | > 3.0 | > 5.0 | Engineering practice |
| Kurtosis | > 4.0 | > 6.0 | BS ISO 13373 |
| Dominant Freq | > 50 Hz | context-dependent | Engineering judgment |

Verdict: **🟢 MACHINE OK** / **🟡 MONITOR CLOSELY** / **🔴 FAULT DETECTED**

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| **File Upload** | Drag & drop `.xlsx`, `.xls`, or `.csv` |
| **Auto Column Detection** | Works with any column naming convention |
| **5 Interactive Charts** | Time Domain, FFT, Welch PSD, Histogram, Dominant Frequencies |
| **Live Threshold Editing** | Change ISO thresholds in the sidebar — verdict updates instantly |
| **KPI Cards** | 8 metrics with color-coded status (green/yellow/red) |
| **HTML Report Download** | Full report with formulas, results, and processing explanation |
| **Result Caching** | `@st.cache_data` — FFT only recomputes on new file upload |

---

## 🔄 Automated Scheduling (main.py)

The CLI version supports automatic execution multiple times per day:

```python
# Option 1 — Python schedule library
import schedule
schedule.every().day.at("08:00").do(run_analysis)
schedule.every().day.at("14:00").do(run_analysis)
schedule.every().day.at("20:00").do(run_analysis)

# Option 2 — Linux cron job
# 0 8,14,20 * * * python /path/to/main.py
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | FFT math, array operations |
| `scipy.fft` | Fast Fourier Transform |
| `scipy.signal` | Welch PSD, peak finding |
| `plotly` | Interactive charts |
| `openpyxl` | Excel file reading |

---

## 📋 Requirements

```
Python >= 3.9
streamlit >= 1.32.0
pandas >= 2.0.0
numpy >= 1.24.0
scipy >= 1.11.0
plotly >= 5.18.0
openpyxl >= 3.1.0
```

---

## 📁 Input Data Format

The app auto-detects column names, but expects:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_s` | float | Time in seconds from start |
| `accel_g` | float | Acceleration in g-units |

Sensor error codes (`999`, `SENSOR_ERROR` strings, `|a| ≥ 100g`) are automatically detected and removed.

---

## 📚 References

- Basic Vibration Analyst: Peak, Peak-to-Peak, RMS, and Crest Factors- Understanding key vibrations — https://www.youtube.com/watch?v=yxYxkDWEDWQ
- Spectrum Analysis in Python - https://www.geeksforgeeks.org/artificial-intelligence/spectrum-analysis-in-python/
- Find peaks in a signal - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
- Scipy FFT documentation — https://docs.scipy.org/doc/scipy/reference/fft.html
- Welch PSD method — https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
- Vibration Analysis of Bearings for Early Fault Detection: A MATLAB Case Study! - https://medium.com/@RashmiW/vibration-analysis-of-bearings-for-early-fault-detection-a-matlab-case-study-1e2ff244c78e
- Vibration measurement and analysis - https://www.spminstrument.com/measuring-techniques/vibration-monitoring/vibration-measurement-and-analysis/
- Peak to peak amplitude of sum of sinusoidals (harmonic frequencies) - https://dsp.stackexchange.com/questions/9724/peak-to-peak-amplitude-of-sum-of-sinusoidals-harmonic-frequencies
---
