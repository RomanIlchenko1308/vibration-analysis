"""
Vibration Analysis Dashboard — Streamlit
=========================================
Based on main.py logic, converted to an interactive Streamlit web app.

How to run:
    pip install streamlit plotly pandas openpyxl scipy numpy
    streamlit run streamlit_app.py

Then open http://localhost:8501 in your browser.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch, find_peaks
from scipy.fft import fft, fftfreq
from datetime import datetime
import io


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibSense Pro — Vibration Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark industrial theme matching main.py dashboard aesthetics)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #080c14; }
    section[data-testid="stSidebar"] { background-color: #0d1520; border-right: 1px solid #1a2d44; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #0d1520;
        border: 1px solid #1a2d44;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="metric-container"] label {
        color: #5a7a9a !important;
        font-size: 11px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-family: monospace;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00c8ff !important;
        font-family: monospace !important;
        font-size: 22px !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-family: monospace;
    }

    /* Headers */
    h1, h2, h3 { color: #00c8ff !important; }
    h1 { border-bottom: 2px solid #1a2d44; padding-bottom: 12px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #0d1520; gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #111d2e;
        color: #5a7a9a;
        border: 1px solid #1a2d44;
        border-radius: 6px 6px 0 0;
        font-family: monospace;
        font-size: 12px;
        letter-spacing: 1px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d1520 !important;
        color: #00c8ff !important;
        border-bottom-color: #0d1520 !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { background: #0d1520; }
    .stDataFrame { border: 1px solid #1a2d44; border-radius: 8px; }

    /* Divider */
    hr { border-color: #1a2d44; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #0d1520;
        border: 2px dashed #1a2d44;
        border-radius: 12px;
        padding: 20px;
    }
    [data-testid="stFileUploader"]:hover { border-color: #00c8ff; }

    /* Expander */
    .streamlit-expanderHeader { background: #0d1520 !important; color: #00c8ff !important; border: 1px solid #1a2d44; border-radius: 8px; }
    .streamlit-expanderContent { background: #080c14 !important; border: 1px solid #1a2d44; border-top: none; border-radius: 0 0 8px 8px; }

    /* Info / warning / error boxes */
    .stAlert { border-radius: 8px; }

    /* General text */
    p, li, .stMarkdown { color: #c8d8e8; }
    code { background: #111d2e !important; color: #00c8ff !important; border: 1px solid #1a2d44; border-radius: 4px; }

    /* Sidebar widgets */
    .stSlider label, .stSelectbox label, .stNumberInput label { color: #5a7a9a !important; font-family: monospace; font-size: 11px; }

    /* Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #0066ff, #00c8ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME  (matches main.py dark palette)
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1520",
    plot_bgcolor="#111d2e",
    font=dict(color="#c8d8e8", family="monospace", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a2d44", zerolinecolor="#1a2d44"),
    yaxis=dict(gridcolor="#1a2d44", zerolinecolor="#1a2d44"),
)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (same as main.py — editable in sidebar)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLDS = dict(
    rms_warn=0.10,
    rms_crit=0.20,
    crest_warn=3.0,
    crest_crit=5.0,
    kurt_warn=4.0,
    kurt_crit=6.0,
    dom_freq_warn=50.0,
    spike_threshold=100.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS  (ported directly from main.py)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes, filename: str, spike_threshold: float):
    """Load Excel/CSV, clean it. Cached so re-runs don't re-parse."""
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    raw_rows = len(df)

    # Detect columns
    t_col = next((c for c in df.columns if any(x in c.lower() for x in ["time", "ts", "timestamp"])), df.columns[0])
    a_col = next((c for c in df.columns if any(x in c.lower() for x in ["acc", "accel", "vibr"])), df.columns[1])

    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df[a_col] = pd.to_numeric(df[a_col], errors="coerce")

    nan_removed = df[[t_col, a_col]].isna().any(axis=1).sum()
    df = df.dropna(subset=[t_col, a_col]).copy()
    df = df.rename(columns={t_col: "timestamp_s", a_col: "accel_g"})

    spike_mask = df["accel_g"].abs() >= spike_threshold
    spike_removed = spike_mask.sum()
    df = df[~spike_mask].copy()

    df = df.sort_values("timestamp_s").reset_index(drop=True)
    return df, raw_rows, nan_removed, int(spike_removed)


def compute_sampling_rate(df):
    dt = df["timestamp_s"].diff().median()
    return 1.0 / dt


@st.cache_data(show_spinner=False)
def time_domain_stats(accel_values: tuple):
    """Same formulas as main.py time_domain_stats()."""
    a = np.array(accel_values)
    rms          = np.sqrt(np.mean(a**2))
    peak         = np.max(np.abs(a))
    crest_factor = peak / rms if rms > 0 else 0
    kurtosis     = pd.Series(a).kurtosis() + 3   # excess → absolute
    peak_to_peak = np.max(a) - np.min(a)
    return dict(
        rms_g=rms,
        peak_g=peak,
        peak_to_peak=peak_to_peak,
        crest_factor=crest_factor,
        kurtosis=kurtosis,
        mean_g=float(np.mean(a)),
        std_g=float(np.std(a)),
        n_samples=len(a),
    )


@st.cache_data(show_spinner=False)
def frequency_analysis(accel_values: tuple, fs: float):
    """Same FFT + Welch logic as main.py frequency_analysis()."""
    a = np.array(accel_values)
    N = len(a)

    # Raw FFT
    fft_vals  = fft(a)
    freqs     = fftfreq(N, d=1/fs)
    pos_mask  = freqs > 0
    freqs_pos = freqs[pos_mask]
    amps      = (2 / N) * np.abs(fft_vals[pos_mask])

    # Welch PSD
    f_welch, psd = welch(a, fs=fs, nperseg=min(1024, N // 4), scaling="density")

    # Dominant peaks
    peak_idx, _ = find_peaks(amps, prominence=np.max(amps) * 0.05, distance=5)
    dominant = sorted(zip(freqs_pos[peak_idx], amps[peak_idx]), key=lambda x: -x[1])[:8]

    return dict(
        freqs_pos=freqs_pos,
        amps=amps,
        f_welch=f_welch,
        psd=psd,
        dominant=dominant,
        fs=fs,
    )


def diagnose(stats, freq_data, thresholds):
    """Same rule-based engine as main.py diagnose()."""
    issues, warnings = [], []
    T = thresholds

    if stats["rms_g"] > T["rms_crit"]:
        issues.append(f"RMS = {stats['rms_g']:.4f} g exceeds critical threshold ({T['rms_crit']} g)")
    elif stats["rms_g"] > T["rms_warn"]:
        warnings.append(f"RMS = {stats['rms_g']:.4f} g above warning threshold ({T['rms_warn']} g)")

    if stats["crest_factor"] > T["crest_crit"]:
        issues.append(f"Crest Factor = {stats['crest_factor']:.2f} → impulsive damage")
    elif stats["crest_factor"] > T["crest_warn"]:
        warnings.append(f"Crest Factor = {stats['crest_factor']:.2f} → monitor closely")

    if stats["kurtosis"] > T["kurt_crit"]:
        issues.append(f"Kurtosis = {stats['kurtosis']:.2f} → early fault signature")
    elif stats["kurtosis"] > T["kurt_warn"]:
        warnings.append(f"Kurtosis = {stats['kurtosis']:.2f} → non-Gaussian vibration")

    if freq_data["dominant"]:
        dom_f, dom_a = freq_data["dominant"][0]
        if dom_f > T["dom_freq_warn"]:
            warnings.append(f"Dominant freq = {dom_f:.1f} Hz ({dom_a:.4f} g) → inspect rotating components")

    if issues:
        status, color = "🔴  FAULT DETECTED", "#ff3366"
    elif warnings:
        status, color = "🟡  MONITOR CLOSELY", "#ffcc00"
    else:
        status, color = "🟢  MACHINE OK", "#00ff88"

    return dict(status=status, color=color, issues=issues, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def chart_time_domain(df, stats):
    fig = go.Figure()
    # Signal
    fig.add_trace(go.Scattergl(
        x=df["timestamp_s"], y=df["accel_g"],
        mode="lines",
        line=dict(color="#00c8ff", width=0.8),
        name="Acceleration (g)",
        hovertemplate="t=%{x:.3f}s<br>a=%{y:.5f}g<extra></extra>",
    ))
    # RMS lines
    rms = stats["rms_g"]
    for sign, label in [(1, f"RMS = {rms:.5f} g"), (-1, None)]:
        fig.add_hline(y=sign*rms, line=dict(color="#00ff88", width=1.5, dash="dash"),
                      annotation_text=label if label else "", annotation_font_color="#00ff88")
    # Peak line
    fig.add_hline(y=stats["peak_g"], line=dict(color="#ffcc00", width=1, dash="dot"),
                  annotation_text=f"Peak = {stats['peak_g']:.5f} g", annotation_font_color="#ffcc00")

    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="⏱  Time Domain — Acceleration Signal", font=dict(color="#00c8ff", size=13)),
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (g)",
        height=300,
        showlegend=True,
        legend=dict(bgcolor="#0d1520", bordercolor="#1a2d44"),
    )
    return fig


def chart_fft(freq_data):
    freqs = freq_data["freqs_pos"]
    amps  = freq_data["amps"]
    dominant = freq_data["dominant"]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=freqs, y=amps,
        mode="lines",
        line=dict(color="#00c8ff", width=0.8),
        fill="tozeroy",
        fillcolor="rgba(0,100,255,0.08)",
        name="Amplitude",
        hovertemplate="%{x:.2f} Hz<br>%{y:.6f} g<extra></extra>",
    ))
    # Mark dominant peaks
    for i, (f, a) in enumerate(dominant[:5]):
        fig.add_vline(x=f, line=dict(color="#ffcc00", width=1, dash="dash"))
        fig.add_annotation(x=f, y=a, text=f"  {f:.1f} Hz",
                           font=dict(color="#ffcc00", size=9),
                           showarrow=False, xanchor="left")

    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="📊  FFT Spectrum — Amplitude vs Frequency", font=dict(color="#00c8ff", size=13)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude (g)",
        yaxis_type="log",
        height=320,
        showlegend=False,
    )
    return fig


def chart_welch(freq_data):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=freq_data["f_welch"], y=freq_data["psd"],
        mode="lines",
        line=dict(color="#ff6b9d", width=0.9),
        fill="tozeroy",
        fillcolor="rgba(255,107,157,0.07)",
        name="PSD",
        hovertemplate="%{x:.2f} Hz<br>%{y:.2e} g²/Hz<extra></extra>",
    ))
    for f, _ in freq_data["dominant"][:3]:
        fig.add_vline(x=f, line=dict(color="#ffcc00", width=1, dash="dash"))

    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="🔊  Welch PSD — Smoothed Power Spectrum", font=dict(color="#00c8ff", size=13)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (g²/Hz)",
        yaxis_type="log",
        height=320,
        showlegend=False,
    )
    return fig


def chart_histogram(df):
    a = df["accel_g"].values
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=a, nbinsx=60,
        marker=dict(color="#00c8ff", opacity=0.7, line=dict(color="#0066ff", width=0.5)),
        name="Count",
        hovertemplate="Bin: %{x:.4f} g<br>Count: %{y}<extra></extra>",
    ))
    # Normal distribution overlay
    mu, sigma = a.mean(), a.std()
    x_range = np.linspace(a.min(), a.max(), 200)
    gauss = (len(a) * (a.max()-a.min()) / 60) * \
            (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mu)/sigma)**2)
    fig.add_trace(go.Scatter(
        x=x_range, y=gauss,
        mode="lines", line=dict(color="#00ff88", width=1.5, dash="dot"),
        name="Gaussian fit",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="📈  Amplitude Distribution (Histogram)", font=dict(color="#00c8ff", size=13)),
        xaxis_title="Acceleration (g)",
        yaxis_title="Count",
        height=300,
        legend=dict(bgcolor="#0d1520"),
        showlegend=True,
    )
    return fig


def chart_dominant_bar(freq_data):
    if not freq_data["dominant"]:
        return go.Figure()
    freqs_d = [f"{f:.2f} Hz" for f, _ in freq_data["dominant"]]
    amps_d  = [a for _, a in freq_data["dominant"]]
    colors  = ["#00c8ff" if i == 0 else "#0066ff" for i in range(len(amps_d))]

    fig = go.Figure(go.Bar(
        x=freqs_d, y=amps_d,
        marker=dict(color=colors, line=dict(color="#1a2d44", width=0.5)),
        hovertemplate="%{x}<br>Amplitude: %{y:.6f} g<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="🎯  Top Dominant Frequencies", font=dict(color="#00c8ff", size=13)),
        xaxis_title="Frequency",
        yaxis_title="Amplitude (g)",
        height=300,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(df, stats, freq_data, diagnosis, thresholds, filename):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fs = freq_data["fs"]
    status_color = {"🟢": "#16a34a", "🟡": "#d97706", "🔴": "#dc2626"}.get(diagnosis["status"][0], "#334155")
    status_bg    = {"🟢": "#f0fdf4", "🟡": "#fffbeb", "🔴": "#fef2f2"}.get(diagnosis["status"][0], "#f8fafc")

    msg_rows = "".join(
        f'<tr style="background:#fef2f2"><td style="color:#dc2626;font-weight:bold">FAULT</td><td>{m}</td></tr>'
        for m in diagnosis["issues"]
    ) + "".join(
        f'<tr style="background:#fffbeb"><td style="color:#d97706;font-weight:bold">WARN</td><td>{m}</td></tr>'
        for m in diagnosis["warnings"]
    ) or '<tr><td colspan="2" style="color:#16a34a">✓ All parameters within normal limits</td></tr>'

    dom_rows = "".join(
        f"""<tr>
            <td>{i+1}</td>
            <td><strong>{f:.2f}</strong> Hz</td>
            <td>{a:.6f} g</td>
            <td>{a / freq_data["dominant"][0][1] * 100:.1f}%</td>
            <td>{"High-freq" if f > 200 else "Mid-freq" if f > 50 else "Dominant" if i==0 else "Harmonic"}</td>
        </tr>"""
        for i, (f, a) in enumerate(freq_data["dominant"])
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Vibration Report — {filename}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 40px 32px; color: #1e293b; line-height: 1.6; }}
  h1 {{ font-size: 26px; font-weight: 800; border-bottom: 3px solid #0066ff; padding-bottom: 10px; }}
  h2 {{ font-size: 17px; font-weight: 700; margin: 32px 0 10px; padding-left: 12px; border-left: 4px solid #0066ff; }}
  h3 {{ font-size: 14px; font-weight: 700; margin: 20px 0 6px; }}
  .meta {{ color: #64748b; font-size: 13px; margin-bottom: 28px; font-family: monospace; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 20px; }}
  th {{ background: #f1f5f9; padding: 9px 13px; text-align: left; font-weight: 700; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 9px 13px; border-bottom: 1px solid #e2e8f0; }}
  .verdict {{ padding: 18px 22px; border-radius: 10px; border-left: 5px solid {status_color}; background: {status_bg}; margin: 16px 0; }}
  .verdict h3 {{ color: {status_color}; font-size: 20px; margin: 0 0 6px; }}
  .formula {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; font-family: monospace; font-size: 13px; margin: 10px 0; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 16px 0; }}
  .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }}
  .kpi .lbl {{ font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}
  .kpi .val {{ font-family: monospace; font-size: 18px; font-weight: 700; }}
  footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 11px; text-align: center; }}
  @media print {{ body {{ padding: 20px; }} }}
</style>
</head><body>

<h1>⚡ Vibration Analysis Report</h1>
<div class="meta">
  File: {filename} &nbsp;|&nbsp; Generated: {ts} &nbsp;|&nbsp;
  Clean Samples: {stats["n_samples"]:,} &nbsp;|&nbsp; Duration: {stats.get("duration_s", df["timestamp_s"].max()-df["timestamp_s"].min()):.2f} s &nbsp;|&nbsp; fs: {fs:.0f} Hz
</div>

<h2>1. Machine Health Verdict</h2>
<div class="verdict">
  <h3>{diagnosis["status"]}</h3>
  <p>{"All vibration parameters within acceptable limits." if not diagnosis["issues"] and not diagnosis["warnings"] else "See findings below."}</p>
</div>
<table>
  <tr><th>Severity</th><th>Finding</th></tr>
  {msg_rows}
</table>

<h2>2. Key Performance Indicators</h2>
<div class="kpis">
  <div class="kpi"><div class="lbl">RMS</div><div class="val">{stats["rms_g"]:.5f} g</div></div>
  <div class="kpi"><div class="lbl">Peak</div><div class="val">{stats["peak_g"]:.5f} g</div></div>
  <div class="kpi"><div class="lbl">Crest Factor</div><div class="val">{stats["crest_factor"]:.2f}</div></div>
  <div class="kpi"><div class="lbl">Kurtosis</div><div class="val">{stats["kurtosis"]:.2f}</div></div>
  <div class="kpi"><div class="lbl">Peak-to-Peak</div><div class="val">{stats["peak_to_peak"]:.5f} g</div></div>
  <div class="kpi"><div class="lbl">Std Dev</div><div class="val">{stats["std_g"]:.5f} g</div></div>
  <div class="kpi"><div class="lbl">Mean</div><div class="val">{stats["mean_g"]:.5f} g</div></div>
  <div class="kpi"><div class="lbl">Dominant Freq</div><div class="val">{freq_data["dominant"][0][0]:.1f} Hz</div></div>
</div>

<h2>3. Processing Steps & Formulas</h2>

<h3>Step 1 — Data Cleaning</h3>
<p>Sensor error codes (|a| ≥ {thresholds["spike_threshold"]} g) removed. Non-numeric strings converted via <code>pd.to_numeric(errors='coerce')</code>.</p>

<h3>Step 2 — Sampling Rate</h3>
<div class="formula">dt = median(diff(timestamp))  →  fs = 1 / dt = <strong>{fs:.1f} Hz</strong>
Nyquist frequency = fs / 2 = {fs/2:.1f} Hz  (max detectable frequency)</div>

<h3>Step 3 — Time Domain Metrics</h3>
<table>
  <tr><th>Metric</th><th>Formula</th><th>Value</th><th>Interpretation</th></tr>
  <tr><td><strong>RMS</strong></td><td><code>√(Σxᵢ² / N)</code></td><td>{stats["rms_g"]:.5f} g</td>
      <td>{"🔴 Critical" if stats["rms_g"]>thresholds["rms_crit"] else "🟡 Warning" if stats["rms_g"]>thresholds["rms_warn"] else "🟢 Normal"}</td></tr>
  <tr><td><strong>Peak</strong></td><td><code>max(|xᵢ|)</code></td><td>{stats["peak_g"]:.5f} g</td><td>Max single event</td></tr>
  <tr><td><strong>Crest Factor</strong></td><td><code>Peak / RMS</code></td><td>{stats["crest_factor"]:.2f}</td>
      <td>{"🔴 >5: bearing damage" if stats["crest_factor"]>5 else "🟡 >3: monitor" if stats["crest_factor"]>3 else "🟢 <3: normal"}</td></tr>
  <tr><td><strong>Kurtosis</strong></td><td><code>E[(x-μ)⁴] / σ⁴</code></td><td>{stats["kurtosis"]:.2f}</td>
      <td>{"🔴 Bearing fault" if stats["kurtosis"]>6 else "🟡 Non-Gaussian" if stats["kurtosis"]>4 else "🟢 Normal (≈3.0)"}</td></tr>
</table>

<h3>Step 4 — FFT Frequency Analysis</h3>
<div class="formula">X[k] = Σ x[n] · e^(−j·2π·k·n/N)   for k = 0, 1, …, N/2
Frequency resolution = fs / N = {fs:.1f} / {stats["n_samples"]:,} = {fs/stats["n_samples"]:.4f} Hz/bin</div>

<h3>Step 5 — Welch PSD</h3>
<p>Signal divided into overlapping Hann-windowed segments (nperseg={min(1024, stats["n_samples"]//4)}). FFT averaged across segments → smoother spectrum in g²/Hz units.</p>

<h2>4. Dominant Frequencies</h2>
<table>
  <tr><th>#</th><th>Frequency</th><th>Amplitude</th><th>Relative</th><th>Note</th></tr>
  {dom_rows}
</table>

<h2>5. Diagnostic Thresholds (ISO 10816)</h2>
<table>
  <tr><th>Metric</th><th>Normal</th><th>Warning</th><th>Critical</th></tr>
  <tr><td>RMS</td><td>&lt; {thresholds["rms_warn"]} g</td><td>{thresholds["rms_warn"]}–{thresholds["rms_crit"]} g</td><td>&gt; {thresholds["rms_crit"]} g</td></tr>
  <tr><td>Crest Factor</td><td>&lt; {thresholds["crest_warn"]}</td><td>{thresholds["crest_warn"]}–{thresholds["crest_crit"]}</td><td>&gt; {thresholds["crest_crit"]}</td></tr>
  <tr><td>Kurtosis</td><td>≈ 3.0</td><td>&gt; {thresholds["kurt_warn"]}</td><td>&gt; {thresholds["kurt_crit"]}</td></tr>
</table>

<footer>VibSense Pro — Generated {ts} | {filename}</footer>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ VibSense Pro")
    st.markdown("---")
    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Drop your Excel or CSV file",
        type=["xlsx", "xls", "csv"],
        help="Must have timestamp (seconds) and acceleration (g) columns",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Thresholds")
    st.caption("Tune these for your specific machine (ISO 10816 defaults)")

    T = DEFAULT_THRESHOLDS.copy()
    T["rms_warn"]        = st.number_input("RMS Warning (g)",     value=0.10, step=0.01, format="%.2f")
    T["rms_crit"]        = st.number_input("RMS Critical (g)",    value=0.20, step=0.01, format="%.2f")
    T["crest_warn"]      = st.number_input("Crest Factor Warn",   value=3.0,  step=0.5,  format="%.1f")
    T["crest_crit"]      = st.number_input("Crest Factor Crit",   value=5.0,  step=0.5,  format="%.1f")
    T["kurt_warn"]       = st.number_input("Kurtosis Warning",    value=4.0,  step=0.5,  format="%.1f")
    T["kurt_crit"]       = st.number_input("Kurtosis Critical",   value=6.0,  step=0.5,  format="%.1f")
    T["dom_freq_warn"]   = st.number_input("Dom. Freq Warn (Hz)", value=50.0, step=5.0,  format="%.1f")
    T["spike_threshold"] = st.number_input("Spike Filter (g)",    value=100.0, step=10.0, format="%.1f")

    st.markdown("---")
    st.caption("VibSense Pro v1.0 | Roman Ilchenko")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# ⚡ Vibration Analysis Dashboard")
st.markdown("*Machine Health Monitoring System — upload your Excel data to begin*")

if uploaded_file is None:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📤 Upload**\nDrop an `.xlsx` or `.csv` file in the sidebar. Needs `timestamp_s` and `accel_g` columns (or similar names).")
    with col2:
        st.info("**📊 Analyze**\nTime domain (RMS, Crest Factor, Kurtosis) + Frequency domain (FFT, Welch PSD) computed instantly.")
    with col3:
        st.info("**📄 Download**\nGet a full HTML report with formulas, step-by-step explanation, and all results.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading and cleaning data..."):
    df, raw_rows, nan_removed, spike_removed = load_and_clean(
        uploaded_file.read(), uploaded_file.name, T["spike_threshold"]
    )
    fs = compute_sampling_rate(df)

with st.spinner("🔢 Computing time-domain statistics..."):
    stats = time_domain_stats(tuple(df["accel_g"].values))
    stats["duration_s"] = float(df["timestamp_s"].max() - df["timestamp_s"].min())

with st.spinner("📡 Running FFT + Welch PSD..."):
    freq_data = frequency_analysis(tuple(df["accel_g"].values), fs)

diagnosis = diagnose(stats, freq_data, T)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📂 File",        uploaded_file.name[:20])
c2.metric("✅ Clean Rows",  f"{len(df):,}")
c3.metric("🗑️ NaN Removed", f"{nan_removed}")
c4.metric("⚡ Spikes Removed", f"{spike_removed}")
c5.metric("📡 Sample Rate", f"{fs:.0f} Hz")


# ─────────────────────────────────────────────────────────────────────────────
# VERDICT BANNER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
color  = diagnosis["color"]
status = diagnosis["status"]

verdict_bg = {"🟢": "rgba(0,255,136,0.07)", "🟡": "rgba(255,204,0,0.07)", "🔴": "rgba(255,51,102,0.07)"}[status[0]]
verdict_border = {"🟢": "#00ff88", "🟡": "#ffcc00", "🔴": "#ff3366"}[status[0]]

issues_html = "".join(f'<div style="color:#ff3366;font-family:monospace;font-size:13px">⚠ {i}</div>' for i in diagnosis["issues"])
warns_html  = "".join(f'<div style="color:#ffcc00;font-family:monospace;font-size:13px">• {w}</div>' for w in diagnosis["warnings"])
ok_html     = '<div style="color:#00ff88;font-family:monospace;font-size:13px">✓ All parameters within normal operating limits</div>' if not diagnosis["issues"] and not diagnosis["warnings"] else ""

st.markdown(f"""
<div style="
    background: {verdict_bg};
    border: 1px solid {verdict_border};
    border-left: 5px solid {verdict_border};
    border-radius: 12px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 8px;
">
    <div style="color:{color}; font-size:32px; font-weight:800; font-family:monospace; white-space:nowrap; min-width:260px">
        {status}
    </div>
    <div>{issues_html}{warns_html}{ok_html}</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📈 Key Performance Indicators")

def kpi_color(val, warn, crit):
    if val > crit: return "inverse"
    if val > warn: return "off"
    return "normal"

k1, k2, k3, k4 = st.columns(4)
k1.metric("⚡ RMS",           f"{stats['rms_g']:.5f} g",
          delta=f"{'⚠ above warn' if stats['rms_g']>T['rms_warn'] else '✓ normal'}",
          delta_color=kpi_color(stats['rms_g'], T['rms_warn'], T['rms_crit']))

k2.metric("🏔️ Peak",           f"{stats['peak_g']:.5f} g")

k3.metric("📊 Crest Factor",  f"{stats['crest_factor']:.2f}",
          delta=f"{'⚠ elevated' if stats['crest_factor']>T['crest_warn'] else '✓ normal'}",
          delta_color=kpi_color(stats['crest_factor'], T['crest_warn'], T['crest_crit']))

k4.metric("📐 Kurtosis",      f"{stats['kurtosis']:.2f}  (normal ≈ 3)",
          delta=f"{'⚠ non-Gaussian' if stats['kurtosis']>T['kurt_warn'] else '✓ normal'}",
          delta_color=kpi_color(stats['kurtosis'], T['kurt_warn'], T['kurt_crit']))

k5, k6, k7, k8 = st.columns(4)
k5.metric("↕️ Peak-to-Peak",  f"{stats['peak_to_peak']:.5f} g")
k6.metric("σ  Std Dev",       f"{stats['std_g']:.5f} g")
k7.metric("〰️ Mean",           f"{stats['mean_g']:.5f} g")
if freq_data["dominant"]:
    k8.metric("🎯 Dominant Freq", f"{freq_data['dominant'][0][0]:.2f} Hz",
              delta=f"{'⚠ inspect' if freq_data['dominant'][0][0]>T['dom_freq_warn'] else '✓ ok'}",
              delta_color="off" if freq_data["dominant"][0][0]>T["dom_freq_warn"] else "normal")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — TABS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📊 Visualizations")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⏱ TIME DOMAIN",
    "📊 FFT SPECTRUM",
    "🔊 WELCH PSD",
    "📈 HISTOGRAM",
    "🎯 DOMINANT FREQS",
])

with tab1:
    st.plotly_chart(chart_time_domain(df, stats), use_container_width=True)
    with st.expander("📚 What is Time Domain Analysis?"):
        st.markdown("""
**Time Domain** means looking at the signal directly as it was recorded — acceleration vs. time.

From this raw signal we extract three key health indicators:

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **RMS** | `√(mean(x²))` | Overall energy level. High RMS = more vibration = potential problem |
| **Crest Factor** | `Peak / RMS` | "Spikiness" ratio. >3 = elevated, >5 = bearing damage |
| **Kurtosis** | `E[(x-μ)⁴] / σ⁴` | Distribution shape. ≈3 = normal, >4 = early bearing fault |

**Reference:** ISO 10816 standard defines acceptable RMS values for different machine classes.
        """)

with tab2:
    st.plotly_chart(chart_fft(freq_data), use_container_width=True)
    with st.expander("📚 How does FFT work?"):
        st.markdown(f"""
**FFT (Fast Fourier Transform)** decomposes your time signal into sine waves at different frequencies.

```
X[k] = Σ x[n] · e^(-j·2π·k·n/N)    for k = 0, 1, …, N/2

Where:
  x[n]  = acceleration sample at time n
  X[k]  = complex number at frequency bin k
  |X[k]|= amplitude at that frequency
  N     = {stats['n_samples']:,} samples
  fs    = {fs:.1f} Hz  →  frequency resolution = {fs/stats['n_samples']:.4f} Hz/bin
```

**Yellow dashed lines** = dominant peaks found by scipy `find_peaks()` with 5% prominence threshold.

**Y-axis is logarithmic** — so small peaks are still visible despite the dominant peak being much larger.
        """)

with tab3:
    st.plotly_chart(chart_welch(freq_data), use_container_width=True)
    with st.expander("📚 Welch PSD vs raw FFT"):
        st.markdown(f"""
**Welch's method** reduces noise in the spectrum by averaging multiple FFT windows:

```
Algorithm:
1. Divide signal into overlapping segments (window size = {min(1024, stats['n_samples']//4)} samples, 50% overlap)
2. Apply Hann window: w[i] = 0.5 × (1 − cos(2π·i / (M−1)))
3. Compute FFT for each windowed segment
4. Average all power spectra  →  much smoother result

Units: g²/Hz  (power per unit frequency, not amplitude)
```

**When to use Welch vs raw FFT:**
- Raw FFT: shows exact amplitude at each frequency bin — use for precise amplitude measurement
- Welch PSD: averaged, stabler — use for identifying which frequency *bands* carry the most energy
        """)

with tab4:
    st.plotly_chart(chart_histogram(df), use_container_width=True)
    with st.expander("📚 What does the histogram tell you?"):
        st.markdown("""
**Amplitude Distribution** shows how often each acceleration value occurs.

- **Gaussian (bell curve) shape** = normal random vibration → machine is healthy
- **Heavy tails / spikes** = non-Gaussian impulsive events → this is what Kurtosis measures
- **Green dashed line** = ideal Gaussian with same mean and std dev as your data

A healthy machine produces vibration that closely follows the Gaussian curve.
A damaged bearing creates sharp impacts that create outlier values — the distribution gets "heavy tails".
        """)

with tab5:
    st.plotly_chart(chart_dominant_bar(freq_data), use_container_width=True)

    # Dominant frequency table
    dom_data = []
    max_amp = freq_data["dominant"][0][1] if freq_data["dominant"] else 1
    for i, (f, a) in enumerate(freq_data["dominant"]):
        note = "High-freq" if f > 200 else "Mid-freq component" if f > 50 else "Dominant / fundamental" if i == 0 else "Harmonic"
        dom_data.append({
            "Rank": i + 1,
            "Frequency (Hz)": round(f, 2),
            "Amplitude (g)": round(a, 6),
            "Relative (%)": round(a / max_amp * 100, 1),
            "Classification": note,
        })

    st.dataframe(
        pd.DataFrame(dom_data),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("📚 What do frequencies mean?"):
        st.markdown("""
| Frequency Pattern | Likely Cause | Action |
|-------------------|-------------|--------|
| 1× rotation speed | Imbalance / bent shaft | Balance shaft |
| 2× rotation speed | Misalignment | Re-align coupling |
| 3–5× rotation speed | Mechanical looseness | Tighten fasteners |
| Ball pass freq (high) | Bearing inner/outer race defect | Replace bearing |
| Broad spectrum increase | Cavitation / turbulence | Check flow conditions |

**How to find rotation speed:** Check machine nameplate RPM, then convert:
`rotation_freq_Hz = RPM / 60`
        """)


# ─────────────────────────────────────────────────────────────────────────────
# DOMINANT FREQUENCY TABLE  (always visible)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🎯 Top Dominant Frequencies")

col_table, col_info = st.columns([3, 2])
with col_table:
    st.dataframe(pd.DataFrame(dom_data), use_container_width=True, hide_index=True)

with col_info:
    st.markdown("**Signal Summary**")
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Duration | `{stats['duration_s']:.2f} s` |
| Sample Rate | `{fs:.0f} Hz` |
| Nyquist Freq | `{fs/2:.0f} Hz` |
| Freq Resolution | `{fs/stats['n_samples']:.4f} Hz/bin` |
| Total Samples | `{stats['n_samples']:,}` |
| Raw Rows | `{raw_rows:,}` |
| Cleaned Away | `{raw_rows - stats['n_samples']:,}` |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD REPORT
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📄 Download Report")

report_html = generate_html_report(df, stats, freq_data, diagnosis, T, uploaded_file.name)
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

col_dl1, col_dl2 = st.columns([2, 5])
with col_dl1:
    st.download_button(
        label="📄 Download Full HTML Report",
        data=report_html,
        file_name=f"vibration_report_{ts_str}.html",
        mime="text/html",
        use_container_width=True,
    )

with col_dl2:
    st.caption("The HTML report includes: all KPIs, formulas with your actual values filled in, dominant frequency table, diagnostic thresholds, step-by-step processing explanation, and ISO 10816 reference. Opens in any browser, prints cleanly to PDF.")