"""
Vibration Analysis System
=========================
Reads vibration data from Excel, performs time-domain and frequency-domain
analysis, detects anomalies, and generates diagnostic reports.

Libraries used:
- pandas       : reading Excel, data manipulation
- numpy        : numerical computing, FFT math
- matplotlib   : plotting all charts
- scipy.signal : signal processing (Welch PSD method)
- scipy.fft    : Fast Fourier Transform

How to run:
    python vibration_analysis.py

To simulate "a few times per day data update", see the scheduler section at bottom.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch, find_peaks
from scipy.fft import fft, fftfreq
from datetime import datetime
import os

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH   = "vibration_data.xlsx"
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Machine health thresholds (tunable)
SPIKE_THRESHOLD_G  = 10.0      # single spike > 10 g  → anomaly
RMS_WARNING_G      = 0.10      # RMS > 0.10 g          → warning
RMS_CRITICAL_G     = 0.20      # RMS > 0.20 g          → critical
DOMINANT_FREQ_WARN = 50.0      # dominant freq > 50 Hz → inspect bearings

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING  (simulates "reading from Excel a few times per day")
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Reads the Excel file, converts types, removes NaN rows, and
    filters out sensor-error spikes (999 g is a known sensor error code).
    Returns a clean DataFrame with columns: timestamp_s, accel_g
    """
    print(f"[{datetime.now():%H:%M:%S}]  Loading data from: {path}")
    df = pd.read_excel(path)

    # Convert acceleration to numeric (some cells may be strings)
    df["accel_g"] = pd.to_numeric(df["accel_g"], errors="coerce")

    # Drop rows with NaN
    df = df.dropna(subset=["timestamp_s", "accel_g"]).copy()

    # Flag and remove sensor-error spikes (999 = saturation / disconnection)
    sensor_errors = df["accel_g"].abs() >= 100        # unrealistic for a car
    n_errors = sensor_errors.sum()
    if n_errors:
        print(f"  ⚠  Removed {n_errors} sensor-error samples (|accel| ≥ 100 g)")
        df = df[~sensor_errors].copy()

    df = df.sort_values("timestamp_s").reset_index(drop=True)
    print(f"  ✓  {len(df):,} clean samples | "
          f"Duration: {df['timestamp_s'].max():.2f} s")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TIME-DOMAIN STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def time_domain_stats(df: pd.DataFrame) -> dict:
    """
    Computes key time-domain metrics used in vibration diagnostics.

    I used this article as a reference for the formulas and interpretation of these metrics:

    https://www.geeksforgeeks.org/artificial-intelligence/spectrum-analysis-in-python/#types-of-spectrum-analysis

    
    
    RMS  (Root Mean Square)  – overall vibration energy level.
    
    https://www.geeksforgeeks.org/maths/root-mean-square-formula/


             RMS = sqrt( mean(x²) )
             High RMS → more energy → possible imbalance or bearing wear.

    Peak  – maximum absolute value. Single large peaks can indicate impacts.

    Crest Factor = Peak / RMS
             Low  (<3)  : normal smooth vibration
             High (>5)  : impulsive events (bearing defect, looseness)

    Kurtosis  – statistical measure of 'tailedness'.
             Normal random vibration ≈ 3
             Kurtosis > 4  → impulsive, non-Gaussian → early bearing fault signal
    """
    a = df["accel_g"].values
    rms           = np.sqrt(np.mean(a**2))
    peak          = np.max(np.abs(a))
    crest_factor  = peak / rms if rms > 0 else 0
    kurtosis      = pd.Series(a).kurtosis() + 3  # scipy/pandas uses excess kurtosis
    peak_to_peak  = np.max(a) - np.min(a)

    stats = {
        "rms_g"        : rms,
        "peak_g"       : peak,
        "peak_to_peak" : peak_to_peak,
        "crest_factor" : crest_factor,
        "kurtosis"     : kurtosis,
        "mean_g"       : np.mean(a),
        "std_g"        : np.std(a),
        "n_samples"    : len(a),
        "duration_s"   : df["timestamp_s"].max() - df["timestamp_s"].min(),
    }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FREQUENCY-DOMAIN ANALYSIS  (the heart of vibration diagnostics)
# ─────────────────────────────────────────────────────────────────────────────
def frequency_analysis(df: pd.DataFrame, fs: float, fft: fft) -> dict:
    """
    Converts time-signal to frequency spectrum using FFT and  PSD.

    FFT (Fast Fourier Transform)
    ----------------------------
    Any repeating vibration pattern can be decomposed into a sum of sine waves
    at different frequencies. FFT reveals WHICH frequencies are present and HOW
    STRONG (amplitude) they are.

    Formula:
        X[k] = Σ x[n] · e^(-j·2π·k·n/N)    for k = 0, 1, …, N/2

    - X[k]  : complex number at frequency bin k
    - |X[k]|: amplitude at that frequency
    - N     : total number of samples
    - fs    : sampling frequency (Hz) = 1 / Δt

     PSD (Power Spectral Density)
    -----------------------------------
    Welch method averages multiple overlapping FFT windows → smoother, less noisy.
    PSD unit: g²/Hz  (energy per unit frequency).
    Better for identifying dominant frequencies in noisy signals.
    """
    a  = df["accel_g"].values
    N  = len(a)

    # ── Raw FFT ──────────────────────────────────────────────────────────────
    fft_vals = fft(a)                        # complex spectrum
    freqs    = fftfreq(N, d=1/fs)            # frequency axis
    # Keep only positive frequencies (signal is real → spectrum is symmetric)
    pos_mask  = freqs > 0
    freqs_pos = freqs[pos_mask]
    amps      = (2 / N) * np.abs(fft_vals[pos_mask])   # two-sided → one-sided

    # ── Welch PSD ─────────────────────────────────────────────────────────────
    f_welch, psd = welch(a, fs=fs,
                         nperseg=min(1024, N//4),   # window size
                         noverlap=None,
                         scaling="density")          # g²/Hz

    # ── Dominant frequencies (peaks in spectrum) ──────────────────────────────
    # find_peaks with prominence filter to ignore noise
    peak_idx, props = find_peaks(amps,
                                 prominence=np.max(amps) * 0.05,  # 5 % of max
                                 distance=5)                       # min 5 bins apart
    dominant = sorted(zip(freqs_pos[peak_idx], amps[peak_idx]),
                      key=lambda x: -x[1])[:5]   # top 5 peaks

    return {
        "freqs_pos"  : freqs_pos,
        "amps"       : amps,
        "f_welch"    : f_welch,
        "psd"        : psd,
        "dominant"   : dominant,   # list of (freq_Hz, amplitude_g) tuples
        "fs"         : fs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ANOMALY DETECTION & MACHINE HEALTH DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
def diagnose(stats: dict, freq_data: dict) -> dict:
    """
    Rule-based diagnostic engine.
    Combines time-domain metrics and frequency peaks to give a health verdict.
    """
    issues   = []
    warnings = []

    # ── RMS check ─────────────────────────────────────────────────────────────
    if stats["rms_g"] > RMS_CRITICAL_G:
        issues.append(f"CRITICAL: RMS = {stats['rms_g']:.4f} g  (threshold {RMS_CRITICAL_G} g)")
    elif stats["rms_g"] > RMS_WARNING_G:
        warnings.append(f"WARNING: RMS = {stats['rms_g']:.4f} g  (threshold {RMS_WARNING_G} g)")

    # ── kCrest factor chec ────────────────────────────────────────────────────
    #Crest factor checks the ratio of a signal's peak amplitude to its root mean square (RMS) value, 
    # measuring waveform "spikiness" or impulsive distortion
    if stats["crest_factor"] > 5:
        issues.append(f"HIGH Crest Factor = {stats['crest_factor']:.2f} → possible impulsive events / bearing damage")
    elif stats["crest_factor"] > 3:
        warnings.append(f"Elevated Crest Factor = {stats['crest_factor']:.2f} → monitor closely")

    # ── Kurtosis check ────────────────────────────────────────────────────────
    if stats["kurtosis"] > 6:
        issues.append(f"HIGH Kurtosis = {stats['kurtosis']:.2f} → early bearing fault signature")
    elif stats["kurtosis"] > 4:
        warnings.append(f"Elevated Kurtosis = {stats['kurtosis']:.2f} → non-Gaussian vibration")

    # ── Dominant frequency check ──────────────────────────────────────────────
    if freq_data["dominant"]:
        dom_f, dom_a = freq_data["dominant"][0]
        if dom_f > DOMINANT_FREQ_WARN:
            warnings.append(f"Dominant freq = {dom_f:.1f} Hz  ({dom_a:.4f} g) → inspect rotating components / bearings")

    # ── Overall verdict ───────────────────────────────────────────────────────
    if issues:
        status = "🔴  FAULT DETECTED"
    elif warnings:
        status = "🟡  MONITOR CLOSELY"
    else:
        status = "🟢  MACHINE OK"

    return {"status": status, "issues": issues, "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALIZATION  (4-panel dashboard)
# ─────────────────────────────────────────────────────────────────────────────
def plot_dashboard(df, stats, freq_data, diagnosis, out_path):
    fig = plt.figure(figsize=(16, 12), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    DARK_BG  = "#0f0f1a"
    PANEL_BG = "#1a1a2e"
    ACCENT   = "#00d4ff"
    WARN     = "#ffaa00"
    GOOD     = "#00ff88"
    TEXT     = "#e0e0e0"
    GRID     = "#2a2a3e"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.spines[:].set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color=ACCENT, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, color=GRID, alpha=0.7, linewidth=0.5)

    # ── Panel 1: Time-domain waveform ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["timestamp_s"], df["accel_g"], color=ACCENT, lw=0.6, alpha=0.85)
    ax1.axhline(stats["rms_g"],  color=GOOD, lw=1, ls="--", label=f"RMS = {stats['rms_g']:.4f} g")
    ax1.axhline(-stats["rms_g"], color=GOOD, lw=1, ls="--")
    ax1.axhline(stats["peak_g"], color=WARN, lw=1, ls=":", label=f"Peak = {stats['peak_g']:.4f} g")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Acceleration (g)")
    ax1.legend(facecolor=PANEL_BG, labelcolor=TEXT, fontsize=8)
    style_ax(ax1, "⏱  Time Domain – Acceleration Signal")

    # ── Panel 2: FFT Spectrum ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(freq_data["freqs_pos"], freq_data["amps"], color=ACCENT, lw=0.7, alpha=0.9)
    for i, (f, a) in enumerate(freq_data["dominant"][:3]):
        ax2.axvline(f, color=WARN, lw=1, ls="--", alpha=0.7)
        ax2.text(f + 0.5, a * 1.3, f"{f:.1f} Hz", color=WARN, fontsize=7)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude (g)")
    ax2.set_xlim(0, freq_data["fs"] / 2)
    style_ax(ax2, "📊  FFT Spectrum (Amplitude vs Frequency)")

    # ── Panel 3: Welch PSD ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(freq_data["f_welch"], freq_data["psd"], color="#ff6b9d", lw=0.8)
    for f, _ in freq_data["dominant"][:3]:
        ax3.axvline(f, color=WARN, lw=1, ls="--", alpha=0.7)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD (g²/Hz)")
    ax3.set_xlim(0, freq_data["fs"] / 2)
    style_ax(ax3, "🔊  Welch PSD – Smoothed Spectrum")

    # ── Panel 4: KPI cards ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(PANEL_BG)
    ax4.axis("off")
    kpis = [
        ("RMS",           f"{stats['rms_g']:.5f} g"),
        ("Peak",          f"{stats['peak_g']:.5f} g"),
        ("Peak-to-Peak",  f"{stats['peak_to_peak']:.5f} g"),
        ("Crest Factor",  f"{stats['crest_factor']:.2f}"),
        ("Kurtosis",      f"{stats['kurtosis']:.2f}"),
        ("Samples",       f"{stats['n_samples']:,}"),
        ("Duration",      f"{stats['duration_s']:.2f} s"),
        ("Sample Rate",   f"{freq_data['fs']:.0f} Hz"),
    ]
    for i, (label, val) in enumerate(kpis):
        y = 0.93 - i * 0.12
        ax4.text(0.05, y, label + ":", color=TEXT, fontsize=9, transform=ax4.transAxes, va="top")
        ax4.text(0.55, y, val, color=ACCENT, fontsize=9, fontweight="bold", transform=ax4.transAxes, va="top")
    ax4.set_title("📈  Key Performance Indicators", color=ACCENT, fontsize=10, fontweight="bold", pad=8)

    # ── Panel 5: Diagnosis ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL_BG)
    ax5.axis("off")
    status_color = {"🟢": GOOD, "🟡": WARN, "🔴": "#ff4444"}.get(
        diagnosis["status"][0], TEXT)
    ax5.text(0.5, 0.92, diagnosis["status"], color=status_color,
             fontsize=13, fontweight="bold", ha="center", va="top",
             transform=ax5.transAxes)
    lines = diagnosis["issues"] + diagnosis["warnings"]
    if not lines:
        lines = ["All parameters within normal limits.", "Machine operating normally."]
    for i, line in enumerate(lines[:5]):
        color = "#ff4444" if line.startswith("CRITICAL") else WARN if line.startswith(("WARNING", "HIGH", "Elevated", "Dominant")) else GOOD
        ax5.text(0.05, 0.75 - i * 0.16, "• " + line[:70], color=color,
                 fontsize=7.5, transform=ax5.transAxes, va="top", wrap=True)

    # Top frequencies table
    if freq_data["dominant"]:
        dom_lines = ["Top Frequencies:"] + [
            f"  {i+1}. {f:.1f} Hz  →  {a:.5f} g"
            for i, (f, a) in enumerate(freq_data["dominant"][:4])
        ]
        for i, line in enumerate(dom_lines):
            ax5.text(0.05, 0.35 - i * 0.10, line, color=TEXT,
                     fontsize=8, transform=ax5.transAxes, va="top",
                     fontweight="bold" if i == 0 else "normal")

    ax5.set_title("🛠  Machine Health Diagnosis", color=ACCENT, fontsize=10, fontweight="bold", pad=8)

    # Title
    fig.suptitle(f"  Vibration Analysis Dashboard  |  {datetime.now():%Y-%m-%d %H:%M:%S}",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓  Dashboard saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(stats, freq_data, diagnosis):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  VIBRATION ANALYSIS REPORT")
    print(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(sep)

    print("\n── Time-Domain Statistics ──")
    for k, v in stats.items():
        print(f"  {k:<18}: {v:.5f}" if isinstance(v, float) else f"  {k:<18}: {v}")

    print("\n── Top Dominant Frequencies ──")
    for i, (f, a) in enumerate(freq_data["dominant"], 1):
        print(f"  {i}. {f:7.2f} Hz  |  Amplitude: {a:.6f} g")

    print(f"\n── Machine Health Status ──")
    print(f"  {diagnosis['status']}")
    for issue in diagnosis["issues"]:
        print(f"  [FAULT]   {issue}")
    for warn in diagnosis["warnings"]:
        print(f"  [WARN]    {warn}")
    if not diagnosis["issues"] and not diagnosis["warnings"]:
        print("  All parameters within normal limits.")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis():
    """
    Full pipeline:
      1. Load data from Excel
      2. Compute time-domain stats
      3. FFT + Welch frequency analysis
      4. Diagnose machine health
      5. Plot dashboard
      6. Print text report
    """
    # Step 1 – Load
    df = load_data(EXCEL_PATH)

    # Step 2 – Sampling frequency (needed for FFT)
    dt = df["timestamp_s"].diff().median()   # median time step (seconds)
    fs = 1.0 / dt                            # samples per second (Hz)
    print(f"  Sampling rate: {fs:.1f} Hz")

    # Step 3 – Time domain
    stats = time_domain_stats(df)

    # Step 4 – Frequency domain
    freq_data = frequency_analysis(df, fs, fft)

    # Step 5 – Diagnose
    diagnosis = diagnose(stats, freq_data)

    # Step 6 – Visualize
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(OUTPUT_DIR, f"vibration_dashboard_{ts}.png")
    plot_dashboard(df, stats, freq_data, diagnosis, plot_path)

    # Step 7 – Report
    print_report(stats, freq_data, diagnosis)

    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SCHEDULER  (runs analysis N times per day automatically)
# ─────────────────────────────────────────────────────────────────────────────
def run_scheduler(times_per_day: int = 3):
    """
    Simulates reading data a few times per day.
    Uses Python 'schedule' library:   pip install schedule

    Example: times_per_day=3  → runs at 08:00, 14:00, 20:00
    """
    try:
        import schedule, time as _time
    except ImportError:
        print("Install scheduler:  pip install schedule")
        return

    interval_hours = 24 // times_per_day
    for i in range(times_per_day):
        t = f"{8 + i * interval_hours:02d}:00"
        schedule.every().day.at(t).do(run_analysis)
        print(f"  Scheduled: {t}")

    print(f"\n  Scheduler running ({times_per_day}×/day). Press Ctrl+C to stop.\n")
    while True:
        schedule.run_pending()
        _time.sleep(30)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_file = run_analysis()
    print(f"\n  Output file: {output_file}")

    # Uncomment below to enable automatic scheduling:
    # run_scheduler(times_per_day=3)