"""
========================================================================================
Universal DSP Signal Analyzer & Filter Design Studio (Streamlit / Python Edition)
========================================================================================
To run this application locally:
1. Install requirements: pip install streamlit numpy scipy matplotlib plotly sounddevice
2. Run command: streamlit run app.py
========================================================================================
"""

import streamlit as st
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DSP Signal Analyzer & Filter Design Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Clean High-Contrast Dark UI
st.markdown("""
<style>
    .main { background-color: #020617; color: #f8fafc; }
    .stMetric { background-color: #0f172a; padding: 12px; border-radius: 12px; border: 1px solid #1e293b; }
    h1, h2, h3, h4 { color: #f1f5f9 !important; font-family: system-ui, -apple-system, sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #0f172a; border-radius: 8px 8px 0px 0px; color: #94a3b8; }
    .stTabs [aria-selected="true"] { background-color: #0284c7 !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Signal Generation Helper Functions
# -----------------------------------------------------------------------------
def generate_signal(source_type, duration=3.0, fs=8000, f0=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    equation = ""
    sig = np.zeros_like(t)

    if source_type == "Pure Sine Wave":
        sig = np.sin(2 * np.pi * f0 * t)
        equation = rf"x(t) = \sin(2\pi \cdot {f0} \cdot t)"
    elif source_type == "Multi-Tone Harmonics":
        sig = 0.6 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t) + 0.15 * np.sin(2 * np.pi * 2400 * t)
        equation = r"x(t) = 0.6\sin(2\pi\cdot 300t) + 0.3\sin(2\pi\cdot 1200t) + 0.15\sin(2\pi\cdot 2400t)"
    elif source_type == "Noisy Sine":
        noise = np.random.normal(0, 0.45, len(t))
        sig = np.sin(2 * np.pi * f0 * t) + noise
        equation = rf"x(t) = \sin(2\pi \cdot {f0} \cdot t) + \mathcal{{N}}(0, \sigma^2)"
    elif source_type == "Frequency Chirp":
        sig = signal.chirp(t, f0=100, t1=duration, f1=3500, method='linear')
        equation = r"x(t) = \cos\left(2\pi \left(f_0 t + \frac{k}{2}t^2\right)\right)"
    elif source_type == "Synthetic ECG (Heartbeat)":
        fs_ecg = 250
        t_ecg = np.linspace(0, 10.0, int(fs_ecg * 10.0), endpoint=False)
        sig = np.zeros_like(t_ecg)
        bpm = 72
        period = 60.0 / bpm
        for beat_t in np.arange(0.2, 10.0, period):
            sig += 0.2 * np.exp(-((t_ecg - (beat_t - 0.15)) ** 2) / (2 * 0.04**2))
            sig -= 0.15 * np.exp(-((t_ecg - (beat_t - 0.05)) ** 2) / (2 * 0.015**2))
            sig += 1.2 * np.exp(-((t_ecg - beat_t) ** 2) / (2 * 0.02**2))
            sig -= 0.25 * np.exp(-((t_ecg - (beat_t + 0.05)) ** 2) / (2 * 0.015**2))
            sig += 0.35 * np.exp(-((t_ecg - (beat_t + 0.25)) ** 2) / (2 * 0.06**2))
        return t_ecg, sig, fs_ecg, r"\text{ECG } (P\text{-}Q\text{-}R\text{-}S\text{-}T \text{ Morphological Waves})"
    elif source_type == "Synthetic Voice (Formants)":
        f_fund = 140
        formants = [700, 1220, 2600]
        sig = np.zeros_like(t)
        for h in range(1, 28):
            fh = h * f_fund
            if fh >= fs / 2:
                break
            weight = sum(np.exp(-((fh - f_c) ** 2) / (2 * (120**2))) for f_c in formants) + 0.05
            sig += (weight / h) * np.sin(2 * np.pi * fh * t)
        sig = sig / np.max(np.abs(sig))
        equation = r"x(t) = \sum \frac{A_k}{k} \sin(2\pi \cdot k f_0 \cdot t)"
    elif source_type == "White Gaussian Noise":
        sig = np.random.normal(0, 1.0, len(t))
        equation = r"x(t) \sim \mathcal{N}(0, 1)"
    elif source_type == "Square Wave":
        sig = signal.square(2 * np.pi * f0 * t)
        equation = rf"x(t) = \text{{sgn}}(\sin(2\pi \cdot {f0} \cdot t))"
    elif source_type == "Sawtooth Wave":
        sig = signal.sawtooth(2 * np.pi * f0 * t)
        equation = rf"x(t) = 2 \left(\frac{{t}}{{T}} - \lfloor \frac{{t}}{{T}} \rfloor\right) - 1"
    else:
        sig = np.sin(2 * np.pi * f0 * t)
        equation = "x(t) = \sin(2\pi f_0 t)"

    return t, sig, fs, equation

# -----------------------------------------------------------------------------
# Streamlit Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ DSP Studio Settings")

signal_options = [
    "Synthetic Voice (Formants)",
    "Synthetic ECG (Heartbeat)",
    "Pure Sine Wave",
    "Multi-Tone Harmonics",
    "Noisy Sine",
    "Frequency Chirp",
    "Square Wave",
    "Sawtooth Wave",
    "White Gaussian Noise",
]
selected_signal = st.sidebar.selectbox("Signal Source Generator", signal_options, index=0)

fs_custom = 8000
duration_custom = 3.0
freq_param = 440

if selected_signal in ["Pure Sine Wave", "Noisy Sine", "Square Wave", "Sawtooth Wave"]:
    freq_param = st.sidebar.slider("Fundamental Frequency (Hz)", min_value=20, max_value=2000, value=440, step=10)

t, sig, fs, eq_latex = generate_signal(selected_signal, duration=duration_custom, fs=fs_custom, f0=freq_param)
nyquist = fs / 2.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Digital Filter Parameters")
filter_family = st.sidebar.radio("Filter Family", ["FIR", "IIR (Butterworth)"], index=0)
filter_type = st.sidebar.selectbox("Filter Type", ["lowpass", "highpass", "bandpass", "notch"], index=0)

cutoff1 = st.sidebar.slider("Cutoff Frequency f1 (Hz)", min_value=10, max_value=int(nyquist - 50), value=min(800, int(nyquist * 0.2)), step=10)
cutoff2 = min(int(nyquist - 10), cutoff1 + 600)
if filter_type in ["bandpass", "notch"]:
    cutoff2 = st.sidebar.slider("Upper Cutoff f2 (Hz)", min_value=cutoff1 + 20, max_value=int(nyquist - 10), value=min(int(nyquist - 20), cutoff1 + 600), step=10)

iir_order = 4
fir_window = "hamming"
if filter_family == "IIR (Butterworth)":
    iir_order = st.sidebar.slider("Butterworth Filter Order (N)", min_value=1, max_value=10, value=4, step=1)
else:
    fir_window = st.sidebar.selectbox("FIR Window", ["hamming", "hann", "blackman", "boxcar"], index=0)

# -----------------------------------------------------------------------------
# Filter Design Engine
# -----------------------------------------------------------------------------
b, a = [1.0], [1.0]
is_stable = True

if filter_family == "FIR":
    numtaps = 101
    if filter_type == "lowpass":
        b = signal.firwin(numtaps, cutoff1, fs=fs, window=fir_window, pass_zero='lowpass')
    elif filter_type == "highpass":
        b = signal.firwin(numtaps, cutoff1, fs=fs, window=fir_window, pass_zero='highpass')
    elif filter_type == "bandpass":
        b = signal.firwin(numtaps, [cutoff1, cutoff2], fs=fs, window=fir_window, pass_zero='bandpass')
    elif filter_type == "notch":
        b = signal.firwin(numtaps, [cutoff1, cutoff2], fs=fs, window=fir_window, pass_zero='bandstop')
    a = [1.0]
else:
    # IIR Butterworth Filter
    Wn = cutoff1 / nyquist if filter_type in ["lowpass", "highpass"] else [cutoff1 / nyquist, cutoff2 / nyquist]
    b_iir, a_iir = signal.butter(iir_order, Wn, btype='bandstop' if filter_type == 'notch' else filter_type)
    b, a = b_iir, a_iir

# Zero-Phase Filtering
try:
    filtered_sig = signal.filtfilt(b, a, sig)
except Exception:
    filtered_sig = signal.lfilter(b, a, sig)

# Compute Zeros & Poles
z_roots, p_roots, k_gain = signal.tf2zpk(b, a)
is_stable = np.all(np.abs(p_roots) < 1.0)

# -----------------------------------------------------------------------------
# Main Application Dashboard
# -----------------------------------------------------------------------------
st.title("⚡ Universal DSP Signal Analyzer & Filter Design Studio")
st.markdown("Interactive 1D Signal Processing • 3D FFT Spectrums • 3D Waterfall Spectrograms • 3D Z-Plane Transfer Function Landscapes")

if eq_latex:
    st.info(f"Signal Mathematical Equation: $${eq_latex}$$")

# Metrics Banner
col1, col2, col3, col4, col5 = st.columns(5)
peak_to_peak = np.ptp(sig)
rms_val = np.sqrt(np.mean(sig**2))
crest_factor = (np.max(np.abs(sig)) / rms_val) if rms_val > 1e-12 else 0.0
mean_val = np.mean(sig)
var_val = np.var(sig)

col1.metric("Peak-to-Peak", f"{peak_to_peak:.4f}")
col2.metric("RMS Power", f"{rms_val:.4f}")
col3.metric("Crest Factor", f"{crest_factor:.4f}")
col4.metric("Mean (DC)", f"{mean_val:.4f}")
col5.metric("Variance (σ²)", f"{var_val:.4f}")

# -----------------------------------------------------------------------------
# Main Visualization Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time & Frequency", 
    "🌊 3D Waterfall Spectrogram", 
    "✨ 3D FFT Spectrum Landscape", 
    "🏔️ 3D Z-Plane Transfer Function", 
    "🎛️ Filter Analysis & Poles/Zeros"
])

# -----------------------------------------------------------------------------
# TAB 1: Time & Frequency Domain
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1. Waveform Comparison (Time Domain)")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t[:1000], y=sig[:1000], mode='lines', name='Original Signal', line=dict(color='#64748b', width=1.5)))
    fig_time.add_trace(go.Scatter(x=t[:1000], y=filtered_sig[:1000], mode='lines', name='Filtered Signal', line=dict(color='#38bdf8', width=2)))
    fig_time.update_layout(
        template="plotly_dark",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("2. FFT Magnitude Spectrum")
    freqs = fft.rfftfreq(len(sig), 1.0 / fs)
    fft_orig = np.abs(fft.rfft(sig))
    fft_filt = np.abs(fft.rfft(filtered_sig))

    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs, y=fft_orig, mode='lines', name='Original FFT', line=dict(color='#64748b', width=1.5)))
    fig_fft.add_trace(go.Scatter(x=freqs, y=fft_filt, mode='lines', name='Filtered FFT', line=dict(color='#f43f5e', width=2)))
    fig_fft.update_layout(
        template="plotly_dark",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    st.plotly_chart(fig_fft, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: 3D Rolling Waterfall Spectrogram
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🌊 3D Rolling Waterfall Spectrogram (Time × Frequency × Energy)")
    st.caption("Interactive 3D topographic surface representing continuous frequency evolutions across time.")

    f_stft, t_stft, Zxx = signal.stft(sig, fs=fs, nperseg=min(256, len(sig)//16))
    Z_db = 20 * np.log10(np.abs(Zxx) + 1e-6)

    fig_waterfall = go.Figure(data=[go.Surface(
        x=t_stft,
        y=f_stft,
        z=Z_db,
        colorscale="Turbo",
        colorbar=dict(title="dB")
    )])
    fig_waterfall.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            zaxis_title="Energy (dB)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=620,
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: 3D FFT Spectrum Landscape (X: Freq, Y: Phase, Z: Magnitude)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("✨ 3D Phase-Magnitude Harmonic Constellation")
    st.caption("Viridis colored 3D spectral cloud mapping frequency harmonics, phase rotation angles, and spectral magnitude.")

    fft_complex = fft.rfft(sig)
    fft_mag = np.abs(fft_complex)
    fft_phase = np.angle(fft_complex)

    # Pick top 1500 points for fluid rendering
    top_indices = np.argsort(fft_mag)[-1500:]

    fig_3d_scatter = go.Figure(data=[go.Scatter3d(
        x=freqs[top_indices],
        y=fft_phase[top_indices],
        z=fft_mag[top_indices],
        mode='markers',
        marker=dict(
            size=3,
            color=fft_mag[top_indices],
            colorscale='Viridis',
            opacity=0.85
        )
    )])
    fig_3d_scatter.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Phase (rad)",
            zaxis_title="Magnitude",
            camera=dict(eye=dict(x=1.7, y=-1.5, z=1.3))
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=620,
    )
    st.plotly_chart(fig_3d_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: 3D Complex Z-Plane Transfer Function Landscape |H(z)|
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🏔️ 3D Complex Z-Plane Transfer Function Landscape |H(z)|")
    st.caption("Volcanic peaks reveal filter poles; sunken valleys reveal zeros. The cylinder outline marks the Unit Circle |z| = 1.")

    re_vals = np.linspace(-1.5, 1.5, 60)
    im_vals = np.linspace(-1.5, 1.5, 60)
    RE, IM = np.meshgrid(re_vals, im_vals)
    Z_complex = RE + 1j * IM

    # Evaluate H(z) = B(z) / A(z) where z^-1 is applied
    Hz_mag = np.zeros_like(RE)
    for i in range(RE.shape[0]):
        for j in range(RE.shape[1]):
            z_pt = Z_complex[i, j]
            if np.abs(z_pt) < 1e-4:
                continue
            num = sum(b[k] * (z_pt ** (-k)) for k in range(len(b)))
            den = sum(a[k] * (z_pt ** (-k)) for k in range(len(a)))
            if np.abs(den) < 1e-4:
                Hz_mag[i, j] = 10.0
            else:
                Hz_mag[i, j] = min(10.0, np.abs(num / den))

    fig_zplane_3d = go.Figure(data=[go.Surface(
        x=RE,
        y=IM,
        z=Hz_mag,
        colorscale="Magma",
        colorbar=dict(title="|H(z)|")
    )])

    # Unit circle ring
    theta = np.linspace(0, 2 * np.pi, 100)
    uc_x = np.cos(theta)
    uc_y = np.sin(theta)
    uc_z = np.zeros_like(theta)
    fig_zplane_3d.add_trace(go.Scatter3d(
        x=uc_x, y=uc_y, z=uc_z,
        mode='lines',
        line=dict(color='#38bdf8', width=5),
        name='Unit Circle (|z|=1)'
    ))

    fig_zplane_3d.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Re(z)",
            yaxis_title="Im(z)",
            zaxis_title="|H(z)| Magnitude",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.4))
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=620,
    )
    st.plotly_chart(fig_zplane_3d, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: Filter Analysis & Pole-Zero Diagram
# -----------------------------------------------------------------------------
with tab5:
    col_pz1, col_pz2 = st.columns([1, 1])

    with col_pz1:
        st.subheader("Pole-Zero Diagram (Complex Z-Plane)")
        fig_pz = go.Figure()
        
        # Unit Circle
        theta = np.linspace(0, 2*np.pi, 200)
        fig_pz.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(dash='dash', color='#38bdf8'), name='Unit Circle'))

        # Zeros
        if len(z_roots) > 0:
            fig_pz.add_trace(go.Scatter(x=np.real(z_roots), y=np.imag(z_roots), mode='markers', marker=dict(symbol='circle-open', size=10, color='#38bdf8', line=dict(width=2)), name='Zeros (o)'))
        # Poles
        if len(p_roots) > 0:
            fig_pz.add_trace(go.Scatter(x=np.real(p_roots), y=np.imag(p_roots), mode='markers', marker=dict(symbol='x', size=10, color='#f43f5e', line=dict(width=2)), name='Poles (x)'))

        fig_pz.update_layout(
            template="plotly_dark",
            xaxis_title="Real Part Re(z)",
            yaxis_title="Imaginary Part Im(z)",
            height=400,
            xaxis=dict(range=[-1.6, 1.6]),
            yaxis=dict(range=[-1.6, 1.6], scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_pz, use_container_width=True)

    with col_pz2:
        st.subheader("Filter Frequency Response |H(f)|")
        w, h = signal.freqz(b, a, worN=1024, fs=fs)
        h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=w, y=h_db, mode='lines', line=dict(color='#10b981', width=2), name='|H(f)| dB'))
        fig_h.update_layout(
            template="plotly_dark",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dB)",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_h, use_container_width=True)

    if not is_stable:
        st.error("⚠️ Filter Instability Warning: Calculated IIR poles reside on or outside the Unit Circle (|p| >= 1.0)!")
    else:
        st.success("✅ Filter is BIBO Stable: All poles strictly reside inside the unit circle.")

st.markdown("---")
st.caption("⚡ Universal DSP Signal Studio • Interactive WebGL / Plotly Scientific Visualization")
