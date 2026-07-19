"""
Universal DSP Signal Analyzer & Filter Design Studio - Web App
----------------------------------------------------------------
Run locally:    streamlit run app.py
Deploy free:    push this file + requirements.txt to a GitHub repo,
                then deploy at https://share.streamlit.io (Streamlit
                Community Cloud reads requirements.txt automatically).
"""

import os
import sys
import subprocess
import tempfile
import io
import urllib.request
import uuid

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile, loadmat
from scipy.signal import spectrogram, firwin, butter, lfilter, filtfilt, freqz, square, sawtooth, chirp, tf2zpk

st.set_page_config(page_title="DSP Signal Analyzer", layout="wide")

# ============================================================================
# DSP ENGINE - identical to the notebook version, no Streamlit-specific code
# ============================================================================

def _strip_comment(line):
    return line.split('#', 1)[0]

def _first_content_line(filename):
    with open(filename, 'r') as f:
        for i, raw_line in enumerate(f):
            content = _strip_comment(raw_line).strip()
            if content:
                return content, i + 1
    return None, 0

def _detect_delimiter(filename):
    line, _ = _first_content_line(filename)
    if line is None:
        return ','
    if ',' in line:
        return ','
    elif '\t' in line:
        return '\t'
    else:
        return None

def _header_skip_count(filename, delim):
    line, n_seen = _first_content_line(filename)
    if line is None:
        return 0
    tokens = line.split(delim) if delim else line.split()
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return n_seen
    return 0

def _fill_nans(arr):
    """Interpolates missing (NaN) values to prevent silent math failures."""
    mask = np.isnan(arr)
    if mask.all():
        raise ValueError("File contains no valid numerical data.")
    if mask.any():
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr

def load_csv_signal(filename, fs_override=None):
    delim = _detect_delimiter(filename)
    skip = _header_skip_count(filename, delim)
    raw = np.genfromtxt(filename, delimiter=delim, skip_header=skip)

    if raw.ndim == 1:
        if fs_override is None:
            raise ValueError("Single-column data: sample rate (Hz) is required.")
        return fs_override, _fill_nans(raw).astype(np.float32)
    else:
        t_col, sig_col = _fill_nans(raw[:, 0]), _fill_nans(raw[:, 1])
        dt = np.diff(t_col)
        if dt.size == 0:
            raise ValueError("Time column has only one row - at least 2 samples are needed.")
        if np.all(dt == 0):
            raise ValueError("Time column is constant. Check data or provide sample rate manually.")
        if not np.all(dt > 0):
            raise ValueError("Time column is not monotonically increasing. Check for out-of-order rows.")
        fs = 1.0 / np.mean(dt)
        return fs, sig_col.astype(np.float32)

def load_wav_signal(filename):
    fs, signal = wavfile.read(filename)
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    elif signal.dtype == np.int32:
        signal = signal.astype(np.float32) / 2147483648.0
    elif signal.dtype == np.uint8:
        signal = (signal.astype(np.float32) - 128) / 128.0
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return fs, signal

# --- REAL-WORLD DEMOS ---
def generate_real_audio_demo():
    url = "https://raw.githubusercontent.com/Uberi/speech_recognition/master/examples/english.wav"
    tmp_path = os.path.join(tempfile.gettempdir(), "real_voice_demo_clear.wav")
    
    if not os.path.exists(tmp_path):
        try:
            # 3-second timeout to prevent app hanging
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3.0) as response, open(tmp_path, 'wb') as out_file:
                out_file.write(response.read())
            return load_wav_signal(tmp_path)
        except Exception:
            # DEPENDENCY FALLBACK: Mathematically synthesize a human-like vowel sound (/a/)
            fs = 8000
            t = np.linspace(0, 3.0, int(fs * 3.0), endpoint=False)
            source = sawtooth(2 * np.pi * 120 * t) # Glottal pitch (120 Hz male)
            b1, a1 = butter(2, [600/(fs/2), 800/(fs/2)], btype='bandpass')
            b2, a2 = butter(2, [1000/(fs/2), 1200/(fs/2)], btype='bandpass')
            b3, a3 = butter(2, [2300/(fs/2), 2700/(fs/2)], btype='bandpass')
            synthetic_voice = lfilter(b1, a1, source) + 0.5 * lfilter(b2, a2, source) + 0.1 * lfilter(b3, a3, source)
            return fs, (synthetic_voice / np.max(np.abs(synthetic_voice))).astype(np.float32)
    return load_wav_signal(tmp_path)

def generate_ecg_demo(duration=10.0, fs=250, heart_rate_bpm=72):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    beat_period = 60.0 / heart_rate_bpm
    sig = np.zeros_like(t)
    for beat_start in np.arange(0, duration, beat_period):
        sig += 0.15 * np.exp(-((t - (beat_start + 0.10)) ** 2) / (2 * 0.020 ** 2))
        sig += 1.00 * np.exp(-((t - (beat_start + 0.20)) ** 2) / (2 * 0.005 ** 2))
        sig += 0.30 * np.exp(-((t - (beat_start + 0.35)) ** 2) / (2 * 0.040 ** 2))
    sig += 0.02 * np.random.randn(len(t))
    sig += 0.05 * np.sin(2 * np.pi * 0.3 * t)
    return fs, sig.astype(np.float32)

def generate_sensor_demo(duration=20.0, fs=100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 15 * t)
    sig += 0.05 * np.random.randn(len(t)) + 0.1 * t / duration
    return fs, sig.astype(np.float32)

def gen_pure_sine(duration=3.0, fs=8000, freq=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return fs, np.sin(2 * np.pi * freq * t).astype(np.float32)

def gen_multi_tone(duration=3.0, fs=8000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t) + 0.25 * np.sin(2 * np.pi * 1320 * t)
    return fs, (sig / np.max(np.abs(sig))).astype(np.float32)

def gen_noisy_sine(duration=3.0, fs=8000, freq=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(len(t))
    return fs, (sig / np.max(np.abs(sig))).astype(np.float32)

def gen_square(duration=3.0, fs=8000, freq=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return fs, square(2 * np.pi * freq * t).astype(np.float32)

def gen_triangle(duration=3.0, fs=8000, freq=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return fs, sawtooth(2 * np.pi * freq * t, width=0.5).astype(np.float32)

def gen_sawtooth(duration=3.0, fs=8000, freq=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return fs, sawtooth(2 * np.pi * freq * t).astype(np.float32)

def gen_chirp(duration=3.0, fs=8000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = chirp(t, f0=20, f1=2000, t1=duration, method='linear')
    return fs, sig.astype(np.float32)

def gen_white_noise(duration=3.0, fs=8000):
    sig = np.random.randn(int(fs * duration))
    return fs, (sig / np.max(np.abs(sig))).astype(np.float32)

def gen_impulse(duration=3.0, fs=8000):
    sig = np.zeros(int(fs * duration))
    sig[len(sig)//2] = 1.0 
    return fs, sig.astype(np.float32)

def gen_step(duration=3.0, fs=8000):
    sig = np.zeros(int(fs * duration))
    sig[len(sig)//2:] = 1.0
    return fs, sig.astype(np.float32)
# ----------------------------------------------

def _ensure_package(pkg_name, import_name=None):
    import_name = import_name or pkg_name
    try:
        return __import__(import_name)
    except ImportError:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg_name],
                                 capture_output=True, text=True)
        if result.returncode != 0 and 'externally-managed-environment' in result.stderr:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                                      '--break-system-packages', pkg_name],
                                     capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Couldn't install {pkg_name}:\n{result.stderr.strip()[-500:]}")
        return __import__(import_name)

AUDIO_EXTS = {'.mp3', '.m4a', '.aac', '.flac', '.ogg', '.aiff', '.aif', '.wma'}

def convert_audio_to_wav(input_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '_converted.wav'
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-acodec', 'pcm_s16le', output_path],
            capture_output=True, text=True
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed (or not on PATH) in this environment. Upload a .wav file instead.")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't convert {input_path}:\n{result.stderr.strip()[-500:]}")
    return output_path

def _is_hdf5(filename):
    with open(filename, 'rb') as f:
        return f.read(8) == b'\x89HDF\r\n\x1a\n'

def load_mat_signal(filename, fs_override=None, var_name=None):
    use_h5py = _is_hdf5(filename)
    if not use_h5py:
        try:
            data = loadmat(filename)
        except (NotImplementedError, ValueError) as e:
            if isinstance(e, NotImplementedError) or 'unknown mat file type' in str(e).lower():
                use_h5py = True
            else:
                raise

    if use_h5py:
        h5py = _ensure_package('h5py')
        data = {}
        with h5py.File(filename, 'r') as f:
            for k in f.keys():
                data[k] = np.array(f[k]).squeeze()

    keys = [k for k in data if not k.startswith('__')]
    if var_name is not None:
        sig = np.asarray(data[var_name]).squeeze()
    else:
        sig_key = max(keys, key=lambda k: np.asarray(data[k]).size)
        sig = np.asarray(data[sig_key]).squeeze()

    fs = None
    for k in keys:
        if any(tag in k.lower() for tag in ('fs', 'rate', 'freq')):
            val = np.asarray(data[k]).squeeze()
            if val.size == 1:
                fs = float(val)
                break
    if fs is None:
        fs = fs_override
    if fs is None:
        raise ValueError(f"Couldn't auto-detect a sample rate (variables found: {keys}).")
    return fs, _fill_nans(sig).astype(np.float32)

def load_edf_signal(filename, channel=0):
    pyedflib = _ensure_package('pyedflib')
    f = pyedflib.EdfReader(filename)
    try:
        n_channels = f.signals_in_file
        if channel >= n_channels:
            raise ValueError(f"channel={channel} out of range - file has {n_channels}: {f.getSignalLabels()}")
        signal = f.readSignal(channel)
        fs = f.getSampleFrequency(channel)
    finally:
        f.close()
    return fs, _fill_nans(signal).astype(np.float32)

def load_any_signal(filename, fs_override=None, mat_var_name=None, edf_channel=0):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.wav':
        return load_wav_signal(filename)
    elif ext in AUDIO_EXTS:
        return load_wav_signal(convert_audio_to_wav(filename))
    elif ext in ('.csv', '.txt'):
        return load_csv_signal(filename, fs_override)
    elif ext == '.mat':
        return load_mat_signal(filename, fs_override, mat_var_name)
    elif ext in ('.edf', '.bdf'):
        return load_edf_signal(filename, edf_channel)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def design_fir(filter_type, fs, cutoff, cutoff2=None, numtaps=101, window='hamming'):
    nyq = fs / 2
    if filter_type == 'lowpass':
        taps = firwin(numtaps, cutoff / nyq, window=window)
    elif filter_type == 'highpass':
        taps = firwin(numtaps, cutoff / nyq, pass_zero=False, window=window)
    elif filter_type == 'bandpass':
        taps = firwin(numtaps, [cutoff / nyq, cutoff2 / nyq], pass_zero=False, window=window)
    elif filter_type == 'notch':
        taps = firwin(numtaps, [cutoff / nyq, cutoff2 / nyq], pass_zero=True, window=window)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    return taps, [1.0]

def design_iir(filter_type, fs, cutoff, cutoff2=None, order=4):
    nyq = fs / 2
    btype_map = {'lowpass': 'low', 'highpass': 'high', 'bandpass': 'band', 'notch': 'bandstop'}
    if filter_type in ('lowpass', 'highpass'):
        b, a = butter(order, cutoff / nyq, btype=btype_map[filter_type])
    else:
        b, a = butter(order, [cutoff / nyq, cutoff2 / nyq], btype=btype_map[filter_type])
    return b, a

def apply_filter(b, a, signal):
    padlen = 3 * max(len(a), len(b))
    if len(signal) > padlen:
        return filtfilt(b, a, signal)
    st.warning(f"Signal too short for zero-phase filtering (needs > {padlen} samples) - using standard filtering.")
    return lfilter(b, a, signal)

# ============================================================================
# WEB UI
# ============================================================================

st.title("Universal DSP Signal Analyzer & Filter Design Studio")

st.sidebar.header("1. Signal Source")
source_options = [
    "Upload a file",
    "Demo: Real Audio (Voice)",
    "Demo: ECG",
    "Demo: Sensor",
    "Demo: Pure Sine Wave",
    "Demo: Multi-Tone Signal",
    "Demo: Noisy Sine Wave",
    "Demo: Square Wave",
    "Demo: Triangle Wave",
    "Demo: Sawtooth Wave",
    "Demo: Chirp Signal",
    "Demo: White Noise",
    "Demo: Impulse (Delta) Signal",
    "Demo: Step Signal"
]
source = st.sidebar.selectbox("Source", source_options)

fs = signal = None

if source == "Upload a file":
    # 1. Create the popover and make the button perfectly fit the sidebar width
    with st.sidebar.popover("ℹ️ View Supported File Types", use_container_width=True):
        
        # 2. Use a tight HTML layout instead of bulky Streamlit columns
        st.markdown(
            """
            <div style="display: flex; gap: 40px;">
                <div>
                    <h3 style="margin: 0px 0px 10px 0px;">Audio</h3>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>mp3</li><li>wav</li><li>m4a</li><li>aac</li>
                        <li>flac</li><li>ogg</li><li>aiff</li><li>wma</li>
                    </ul>
                </div>
                <div>
                    <h3 style="margin: 0px 0px 10px 0px;">Data</h3>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>csv</li><li>txt</li><li>mat</li><li>edf</li><li>bdf</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 3. Add the tiny micro-gap 
    st.sidebar.markdown('<p style="margin-bottom: 4px;"></p>', unsafe_allow_html=True)
    
    # 4. Draw the upload box
    uploaded = st.sidebar.file_uploader("hidden_label", label_visibility="collapsed")
    
    if uploaded is not None:
        ext = os.path.splitext(uploaded.name)[1].lower()
        # FIX 1: Generate a unique ID to prevent multi-user file overwrite collisions
        safe_filename = f"{uuid.uuid4().hex}_{uploaded.name}"
        tmp_path = os.path.join(tempfile.gettempdir(), safe_filename)
        
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        fs_override = None
        if ext in ('.csv', '.txt'):
            manual_fs = st.sidebar.number_input(
                "Sample rate in Hz (only needed for single-column CSV)",
                min_value=0, value=0, step=1
            )
            fs_override = float(manual_fs) if manual_fs > 0 else None

        try:
            fs, signal = load_any_signal(tmp_path, fs_override=fs_override)
        except ValueError as e:
            if ext == '.mat':
                st.sidebar.warning(f"{e} Enter a sample rate below to proceed.")
                manual_fs = st.sidebar.number_input(
                    "Sample rate in Hz", min_value=0, value=0, step=1, key="mat_fs_fallback"
                )
                if manual_fs <= 0:
                    st.stop()
                try:
                    fs, signal = load_any_signal(tmp_path, fs_override=float(manual_fs))
                except Exception as e2:
                    st.sidebar.error(str(e2))
                    st.stop()
            else:
                st.sidebar.error(str(e))
                st.stop()
        except Exception as e:
            st.sidebar.error(str(e))
            st.stop()
    else:
        st.info("Upload a file in the sidebar, or pick a demo source to try it instantly.")
        st.stop()
elif source == "Demo: Real Audio (Voice)":
    with st.spinner("Loading real audio sample..."):
        try:
            fs, signal = generate_real_audio_demo()
        except Exception as e:
            st.sidebar.error(str(e))
            st.stop()
elif source == "Demo: ECG":
    fs, signal = generate_ecg_demo()
elif source == "Demo: Sensor":
    fs, signal = generate_sensor_demo()
elif source == "Demo: Pure Sine Wave":
    fs, signal = gen_pure_sine()
elif source == "Demo: Multi-Tone Signal":
    fs, signal = gen_multi_tone()
elif source == "Demo: Noisy Sine Wave":
    fs, signal = gen_noisy_sine()
elif source == "Demo: Square Wave":
    fs, signal = gen_square()
elif source == "Demo: Triangle Wave":
    fs, signal = gen_triangle()
elif source == "Demo: Sawtooth Wave":
    fs, signal = gen_sawtooth()
elif source == "Demo: Chirp Signal":
    fs, signal = gen_chirp()
elif source == "Demo: White Noise":
    fs, signal = gen_white_noise()
elif source == "Demo: Impulse (Delta) Signal":
    fs, signal = gen_impulse()
elif source == "Demo: Step Signal":
    fs, signal = gen_step()

st.sidebar.success(f"{fs:.1f} Hz  |  {len(signal) / fs:.2f} sec  |  {len(signal)} samples")

full_signal = signal

st.sidebar.markdown("---")
st.sidebar.header("2. Trim Signal (Optional)")
full_duration = float(len(full_signal) / fs)
trim_start, trim_end = st.sidebar.slider(
    "Select time range (seconds)", min_value=0.0, max_value=full_duration, value=(0.0, full_duration)
)

if trim_end > trim_start:
    start_sample = int(trim_start * fs)
    end_sample = int(trim_end * fs)
    signal = full_signal[start_sample:end_sample]
    st.sidebar.caption(f"✂️ Analyzing: {len(signal)} samples ({len(signal)/fs:.2f}s)")
else:
    st.sidebar.error("Start time must be strictly before end time.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("3. Export Settings")
graph_format = st.sidebar.selectbox("Graph Format", ["PNG", "PDF", "SVG"]).lower()
audio_format = st.sidebar.selectbox("Audio Format", ["WAV", "MP3", "FLAC"]).lower()

graph_mime = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}.get(graph_format, f"image/{graph_format}")
audio_mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(audio_format, f"audio/{audio_format}")

st.header("Signal Metrics & Statistics")
peak_to_peak = float(np.max(signal) - np.min(signal))
rms = float(np.sqrt(np.mean(np.square(signal))))
peak_abs = float(np.max(np.abs(signal)))
crest_factor = (peak_abs / rms) if rms > 0 else 0.0
mean_val = float(np.mean(signal))
variance_val = float(np.var(signal))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Peak-to-Peak Amplitude", f"{peak_to_peak:.4g}", help="Difference between the maximum and minimum values.")
m2.metric("RMS Power", f"{rms:.4g}", help="Root Mean Square - the effective amplitude of the signal.")
m3.metric("Crest Factor", f"{crest_factor:.4g}", help="Peak amplitude divided by RMS.")
m4.metric("Mean", f"{mean_val:.4g}", help="Average sample value (DC offset).")
m5.metric("Variance", f"{variance_val:.4g}", help="Average squared deviation from the mean.")

n = len(signal)
t = np.arange(n) / fs
freqs = np.fft.rfftfreq(n, 1 / fs)
magnitude = np.abs(np.fft.rfft(signal)) / n

st.header("Signal Analysis")
fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8))
axs1[0].plot(t + trim_start, signal, linewidth=0.7)
axs1[0].set_title("Waveform (Time Domain)")
axs1[0].set_xlabel("Time (s)"); axs1[0].set_ylabel("Amplitude")

axs1[1].plot(freqs, magnitude, linewidth=0.7)
axs1[1].set_title("Magnitude Spectrum (FFT)")
axs1[1].set_xlabel("Frequency (Hz)"); axs1[1].set_ylabel("Magnitude")
axs1[1].set_xlim(0, fs / 2)

# FIX 2: Guardrail - Only compute Spectrogram if we have enough data points
if n >= 256:
    f_spec, t_spec, Sxx = spectrogram(signal, fs, nperseg=min(1024, n), noverlap=min(512, n // 2))
    im = axs1[2].pcolormesh(t_spec + trim_start, f_spec, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    axs1[2].set_title("Spectrogram")
    axs1[2].set_xlabel("Time (s)")
    axs1[2].set_ylabel("Frequency (Hz)")
    fig1.colorbar(im, ax=axs1[2], label="dB")
else:
    axs1[2].text(0.5, 0.5, "Not enough data for Spectrogram", ha='center', va='center', fontsize=12)
    axs1[2].set_axis_off()

plt.tight_layout()
st.pyplot(fig1)

buf1 = io.BytesIO()
fig1.savefig(buf1, format=graph_format)
st.download_button(
    label=f"📥 Download Analysis Graph ({graph_format.upper()})", data=buf1.getvalue(),
    file_name=f"signal_analysis.{graph_format}", mime=graph_mime
)
plt.close(fig1)

st.header("Filter Design")
c1, c2, c3 = st.columns(3)
family = c1.selectbox("Family", ["FIR", "IIR"])
ftype = c2.selectbox("Filter type", ["lowpass", "highpass", "bandpass", "notch"])

window_type = "hamming"
iir_order = 4
if family == "FIR":
    window_type = c3.selectbox("Window Function", ["hamming", "hann", "blackman", "bartlett", "boxcar"])
else:
    iir_order = c3.number_input("Filter Order", min_value=1, max_value=12, value=4, step=1)

nyq = fs / 2
smax = round(nyq * 0.98, 1)
if smax < 1.0:
    st.error("Sample rate is too low to design a filter here. Check the detected/entered sample rate.")
    st.stop()
cutoff = st.slider("Cutoff (Hz)", min_value=1.0, max_value=float(smax), value=float(min(200.0, smax * 0.2)))
cutoff2 = None
needs_band = ftype in ("bandpass", "notch")
if needs_band:
    cutoff2 = st.slider("Cutoff2 (Hz)", min_value=1.0, max_value=float(smax), value=float(min(1500.0, smax * 0.5)))
    if cutoff2 <= cutoff:
        st.error(f"Cutoff2 ({cutoff2:.0f} Hz) must be greater than Cutoff ({cutoff:.0f} Hz).")
        st.stop()

if family == "FIR":
    b, a = design_fir(ftype, fs, cutoff, cutoff2, window=window_type)
else:
    b, a = design_iir(ftype, fs, cutoff, cutoff2, order=iir_order)

# Extract Poles and check for IIR stability explicitly
z, p, k = tf2zpk(b, a)
if len(p) > 0 and np.any(np.abs(p) >= 1.0):
    st.error("⚠️ **Filter Instability Detected!** The calculated poles fall on or outside the Unit Circle. This IIR filter will mathematically explode. Please reduce the Filter Order or adjust the Cutoff frequency.")
    st.stop()  # <--- THIS PREVENTS THE CRASH

try:
    filtered = apply_filter(b, a, signal)
except Exception as e:
    st.error(f"Filtering failed (likely due to instability): {e}")
    st.stop()

freqs_f = np.fft.rfftfreq(len(filtered), 1 / fs)
mag_f = np.abs(np.fft.rfft(filtered)) / len(filtered)
w, h = freqz(b, a, worN=2048, fs=fs)

title = f"{family} {ftype}  cutoff={cutoff:.0f}Hz" + (f", {cutoff2:.0f}Hz" if needs_band else "")

fig2, axs2 = plt.subplots(4, 1, figsize=(10, 12))
axs2[0].plot(t + trim_start, signal, label="Original", alpha=0.6)
axs2[0].plot(t + trim_start, filtered, label="Filtered", alpha=0.8)
axs2[0].set_title(title + " - Waveform")
axs2[0].set_xlabel("Time (s)"); axs2[0].legend()

axs2[1].plot(freqs, magnitude, label="Original", alpha=0.6)
axs2[1].plot(freqs_f, mag_f, label="Filtered", alpha=0.8)
axs2[1].set_title("Spectrum Comparison")
axs2[1].set_xlabel("Frequency (Hz)"); axs2[1].set_xlim(0, fs / 2); axs2[1].legend()

# FIX 3: Use 20*log10 for standard Amplitude dB, not 40
axs2[2].plot(w, 20 * np.log10(np.abs(h) + 1e-12))
axs2[2].set_title(f"{family} {ftype} Effective Frequency Response")
axs2[2].set_xlabel("Frequency (Hz)"); axs2[2].set_xlim(0, fs / 2); axs2[2].grid(alpha=0.3)

axs2[3].add_patch(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5, label='Unit Circle'))

# FIX 4: Protect the Zeros array from throwing empty array errors
if len(z) > 0:
    axs2[3].scatter(np.real(z), np.imag(z), marker='o', edgecolors='b', facecolors='none', label='Zeros (O)')
if len(p) > 0:
    axs2[3].scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles (X)')
axs2[3].axhline(0, color='gray', alpha=0.3)
axs2[3].axvline(0, color='gray', alpha=0.3)
axs2[3].set_title("Filter Stability: Pole-Zero Plot (Z-Plane)")
axs2[3].set_xlabel("Real Axis"); axs2[3].set_ylabel("Imaginary Axis")
axs2[3].axis('equal') 
axs2[3].legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig2)

buf2 = io.BytesIO()
fig2.savefig(buf2, format=graph_format)
st.download_button(
    label=f"📥 Download Filter Graph ({graph_format.upper()})", data=buf2.getvalue(),
    file_name=f"filter_comparison.{graph_format}", mime=graph_mime
)
plt.close(fig2)

st.header("Listen")

# Global Normalization (Prevents trimmed volume discrepancies)
global_peak = max(np.max(np.abs(full_signal)), np.max(np.abs(filtered)), 1e-9)

full_play_int = np.int16((full_signal / global_peak) * 32767)
filt_play_int = np.int16((filtered / global_peak) * 32767)

ac1, ac2 = st.columns(2)
with ac1:
    st.caption("Original (Full File)")
    st.audio(full_play_int, sample_rate=int(fs))
with ac2:
    st.caption(f"Filtered & Trimmed ({title})")
    st.audio(filt_play_int, sample_rate=int(fs))

    if audio_format == "wav":
        wav_buf = io.BytesIO()
        wavfile.write(wav_buf, int(fs), filt_play_int)
        audio_data = wav_buf.getvalue()
    else:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wavfile.write(tmp_wav.name, int(fs), filt_play_int)
            tmp_out = tmp_wav.name.replace('.wav', f'.{audio_format}')
            try:
                result = subprocess.run(['ffmpeg', '-y', '-i', tmp_wav.name, tmp_out], capture_output=True, text=True)
            except FileNotFoundError:
                os.remove(tmp_wav.name)
                st.error("ffmpeg is not installed. Choose WAV instead.")
                st.stop()
            if result.returncode != 0 or not os.path.exists(tmp_out):
                os.remove(tmp_wav.name)
                st.error(f"Encoding failed: {result.stderr.strip()[-500:]}")
                st.stop()
            with open(tmp_out, 'rb') as f_out:
                audio_data = f_out.read()
            os.remove(tmp_wav.name)
            os.remove(tmp_out)

    st.download_button(
        label=f"🎵 Download Filtered Audio ({audio_format.upper()})", data=audio_data,
        file_name=f"filtered_audio.{audio_format}", mime=audio_mime
    )
