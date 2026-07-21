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
from scipy import ndimage
from PIL import Image

st.set_page_config(page_title="DSP Signal Analyzer", layout="wide")

# ============================================================================
# UI CLEANUP - Hide GitHub Icon and Streamlit Menu (Keeping Sidebar Toggle!)
# ============================================================================
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Hides the top-right toolbar (GitHub, Fork, Deploy, Menu) but keeps the sidebar toggle */
    [data-testid="stToolbar"] {display: none !important;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ============================================================================
# DSP ENGINE - Pure Logic (No Streamlit UI inside these functions)
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

def _fill_nans(arr, label="data"):
    """Interpolates ISOLATED missing (NaN) values. Refuses to guess if too much
    of the column is missing."""
    mask = np.isnan(arr)
    n_missing = int(mask.sum())
    if n_missing == 0:
        return arr, 0
    frac_missing = n_missing / len(arr)
    if frac_missing >= 1.0:
        raise ValueError(f"The {label} column contains no valid numerical data.")
    if frac_missing > 0.10:
        raise ValueError(
            f"{n_missing} of {len(arr)} values ({frac_missing:.0%}) in the {label} "
            f"column are missing - too many to interpolate reliably. Check the file."
        )
    arr = arr.copy()
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr, n_missing

def load_csv_signal(filename, fs_override=None):
    """Returns (fs, signal, n_interpolated)."""
    delim = _detect_delimiter(filename)
    skip = _header_skip_count(filename, delim)
    try:
        raw = np.genfromtxt(filename, delimiter=delim, skip_header=skip)
    except Exception:
        raise ValueError("Could not parse CSV/TXT data safely. Ensure data is numeric.")

    if raw.ndim == 1:
        if fs_override is None:
            raise ValueError("Single-column data: sample rate (Hz) is required.")
        filled, n_missing = _fill_nans(raw, "value")
        return fs_override, filled.astype(np.float32), n_missing
    else:
        t_col, n_missing_t = _fill_nans(raw[:, 0], "time")
        sig_col, n_missing_v = _fill_nans(raw[:, 1], "value")
        dt = np.diff(t_col)
        if dt.size == 0:
            raise ValueError("Time column has only one row - at least 2 samples are needed.")
        if np.all(dt == 0):
            raise ValueError("Time column is constant. Check data or provide sample rate manually.")
        if not np.all(dt > 0):
            raise ValueError("Time column is not monotonically increasing. Check for out-of-order rows.")
        fs = 1.0 / np.mean(dt)
        return fs, sig_col.astype(np.float32), n_missing_t + n_missing_v

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
    fail_marker = tmp_path + ".failed"

    def _synthetic_fallback():
        fs = 8000
        t = np.linspace(0, 3.0, int(fs * 3.0), endpoint=False)
        source = sawtooth(2 * np.pi * 120 * t)
        b1, a1 = butter(2, [600/(fs/2), 800/(fs/2)], btype='bandpass')
        b2, a2 = butter(2, [1000/(fs/2), 1200/(fs/2)], btype='bandpass')
        b3, a3 = butter(2, [2300/(fs/2), 2700/(fs/2)], btype='bandpass')
        synthetic_voice = lfilter(b1, a1, source) + 0.5 * lfilter(b2, a2, source) + 0.1 * lfilter(b3, a3, source)
        return fs, (synthetic_voice / np.max(np.abs(synthetic_voice))).astype(np.float32)

    if os.path.exists(fail_marker):
        return _synthetic_fallback()
    if os.path.exists(tmp_path):
        return load_wav_signal(tmp_path)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3.0) as response, open(tmp_path, 'wb') as out_file:
            out_file.write(response.read())
        return load_wav_signal(tmp_path)
    except Exception:
        open(fail_marker, 'w').close()  
        return _synthetic_fallback()

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
        raise RuntimeError("FFmpeg is not installed on this system. Upload a .wav file instead.")
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

    detected_fs = None
    for k in keys:
        if any(tag in k.lower() for tag in ('fs', 'rate', 'freq', 'samp')):
            val = np.asarray(data[k]).squeeze()
            if val.size == 1:
                detected_fs = float(val)
                break

    warning = None
    if fs_override is not None:
        fs = fs_override
        if detected_fs is not None and detected_fs != fs_override:
            warning = (f"Using your entered rate ({fs_override} Hz) - the file also embeds "
                       f"{detected_fs} Hz, which was NOT used. Double check which is correct.")
    elif detected_fs is not None:
        fs = detected_fs
    else:
        raise ValueError(f"Couldn't auto-detect a sample rate (variables found: {keys}).")

    filled, n_missing = _fill_nans(sig, "signal")
    return fs, filled.astype(np.float32), n_missing, warning

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
    filled, n_missing = _fill_nans(signal, "signal")
    return fs, filled.astype(np.float32), n_missing

def load_any_signal(filename, fs_override=None, mat_var_name=None, edf_channel=0):
    """Always returns (fs, signal, n_interpolated, warning) - the last two are
    0/None for loaders that don't have those concepts (wav/edf)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.wav':
        fs, sig = load_wav_signal(filename)
        return fs, sig, 0, None
    elif ext in AUDIO_EXTS:
        fs, sig = load_wav_signal(convert_audio_to_wav(filename))
        return fs, sig, 0, None
    elif ext in ('.csv', '.txt'):
        fs, sig, n_missing = load_csv_signal(filename, fs_override)
        return fs, sig, n_missing, None
    elif ext == '.mat':
        return load_mat_signal(filename, fs_override, mat_var_name)
    elif ext in ('.edf', '.bdf'):
        fs, sig, n_missing = load_edf_signal(filename, edf_channel)
        return fs, sig, n_missing, None
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
    """Applies filter and returns (filtered_signal, fallback_warning_flag)"""
    padlen = 3 * max(len(a), len(b))
    if len(signal) > padlen:
        return filtfilt(b, a, signal), False
    # If signal is too short for zero-phase filtfilt, fallback to lfilter
    return lfilter(b, a, signal), True


# ============================================================================
# 2D IMAGE DSP ENGINE 
# ============================================================================

@st.cache_data(show_spinner=False)
def load_image_array(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)

@st.cache_data(show_spinner=False)
def gen_zone_plate(size=512):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = X**2 + Y**2
    Z = np.sin(50 * R * np.pi)
    Z = ((Z + 1) * 127.5).astype(np.uint8)
    return np.stack([Z, Z, Z], axis=-1)

@st.cache_data(show_spinner=False)
def gen_2d_grating(size=512):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(20 * np.pi * X) + np.cos(20 * np.pi * Y)  
    Z = ((Z + 2) / 4 * 255).astype(np.uint8)
    return np.stack([Z, Z, Z], axis=-1)

@st.cache_data(show_spinner="Detecting edges...")
def apply_sobel_edge_detection(image_array):
    grayscale = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    gx = ndimage.sobel(grayscale, axis=1, mode="reflect")
    gy = ndimage.sobel(grayscale, axis=0, mode="reflect")
    magnitude = np.hypot(gx, gy)
    peak = magnitude.max()
    if peak > 0:
        magnitude = (magnitude / peak) * 255.0
    return magnitude.astype(np.uint8)

@st.cache_data(show_spinner="Applying Gaussian blur...")
def apply_gaussian_blur(image_array, sigma):
    blurred = ndimage.gaussian_filter(
        image_array.astype(np.float64), sigma=(sigma, sigma, 0), mode="reflect"
    )
    return np.clip(blurred, 0, 255).astype(np.uint8)

@st.cache_data(show_spinner="Sharpening...")
def apply_sharpening(image_array):
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ])
    sharpened = np.empty_like(image_array, dtype=np.float64)
    for channel in range(image_array.shape[2]):
        sharpened[..., channel] = ndimage.convolve(
            image_array[..., channel].astype(np.float64), sharpen_kernel, mode="reflect"
        )
    return np.clip(sharpened, 0, 255).astype(np.uint8)

@st.cache_data(show_spinner="Computing 2D FFT...")
def apply_fft2d(image_array):
    gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1e-8)
    spread = magnitude_spectrum.max() - magnitude_spectrum.min()
    if spread < 1e-9:
        return np.full(gray.shape, 128, dtype=np.uint8)
    norm_mag = (magnitude_spectrum - magnitude_spectrum.min()) / spread * 255
    return norm_mag.astype(np.uint8)

@st.cache_data(show_spinner="Applying Median Filter...")
def apply_median_filter(image_array, size):
    return ndimage.median_filter(image_array, size=(size, size, 1))

@st.cache_data(show_spinner="Applying Thresholding...")
def apply_binarization(image_array, threshold):
    gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    bw = (gray > threshold).astype(np.uint8) * 255
    return np.stack([bw, bw, bw], axis=-1)

@st.cache_data(show_spinner="Applying Erosion...")
def apply_erosion(image_array, size):
    return ndimage.grey_erosion(image_array, size=(size, size, 1))

@st.cache_data(show_spinner="Applying Dilation...")
def apply_dilation(image_array, size):
    return ndimage.grey_dilation(image_array, size=(size, size, 1))


# ============================================================================
# WEB UI
# ============================================================================

st.title("Universal DSP Signal Analyzer & Filter Design Studio")

app_mode = st.sidebar.selectbox(
    "🎛️ Select Studio Mode",
    ["📈 1D Signal Studio", "🖼️ 2D Image Studio"]
)
st.sidebar.markdown("---")

# ============================================================================
# 1D SIGNAL STUDIO
# ============================================================================
if app_mode == "📈 1D Signal Studio":
    
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
        with st.sidebar.popover("ℹ️ View Supported File Types", use_container_width=True):
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

        st.sidebar.markdown('<p style="margin-bottom: 4px;"></p>', unsafe_allow_html=True)
        uploaded = st.sidebar.file_uploader("hidden_label", label_visibility="collapsed")

        if uploaded is not None:
            ext = os.path.splitext(uploaded.name)[1].lower()
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
                fs, signal, n_interp, fs_warning = load_any_signal(tmp_path, fs_override=fs_override)
                if n_interp:
                    st.sidebar.warning(f"Interpolated {n_interp} missing value(s) found in the file.")
                if fs_warning:
                    st.sidebar.warning(fs_warning)
            except ValueError as e:
                if ext == '.mat':
                    st.sidebar.warning(f"{e} Enter a sample rate below to proceed.")
                    manual_fs = st.sidebar.number_input(
                        "Sample rate in Hz", min_value=0, value=0, step=1, key="mat_fs_fallback"
                    )
                    if manual_fs <= 0:
                        st.stop()
                    try:
                        fs, signal, n_interp, fs_warning = load_any_signal(tmp_path, fs_override=float(manual_fs))
                        if n_interp:
                            st.sidebar.warning(f"Interpolated {n_interp} missing value(s) found in the file.")
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
    st.sidebar.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)

    graph_mime = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}.get(graph_format, f"image/{graph_format}")
    audio_mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(audio_format, f"audio/{audio_format}")

    # --- DYNAMIC SIGNAL EQUATIONS ---
    equation = None
    if "Impulse" in source:
        equation = r"x[n] = \delta\left[n - \frac{N}{2}\right]"
    elif "Step" in source:
        equation = r"x[n] = u\left[n - \frac{N}{2}\right]"
    elif "Pure Sine" in source:
        equation = r"x(t) = \sin(2\pi f_0 t)"
    elif "Multi-Tone" in source:
        equation = r"x(t) = \sin(2\pi f_1 t) + 0.5\sin(2\pi f_2 t) + 0.25\sin(2\pi f_3 t)"
    elif "Square" in source:
        equation = r"x(t) = \operatorname{sgn}(\sin(2\pi f_0 t))"
    elif "Triangle" in source:
        equation = r"x(t) = \frac{2}{\pi} \arcsin(\sin(2\pi f_0 t))"
    elif "Sawtooth" in source:
        equation = r"x(t) = 2 \left( \frac{t}{T} - \left\lfloor \frac{t}{T} + \frac{1}{2} \right\rfloor \right)"
    elif "Chirp" in source:
        equation = r"x(t) = \sin\left(2\pi \left( f_0 + \frac{f_1 - f_0}{2T}t \right) t \right)"
    elif "White Noise" in source:
        equation = r"x[n] \sim \mathcal{N}(0, \sigma^2)"
    elif "Noisy Sine" in source:
        equation = r"x(t) = \sin(2\pi f_0 t) + \mathcal{N}(0, \sigma^2)"

    if equation:
        st.markdown(f"### Signal Equation &nbsp;&nbsp;&nbsp; ${equation}$")

    st.header("Signal Metrics & Statistics")
    peak_to_peak = float(np.max(signal) - np.min(signal))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak_abs = float(np.max(np.abs(signal)))
    crest_factor = (peak_abs / rms) if rms > 0 else 0.0
    mean_val = float(np.mean(signal))
    variance_val = float(np.var(signal))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Peak-to-Peak Amplitude", f"{peak_to_peak:.4g}")
    m2.metric("RMS Power", f"{rms:.4g}")
    m3.metric("Crest Factor", f"{crest_factor:.4g}")
    m4.metric("Mean", f"{mean_val:.4g}")
    m5.metric("Variance", f"{variance_val:.4g}")

    n = len(signal)
    t = np.arange(n) / fs
    freqs = np.fft.rfftfreq(n, 1 / fs)
    
    if source == "Demo: Impulse (Delta) Signal":
        magnitude = np.abs(np.fft.rfft(signal))
    else:
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

    z, p, k = tf2zpk(b, a)
    
    if family == "FIR" and len(z) > 0:
        p = np.zeros(len(z))

    if len(p) > 0 and np.any(np.abs(p) >= 1.0 + 1e-5):
        st.error("⚠️ **Filter Instability Detected!** The calculated poles fall on or outside the Unit Circle. This IIR filter will mathematically explode.")
        st.stop()  

    try:
        filtered, fallback_warning = apply_filter(b, a, signal)
        if fallback_warning:
            padlen = 3 * max(len(a), len(b))
            st.warning(f"Signal too short for zero-phase filtering (needs > {padlen} samples) - using standard filtering.")
            with st.popover("ℹ️ What is Zero-Phase Filtering?"):
                st.write("**Zero-Phase Filtering** is a technique where a signal is filtered twice: once forward, and once backward.")
                st.write("Standard filters introduce a slight time delay (phase shift), which pushes the signal slightly to the right. By running the filter backward the second time, this delay is completely canceled out. This ensures the peaks and valleys of your filtered signal remain perfectly aligned in time with the original data.")
    except Exception as e:
        st.error(f"Filtering failed: {e}")
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

    axs2[2].plot(w, 20 * np.log10(np.abs(h) + 1e-12))
    axs2[2].set_title(f"{family} {ftype} Effective Frequency Response")
    axs2[2].set_xlabel("Frequency (Hz)"); axs2[2].set_xlim(0, fs / 2); axs2[2].grid(alpha=0.3)

    axs2[3].add_patch(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5, label='Unit Circle'))

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
                    st.error("FFmpeg is not installed on the system. Choose WAV format instead.")
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


# ============================================================================
# 2D IMAGE STUDIO
# ============================================================================
elif app_mode == "🖼️ 2D Image Studio":
    
    st.sidebar.header("1. Image Source")
    
    source_2d = st.sidebar.selectbox(
        "Source", 
        ["Upload a file", "Demo: 2D Zone Plate (Chirp)", "Demo: 2D Spatial Grating"]
    )

    original_array = None

    if source_2d == "Upload a file":
        with st.sidebar.popover("ℹ️ View Supported File Types", use_container_width=True):
            st.markdown(
                """
                <div style="display: flex; gap: 40px;">
                    <div>
                        <h3 style="margin: 0px 0px 10px 0px;">Image Formats</h3>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li>png</li><li>jpg</li><li>jpeg</li>
                            <li>bmp</li><li>webp</li><li>tiff</li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.sidebar.markdown('<p style="margin-bottom: 4px;"></p>', unsafe_allow_html=True)
        uploaded_image = st.sidebar.file_uploader(
            "hidden_label",
            label_visibility="collapsed",
            type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
            key="image_2d_uploader"
        )

        if uploaded_image is not None:
            try:
                image_bytes = uploaded_image.getvalue()
                original_array = load_image_array(image_bytes)
            except Exception as e:
                st.sidebar.error(f"Couldn't read this image: {e}")
                st.stop()
        else:
            st.info("Upload an image in the sidebar, or pick a mathematical demo source to test 2D DSP.")
            st.stop()
            
    elif source_2d == "Demo: 2D Zone Plate (Chirp)":
        original_array = gen_zone_plate()
    elif source_2d == "Demo: 2D Spatial Grating":
        original_array = gen_2d_grating()

    if original_array is not None:
        pil_original = Image.fromarray(original_array)
        
        st.sidebar.markdown("---")
        st.sidebar.header("2. Export Settings")
        
        image_format = st.sidebar.selectbox("Export Format", ["PNG", "JPG", "BMP", "PDF"]).lower()
        
        st.sidebar.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)
        
        pil_format = "JPEG" if image_format == "jpg" else image_format.upper()
        mime_format = "application/pdf" if image_format == "pdf" else f"image/{'jpeg' if image_format == 'jpg' else image_format}"

        st.subheader("Original Image")
        st.image(pil_original, use_container_width=True)
        
        st.markdown("---")
        st.subheader("DSP Operations")
        
        operation = st.selectbox(
            "2D DSP Operation",
            [
                "Sobel Edge Detection", 
                "Gaussian Blur", 
                "Image Sharpening",
                "2D FFT (Frequency Spectrum)",
                "Median Filtering",
                "Image Binarization (Threshold)",
                "Morphological Erosion",
                "Morphological Dilation"
            ],
            key="image_2d_operation"
        )

        sigma = 2.0
        kernel_size = 3
        threshold = 128

        if operation == "Gaussian Blur":
            sigma = st.slider("Sigma (blur strength)", 0.1, 10.0, 2.0, 0.1)
        elif operation in ["Median Filtering", "Morphological Erosion", "Morphological Dilation"]:
            kernel_size = st.slider("Kernel Size (pixels)", min_value=3, max_value=21, value=5, step=2)
        elif operation == "Image Binarization (Threshold)":
            threshold = st.slider("Threshold Level", 0, 255, 128, 1)

        if operation == "Sobel Edge Detection":
            filtered_array = apply_sobel_edge_detection(original_array)
            result_caption = "Gradient magnitude: sqrt(Gx^2 + Gy^2)"
            
        elif operation == "Gaussian Blur":
            filtered_array = apply_gaussian_blur(original_array, sigma)
            result_caption = f"Lowpass filter, sigma = {sigma}"
            
        elif operation == "Image Sharpening":  
            filtered_array = apply_sharpening(original_array)
            result_caption = "Highpass 3x3 convolution kernel"
            
        elif operation == "2D FFT (Frequency Spectrum)":
            filtered_array = apply_fft2d(original_array)
            result_caption = "2D Magnitude Spectrum (Log Scale) shifted to center"
            
        elif operation == "Median Filtering":
            filtered_array = apply_median_filter(original_array, kernel_size)
            result_caption = f"Median Filter (Non-linear Denoising), {kernel_size}x{kernel_size} footprint"
            
        elif operation == "Image Binarization (Threshold)":
            filtered_array = apply_binarization(original_array, threshold)
            result_caption = f"Binarized (Threshold > {threshold})"
            
        elif operation == "Morphological Erosion":
            filtered_array = apply_erosion(original_array, kernel_size)
            result_caption = f"Erosion (Local Minimum), {kernel_size}x{kernel_size} footprint"
            
        elif operation == "Morphological Dilation":
            filtered_array = apply_dilation(original_array, kernel_size)
            result_caption = f"Dilation (Local Maximum), {kernel_size}x{kernel_size} footprint"

        pil_filtered = Image.fromarray(filtered_array)

        st.markdown("---")
        
        st.subheader(f"Filtered Result: {operation}")
        st.caption(result_caption)
        
        if operation == "2D FFT (Frequency Spectrum)":
            st.image(pil_filtered, use_container_width=True, clamp=True)
        else:
            st.image(pil_filtered, use_container_width=True)

        operation_slugs = {
            "Sobel Edge Detection": "sobel_edges",
            "Gaussian Blur": "gaussian_blur",
            "Image Sharpening": "sharpened",
            "2D FFT (Frequency Spectrum)": "fft2d",
            "Median Filtering": "median",
            "Image Binarization (Threshold)": "binary",
            "Morphological Erosion": "erosion",
            "Morphological Dilation": "dilation"
        }
        
        download_buf = io.BytesIO()
        pil_filtered.save(download_buf, format=pil_format)
        
        st.markdown("---")
        st.download_button(
            label=f"📥 Download Filtered Image ({image_format.upper()})",
            data=download_buf.getvalue(),
            file_name=f"filtered_{operation_slugs[operation]}.{image_format}",
            mime=mime_format,
            key="image_2d_download",
        )
