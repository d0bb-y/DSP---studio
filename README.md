
# Universal DSP Signal Analyzer & Filter Design Studio

**Live Application:** [https://kdev-dsp-analyzer.streamlit.app/](https://kdev-dsp-analyzer.streamlit.app/)

## Overview
This project is a production-grade, cloud-deployed Digital Signal Processing (DSP) application built with Python, Streamlit, and SciPy. It provides a full-suite graphical interface for analyzing real-world and synthetic signals in the time and frequency domains, alongside an interactive FIR/IIR filter design studio.

## Core Features
* **Universal Signal Ingestion:** Automatically parses and standardizes raw data from audio formats (WAV, MP3, FLAC, M4A), medical/scientific formats (EDF, BDF, MAT), and flat files (CSV, TXT). Uses system-level `ffmpeg` for background audio conversion.
* **Comprehensive Signal Analysis:** Real-time computation of Time-Domain Waveforms, Fast Fourier Transforms (FFT), and Spectrograms.
* **Interactive Filter Design:** Design Lowpass, Highpass, Bandpass, and Notch filters using FIR (Windowed) or IIR (Butterworth) topologies.
* **Filter Evaluation & Stability:** Visualizes Effective Frequency Response (in dB) and generates Z-Plane Pole-Zero plots to mathematically evaluate IIR stability. 
* **Audio Playback & Export:** Features dynamically normalized audio playback comparing original vs. filtered signals, with safe WAV/MP3/FLAC export capabilities.

## Technical Safeguards
This application was engineered with robust edge-case handling for a production cloud environment:
* **Linear Interpolation:** Prevents mathematical crashes from silent `NaN` values or missing rows in user-uploaded CSVs.
* **IIR Instability Guardrails:** Automatically monitors Z-Plane poles and halts execution if an unstable high-order IIR filter threatens to cause a mathematical overflow.
* **UUID File Hashing:** Prevents race conditions and overwrites during simultaneous multi-user file uploads.
* **Spectrogram Guardrails:** Prevents array shape errors when analyzing heavily micro-trimmed signal segments.
* **Dependency Fallbacks:** Generates synthetic sawtooth-based vocal models if external acoustic datasets fail to load over the network.
