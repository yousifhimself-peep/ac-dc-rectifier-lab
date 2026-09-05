"""Streamlit dashboard for an AC-to-DC power-supply simulator."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from simulation.engine import SimulationMetrics, simulate_power_supply


st.set_page_config(page_title="AC–DC Rectifier Lab", page_icon="⚡", layout="wide")


def _downsample(values: np.ndarray, limit: int = 7000) -> np.ndarray:
    """Keep Plotly responsive while retaining the shape of long waveforms."""
    if len(values) <= limit:
        return np.arange(len(values))
    return np.linspace(0, len(values) - 1, limit, dtype=int)


def _metric_card(label: str, value: str, help_text: str) -> None:
    st.metric(label, value, help=help_text)


def _waveform_figure(result: dict) -> go.Figure:
    time = result["time"]
    indices = _downsample(time)
    x_ms = time[indices] * 1000.0
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x_ms, y=result["ac_input"][indices], name="AC input", line=dict(width=1.4)))
    figure.add_trace(go.Scatter(x=x_ms, y=result["raw_rectified"][indices], name="Raw rectified", line=dict(width=1.5)))
    figure.add_trace(go.Scatter(x=x_ms, y=result["filtered_dc"][indices], name="Filtered DC", line=dict(width=2.2)))
    figure.update_layout(
        template="plotly_dark", height=510, hovermode="x unified",
        xaxis_title="Time (ms)", yaxis_title="Voltage (V)", legend=dict(orientation="h", y=1.08),
    )
    return figure


def _conditioning_figure(result: dict, adc_max: float) -> go.Figure:
    time = result["time"]
    indices = _downsample(time)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time[indices] * 1000.0, y=result["conditioned_output"][indices], name="Conditioned ADC", line=dict(color="#00d4ff", width=2.2)))
    figure.add_hline(y=adc_max, line_dash="dash", line_color="#ff7b72", annotation_text="ADC maximum")
    figure.add_hline(y=0, line_dash="dot", line_color="#8b949e", annotation_text="0 V")
    figure.update_layout(template="plotly_dark", height=510, hovermode="x unified", xaxis_title="Time (ms)", yaxis_title="ADC input (V)")
    figure.update_yaxes(range=[0, adc_max * 1.08])
    return figure


def _spectrum_figure(result: dict, frequency_hz: float) -> go.Figure:
    frequencies = result["fft_frequencies"]
    magnitude = result["fft_magnitude"]
    # The useful rectifier harmonics are in the low-kHz region.
    mask = frequencies <= max(500.0, frequency_hz * 8.0)
    figure = go.Figure(go.Bar(x=frequencies[mask], y=magnitude[mask], name="Amplitude", marker_color="#7ee787"))
    figure.update_layout(template="plotly_dark", height=510, bargap=0, xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (V)")
    return figure


st.title("⚡ AC–DC Rectifier & Signal Conditioning Lab")
st.caption("A numerical laboratory for diode rectification, capacitor-input filtering, and op-amp-style ADC conditioning.")

with st.sidebar:
    st.header("Circuit controls")
    topology = st.selectbox("Rectifier topology", ["Full-Wave Bridge", "Half-Wave"])
    st.subheader("AC source")
    v_rms = st.slider("RMS voltage (V)", 6.0, 240.0, 12.0, 1.0)
    frequency_hz = 50.0 if st.radio("Frequency", ["50 Hz", "60 Hz"], horizontal=True) == "50 Hz" else 60.0
    st.subheader("Diodes")
    diode_vf = st.slider("Forward drop Vf (V)", 0.2, 1.2, 0.7, 0.05)
    st.subheader("Filter and load")
    load_resistance = st.slider("Load resistance RL (Ω)", 10, 5000, 1000, 10)
    capacitance_uf = st.slider("Capacitance C (µF)", 10, 4700, 470, 10)
    st.subheader("Signal conditioning")
    adc_max = st.selectbox("Target ADC maximum", [3.3, 5.0], format_func=lambda v: f"{v:.1f} V")
    cutoff_hz = st.slider("Active low-pass cutoff (Hz)", 1.0, 500.0, 25.0, 1.0)

try:
    result = simulate_power_supply(
        topology=topology, v_rms=v_rms, frequency_hz=frequency_hz, diode_vf=diode_vf,
        load_resistance_ohm=float(load_resistance), capacitance_f=capacitance_uf * 1e-6,
        adc_max_voltage=adc_max, cutoff_hz=cutoff_hz,
    )
except ValueError as error:
    st.error(f"Simulation input error: {error}")
    st.stop()

metrics: SimulationMetrics = result["metrics"]
st.subheader("Steady-state measurements")
metric_columns = st.columns(4)
with metric_columns[0]:
    _metric_card("Vdc (conditioned)", f"{metrics.conditioned_dc_voltage:.3f} V", "Mean conditioned output after one-cycle warm-up")
with metric_columns[1]:
    _metric_card("Ripple voltage", f"{metrics.ripple_pp:.3f} Vpp", "Peak-to-peak capacitor ripple after warm-up")
with metric_columns[2]:
    _metric_card("Ripple factor", f"{metrics.ripple_factor_percent:.2f}%", "AC RMS divided by DC average")
with metric_columns[3]:
    _metric_card("Diode PIV", f"{metrics.piv_voltage:.2f} V", "Estimated peak inverse voltage per diode")

waveform_tab, conditioning_tab, spectrum_tab = st.tabs(["Waveform", "Signal conditioning", "Frequency spectrum"])
with waveform_tab:
    st.plotly_chart(_waveform_figure(result), use_container_width=True)
    st.info(f"Unconditioned DC average: {metrics.dc_voltage:.3f} V · Unconditioned RMS: {metrics.rms_voltage:.3f} V · τ = {load_resistance * capacitance_uf * 1e-6 * 1000:.2f} ms")
with conditioning_tab:
    st.plotly_chart(_conditioning_figure(result, adc_max), use_container_width=True)
    st.caption(f"The filtered output is scaled to 95% of the selected {adc_max:.1f} V ADC range, then low-pass filtered at {cutoff_hz:.0f} Hz and clipped defensively.")
with spectrum_tab:
    st.plotly_chart(_spectrum_figure(result, frequency_hz), use_container_width=True)
    st.caption(f"Expected dominant ripple region: {frequency_hz * (2 if topology == 'Full-Wave Bridge' else 1):.0f} Hz for this topology.")

with st.expander("Circuit equations"):
    st.markdown(r"""
    - Source: (v_{in}(t)=sqrt{2}V_{rms}sin(2pi ft))
    - Bridge rectifier: (v_{rect}=max(|v_{in}|-2V_f,0)); half-wave: (v_{rect}=max(v_{in}-V_f,0))
    - Capacitor charge/discharge: (v_{out}=v_{rect}) while conducting, otherwise (v_{out}=v_{out,prev}e^{-Delta t/(R_LC)})
    - Ripple factor: (r=\sqrt{(V_{rms}/V_{dc})^2-1}\times100\%)
    - Conditioning uses a first-order Butterworth low-pass and scales the observed DC waveform to 95% of the ADC limit.
    """)
