"""Numerical engine for AC rectification, capacitor filtering, and conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SimulationMetrics:
    """Summary values calculated after the startup transient."""

    dc_voltage: float
    rms_voltage: float
    ripple_pp: float
    ripple_factor_percent: float
    piv_voltage: float
    conditioned_dc_voltage: float


def _validate_inputs(
    topology: str,
    v_rms: float,
    frequency_hz: float,
    diode_vf: float,
    load_resistance_ohm: float,
    capacitance_f: float,
    adc_max_voltage: float,
    cutoff_hz: float,
) -> None:
    if topology not in {"Full-Wave Bridge", "Half-Wave"}:
        raise ValueError("topology must be 'Full-Wave Bridge' or 'Half-Wave'")
    if v_rms <= 0 or frequency_hz <= 0:
        raise ValueError("source voltage and frequency must be positive")
    if diode_vf < 0 or load_resistance_ohm <= 0 or capacitance_f <= 0:
        raise ValueError("diode drop must be non-negative; R and C must be positive")
    if adc_max_voltage <= 0 or cutoff_hz <= 0:
        raise ValueError("ADC range and cutoff frequency must be positive")


def _lowpass(values: np.ndarray, cutoff_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Apply a zero-phase, active first-order Butterworth low-pass filter."""
    nyquist = sample_rate_hz / 2.0
    safe_cutoff = min(float(cutoff_hz), nyquist * 0.95)
    if safe_cutoff <= 0 or len(values) < 8:
        return values.copy()
    b, a = signal.butter(1, safe_cutoff / nyquist, btype="low")
    # filtfilt avoids introducing a visual phase shift in this measurement display.
    pad_length = min(3 * max(len(a), len(b)), len(values) - 1)
    return signal.filtfilt(b, a, values, padlen=max(1, pad_length))


def _fft(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-sided amplitude spectrum, including its DC component."""
    centered = np.asarray(values, dtype=float)
    spectrum = np.fft.rfft(centered)
    magnitude = np.abs(spectrum) / max(len(centered), 1)
    if len(magnitude) > 2:
        magnitude[1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    return frequencies, magnitude


def simulate_power_supply(
    *,
    topology: str = "Full-Wave Bridge",
    v_rms: float = 12.0,
    frequency_hz: float = 50.0,
    diode_vf: float = 0.7,
    load_resistance_ohm: float = 1000.0,
    capacitance_f: float = 470e-6,
    adc_max_voltage: float = 3.3,
    cutoff_hz: float = 25.0,
    sample_rate_hz: float = 100_000.0,
    cycles: int = 4,
) -> dict[str, np.ndarray | float | SimulationMetrics]:
    """Simulate the complete power-supply signal chain.

    The returned arrays use SI units (seconds and volts). The first AC cycle is
    treated as startup warm-up for steady-state metrics.
    """
    _validate_inputs(
        topology,
        v_rms,
        frequency_hz,
        diode_vf,
        load_resistance_ohm,
        capacitance_f,
        adc_max_voltage,
        cutoff_hz,
    )
    if sample_rate_hz < 100_000:
        raise ValueError("sample_rate_hz must be at least 100,000 Hz")
    if cycles < 2:
        raise ValueError("cycles must be at least 2 to allow transient warm-up")

    sample_count = int(np.ceil(sample_rate_hz * cycles / frequency_hz)) + 1
    time = np.arange(sample_count, dtype=float) / sample_rate_hz
    peak_voltage = np.sqrt(2.0) * v_rms
    ac_input = peak_voltage * np.sin(2.0 * np.pi * frequency_hz * time)

    if topology == "Full-Wave Bridge":
        raw_rectified = np.maximum(np.abs(ac_input) - 2.0 * diode_vf, 0.0)
        # A bridge's reverse stress is approximately one source peak.
        piv_voltage = peak_voltage
    else:
        raw_rectified = np.maximum(ac_input - diode_vf, 0.0)
        # A simple half-wave diode sees the opposite source peak when off.
        piv_voltage = peak_voltage

    dt = 1.0 / sample_rate_hz
    time_constant = load_resistance_ohm * capacitance_f
    decay = np.exp(-dt / time_constant)
    filtered_dc = np.empty_like(raw_rectified)
    filtered_dc[0] = raw_rectified[0]
    for index in range(1, len(raw_rectified)):
        previous = filtered_dc[index - 1]
        if raw_rectified[index] > previous:
            filtered_dc[index] = raw_rectified[index]
        else:
            filtered_dc[index] = previous * decay

    # Scale the signal against the observed steady-state peak, leaving headroom.
    warmup_start = min(int(np.ceil(sample_rate_hz / frequency_hz)), len(filtered_dc) - 1)
    steady_state = filtered_dc[warmup_start:]
    scale_reference = max(float(np.max(steady_state)) if len(steady_state) else 0.0, 1e-12)
    scale = (0.95 * adc_max_voltage) / scale_reference
    conditioned_unfiltered = np.clip(filtered_dc * scale, 0.0, adc_max_voltage)
    conditioned_output = np.clip(
        _lowpass(conditioned_unfiltered, cutoff_hz, sample_rate_hz),
        0.0,
        adc_max_voltage,
    )

    measured = filtered_dc[warmup_start:]
    dc_voltage = float(np.mean(measured)) if len(measured) else 0.0
    rms_voltage = float(np.sqrt(np.mean(np.square(measured)))) if len(measured) else 0.0
    ripple_pp = float(np.ptp(measured)) if len(measured) else 0.0
    if dc_voltage > 1e-12:
        ratio = max((rms_voltage / dc_voltage) ** 2 - 1.0, 0.0)
        ripple_factor_percent = float(np.sqrt(ratio) * 100.0)
    else:
        ripple_factor_percent = 0.0
    conditioned_dc_voltage = (
        float(np.mean(conditioned_output[warmup_start:])) if len(conditioned_output) else 0.0
    )
    metrics = SimulationMetrics(
        dc_voltage=dc_voltage,
        rms_voltage=rms_voltage,
        ripple_pp=ripple_pp,
        ripple_factor_percent=ripple_factor_percent,
        piv_voltage=float(piv_voltage),
        conditioned_dc_voltage=conditioned_dc_voltage,
    )
    frequencies, magnitude = _fft(measured, sample_rate_hz)
    return {
        "time": time,
        "ac_input": ac_input,
        "raw_rectified": raw_rectified,
        "filtered_dc": filtered_dc,
        "conditioned_output": conditioned_output,
        "fft_frequencies": frequencies,
        "fft_magnitude": magnitude,
        "metrics": metrics,
        "sample_rate_hz": float(sample_rate_hz),
        "warmup_start": int(warmup_start),
    }
