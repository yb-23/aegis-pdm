"""Unit tests for the synthetic bearing vibration generator."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import hilbert

from digital_twin.faults.bearing_frequencies import compute_fault_frequencies
from digital_twin.faults.bearing_signal import generate_bearing_signal
from services.common.schemas import FaultType, SimulationConfig


def _config(fault: FaultType, severity: float, **kwargs: float) -> SimulationConfig:
    return SimulationConfig(
        asset_id="motor-01",
        fault=fault,
        severity=severity,
        duration_s=2.0,
        sampling_rate_hz=12_000,
        noise_std=0.05,
        seed=7,
        **kwargs,  # type: ignore[arg-type]
    )


def _envelope_spectrum(
    signal: NDArray[np.float64], fs: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    envelope = np.abs(hilbert(signal))
    envelope = envelope - envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope)) / len(envelope)
    freqs = np.fft.rfftfreq(len(envelope), d=1.0 / fs)
    return freqs, spectrum


def _peak_near(
    freqs: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    target_hz: float,
    tolerance_hz: float = 2.0,
) -> float:
    mask = np.abs(freqs - target_hz) <= tolerance_hz
    return float(spectrum[mask].max())


def test_signal_length_matches_config() -> None:
    cfg = _config(FaultType.HEALTHY, 0.0)
    signal = generate_bearing_signal(cfg)
    assert signal.shape == (24_000,)
    assert signal.dtype == np.float64


def test_generation_is_deterministic_for_a_given_seed() -> None:
    cfg = _config(FaultType.BEARING_OUTER_RACE, 0.6)
    assert np.array_equal(generate_bearing_signal(cfg), generate_bearing_signal(cfg))


def test_different_seeds_give_different_signals() -> None:
    a = generate_bearing_signal(_config(FaultType.BEARING_OUTER_RACE, 0.6))
    cfg_b = SimulationConfig(
        asset_id="motor-01",
        fault=FaultType.BEARING_OUTER_RACE,
        severity=0.6,
        duration_s=2.0,
        seed=8,
    )
    assert not np.array_equal(a, generate_bearing_signal(cfg_b))


def test_outer_race_fault_shows_bpfo_in_envelope_spectrum() -> None:
    cfg = _config(FaultType.BEARING_OUTER_RACE, 0.8)
    bpfo = compute_fault_frequencies(cfg.bearing, cfg.shaft_speed_hz).bpfo_hz
    freqs, spectrum = _envelope_spectrum(generate_bearing_signal(cfg), 12_000.0)
    peak = _peak_near(freqs, spectrum, bpfo)
    # The BPFO line must stand well above the median spectral floor in the 20-500 Hz band.
    band = (freqs > 20.0) & (freqs < 500.0)
    assert peak > 5.0 * float(np.median(spectrum[band]))


def test_healthy_signal_has_no_bpfo_line() -> None:
    healthy = _config(FaultType.HEALTHY, 0.0)
    faulty = _config(FaultType.BEARING_OUTER_RACE, 0.8)
    bpfo = compute_fault_frequencies(healthy.bearing, healthy.shaft_speed_hz).bpfo_hz
    f_h, s_h = _envelope_spectrum(generate_bearing_signal(healthy), 12_000.0)
    f_f, s_f = _envelope_spectrum(generate_bearing_signal(faulty), 12_000.0)
    assert _peak_near(f_f, s_f, bpfo) > 5.0 * _peak_near(f_h, s_h, bpfo)


def test_severity_increases_vibration_energy() -> None:
    mild = generate_bearing_signal(_config(FaultType.BEARING_OUTER_RACE, 0.2))
    severe = generate_bearing_signal(_config(FaultType.BEARING_OUTER_RACE, 0.9))
    assert float(np.std(severe)) > float(np.std(mild))


def test_non_bearing_fault_is_rejected() -> None:
    with pytest.raises(ValueError, match="not handled"):
        generate_bearing_signal(_config(FaultType.UNBALANCE, 0.5))
