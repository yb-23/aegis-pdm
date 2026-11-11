"""Unit tests for rotor fault signals and the generator dispatcher."""

import numpy as np
import pytest
from numpy.typing import NDArray

from digital_twin.faults.rotor_signal import generate_rotor_signal
from digital_twin.generator import generate_vibration
from services.common.schemas import FaultType, SimulationConfig

FS = 12_000.0


def _config(fault: FaultType, severity: float) -> SimulationConfig:
    return SimulationConfig(
        asset_id="motor-01",
        fault=fault,
        severity=severity,
        duration_s=2.0,
        sampling_rate_hz=12_000,
        shaft_speed_hz=30.0,
        noise_std=0.02,
        seed=3,
    )


def _amplitude_at(signal: NDArray[np.float64], target_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal)) * 2.0 / len(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / FS)
    mask = np.abs(freqs - target_hz) <= 1.0
    return float(spectrum[mask].max())


def test_unbalance_dominates_at_1x() -> None:
    signal = generate_rotor_signal(_config(FaultType.UNBALANCE, 0.8))
    a1 = _amplitude_at(signal, 30.0)
    assert a1 > 5.0 * _amplitude_at(signal, 60.0)
    assert a1 > 1.0


def test_unbalance_amplitude_grows_with_severity() -> None:
    mild = _amplitude_at(generate_rotor_signal(_config(FaultType.UNBALANCE, 0.2)), 30.0)
    severe = _amplitude_at(generate_rotor_signal(_config(FaultType.UNBALANCE, 0.9)), 30.0)
    assert severe > 2.0 * mild


def test_misalignment_dominates_at_2x() -> None:
    signal = generate_rotor_signal(_config(FaultType.MISALIGNMENT, 0.8))
    a2 = _amplitude_at(signal, 60.0)
    assert a2 > _amplitude_at(signal, 30.0)
    assert a2 > _amplitude_at(signal, 90.0)


def test_healthy_rotor_has_small_1x_only() -> None:
    signal = generate_rotor_signal(_config(FaultType.HEALTHY, 0.0))
    assert _amplitude_at(signal, 30.0) == pytest.approx(0.1, abs=0.02)
    assert _amplitude_at(signal, 60.0) < 0.02


def test_bearing_fault_is_rejected_by_rotor_generator() -> None:
    with pytest.raises(ValueError, match="not handled"):
        generate_rotor_signal(_config(FaultType.BEARING_OUTER_RACE, 0.5))


@pytest.mark.parametrize("fault", list(FaultType))
def test_dispatcher_returns_labelled_window_for_every_fault(fault: FaultType) -> None:
    severity = 0.0 if fault is FaultType.HEALTHY else 0.5
    window = generate_vibration(_config(fault, severity))
    assert window.label is fault
    assert window.severity == severity
    assert len(window.samples) == 24_000
    assert window.sampling_rate_hz == 12_000
