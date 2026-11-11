"""Synthetic vibration signals for rotor faults (unbalance and misalignment)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from services.common.schemas import FaultType, SimulationConfig

# Peak amplitudes at severity 1.0, in arbitrary acceleration units.
UNBALANCE_1X_AMPLITUDE = 2.0
MISALIGNMENT_2X_AMPLITUDE = 1.5
MISALIGNMENT_1X_RATIO = 0.4
MISALIGNMENT_3X_RATIO = 0.3
BASELINE_1X_AMPLITUDE = 0.1

ROTOR_FAULTS = frozenset({FaultType.UNBALANCE, FaultType.MISALIGNMENT})


def generate_rotor_signal(config: SimulationConfig) -> NDArray[np.float64]:
    """Generate one vibration window for a healthy, unbalanced or misaligned machine."""
    if config.fault not in ROTOR_FAULTS and config.fault is not FaultType.HEALTHY:
        msg = f"{config.fault.value} is not handled by the rotor generator"
        raise ValueError(msg)

    rng = np.random.default_rng(config.seed)
    n_samples = int(round(config.duration_s * config.sampling_rate_hz))
    time_s = np.arange(n_samples, dtype=np.float64) / config.sampling_rate_hz
    omega = 2.0 * np.pi * config.shaft_speed_hz
    phase = rng.uniform(0.0, 2.0 * np.pi, size=3)

    signal: NDArray[np.float64] = BASELINE_1X_AMPLITUDE * np.sin(omega * time_s + phase[0])

    if config.fault is FaultType.UNBALANCE:
        signal += UNBALANCE_1X_AMPLITUDE * config.severity * np.sin(omega * time_s + phase[0])
    elif config.fault is FaultType.MISALIGNMENT:
        amp_2x = MISALIGNMENT_2X_AMPLITUDE * config.severity
        signal += MISALIGNMENT_1X_RATIO * amp_2x * np.sin(omega * time_s + phase[0])
        signal += amp_2x * np.sin(2.0 * omega * time_s + phase[1])
        signal += MISALIGNMENT_3X_RATIO * amp_2x * np.sin(3.0 * omega * time_s + phase[2])

    signal += config.noise_std * rng.standard_normal(n_samples)
    return signal
