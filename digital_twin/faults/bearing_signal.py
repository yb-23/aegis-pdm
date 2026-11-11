"""Synthetic vibration signal generator for rolling-element bearings.

A localized bearing fault produces periodic impacts at the characteristic fault frequency.
Each impact excites a structural resonance modelled as an exponentially damped sinusoid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from digital_twin.faults.bearing_frequencies import compute_fault_frequencies
from services.common.schemas import FaultType, SimulationConfig

# Structural resonance excited by the impacts (typical accelerometer-mounted range).
RESONANCE_HZ = 3_000.0
DAMPING_RATIO = 0.05
# Impulse amplitude at severity 1.0, in arbitrary acceleration units.
MAX_IMPACT_AMPLITUDE = 3.0
# Relative standard deviation of the inter-impact interval (ball slip).
SLIP_JITTER = 0.01

BEARING_FAULTS = frozenset(
    {
        FaultType.BEARING_OUTER_RACE,
        FaultType.BEARING_INNER_RACE,
        FaultType.BEARING_BALL,
    }
)


def _fault_frequency_hz(config: SimulationConfig) -> float:
    freqs = compute_fault_frequencies(config.bearing, config.shaft_speed_hz)
    if config.fault is FaultType.BEARING_OUTER_RACE:
        return freqs.bpfo_hz
    if config.fault is FaultType.BEARING_INNER_RACE:
        return freqs.bpfi_hz
    if config.fault is FaultType.BEARING_BALL:
        return 2.0 * freqs.bsf_hz
    msg = f"{config.fault.value} is not a bearing fault"
    raise ValueError(msg)


def _impact_response(
    time_s: NDArray[np.float64], impact_time_s: float, amplitude: float
) -> NDArray[np.float64]:
    """Damped sinusoid response to one impact (zero before the impact)."""
    omega_n = 2.0 * np.pi * RESONANCE_HZ
    delta = time_s - impact_time_s
    # Clip the exponent argument so that np.exp never overflows before the impact.
    causal = np.maximum(delta, 0.0)
    envelope = np.exp(-DAMPING_RATIO * omega_n * causal)
    response = amplitude * envelope * np.sin(omega_n * causal)
    return np.where(delta >= 0.0, response, 0.0).astype(np.float64)


def generate_bearing_signal(config: SimulationConfig) -> NDArray[np.float64]:
    """Generate one vibration window for a healthy or bearing-faulty machine."""
    if config.fault not in BEARING_FAULTS and config.fault is not FaultType.HEALTHY:
        msg = f"{config.fault.value} is not handled by the bearing generator"
        raise ValueError(msg)

    rng = np.random.default_rng(config.seed)
    n_samples = int(round(config.duration_s * config.sampling_rate_hz))
    time_s = np.arange(n_samples, dtype=np.float64) / config.sampling_rate_hz

    signal = np.zeros(n_samples, dtype=np.float64)

    if config.fault in BEARING_FAULTS and config.severity > 0.0:
        fault_hz = _fault_frequency_hz(config)
        period_s = 1.0 / fault_hz
        amplitude = MAX_IMPACT_AMPLITUDE * config.severity
        n_impacts = int(config.duration_s * fault_hz) + 1
        # Start at a random phase inside the first period.
        t_k = rng.uniform(0.0, period_s)
        for _ in range(n_impacts + 1):
            if t_k >= config.duration_s:
                break
            gain = 1.0
            if config.fault is FaultType.BEARING_INNER_RACE:
                gain = 1.0 + 0.5 * np.cos(2.0 * np.pi * config.shaft_speed_hz * t_k)
            signal += _impact_response(time_s, t_k, amplitude * gain)
            t_k += period_s * (1.0 + SLIP_JITTER * rng.standard_normal())

    # Baseline shaft-rotation component present on any machine, plus sensor noise.
    signal += 0.1 * np.sin(2.0 * np.pi * config.shaft_speed_hz * time_s)
    signal += config.noise_std * rng.standard_normal(n_samples)
    return signal
