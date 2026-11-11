"""Single entry point of the digital twin: config in, labelled vibration window out."""

from __future__ import annotations

from digital_twin.faults.bearing_signal import BEARING_FAULTS, generate_bearing_signal
from digital_twin.faults.rotor_signal import ROTOR_FAULTS, generate_rotor_signal
from services.common.schemas import FaultType, SimulationConfig, VibrationWindow


def generate_vibration(config: SimulationConfig) -> VibrationWindow:
    """Simulate one window for any supported fault type and wrap it in the data contract."""
    if config.fault in BEARING_FAULTS:
        samples = generate_bearing_signal(config)
    elif config.fault in ROTOR_FAULTS or config.fault is FaultType.HEALTHY:
        samples = generate_rotor_signal(config)
    else:
        msg = f"unsupported fault type: {config.fault.value}"
        raise ValueError(msg)

    return VibrationWindow(
        asset_id=config.asset_id,
        sampling_rate_hz=config.sampling_rate_hz,
        shaft_speed_hz=config.shaft_speed_hz,
        samples=samples.tolist(),
        label=config.fault,
        severity=config.severity,
    )
