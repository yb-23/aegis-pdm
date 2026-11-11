"""Unit tests for the shared data contracts."""

import pytest
from pydantic import ValidationError

from services.common.schemas import (
    BearingGeometry,
    EdgeInference,
    FaultType,
    SimulationConfig,
    VibrationWindow,
)


def test_default_simulation_config_is_valid() -> None:
    cfg = SimulationConfig(asset_id="motor-01")
    assert cfg.fault is FaultType.HEALTHY
    assert cfg.severity == 0.0
    assert cfg.sampling_rate_hz == 12_000


def test_healthy_with_nonzero_severity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(asset_id="motor-01", fault=FaultType.HEALTHY, severity=0.5)


def test_faulty_config_accepts_severity() -> None:
    cfg = SimulationConfig(asset_id="motor-01", fault=FaultType.BEARING_OUTER_RACE, severity=0.7)
    assert cfg.severity == 0.7


def test_severity_out_of_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(asset_id="motor-01", fault=FaultType.UNBALANCE, severity=1.5)


def test_config_is_immutable() -> None:
    cfg = SimulationConfig(asset_id="motor-01")
    with pytest.raises(ValidationError):
        cfg.severity = 0.3


def test_bearing_ball_larger_than_pitch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BearingGeometry(ball_diameter_mm=50.0, pitch_diameter_mm=39.04)


def test_vibration_window_roundtrip_json() -> None:
    window = VibrationWindow(
        asset_id="motor-01",
        sampling_rate_hz=12_000,
        shaft_speed_hz=29.95,
        samples=[0.0, 0.1, -0.1],
        label=FaultType.UNBALANCE,
        severity=0.4,
    )
    restored = VibrationWindow.model_validate_json(window.model_dump_json())
    assert restored == window


def test_empty_vibration_window_is_rejected() -> None:
    with pytest.raises(ValidationError):
        VibrationWindow(
            asset_id="motor-01", sampling_rate_hz=12_000, shaft_speed_hz=29.95, samples=[]
        )


def test_edge_inference_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        EdgeInference(
            asset_id="motor-01",
            anomaly_score=0.1,
            predicted_class=FaultType.HEALTHY,
            latency_us=-5,
            model_version="0.1.0",
        )
