"""Shared data contracts for AEGIS-PdM.

These schemas are the single source of truth for every message exchanged between
the digital twin, the STM32 bridge, the cloud services and the dashboard.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FaultType(StrEnum):
    """Fault classes simulated by the digital twin (also the classifier labels)."""

    HEALTHY = "healthy"
    UNBALANCE = "unbalance"
    MISALIGNMENT = "misalignment"
    BEARING_OUTER_RACE = "bearing_outer_race"
    BEARING_INNER_RACE = "bearing_inner_race"
    BEARING_BALL = "bearing_ball"


class AssetType(StrEnum):
    """Kinds of industrial equipment supported by the platform."""

    INDUCTION_MOTOR = "induction_motor"
    CENTRIFUGAL_PUMP = "centrifugal_pump"
    TURBINE = "turbine"


class BearingGeometry(BaseModel):
    """Geometry of a rolling-element bearing (used to derive fault frequencies).

    Defaults correspond to the 6205-2RS JEM SKF bearing used in the public CWRU dataset,
    so that synthetic signals can be compared with real recordings.
    """

    model_config = ConfigDict(frozen=True)

    n_balls: Annotated[int, Field(gt=0)] = 9
    ball_diameter_mm: Annotated[float, Field(gt=0)] = 7.94
    pitch_diameter_mm: Annotated[float, Field(gt=0)] = 39.04
    contact_angle_deg: Annotated[float, Field(ge=0, lt=90)] = 0.0

    @model_validator(mode="after")
    def _ball_smaller_than_pitch(self) -> BearingGeometry:
        if self.ball_diameter_mm >= self.pitch_diameter_mm:
            msg = "ball_diameter_mm must be smaller than pitch_diameter_mm"
            raise ValueError(msg)
        return self


class SimulationConfig(BaseModel):
    """Parameters of one simulation run of the digital twin."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1, max_length=64)
    asset_type: AssetType = AssetType.INDUCTION_MOTOR
    fault: FaultType = FaultType.HEALTHY
    severity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    shaft_speed_hz: Annotated[float, Field(gt=0.0, le=500.0)] = 29.95
    sampling_rate_hz: Annotated[int, Field(ge=1_000, le=100_000)] = 12_000
    duration_s: Annotated[float, Field(gt=0.0, le=600.0)] = 1.0
    noise_std: Annotated[float, Field(ge=0.0)] = 0.05
    seed: int = 0
    bearing: BearingGeometry = Field(default_factory=BearingGeometry)

    @model_validator(mode="after")
    def _severity_consistent_with_fault(self) -> SimulationConfig:
        if self.fault is FaultType.HEALTHY and self.severity != 0.0:
            msg = "severity must be 0.0 when fault is 'healthy'"
            raise ValueError(msg)
        return self


class VibrationWindow(BaseModel):
    """A fixed-length window of vibration samples with its metadata and ground truth."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sampling_rate_hz: Annotated[int, Field(ge=1_000, le=100_000)]
    shaft_speed_hz: Annotated[float, Field(gt=0.0)]
    samples: Annotated[list[float], Field(min_length=1)]
    label: FaultType = FaultType.HEALTHY
    severity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class EdgeInference(BaseModel):
    """Result reported by the STM32 after running the on-device model."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    anomaly_score: Annotated[float, Field(ge=0.0)]
    predicted_class: FaultType
    latency_us: Annotated[int, Field(ge=0)]
    model_version: str = Field(min_length=1)
