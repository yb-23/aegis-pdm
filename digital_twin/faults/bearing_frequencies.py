"""Characteristic fault frequencies of rolling-element bearings.

Assumes a fixed outer race and an inner race rotating with the shaft.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from services.common.schemas import BearingGeometry


@dataclass(frozen=True)
class BearingFaultFrequencies:
    """Fault frequencies in Hz for one bearing at one shaft speed."""

    shaft_hz: float
    bpfo_hz: float
    bpfi_hz: float
    bsf_hz: float
    ftf_hz: float


def compute_fault_frequencies(
    bearing: BearingGeometry, shaft_speed_hz: float
) -> BearingFaultFrequencies:
    """Compute BPFO, BPFI, BSF and FTF from the bearing geometry and shaft speed."""
    if shaft_speed_hz <= 0.0:
        msg = "shaft_speed_hz must be strictly positive"
        raise ValueError(msg)

    cos_theta = math.cos(math.radians(bearing.contact_angle_deg))
    ratio = bearing.ball_diameter_mm / bearing.pitch_diameter_mm * cos_theta
    half_balls = bearing.n_balls / 2.0

    bpfo = half_balls * shaft_speed_hz * (1.0 - ratio)
    bpfi = half_balls * shaft_speed_hz * (1.0 + ratio)
    bsf = (
        bearing.pitch_diameter_mm
        / (2.0 * bearing.ball_diameter_mm)
        * shaft_speed_hz
        * (1.0 - ratio**2)
    )
    ftf = 0.5 * shaft_speed_hz * (1.0 - ratio)

    return BearingFaultFrequencies(
        shaft_hz=shaft_speed_hz,
        bpfo_hz=bpfo,
        bpfi_hz=bpfi,
        bsf_hz=bsf,
        ftf_hz=ftf,
    )
