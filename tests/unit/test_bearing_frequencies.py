"""Unit tests for bearing fault frequency computation."""

import pytest

from digital_twin.faults.bearing_frequencies import compute_fault_frequencies
from services.common.schemas import BearingGeometry


def test_cwru_fault_orders_match_published_values() -> None:
    freqs = compute_fault_frequencies(BearingGeometry(), shaft_speed_hz=1.0)
    assert freqs.bpfo_hz == pytest.approx(3.5848, abs=1e-3)
    assert freqs.bpfi_hz == pytest.approx(5.4152, abs=1e-3)
    assert freqs.ftf_hz == pytest.approx(0.3983, abs=1e-3)
    assert freqs.bsf_hz == pytest.approx(2.3567, abs=1e-3)
    assert 2.0 * freqs.bsf_hz == pytest.approx(4.7135, abs=2e-3)


def test_frequencies_at_cwru_nominal_speed() -> None:
    freqs = compute_fault_frequencies(BearingGeometry(), shaft_speed_hz=29.95)
    assert freqs.bpfo_hz == pytest.approx(107.36, abs=0.05)
    assert freqs.bpfi_hz == pytest.approx(162.19, abs=0.05)


def test_bpfo_plus_bpfi_equals_ball_count_times_shaft_speed() -> None:
    bearing = BearingGeometry(contact_angle_deg=15.0)
    freqs = compute_fault_frequencies(bearing, shaft_speed_hz=25.0)
    assert freqs.bpfo_hz + freqs.bpfi_hz == pytest.approx(bearing.n_balls * 25.0)


def test_frequencies_scale_linearly_with_speed() -> None:
    slow = compute_fault_frequencies(BearingGeometry(), shaft_speed_hz=10.0)
    fast = compute_fault_frequencies(BearingGeometry(), shaft_speed_hz=20.0)
    assert fast.bpfo_hz == pytest.approx(2.0 * slow.bpfo_hz)
    assert fast.bsf_hz == pytest.approx(2.0 * slow.bsf_hz)


@pytest.mark.parametrize("speed", [0.0, -5.0])
def test_non_positive_speed_is_rejected(speed: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        compute_fault_frequencies(BearingGeometry(), shaft_speed_hz=speed)
