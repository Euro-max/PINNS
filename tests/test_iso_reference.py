"""The obstacle-avoidance reference through the ISO 3888-2:2011 cone layout (pinc/refs.py, IsoLaneChange)."""
import numpy as np
import pytest

from pinc.config import load_config
from pinc.refs import iso3888_lanes, make_reference


def test_lane_geometry_matches_the_standard():
    lanes = iso3888_lanes(1.8)
    # widths 1.1 W + 0.25, W + 1 and max(1.3 W + 0.25, 3); 1 m clear gap -> centre shifts of 3.515 and 3.9 m
    assert [round(2*sl + 1.8, 3) for *_, sl in lanes] == [2.23, 2.8, 3.0]
    assert lanes[1][2] == pytest.approx(3.515) and lanes[1][2] - lanes[2][2] == pytest.approx(3.9)
    assert [(a, b) for a, b, *_ in lanes] == [(0.0, 12.0), (25.5, 36.5), (49.0, 61.0)]


def test_path_stays_in_every_lane_and_below_the_friction_limit():
    ref = make_reference("iso_lane_change", load_config(None))
    X = np.linspace(0, 100, 20001)
    Y, dY = ref.path(X)
    for a, b, c, slack in iso3888_lanes(1.8):
        m = (X - ref.run_in >= a) & (X - ref.run_in <= b)
        assert np.max(np.abs(Y[m] - c)) <= slack + 1e-6
    assert abs(Y[0]) < 1e-3 and np.allclose(np.gradient(Y, X), dY, atol=1e-4)
    assert ref.peak_lateral_acceleration() < 8.0           # m/s^2; the tyre limit at static load is about 8.6
