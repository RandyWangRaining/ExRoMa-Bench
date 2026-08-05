from __future__ import annotations

import math

import pytest

from exroma_bench.sim.domains import DOMAINS, NASA_OUTPOST_SCALE


def test_nasa_outpost_uses_source_asset_scale() -> None:
    assert NASA_OUTPOST_SCALE == pytest.approx(0.02)


def _quat_to_roll_pitch(quaternion) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(2.0 * (w * y - z * x))
    return math.degrees(roll), math.degrees(pitch)


def test_moon_lighting_matches_reference_seed_zero_reset() -> None:
    moon = DOMAINS["moon"]
    assert moon.sun_intensity == pytest.approx(1360.8160400390625)
    assert _quat_to_roll_pitch(moon.sun_orientation) == pytest.approx(
        (-12.11445, 2.001503), abs=1e-5
    )
    assert moon.sky_orientation == pytest.approx(
        (
            0.023698091506958008,
            -0.6120024919509888,
            -0.009596887975931168,
            0.7904424667358398,
        )
    )
