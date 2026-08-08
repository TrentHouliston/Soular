"""Tests for the graded transmittance grid and its parsers.

The failure mode these guard against is silent and plausible: a shading map that
is mirrored or rotated still produces a smooth, believable-looking forecast, it
is just wrong. So the conventions are asserted directly rather than inferred from
end-to-end behaviour.
"""

import io
from typing import Any

import numpy as np
import pytest

from custom_components.soular.core.shading import (
    ShadingFormatError,
    TransmittanceGrid,
    from_csv,
    from_horizon,
    from_npz,
)


def build_grid(values: np.ndarray, azimuth_step: float = 90.0, elevation_step: float = 30.0) -> TransmittanceGrid:
    """Build a grid from a float array of transmittances."""
    azimuth = np.arange(0.0, 360.0, azimuth_step)
    elevation = np.arange(0.0, 91.0, elevation_step)
    return TransmittanceGrid(
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        values=np.rint(values * 255.0).astype(np.uint8),
    )


def uniform_grid(value: float) -> TransmittanceGrid:
    """Build a grid with the same transmittance everywhere."""
    return build_grid(np.full((4, 4), value))


def test_uniform_grid_returns_its_value() -> None:
    """A constant grid is constant everywhere, including at cell boundaries."""
    grid = uniform_grid(0.5)
    azimuth = np.array([0.0, 45.0, 90.0, 271.3, 359.9])
    elevation = np.array([0.0, 15.0, 30.0, 61.0, 90.0])
    result = grid.lookup(azimuth, elevation)
    np.testing.assert_allclose(result, np.full(5, np.rint(0.5 * 255) / 255.0))


def test_lookup_is_bilinear_between_cells() -> None:
    """Halfway between two elevation rows gives the mean of the two."""
    values = np.zeros((4, 4))
    values[:, 1] = 0.0
    values[:, 2] = 1.0
    grid = build_grid(values)
    # Elevation axis is 0/30/60/90; 45 degrees sits midway between rows 1 and 2.
    midpoint = grid.lookup(np.array([0.0]), np.array([45.0]))
    assert midpoint[0] == pytest.approx(0.5, abs=1.0 / 255.0)


def test_azimuth_wraps_across_north() -> None:
    """Interpolation is continuous through 0/360, not discontinuous at the seam.

    Getting this wrong shows up as a hard step in the forecast at true north,
    which at this latitude is the middle of the day.
    """
    values = np.zeros((4, 4))
    values[0, :] = 1.0  # the cell centred on azimuth 0
    values[3, :] = 0.0  # the cell centred on azimuth 270
    grid = build_grid(values)

    just_before = grid.lookup(np.array([359.9]), np.array([45.0]))[0]
    at_north = grid.lookup(np.array([0.0]), np.array([45.0]))[0]
    just_after = grid.lookup(np.array([0.1]), np.array([45.0]))[0]

    assert just_before == pytest.approx(at_north, abs=0.01)
    assert just_after == pytest.approx(at_north, abs=0.01)


def test_azimuth_is_periodic() -> None:
    """Adding a full turn changes nothing."""
    grid = build_grid(np.linspace(0.0, 1.0, 16).reshape(4, 4))
    azimuth = np.array([12.0, 100.0, 250.0])
    elevation = np.array([20.0, 40.0, 70.0])
    np.testing.assert_allclose(
        grid.lookup(azimuth, elevation),
        grid.lookup(azimuth + 720.0, elevation),
    )


def test_elevation_clamps_rather_than_extrapolates() -> None:
    """Below the lowest measured row, keep that row rather than inventing sky."""
    values = np.zeros((4, 4))
    values[:, 0] = 0.2
    values[:, -1] = 1.0
    grid = build_grid(values)

    below = grid.lookup(np.array([0.0]), np.array([-10.0]))[0]
    at_bottom = grid.lookup(np.array([0.0]), np.array([0.0]))[0]
    above = grid.lookup(np.array([0.0]), np.array([120.0]))[0]

    assert below == pytest.approx(at_bottom)
    assert above == pytest.approx(1.0, abs=1.0 / 255.0)


def test_lookup_stays_in_unit_range() -> None:
    """Transmittance is a fraction; interpolation must not escape [0, 1]."""
    grid = build_grid(np.random.default_rng(0).uniform(0.0, 1.0, (4, 4)))
    azimuth = np.linspace(-720.0, 1080.0, 2001)
    elevation = np.linspace(-30.0, 120.0, 2001)
    result = grid.lookup(azimuth, elevation)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_rejects_unordered_axes() -> None:
    """A descending axis would silently mirror the map."""
    with pytest.raises(ShadingFormatError, match="strictly increasing"):
        TransmittanceGrid(
            azimuth_deg=np.array([180.0, 90.0, 0.0]),
            elevation_deg=np.array([0.0, 45.0, 90.0]),
            values=np.zeros((3, 3), dtype=np.uint8),
        )


def test_rejects_mismatched_shape() -> None:
    """Axis lengths and value shape must agree."""
    with pytest.raises(ShadingFormatError, match="does not match axes"):
        TransmittanceGrid(
            azimuth_deg=np.array([0.0, 90.0, 180.0]),
            elevation_deg=np.array([0.0, 45.0]),
            values=np.zeros((2, 3), dtype=np.uint8),
        )


def npz_bytes(**arrays: Any) -> bytes:
    """Serialise arrays to an in-memory npz archive."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def test_from_npz_reads_named_array() -> None:
    """The npz layout produced by the site analysis round-trips."""
    azimuth = np.arange(0.0, 361.0, 1.0)  # includes both endpoints, as the real file does
    elevation = np.arange(0.0, 91.0, 1.0)
    east = np.full((361, 91), 0.6)
    west = np.full((361, 91), 0.9)

    grid = from_npz(npz_bytes(azimuth_deg=azimuth, elevation_deg=elevation, T_east=east, T_west=west), "east")

    # The duplicated 360-degree column is dropped so the axis is a half-open period.
    assert grid.azimuth_deg.size == 360
    assert grid.lookup(np.array([123.0]), np.array([45.0]))[0] == pytest.approx(0.6, abs=1.0 / 255.0)


def test_from_npz_names_the_arrays_it_does_have() -> None:
    """A wrong array name is a common misconfiguration; say what is available."""
    data = npz_bytes(
        azimuth_deg=np.arange(0.0, 360.0),
        elevation_deg=np.arange(0.0, 91.0),
        T_east=np.zeros((360, 91)),
    )
    with pytest.raises(ShadingFormatError, match=r"\['east'\]"):
        from_npz(data, "north")


def test_from_csv_pivots_long_format() -> None:
    """The long CSV form pivots back into the same grid."""
    rows = ["array,azimuth_deg,elevation_deg,transmittance"]
    for azimuth in (0.0, 90.0, 180.0, 270.0):
        for elevation in (0.0, 45.0, 90.0):
            rows.append(f"east,{azimuth},{elevation},0.4")
            rows.append(f"west,{azimuth},{elevation},0.8")

    grid = from_csv("\n".join(rows), "east")
    assert grid.lookup(np.array([45.0]), np.array([20.0]))[0] == pytest.approx(0.4, abs=1.0 / 255.0)


def test_from_csv_rejects_incomplete_grid() -> None:
    """A hole in a shading map reads as open sky, so refuse to guess."""
    rows = [
        "array,azimuth_deg,elevation_deg,transmittance",
        "east,0,0,0.5",
        "east,0,45,0.5",
        "east,90,0,0.5",
        # (90, 45) deliberately missing
    ]
    with pytest.raises(ShadingFormatError, match="not a complete grid"):
        from_csv("\n".join(rows), "east")


def test_from_horizon_blocks_below_the_skyline() -> None:
    """A hard horizon becomes zero below the skyline and one well above it."""
    text = "\n".join(f"{azimuth}\t{30.0 if azimuth < 180 else 0.0}" for azimuth in range(0, 361, 2))
    grid = from_horizon(text)

    blocked = grid.lookup(np.array([90.0]), np.array([10.0]))[0]
    lit = grid.lookup(np.array([90.0]), np.array([50.0]))[0]
    open_side = grid.lookup(np.array([270.0]), np.array([10.0]))[0]

    assert blocked == pytest.approx(0.0, abs=1e-6)
    assert lit == pytest.approx(1.0, abs=1.0 / 255.0)
    assert open_side == pytest.approx(1.0, abs=1.0 / 255.0)


def test_from_horizon_edge_is_soft() -> None:
    """The transition ramps rather than stepping.

    A five-minute sun track covers about 1.25 degrees of azimuth per sample. A
    hard step makes the array flick between lit and dark on consecutive samples
    with nothing in between, which is neither physical nor useful to an optimiser.
    """
    text = "\n".join(f"{azimuth}\t20.0" for azimuth in range(0, 361, 2))
    grid = from_horizon(text, edge_width_deg=2.0)

    ramp = grid.lookup(np.full(3, 90.0), np.array([20.0, 21.0, 22.0]))
    assert ramp[0] == pytest.approx(0.0, abs=1e-6)
    assert ramp[1] == pytest.approx(0.5, abs=0.02)
    assert ramp[2] == pytest.approx(1.0, abs=1.0 / 255.0)


def test_from_horizon_rejects_junk() -> None:
    """A non-numeric row is a corrupt file, not a comment."""
    with pytest.raises(ShadingFormatError, match="not numeric"):
        from_horizon("0\t10\n90\tbanana\n180\t5\n")
