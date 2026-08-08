"""Graded beam transmittance: what fraction of direct sunlight actually arrives.

A horizon polyline answers "is the sun blocked?" with yes or no. Real obstructions
are mostly trees, and trees are porous: they attenuate across a band of elevations
rather than cutting off at one. A graded transmittance field ``T(azimuth,
elevation)`` measures that directly, and on the site this was built for it cut
five-minute RMSE by 10-19% against a no-shading baseline where the hard horizon
managed noticeably less.

Two rules, both load-bearing:

* **Apply ``T`` to the beam only.** A tree blocking the sun disk still passes most
  of the sky. Scaling total irradiance by ``T`` over-penalises badly.
* **Absolute values are good to roughly +/-25%; the shape is what is reliable.**
  Storing at 8-bit precision is therefore lossless in every sense that matters,
  and turns a 262 kB grid per array into 33 kB.
"""

from dataclasses import dataclass
import io

import numpy as np
from numpy.typing import NDArray

from custom_components.soular.core.types import FloatArray

# Width of the soft edge used when converting a hard horizon to a graded grid.
# A step function aliases badly against a five-minute sun track, which moves
# ~1.25 degrees of azimuth per step: the array would flick between lit and dark
# on consecutive samples with nothing in between.
HORIZON_EDGE_WIDTH_DEG = 2.0

# Bilinear interpolation needs a cell, so each axis needs at least two samples,
# and a horizon row needs an azimuth and an elevation.
MIN_AXIS_SAMPLES = 2
HORIZON_COLUMNS = 2


class ShadingFormatError(ValueError):
    """A shading file could not be parsed, or did not contain the named array."""


@dataclass(frozen=True, slots=True)
class TransmittanceGrid:
    """Beam transmittance on an azimuth x elevation grid, stored as 8-bit.

    ``azimuth_deg`` is compass degrees from north, clockwise, ascending and
    within [0, 360). ``elevation_deg`` is degrees above the horizon, ascending.
    ``values`` is indexed ``[azimuth, elevation]``.
    """

    azimuth_deg: FloatArray
    elevation_deg: FloatArray
    values: NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Validate axis ordering and shape."""
        if self.values.shape != (self.azimuth_deg.size, self.elevation_deg.size):
            msg = (
                f"values shape {self.values.shape} does not match axes "
                f"({self.azimuth_deg.size}, {self.elevation_deg.size})"
            )
            raise ShadingFormatError(msg)
        if self.azimuth_deg.size < MIN_AXIS_SAMPLES or self.elevation_deg.size < MIN_AXIS_SAMPLES:
            msg = "a transmittance grid needs at least two samples on each axis"
            raise ShadingFormatError(msg)
        if not (np.all(np.diff(self.azimuth_deg) > 0) and np.all(np.diff(self.elevation_deg) > 0)):
            msg = "transmittance grid axes must be strictly increasing"
            raise ShadingFormatError(msg)

    def lookup(self, azimuth_deg: FloatArray, elevation_deg: FloatArray) -> FloatArray:
        """Bilinearly interpolate transmittance at sun positions.

        Azimuth wraps at 360; elevation clamps to the ends of the grid. Clamping
        rather than extrapolating means a grid that starts at 4 degrees treats
        everything below it as being as blocked as its lowest measured row, which
        is the conservative and usually correct reading near the horizon.
        """
        # Close the azimuth axis so interpolation across north is continuous.
        az_axis = np.concatenate([self.azimuth_deg, self.azimuth_deg[:1] + 360.0])
        values = np.concatenate([self.values, self.values[:1]], axis=0).astype(np.float64) / 255.0

        az = np.mod(np.asarray(azimuth_deg, dtype=np.float64), 360.0)
        # Below az_axis[0] the wrapped interval is the one ending at az_axis[0],
        # i.e. the last cell; shifting by a period puts it back inside the axis.
        az = np.where(az < az_axis[0], az + 360.0, az)
        el = np.clip(np.asarray(elevation_deg, dtype=np.float64), self.elevation_deg[0], self.elevation_deg[-1])

        ia, fa = _bracket(az_axis, az)
        ie, fe = _bracket(self.elevation_deg, el)

        v00 = values[ia, ie]
        v10 = values[ia + 1, ie]
        v01 = values[ia, ie + 1]
        v11 = values[ia + 1, ie + 1]
        lower = v00 * (1.0 - fa) + v10 * fa
        upper = v01 * (1.0 - fa) + v11 * fa
        result = lower * (1.0 - fe) + upper * fe
        return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)


def _bracket(axis: FloatArray, values: FloatArray) -> tuple[NDArray[np.intp], FloatArray]:
    """Return the lower cell index and the fractional position within it."""
    idx = np.clip(np.searchsorted(axis, values, side="right") - 1, 0, axis.size - 2)
    span = axis[idx + 1] - axis[idx]
    frac = np.clip((values - axis[idx]) / span, 0.0, 1.0)
    return idx, np.asarray(frac, dtype=np.float64)


def _quantise(values: FloatArray) -> NDArray[np.uint8]:
    """Round transmittance to 8 bits, the precision the measurement supports."""
    return np.asarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0), dtype=np.uint8)


def _drop_duplicate_endpoint(azimuth: FloatArray, values: NDArray[np.uint8]) -> tuple[FloatArray, NDArray[np.uint8]]:
    """Trim a redundant 360-degree column so the axis is a half-open period."""
    if azimuth.size >= MIN_AXIS_SAMPLES and np.isclose(azimuth[-1] - azimuth[0], 360.0):
        return azimuth[:-1], values[:-1]
    return azimuth, values


def from_npz(data: bytes, array_name: str) -> TransmittanceGrid:
    """Parse a graded grid from a ``.npz`` archive.

    Expects ``azimuth_deg``, ``elevation_deg`` and ``T_<array_name>``, matching
    the layout produced by the site analysis.
    """
    key = f"T_{array_name}"
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        available = list(archive.files)
        if key not in available:
            grids = sorted(name.removeprefix("T_") for name in available if name.startswith("T_"))
            msg = f"no transmittance grid named {array_name!r}; this file has {grids or 'none'}"
            raise ShadingFormatError(msg)
        azimuth = np.asarray(archive["azimuth_deg"], dtype=np.float64)
        elevation = np.asarray(archive["elevation_deg"], dtype=np.float64)
        values = _quantise(np.asarray(archive[key], dtype=np.float64))

    azimuth, values = _drop_duplicate_endpoint(azimuth, values)
    return TransmittanceGrid(azimuth_deg=azimuth, elevation_deg=elevation, values=values)


def from_csv(text: str, array_name: str) -> TransmittanceGrid:
    """Parse a graded grid from the long-format CSV.

    Columns ``array, azimuth_deg, elevation_deg, transmittance``. Pivoted back
    into a dense grid; missing cells are an error rather than a silent gap,
    because a hole in a shading map reads as open sky.
    """
    azimuths: list[float] = []
    elevations: list[float] = []
    readings: dict[tuple[float, float], float] = {}

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        msg = "shading CSV is empty"
        raise ShadingFormatError(msg)
    header = [column.strip().lower() for column in lines[0].split(",")]
    try:
        col_array = header.index("array")
        col_az = header.index("azimuth_deg")
        col_el = header.index("elevation_deg")
        col_t = header.index("transmittance")
    except ValueError as err:
        msg = f"shading CSV header must contain array, azimuth_deg, elevation_deg, transmittance; got {header}"
        raise ShadingFormatError(msg) from err

    for line in lines[1:]:
        fields = line.split(",")
        if fields[col_array].strip() != array_name:
            continue
        azimuth = float(fields[col_az])
        elevation = float(fields[col_el])
        azimuths.append(azimuth)
        elevations.append(elevation)
        readings[azimuth, elevation] = float(fields[col_t])

    if not readings:
        msg = f"no rows for array {array_name!r} in shading CSV"
        raise ShadingFormatError(msg)

    az_axis = np.array(sorted(set(azimuths)), dtype=np.float64)
    el_axis = np.array(sorted(set(elevations)), dtype=np.float64)
    dense = np.full((az_axis.size, el_axis.size), np.nan, dtype=np.float64)
    az_pos = {value: i for i, value in enumerate(az_axis)}
    el_pos = {value: i for i, value in enumerate(el_axis)}
    for (azimuth, elevation), value in readings.items():
        dense[az_pos[azimuth], el_pos[elevation]] = value

    if np.isnan(dense).any():
        missing = int(np.isnan(dense).sum())
        msg = f"shading CSV for {array_name!r} is not a complete grid: {missing} cells missing"
        raise ShadingFormatError(msg)

    azimuth_axis, values = _drop_duplicate_endpoint(az_axis, _quantise(dense))
    return TransmittanceGrid(azimuth_deg=azimuth_axis, elevation_deg=el_axis, values=values)


def from_horizon(text: str, *, edge_width_deg: float = HORIZON_EDGE_WIDTH_DEG) -> TransmittanceGrid:
    """Convert a two-column ``azimuth<TAB>elevation`` horizon into a graded grid.

    This is the format the incumbent integration reads, so an existing setup can
    be carried over unchanged. The result is strictly less informative than a
    measured grid -- a hard skyline cannot express a porous canopy -- but the soft
    edge at least stops the sun snapping between lit and dark between samples.
    """
    horizon_az: list[float] = []
    horizon_el: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.replace(",", "\t").split()
        if len(fields) < HORIZON_COLUMNS:
            msg = f"horizon file rows need two columns, got {line!r}"
            raise ShadingFormatError(msg)
        try:
            horizon_az.append(float(fields[0]))
            horizon_el.append(float(fields[1]))
        except ValueError as err:
            msg = f"horizon file row is not numeric: {line!r}"
            raise ShadingFormatError(msg) from err

    if len(horizon_az) < MIN_AXIS_SAMPLES:
        msg = "horizon file needs at least two points"
        raise ShadingFormatError(msg)

    order = np.argsort(np.asarray(horizon_az, dtype=np.float64))
    src_az = np.asarray(horizon_az, dtype=np.float64)[order]
    src_el = np.asarray(horizon_el, dtype=np.float64)[order]

    az_axis: FloatArray = np.arange(0.0, 360.0, 1.0)
    el_axis: FloatArray = np.arange(0.0, 91.0, 1.0)
    # Periodic interpolation of the skyline onto a regular azimuth axis.
    skyline = np.interp(az_axis, src_az, src_el, period=360.0)

    # Transmittance ramps from 0 at the skyline to 1 one edge-width above it.
    above = el_axis[None, :] - skyline[:, None]
    graded = np.clip(above / edge_width_deg, 0.0, 1.0)
    return TransmittanceGrid(azimuth_deg=az_axis, elevation_deg=el_axis, values=_quantise(graded))


def open_sky(azimuth_deg: FloatArray) -> FloatArray:
    """Transmittance for an array with no configured shading."""
    return np.ones_like(np.asarray(azimuth_deg, dtype=np.float64))
