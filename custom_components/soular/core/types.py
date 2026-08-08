"""Value types shared across the forecasting core.

Everything here is frozen and slotted. The core is pure computation, so a spec
that could change under a caller mid-pipeline would be a bug rather than a
feature. Angles are degrees and powers are watts throughout; the one convention
that bites is azimuth, which is compass degrees from north, clockwise (0 = N,
90 = E, 180 = S, 270 = W) everywhere in this package.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.floating[Any]]
type TimeArray = NDArray[np.datetime64]

type TranspositionModel = Literal["perez", "haydavies", "isotropic"]


@dataclass(frozen=True, slots=True)
class SiteSpec:
    """A location. Every array on a site shares it."""

    latitude: float
    longitude: float
    elevation_m: float = 0.0
    albedo: float = 0.20
    transposition_model: TranspositionModel = "perez"


@dataclass(frozen=True, slots=True)
class InverterSpec:
    """An inverter, so clipping is modelled where it physically happens.

    Clipping the mean of an interval understates the energy lost inside it, so
    the pipeline evaluates this on its fine grid and aggregates afterwards.
    """

    name: str
    ac_limit_w: float
    dc_limit_w: float | None = None
    eta_nom: float = 0.96
    model: Literal["constant", "pvwatts"] = "pvwatts"


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """One coplanar group of modules on one inverter input.

    ``dc_loss_fraction`` deliberately collapses soiling, wiring, mismatch and
    connection losses into a single number. They are not separately identifiable
    from production data, and the online correction re-estimates their product
    anyway, so offering one slider per mechanism would invent precision.
    """

    name: str
    azimuth_deg: float
    tilt_deg: float
    dc_capacity_w: float
    gamma_pdc: float = -0.0035
    dc_loss_fraction: float = 0.14
    inverter: str = "default"
    # Wind is reported at 10 m; Faiman's coefficients want it at module height.
    # See core.pvmodel.cell_temperature for why this is not folded into u0/u1.
    wind_height_factor: float = 0.67


@dataclass(frozen=True, slots=True)
class SolarGeometry:
    """Sun position and airmass, evaluated at a set of instants."""

    apparent_zenith: FloatArray
    apparent_elevation: FloatArray
    azimuth: FloatArray
    airmass_relative: FloatArray
    airmass_absolute: FloatArray
    dni_extra: FloatArray


@dataclass(frozen=True, slots=True)
class ClearSky:
    """Ineichen-Perez clear-sky irradiance components, W/m^2."""

    ghi: FloatArray
    dni: FloatArray
    dhi: FloatArray


@dataclass(frozen=True, slots=True)
class PlaneOfArray:
    """Plane-of-array irradiance for one array, W/m^2.

    Beam and diffuse stay separate all the way through because shading acts on
    the beam alone: a tree blocking the sun disk still passes most of the sky.
    Collapsing to a total early would put a floor under the transmittance that
    looks like partial shading everywhere.
    """

    aoi: FloatArray
    poa_beam: FloatArray
    poa_diffuse: FloatArray
    iam_beam: FloatArray

    @property
    def poa_global(self) -> FloatArray:
        """Total effective plane-of-array irradiance."""
        return self.poa_beam + self.poa_diffuse


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """The instants a forecast is evaluated at, and the intervals they bound.

    ``times`` are instants, not interval labels. Consumers that need energy
    integrate; consumers that need power sample. Keeping this explicit avoids
    the half-interval phase error that comes from labelling an interval mean
    with one of its endpoints.
    """

    times: TimeArray
    step_seconds: FloatArray = field(repr=False)

    def __post_init__(self) -> None:
        """Validate that the grid is non-empty, ordered and consistently sized."""
        if self.times.size == 0:
            msg = "TimeGrid must contain at least one instant"
            raise ValueError(msg)
        if self.times.size != self.step_seconds.size:
            msg = f"times ({self.times.size}) and step_seconds ({self.step_seconds.size}) must be the same length"
            raise ValueError(msg)
        if self.times.size > 1 and not np.all(np.diff(self.times.astype("datetime64[s]")) > np.timedelta64(0, "s")):
            msg = "TimeGrid times must be strictly increasing"
            raise ValueError(msg)
