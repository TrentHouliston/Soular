"""The one entry point: weather in, per-array and site power out.

Everything that forecasts calls :func:`forecast` -- the integration, the unit
tests, and the offline backtest harness. That is the whole point of keeping this
package free of Home Assistant. A skill number measured by the backtest describes
the model that actually ships, not a reimplementation of it that has since
drifted.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.irradiance import decompose, plane_of_array
from custom_components.soular.core.pvmodel import ac_power, cell_temperature, dc_power
from custom_components.soular.core.shading import TransmittanceGrid, open_sky
from custom_components.soular.core.types import (
    ArraySpec,
    ClearSky,
    FloatArray,
    InverterSpec,
    SiteSpec,
    TimeArray,
    TimeGrid,
)

# The near-term grid has to be fine enough to resolve clipping and to give haeo
# something to trapezoid-integrate over its one-minute leading tier; beyond a few
# hours the weather models have nothing that fine to say.
NEAR_TERM_STEP_MINUTES = 5
FAR_TERM_STEP_MINUTES = 15
NEAR_TERM_HOURS = 6
DEFAULT_HORIZON_HOURS = 48


@dataclass(frozen=True, slots=True)
class WeatherSeries:
    """Irradiance and ambient conditions sampled on the forecast grid.

    ``dni`` and ``dhi`` are optional. When absent the beam/diffuse split is
    derived from GHI, which is what happens whenever a satellite observation has
    contributed, since those products carry shortwave radiation only.
    """

    ghi: FloatArray
    temp_air: FloatArray
    wind_speed_10m: FloatArray
    dni: FloatArray | None = None
    dhi: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class SystemSpec:
    """A configured site: its location, its arrays, and how they are wired."""

    site: SiteSpec
    arrays: Sequence[ArraySpec]
    inverters: Mapping[str, InverterSpec]
    shading: Mapping[str, TransmittanceGrid] = field(default_factory=dict)

    def inverter_for(self, array: ArraySpec) -> InverterSpec:
        """Look up an array's inverter, failing loudly on a misconfiguration."""
        try:
            return self.inverters[array.inverter]
        except KeyError as err:
            known = sorted(self.inverters)
            msg = f"array {array.name!r} references unknown inverter {array.inverter!r}; configured: {known}"
            raise KeyError(msg) from err


@dataclass(frozen=True, slots=True)
class ArrayForecast:
    """Per-array output. Diagnostic fields are carried so entities stay dumb."""

    name: str
    dc_power_w: FloatArray
    ac_power_w: FloatArray
    poa_global: FloatArray
    poa_beam: FloatArray
    cell_temperature: FloatArray
    transmittance: FloatArray


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """A complete forecast on one time grid."""

    times: TimeArray
    arrays: tuple[ArrayForecast, ...]
    ac_power_w: FloatArray
    clearsky: ClearSky

    def array(self, name: str) -> ArrayForecast:
        """Look up one array's forecast by name."""
        for entry in self.arrays:
            if entry.name == name:
                return entry
        msg = f"no array named {name!r} in this forecast"
        raise KeyError(msg)


def build_time_grid(
    start: np.datetime64,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    near_term_hours: int = NEAR_TERM_HOURS,
) -> TimeGrid:
    """Build the standard mixed-resolution grid: fine near-term, coarse beyond.

    The horizon is a whole number of hours starting at ``start`` because haeo pads
    a forecast series to whole days by wrapping its head onto its tail. A series
    that stops mid-day gets this morning spliced onto its end.
    """
    origin = start.astype("datetime64[s]")
    near_end = origin + np.timedelta64(near_term_hours * 3600, "s")
    far_end = origin + np.timedelta64(horizon_hours * 3600, "s")

    near = np.arange(origin, near_end, np.timedelta64(NEAR_TERM_STEP_MINUTES * 60, "s")).astype("datetime64[s]")
    far = np.arange(near_end, far_end, np.timedelta64(FAR_TERM_STEP_MINUTES * 60, "s")).astype("datetime64[s]")
    times = np.concatenate([near, far])

    steps = np.concatenate(
        [
            np.full(near.size, NEAR_TERM_STEP_MINUTES * 60.0),
            np.full(far.size, FAR_TERM_STEP_MINUTES * 60.0),
        ]
    )
    return TimeGrid(times=times, step_seconds=steps)


def forecast(system: SystemSpec, grid: TimeGrid, weather: WeatherSeries) -> ForecastResult:
    """Run the full model: geometry, transposition, shading, PV, inverter.

    Inverter limits are applied to the summed DC of everything on that inverter,
    on this grid, before any aggregation to coarser intervals. Per-array AC is
    then attributed back pro rata, so per-array sensors sum to the site total
    even when several arrays clip together.
    """
    _validate_weather(grid, weather)

    geometry = solar_geometry(grid.times, system.site)
    clearsky = clear_sky(grid.times, system.site, geometry)

    if weather.dni is None or weather.dhi is None:
        dni, dhi = decompose(grid.times, weather.ghi, geometry)
    else:
        dni, dhi = weather.dni, weather.dhi

    dc_by_array: dict[str, FloatArray] = {}
    parts: dict[str, tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = {}

    for array in system.arrays:
        poa = plane_of_array(array, system.site, geometry, weather.ghi, dni, dhi)

        shading = system.shading.get(array.name)
        transmittance = (
            shading.lookup(geometry.azimuth, geometry.apparent_elevation)
            if shading is not None
            else open_sky(geometry.azimuth)
        )
        # Transmittance multiplies the beam alone. The diffuse a tree lets past is
        # most of the diffuse there was.
        poa_effective = poa.poa_beam * transmittance + poa.poa_diffuse

        temp_cell = cell_temperature(poa_effective, weather.temp_air, weather.wind_speed_10m, array)
        dc = dc_power(poa_effective, temp_cell, array)

        dc_by_array[array.name] = dc
        parts[array.name] = (poa_effective, poa.poa_beam * transmittance, temp_cell, transmittance)

    ac_by_array = _distribute_ac(system, dc_by_array, grid.times.size)

    forecasts = tuple(
        ArrayForecast(
            name=array.name,
            dc_power_w=dc_by_array[array.name],
            ac_power_w=ac_by_array[array.name],
            poa_global=parts[array.name][0],
            poa_beam=parts[array.name][1],
            cell_temperature=parts[array.name][2],
            transmittance=parts[array.name][3],
        )
        for array in system.arrays
    )
    site_ac = (
        np.sum([entry.ac_power_w for entry in forecasts], axis=0)
        if forecasts
        else np.zeros(grid.times.size, dtype=np.float64)
    )

    return ForecastResult(
        times=grid.times,
        arrays=forecasts,
        ac_power_w=np.asarray(site_ac, dtype=np.float64),
        clearsky=clearsky,
    )


def _distribute_ac(
    system: SystemSpec,
    dc_by_array: Mapping[str, FloatArray],
    steps: int,
) -> dict[str, FloatArray]:
    """Clip per inverter, then split the AC back over that inverter's arrays."""
    groups: dict[str, list[ArraySpec]] = {}
    for array in system.arrays:
        groups.setdefault(array.inverter, []).append(array)

    result: dict[str, FloatArray] = {}
    for inverter_name, arrays in groups.items():
        inverter = system.inverter_for(arrays[0])
        if inverter.name != inverter_name:
            msg = f"inverter registered as {inverter_name!r} reports its name as {inverter.name!r}"
            raise ValueError(msg)

        total_dc = np.sum([dc_by_array[array.name] for array in arrays], axis=0)
        total_ac = ac_power(np.asarray(total_dc, dtype=np.float64), inverter)

        with np.errstate(invalid="ignore", divide="ignore"):
            for array in arrays:
                share = np.where(total_dc > 0.0, dc_by_array[array.name] / total_dc, 0.0)
                result[array.name] = np.asarray(total_ac * share, dtype=np.float64)

    for array in system.arrays:
        if array.name not in result:  # pragma: no cover - defensive
            result[array.name] = np.zeros(steps, dtype=np.float64)
    return result


def _validate_weather(grid: TimeGrid, weather: WeatherSeries) -> None:
    """Reject a weather series that is not aligned with the grid.

    A length mismatch here would broadcast into a plausible-looking forecast that
    is silently shifted in time, so it is worth an explicit check.
    """
    expected = grid.times.size
    fields = {
        "ghi": weather.ghi,
        "temp_air": weather.temp_air,
        "wind_speed_10m": weather.wind_speed_10m,
        "dni": weather.dni,
        "dhi": weather.dhi,
    }
    for name, values in fields.items():
        if values is not None and values.size != expected:
            msg = f"weather.{name} has {values.size} samples but the grid has {expected}"
            raise ValueError(msg)
