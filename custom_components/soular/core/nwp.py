"""Assemble a weather series on the forecast grid from coarse forecast data.

Shared by the integration and the offline backtest deliberately. This conversion
is where several easy-to-get-wrong things live -- interval means versus instants,
horizontal versus normal-incidence beam, what to do when a source only carries a
total -- and a second copy would drift from the measured one.
"""

import numpy as np

from custom_components.soular.core.clearsky import clear_sky, resample_to_grid
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.irradiance import decompose
from custom_components.soular.core.pipeline import WeatherSeries
from custom_components.soular.core.types import FloatArray, SiteSpec, TimeArray, TimeGrid

# Below this the beam is spread over so much atmosphere that dividing by its
# cosine amplifies noise without adding information.
MIN_COS_ZENITH = 0.05

DEFAULT_TEMPERATURE_C = 20.0
DEFAULT_WIND_SPEED_MS = 2.0


def weather_from_hourly(
    hourly_times: TimeArray,
    grid: TimeGrid,
    site: SiteSpec,
    ghi: FloatArray,
    temperature: FloatArray,
    wind_speed_10m: FloatArray,
    direct_horizontal: FloatArray | None = None,
    diffuse: FloatArray | None = None,
) -> WeatherSeries:
    """Resample an hourly forecast onto the fine grid.

    ``hourly_times`` label the *end* of each interval, which is Open-Meteo's
    convention for its radiation variables.

    When the forecast carries its own beam and diffuse split, that split is used:
    it is model-informed and better than re-deriving one from the total with a
    correlation. When it does not -- as with satellite products, which carry
    shortwave radiation only -- the split is derived instead.
    """
    if hourly_times.size < 2:  # noqa: PLR2004
        msg = "need at least two hourly samples to form an interval"
        raise ValueError(msg)

    geometry = solar_geometry(grid.times, site)
    clearsky = clear_sky(grid.times, site, geometry)
    starts, ends = hourly_times[:-1], hourly_times[1:]

    def radiation(values: FloatArray | None) -> FloatArray | None:
        """Resample one interval-mean radiation series, or nothing."""
        if values is None:
            return None
        interval_values = values[1:]
        if not np.isfinite(interval_values).any():
            return None
        return resample_to_grid(
            starts,
            ends,
            np.nan_to_num(interval_values, nan=0.0),
            grid.times,
            clearsky.ghi,
            site,
        )

    def ambient(values: FloatArray, default: float) -> FloatArray:
        """Interpolate an hourly ambient variable, which is an instant, not a mean."""
        finite = np.isfinite(values)
        if not finite.any():
            return np.full(grid.times.size, default)
        return np.asarray(
            np.interp(
                grid.times.astype("datetime64[s]").astype(np.int64).astype(np.float64),
                hourly_times[finite].astype("datetime64[s]").astype(np.int64).astype(np.float64),
                values[finite],
            ),
            dtype=np.float64,
        )

    resampled_ghi = radiation(ghi)
    if resampled_ghi is None:
        msg = "forecast carried no usable shortwave radiation"
        raise ValueError(msg)

    resampled_direct = radiation(direct_horizontal)
    resampled_diffuse = radiation(diffuse)

    if resampled_direct is not None and resampled_diffuse is not None:
        # direct_radiation is on the horizontal plane; the transposition needs it
        # at normal incidence.
        cos_zenith = np.clip(np.cos(np.radians(geometry.apparent_zenith)), MIN_COS_ZENITH, None)
        dni: FloatArray = resampled_direct / cos_zenith
        dhi: FloatArray = resampled_diffuse
    else:
        dni, dhi = decompose(grid.times, resampled_ghi, geometry)

    return WeatherSeries(
        ghi=resampled_ghi,
        dni=dni,
        dhi=dhi,
        temp_air=ambient(temperature, DEFAULT_TEMPERATURE_C),
        wind_speed_10m=ambient(wind_speed_10m, DEFAULT_WIND_SPEED_MS),
    )
