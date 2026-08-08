"""Solar position, airmass and extraterrestrial irradiance.

Thin wrappers over pvlib that take and return numpy arrays. The rest of the core
never sees a DataFrame, so callers can reason about shapes rather than index
alignment, and a mis-aligned index cannot silently reorder a forecast.

These match the calls in the site analysis that produced the shading maps
(``solar-data/src/skymap/model.py``), because a transmittance map is only valid
against the geometry it was measured with.
"""

import numpy as np
import pandas as pd
import pvlib

from custom_components.soular.core.types import FloatArray, SiteSpec, SolarGeometry, TimeArray


def to_index(times: TimeArray) -> pd.DatetimeIndex:
    """Convert a UTC ``datetime64`` array to the tz-aware index pvlib expects."""
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True))


def solar_geometry(times: TimeArray, site: SiteSpec) -> SolarGeometry:
    """Sun position and airmass at ``times``.

    Azimuth is compass degrees from north, clockwise, matching both the array
    azimuths and the shading grids.
    """
    index = to_index(times)
    position = pvlib.solarposition.get_solarposition(
        index,
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.elevation_m,
    )
    apparent_zenith: FloatArray = np.asarray(position["apparent_zenith"], dtype=np.float64)

    # Kasten-Young 1989. Undefined below the horizon, where pvlib returns NaN;
    # downstream models treat NaN airmass as "no beam", which is correct.
    airmass_relative: FloatArray = np.asarray(
        pvlib.atmosphere.get_relative_airmass(position["apparent_zenith"]), dtype=np.float64
    )
    pressure = float(pvlib.atmosphere.alt2pres(site.elevation_m))
    airmass_absolute: FloatArray = np.asarray(
        pvlib.atmosphere.get_absolute_airmass(airmass_relative, pressure), dtype=np.float64
    )

    return SolarGeometry(
        apparent_zenith=apparent_zenith,
        apparent_elevation=np.asarray(position["apparent_elevation"], dtype=np.float64),
        azimuth=np.asarray(position["azimuth"], dtype=np.float64),
        airmass_relative=airmass_relative,
        airmass_absolute=airmass_absolute,
        dni_extra=np.asarray(pvlib.irradiance.get_extra_radiation(index), dtype=np.float64),
    )


def angle_of_incidence(azimuth_deg: float, tilt_deg: float, geometry: SolarGeometry) -> FloatArray:
    """Angle between the sun and an array's surface normal, degrees."""
    return np.asarray(
        pvlib.irradiance.aoi(tilt_deg, azimuth_deg, geometry.apparent_zenith, geometry.azimuth),
        dtype=np.float64,
    )
