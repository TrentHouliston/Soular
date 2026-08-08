"""Clear-sky irradiance, and the clear-sky index that everything is blended in.

Clear sky is the backbone of the whole forecast, not just a fallback. Every
irradiance source -- deterministic NWP, satellite observation, ensemble member --
is converted to a clear-sky index ``k = GHI / GHI_clearsky`` before it is
combined with any other, then converted back. Two reasons:

* ``k`` is smooth and bounded where GHI is neither. GHI has a hard diurnal
  envelope, so linearly interpolating it between coarse samples clips the peak
  and smears sunrise and sunset. Interpolating ``k`` does not.
* Sources disagree about absolute irradiance far more than they disagree about
  cloudiness. Differencing against a common clear-sky removes the part of the
  disagreement that is nobody's forecast error.
"""

import numpy as np
import pvlib

from custom_components.soular.core.geometry import solar_geometry, to_index
from custom_components.soular.core.types import ClearSky, FloatArray, SiteSpec, SolarGeometry, TimeArray

# Cloud enhancement is real: forward scattering off cloud edges genuinely pushes
# GHI above the clear-sky value, by up to ~35% for minutes at a time. The ceiling
# is here to stop a near-zero clear-sky denominator near sunrise turning a
# rounding error into a 40 kW spike.
K_MIN = 0.0
K_MAX = 1.35

# Below this the clear-sky value is too small to divide by meaningfully.
MIN_CLEARSKY_GHI = 5.0


def clear_sky(times: TimeArray, site: SiteSpec, geometry: SolarGeometry) -> ClearSky:
    """Ineichen-Perez clear-sky GHI/DNI/DHI with climatological Linke turbidity.

    Note that ``lookup_linke_turbidity`` reads a bundled HDF5 file, so this is
    blocking I/O and must not be called on the Home Assistant event loop.
    """
    index = to_index(times)
    turbidity = pvlib.clearsky.lookup_linke_turbidity(index, site.latitude, site.longitude)
    # pvlib annotates altitude as int and dni_extra as float, but both accept
    # arrays and floats at runtime -- its own clear-sky examples pass a series.
    result = pvlib.clearsky.ineichen(
        geometry.apparent_zenith,
        geometry.airmass_absolute,
        turbidity,
        altitude=site.elevation_m,  # pyright: ignore[reportArgumentType]
        dni_extra=geometry.dni_extra,  # pyright: ignore[reportArgumentType]
    )
    return ClearSky(
        ghi=np.asarray(result["ghi"], dtype=np.float64),
        dni=np.asarray(result["dni"], dtype=np.float64),
        dhi=np.asarray(result["dhi"], dtype=np.float64),
    )


def clear_sky_ghi_mean(
    interval_starts: TimeArray,
    interval_ends: TimeArray,
    site: SiteSpec,
    *,
    substeps: int = 15,
) -> FloatArray:
    """Mean clear-sky GHI over each interval, by sub-sampling within it.

    Open-Meteo's radiation variables are means over the *preceding* interval, not
    instantaneous values. Dividing an interval mean by an instantaneous clear-sky
    value evaluated at one endpoint produces a clear-sky index that is wrong by
    the curvature of the diurnal cycle -- several percent at low sun, and biased
    in opposite directions morning and evening. Averaging the denominator over
    the same interval removes that entirely.

    Cheap enough to be unconditional: 48 h of 15-minute intervals at 15 substeps
    is ~2900 sun-position evaluations, a couple of milliseconds vectorised.
    """
    if interval_starts.shape != interval_ends.shape:
        msg = "interval_starts and interval_ends must have the same shape"
        raise ValueError(msg)

    starts = interval_starts.astype("datetime64[s]").astype(np.int64)
    ends = interval_ends.astype("datetime64[s]").astype(np.int64)

    # Midpoints of `substeps` equal slices of each interval: an open-interval
    # rule, so an interval that starts exactly at sunrise is not dominated by its
    # zero endpoint.
    offsets = (np.arange(substeps, dtype=np.float64) + 0.5) / substeps
    sample_seconds = starts[:, None] + (ends - starts)[:, None] * offsets[None, :]
    sample_times = sample_seconds.round().astype(np.int64).astype("datetime64[s]")

    flat = sample_times.reshape(-1)
    geometry = solar_geometry(flat, site)
    ghi = clear_sky(flat, site, geometry).ghi
    return np.asarray(np.nanmean(ghi.reshape(sample_times.shape), axis=1), dtype=np.float64)


def clear_sky_index(ghi: FloatArray, clearsky_ghi: FloatArray) -> FloatArray:
    """Convert irradiance to a clear-sky index, NaN where clear sky is negligible.

    NaN rather than zero: night is *unknown* cloudiness, not clear sky. Filling it
    with a number here would let darkness vote in the blending average.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(clearsky_ghi > MIN_CLEARSKY_GHI, ghi / clearsky_ghi, np.nan)
    return np.asarray(np.clip(k, K_MIN, K_MAX), dtype=np.float64)


def apply_clear_sky_index(k: FloatArray, clearsky_ghi: FloatArray) -> FloatArray:
    """Reconstruct GHI from a clear-sky index. Inverse of :func:`clear_sky_index`."""
    return np.asarray(np.where(np.isnan(k), 0.0, k * clearsky_ghi), dtype=np.float64)


def fill_gaps(k: FloatArray) -> FloatArray:
    """Carry the nearest defined clear-sky index across night and missing samples.

    Cloudiness is persistent, so the nearest known value is a far better guess
    than any constant. This matters most at dawn, where the first defined sample
    of the day would otherwise have nothing behind it to interpolate from.
    """
    values = np.asarray(k, dtype=np.float64)
    defined = ~np.isnan(values)
    if not defined.any():
        return np.ones_like(values)

    positions = np.arange(values.size)
    return np.asarray(np.interp(positions, positions[defined], values[defined]), dtype=np.float64)


def resample_to_grid(
    interval_starts: TimeArray,
    interval_ends: TimeArray,
    interval_mean_ghi: FloatArray,
    grid_times: TimeArray,
    grid_clearsky_ghi: FloatArray,
    site: SiteSpec,
) -> FloatArray:
    """Turn coarse interval-mean GHI into instantaneous GHI on a fine grid.

    The obvious approach -- interpolate GHI directly -- is wrong twice over. It
    treats an interval mean as an instantaneous value, which shifts the whole
    series by half an interval; and it interpolates a quantity with a hard
    diurnal envelope, which shaves the peak and smears sunrise and sunset.

    Working in clear-sky index fixes both. The mean is divided by clear sky
    averaged over the *same* interval, so the ratio is dimensionally what it
    claims to be; and the ratio is smooth, so interpolating it is benign. The
    diurnal shape then comes back from the fine grid's own clear sky.
    """
    if not (interval_starts.shape == interval_ends.shape == interval_mean_ghi.shape):
        msg = "interval bounds and values must have matching shapes"
        raise ValueError(msg)
    if interval_starts.size == 0:
        msg = "cannot resample from an empty set of intervals"
        raise ValueError(msg)

    clearsky_mean = clear_sky_ghi_mean(interval_starts, interval_ends, site)
    k = fill_gaps(clear_sky_index(interval_mean_ghi, clearsky_mean))

    # An interval mean is most representative of the interval's midpoint, so that
    # is where the ratio is anchored before interpolating.
    starts = interval_starts.astype("datetime64[s]").astype(np.int64)
    ends = interval_ends.astype("datetime64[s]").astype(np.int64)
    midpoints = (starts + ends) / 2.0

    grid_seconds = grid_times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    k_grid = np.interp(grid_seconds, midpoints, k)
    return apply_clear_sky_index(np.asarray(k_grid, dtype=np.float64), grid_clearsky_ghi)
