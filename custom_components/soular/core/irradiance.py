"""Transposition to the plane of array, incidence-angle losses, and decomposition.

This is where Soular departs most sharply from the integration it replaces.

``open_meteo_solar_forecast`` asks Open-Meteo for ``global_tilted_irradiance``,
which Open-Meteo computes with an isotropic sky model and a hardcoded albedo of
0.2. Isotropic has no circumsolar brightening and no horizon band, so it
systematically under-reads plane-of-array irradiance in clear conditions, worst
around solar noon and low incidence angles -- exactly when most energy arrives.

Soular requests GHI/DNI/DHI instead and transposes with Perez 1990, which models
those two terms explicitly, and with a configurable ground reflectance.
"""

from functools import lru_cache

import numpy as np
import pvlib

from custom_components.soular.core.geometry import angle_of_incidence, to_index
from custom_components.soular.core.types import ArraySpec, FloatArray, PlaneOfArray, SiteSpec, SolarGeometry, TimeArray

# Beyond this the sun is behind the plane and no beam is geometrically possible.
BEHIND_PANEL_AOI = 90.0


@lru_cache(maxsize=32)
def diffuse_iam(tilt_deg: float) -> tuple[float, float]:
    """Effective incidence-angle modifiers for sky and ground diffuse.

    Diffuse light arrives from the whole dome, so its incidence-angle loss is an
    integral over the dome rather than a value at one angle. Marion's numerical
    integration of the physical IAM is the standard way to do it.

    Cached on tilt because a fixed array's tilt never changes, and the integral
    is a couple of hundred evaluations that would otherwise repeat every refresh.
    """
    result = pvlib.iam.marion_diffuse("physical", tilt_deg)
    return (
        float(np.asarray(result["sky"]).item()),
        float(np.asarray(result["ground"]).item()),
    )


def decompose(times: TimeArray, ghi: FloatArray, geometry: SolarGeometry) -> tuple[FloatArray, FloatArray]:
    """Split GHI into DNI and DHI with the Erbs correlation.

    Needed because the sources disagree about what they provide. The satellite
    product (Himawari via Open-Meteo) ships shortwave radiation only, and the
    DNI/DHI Open-Meteo reports alongside it come from its own separation model.
    Blending a vendor's split with our own would make the beam and diffuse
    inconsistent with the total. Splitting the blended GHI ourselves keeps the
    three mutually consistent no matter which sources contributed.
    """
    index = to_index(times)
    result = pvlib.irradiance.erbs(ghi, geometry.apparent_zenith, index)
    return (
        np.asarray(result["dni"], dtype=np.float64),
        np.asarray(result["dhi"], dtype=np.float64),
    )


def plane_of_array(
    array: ArraySpec,
    site: SiteSpec,
    geometry: SolarGeometry,
    ghi: FloatArray,
    dni: FloatArray,
    dhi: FloatArray,
) -> PlaneOfArray:
    """Transpose horizontal irradiance onto one array's plane.

    Mirrors ``solar-data/src/skymap/model.py::plane_of_array``, which is the
    model the site's transmittance maps were measured against. A map is only
    valid against the optics it was derived with, so this must stay in step.
    """
    aoi = angle_of_incidence(array.azimuth_deg, array.tilt_deg, geometry)

    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=array.tilt_deg,
        surface_azimuth=array.azimuth_deg,
        solar_zenith=geometry.apparent_zenith,
        solar_azimuth=geometry.azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=geometry.dni_extra,
        # pvlib documents this as *relative* airmass. The reference analysis
        # passed absolute; at this site's elevation the two differ by ~0.1%,
        # which moves only the Perez sky-diffuse term and is far inside the
        # +/-25% absolute accuracy the transmittance map itself claims.
        airmass=geometry.airmass_relative,
        albedo=site.albedo,
        model=site.transposition_model,
    )

    beam_geometric = _clean(total["poa_direct"])
    sky_diffuse = _clean(total["poa_sky_diffuse"])
    ground_diffuse = _clean(total["poa_ground_diffuse"])

    # Glass reflection reaches 10-30% by AOI 60-80 degrees. Because the loss
    # depends on incidence angle it cannot be absorbed into a scalar efficiency:
    # omitting it paints a false shadow along the whole winter-solstice sun track
    # and at both low-sun extremes, in every array identically. That signature is
    # what gives it away as model error rather than trees.
    iam_beam = _clean(pvlib.iam.physical(np.clip(aoi, None, BEHIND_PANEL_AOI)))
    iam_sky, iam_ground = diffuse_iam(array.tilt_deg)

    behind = aoi >= BEHIND_PANEL_AOI
    poa_beam = np.where(behind, 0.0, beam_geometric * iam_beam)

    return PlaneOfArray(
        aoi=aoi,
        poa_beam=np.asarray(poa_beam, dtype=np.float64),
        poa_diffuse=sky_diffuse * iam_sky + ground_diffuse * iam_ground,
        iam_beam=iam_beam,
    )


def _clean(values: object) -> FloatArray:
    """Coerce a pvlib result to a non-negative float array with no NaNs.

    pvlib returns NaN below the horizon and for undefined airmass. Those are
    "no irradiance", not "unknown", so zero is the right fill.
    """
    array = np.asarray(values, dtype=np.float64)
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
