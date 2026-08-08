"""The acceptance gate for the physics port.

A transmittance map is a measurement made *through* an optical model. It is only
valid for a model that reproduces the one it was measured against. These tests
re-implement ``solar-data/src/skymap/model.py`` inline, verbatim, and assert
Soular's core agrees with it.

The reimplementation is deliberately duplicated here rather than imported: the
analysis repo is not a dependency, and pinning the contract in this file means it
keeps holding even if that repo moves or changes.

One intentional deviation. The reference passes *absolute* airmass to
``pvlib.irradiance.perez``, which documents its parameter as *relative* airmass.
Soular passes relative. At this site's elevation the two differ by about 0.1%,
it moves only the Perez sky-diffuse term, and it is far inside the +/-25%
absolute accuracy the map itself claims -- so the tolerances below are on diffuse
only, and beam is required to agree exactly.
"""

import numpy as np
import pandas as pd
import pvlib
import pytest

from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.irradiance import plane_of_array
from custom_components.soular.core.tests.conftest import ARRAYS, day_times
from custom_components.soular.core.types import ArraySpec, FloatArray, SiteSpec, TimeArray

LATITUDE = -33.11915471966274
LONGITUDE = 151.53401076793673
ALTITUDE = 10.0

# pvlib's get_total_irradiance defaults to albedo=0.25 and the reference analysis
# never overrode it, so that is the value the maps were derived under.
REFERENCE_ALBEDO = 0.25

REFERENCE_SITE = SiteSpec(
    latitude=LATITUDE,
    longitude=LONGITUDE,
    elevation_m=ALTITUDE,
    albedo=REFERENCE_ALBEDO,
)


def skymap_reference(array: ArraySpec, times: TimeArray) -> dict[str, FloatArray]:
    """Verbatim re-implementation of ``skymap/model.py`` for one array."""
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))

    position = pvlib.solarposition.get_solarposition(index, latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE)
    relative = pvlib.atmosphere.get_relative_airmass(position["apparent_zenith"])
    airmass = pvlib.atmosphere.get_absolute_airmass(relative, pvlib.atmosphere.alt2pres(ALTITUDE))

    turbidity = pvlib.clearsky.lookup_linke_turbidity(index, LATITUDE, LONGITUDE)
    dni_extra = pvlib.irradiance.get_extra_radiation(index)
    clearsky = pvlib.clearsky.ineichen(
        position["apparent_zenith"],
        airmass,
        turbidity,
        altitude=ALTITUDE,  # pyright: ignore[reportArgumentType]
        dni_extra=dni_extra,  # pyright: ignore[reportArgumentType]
    )

    aoi = pvlib.irradiance.aoi(array.tilt_deg, array.azimuth_deg, position["apparent_zenith"], position["azimuth"])
    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=array.tilt_deg,
        surface_azimuth=array.azimuth_deg,
        solar_zenith=position["apparent_zenith"],
        solar_azimuth=position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        dni_extra=dni_extra,
        airmass=airmass,
        model="perez",
    )

    # pvlib.iam.physical is annotated as returning an array but preserves a Series.
    iam_beam = pvlib.iam.physical(aoi.clip(upper=90.0)).fillna(0.0).clip(lower=0.0)  # pyright: ignore[reportAttributeAccessIssue]
    marion = pvlib.iam.marion_diffuse("physical", array.tilt_deg)
    iam_sky = float(np.asarray(marion["sky"]).item())
    iam_ground = float(np.asarray(marion["ground"]).item())

    poa_beam = total["poa_direct"].fillna(0.0).clip(lower=0.0) * iam_beam
    poa_beam = poa_beam.where(aoi < 90.0, 0.0)
    poa_diffuse = (
        total["poa_sky_diffuse"].fillna(0.0).clip(lower=0.0) * iam_sky
        + total["poa_ground_diffuse"].fillna(0.0).clip(lower=0.0) * iam_ground
    )

    return {
        "ghi": np.asarray(clearsky["ghi"], dtype=np.float64),
        "dni": np.asarray(clearsky["dni"], dtype=np.float64),
        "dhi": np.asarray(clearsky["dhi"], dtype=np.float64),
        "aoi": np.asarray(aoi, dtype=np.float64),
        "poa_beam": np.asarray(poa_beam, dtype=np.float64),
        "poa_diffuse": np.asarray(poa_diffuse, dtype=np.float64),
    }


def soular_result(array: ArraySpec, times: TimeArray) -> dict[str, FloatArray]:
    """Compute the same quantities via Soular's core."""
    geometry = solar_geometry(times, REFERENCE_SITE)
    clearsky = clear_sky(times, REFERENCE_SITE, geometry)
    poa = plane_of_array(array, REFERENCE_SITE, geometry, clearsky.ghi, clearsky.dni, clearsky.dhi)
    return {
        "ghi": clearsky.ghi,
        "dni": clearsky.dni,
        "dhi": clearsky.dhi,
        "aoi": poa.aoi,
        "poa_beam": poa.poa_beam,
        "poa_diffuse": poa.poa_diffuse,
    }


@pytest.mark.parametrize("array", ARRAYS, ids=lambda a: a.name)
@pytest.mark.parametrize("date", ["2026-01-15", "2026-06-21", "2026-03-21"])
def test_clearsky_and_beam_match_exactly(array: ArraySpec, date: str) -> None:
    """Clear sky, incidence angle and beam POA must be bit-for-bit identical.

    These feed the transmittance denominator directly. Any drift here would
    rescale the shading map rather than merely perturb it.
    """
    times = day_times(date)
    ours = soular_result(array, times)
    reference = skymap_reference(array, times)

    for field in ("ghi", "dni", "dhi", "aoi", "poa_beam"):
        np.testing.assert_array_equal(ours[field], reference[field], err_msg=field)


# Measured worst cases across all four arrays at five points around the year:
# diffuse 2.1e-4 of peak, daily energy 1.0e-5. The bounds below sit roughly 3x
# above those, tight enough that any real regression trips them and loose enough
# that a pvlib patch release does not.
MAX_DIFFUSE_REL_TO_PEAK = 5e-4
MAX_DAILY_ENERGY_REL = 3e-5

SEASONS = ["2026-01-15", "2026-03-21", "2026-06-21", "2026-09-21", "2026-12-21"]


@pytest.mark.parametrize("array", ARRAYS, ids=lambda a: a.name)
@pytest.mark.parametrize("date", SEASONS)
def test_diffuse_matches_within_airmass_convention(array: ArraySpec, date: str) -> None:
    """Diffuse POA agrees to well within the map's own stated accuracy.

    The only difference is relative versus absolute airmass in Perez. Asserting a
    tight bound here is what stops that known deviation quietly growing into
    something else.
    """
    times = day_times(date)
    ours = soular_result(array, times)
    reference = skymap_reference(array, times)

    peak = float(np.max(reference["poa_diffuse"]))
    worst = float(np.max(np.abs(ours["poa_diffuse"] - reference["poa_diffuse"])))
    assert worst / peak < MAX_DIFFUSE_REL_TO_PEAK, f"diffuse differs by {worst:.4f} W/m2, {worst / peak:.2e} of peak"


@pytest.mark.parametrize("array", ARRAYS, ids=lambda a: a.name)
@pytest.mark.parametrize("date", SEASONS)
def test_daily_poa_energy_matches(array: ArraySpec, date: str) -> None:
    """Daily plane-of-array energy agrees to a part in a hundred thousand.

    Daily energy is what the shading map's efficiency terms were calibrated
    against, so this is the figure that has to transfer.
    """
    times = day_times(date)
    ours = soular_result(array, times)
    reference = skymap_reference(array, times)

    ours_total = float(np.sum(ours["poa_beam"] + ours["poa_diffuse"]))
    reference_total = float(np.sum(reference["poa_beam"] + reference["poa_diffuse"]))
    assert ours_total == pytest.approx(reference_total, rel=MAX_DAILY_ENERGY_REL)
