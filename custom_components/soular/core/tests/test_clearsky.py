"""Tests for clear sky and the clear-sky index basis.

The clear-sky index is the common currency every source is converted into before
blending, so its edge cases -- night, sunrise, cloud enhancement -- are where a
whole forecast can quietly go wrong.
"""

import numpy as np
import pytest

from custom_components.soular.core.clearsky import (
    K_MAX,
    apply_clear_sky_index,
    clear_sky,
    clear_sky_ghi_mean,
    clear_sky_index,
)
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.tests.conftest import SITE
from custom_components.soular.core.types import SiteSpec, TimeArray


def test_clear_sky_is_zero_at_night(site: SiteSpec, summer_day: TimeArray) -> None:
    """No sun below the horizon, and no negative irradiance anywhere."""
    geometry = solar_geometry(summer_day, site)
    result = clear_sky(summer_day, site, geometry)
    night = geometry.apparent_elevation < -1.0

    assert np.all(result.ghi[night] == 0.0)
    assert np.all(result.ghi >= 0.0)
    assert np.all(result.dni >= 0.0)
    assert np.all(result.dhi >= 0.0)


def test_clear_sky_components_are_consistent(site: SiteSpec, summer_day: TimeArray) -> None:
    """GHI equals the horizontal projection of DNI plus DHI."""
    geometry = solar_geometry(summer_day, site)
    result = clear_sky(summer_day, site, geometry)
    daylight = geometry.apparent_elevation > 5.0

    cos_zenith = np.cos(np.radians(geometry.apparent_zenith))
    reconstructed = result.dni * cos_zenith + result.dhi
    np.testing.assert_allclose(reconstructed[daylight], result.ghi[daylight], rtol=1e-9)


def test_index_is_nan_at_night_not_zero(site: SiteSpec, summer_day: TimeArray) -> None:
    """Night is unknown cloudiness, not clear sky.

    Filling it with a number would let darkness vote in the blending average and
    drag every dawn forecast toward whatever that number was.
    """
    geometry = solar_geometry(summer_day, site)
    result = clear_sky(summer_day, site, geometry)
    night = geometry.apparent_elevation < -5.0

    k = clear_sky_index(result.ghi, result.ghi)
    assert np.all(np.isnan(k[night]))
    assert not np.any(np.isnan(k[geometry.apparent_elevation > 20.0]))


def test_index_of_clear_sky_is_one(site: SiteSpec, summer_day: TimeArray) -> None:
    """Feeding clear sky back in gives an index of exactly one in daylight."""
    geometry = solar_geometry(summer_day, site)
    result = clear_sky(summer_day, site, geometry)
    daylight = geometry.apparent_elevation > 10.0

    k = clear_sky_index(result.ghi, result.ghi)
    np.testing.assert_allclose(k[daylight], 1.0, rtol=1e-12)


def test_index_round_trips(site: SiteSpec, summer_day: TimeArray) -> None:
    """Converting to an index and back reproduces the original irradiance."""
    geometry = solar_geometry(summer_day, site)
    result = clear_sky(summer_day, site, geometry)
    observed = result.ghi * 0.6

    k = clear_sky_index(observed, result.ghi)
    restored = apply_clear_sky_index(k, result.ghi)
    daylight = geometry.apparent_elevation > 10.0
    np.testing.assert_allclose(restored[daylight], observed[daylight], rtol=1e-12)


def test_cloud_enhancement_is_allowed_but_bounded(site: SiteSpec, summer_day: TimeArray) -> None:
    """Indices above one are real; a runaway near sunrise is not.

    Forward scattering off cloud edges genuinely pushes GHI above clear sky. The
    ceiling exists so a near-zero denominator at dawn cannot turn a rounding
    error into a forecast of tens of kilowatts.
    """
    geometry = solar_geometry(summer_day, site)
    result = clear_sky(summer_day, site, geometry)

    enhanced = clear_sky_index(result.ghi * 1.2, result.ghi)
    daylight = geometry.apparent_elevation > 10.0
    np.testing.assert_allclose(enhanced[daylight], 1.2, rtol=1e-12)

    absurd = clear_sky_index(result.ghi * 50.0, result.ghi)
    assert np.nanmax(absurd) <= K_MAX


def test_interval_mean_differs_from_endpoint_sample(site: SiteSpec) -> None:
    """Interval-mean clear sky is not the value at either endpoint.

    Open-Meteo reports radiation as a mean over the preceding interval. Dividing
    that by an instantaneous clear-sky value biases the index by the curvature of
    the diurnal cycle -- and in opposite directions morning and evening, so it
    does not cancel over a day.
    """
    # An hour around sunrise, where curvature is strongest.
    starts = np.arange(
        np.datetime64("2026-01-15T18:00:00", "s"),
        np.datetime64("2026-01-15T20:00:00", "s"),
        np.timedelta64(900, "s"),
    ).astype("datetime64[s]")
    # numpy's stubs type datetime64 + timedelta64 as timedelta64; astype restores it.
    ends = (starts + np.timedelta64(900, "s")).astype("datetime64[s]")

    means = clear_sky_ghi_mean(starts, ends, site)
    at_end = clear_sky(ends, site, solar_geometry(ends, site)).ghi
    at_start = clear_sky(starts, site, solar_geometry(starts, site)).ghi

    rising = means > 1.0
    assert rising.any()
    # The interval mean must lie between the endpoints on a monotone stretch.
    lower = np.minimum(at_start, at_end)[rising]
    upper = np.maximum(at_start, at_end)[rising]
    assert np.all(means[rising] >= lower - 1e-9)
    assert np.all(means[rising] <= upper + 1e-9)
    # And it must differ from both, otherwise this correction would be pointless.
    assert np.max(np.abs(means[rising] - at_end[rising])) > 1.0


def test_interval_mean_converges_to_the_instant_for_short_intervals(site: SiteSpec) -> None:
    """As the interval shrinks, its mean approaches the instantaneous value."""
    starts = np.array([np.datetime64("2026-01-15T02:00:00", "s")])
    ends = starts + np.timedelta64(2, "s")

    mean = clear_sky_ghi_mean(starts, ends, site)
    midpoint = starts + np.timedelta64(1, "s")
    instant = clear_sky(midpoint, site, solar_geometry(midpoint, site)).ghi
    assert mean[0] == pytest.approx(float(instant[0]), rel=1e-4)


def test_interval_mean_rejects_mismatched_bounds() -> None:
    """Starts and ends must pair up."""
    starts = np.arange(
        np.datetime64("2026-01-15T00:00:00", "s"),
        np.datetime64("2026-01-15T01:00:00", "s"),
        np.timedelta64(900, "s"),
    ).astype("datetime64[s]")
    with pytest.raises(ValueError, match="same shape"):
        clear_sky_ghi_mean(starts, starts[:2], SITE)
