"""End-to-end properties of the forecasting pipeline.

These are the invariants that hold regardless of weather: signs, conservation,
and the behaviour of shading and clipping. A physics port can pass an equivalence
test against a reference and still be wired up wrongly; these catch the wiring.
"""

import numpy as np
import pytest

from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.irradiance import diffuse_iam, plane_of_array
from custom_components.soular.core.pipeline import (
    DEFAULT_HORIZON_HOURS,
    FAR_TERM_STEP_MINUTES,
    NEAR_TERM_HOURS,
    NEAR_TERM_STEP_MINUTES,
    SystemSpec,
    WeatherSeries,
    build_time_grid,
    forecast,
)
from custom_components.soular.core.pvmodel import ac_power
from custom_components.soular.core.shading import TransmittanceGrid
from custom_components.soular.core.tests.conftest import ARRAYS, INVERTER, SITE
from custom_components.soular.core.types import ArraySpec, FloatArray, InverterSpec, SiteSpec, TimeArray, TimeGrid


def clear_weather(times: TimeArray, site: SiteSpec = SITE, *, scale: float = 1.0) -> WeatherSeries:
    """Clear-sky irradiance as a stand-in weather series, optionally dimmed."""
    geometry = solar_geometry(times, site)
    clearsky = clear_sky(times, site, geometry)
    return WeatherSeries(
        ghi=clearsky.ghi * scale,
        dni=clearsky.dni * scale,
        dhi=clearsky.dhi * scale,
        temp_air=np.full(times.size, 25.0),
        wind_speed_10m=np.full(times.size, 2.0),
    )


def grid_for(times: TimeArray) -> TimeGrid:
    """Wrap a time array as a uniform grid."""
    step = float((times[1] - times[0]).astype("timedelta64[s]").astype(np.int64))
    return TimeGrid(times=times, step_seconds=np.full(times.size, step))


def uniform_shading(value: float) -> TransmittanceGrid:
    """Build a grid that attenuates the beam equally from every direction."""
    return TransmittanceGrid(
        azimuth_deg=np.arange(0.0, 360.0, 45.0),
        elevation_deg=np.arange(0.0, 91.0, 30.0),
        values=np.full((8, 4), round(value * 255)).astype(np.uint8),
    )


def test_poa_is_never_negative(system: SystemSpec, summer_day: TimeArray) -> None:
    """Irradiance and power are non-negative at every instant, including night."""
    result = forecast(system, grid_for(summer_day), clear_weather(summer_day))
    for entry in result.arrays:
        assert np.all(entry.poa_global >= 0.0), entry.name
        assert np.all(entry.poa_beam >= 0.0), entry.name
        assert np.all(entry.dc_power_w >= 0.0), entry.name
        assert np.all(entry.ac_power_w >= 0.0), entry.name
    assert np.all(result.ac_power_w >= 0.0)


def test_no_production_when_sun_is_down(system: SystemSpec, summer_day: TimeArray) -> None:
    """Below the horizon there is no beam and no power."""
    geometry = solar_geometry(summer_day, SITE)
    night = geometry.apparent_elevation < 0.0
    assert night.any(), "test day must contain some night"

    result = forecast(system, grid_for(summer_day), clear_weather(summer_day))
    for entry in result.arrays:
        assert np.all(entry.poa_beam[night] == 0.0), entry.name
        assert np.all(entry.dc_power_w[night] == 0.0), entry.name


def test_horizontal_geometric_poa_equals_ghi(summer_day: TimeArray) -> None:
    """A horizontal plane receives exactly GHI, once optics are divided back out.

    This is the strongest available check that the transposition is wired up with
    the right angles: any azimuth sign error or degree/radian slip breaks it.
    """
    flat = ArraySpec(name="flat", azimuth_deg=180.0, tilt_deg=0.0, dc_capacity_w=1000.0)
    geometry = solar_geometry(summer_day, SITE)
    clearsky = clear_sky(summer_day, SITE, geometry)
    poa = plane_of_array(flat, SITE, geometry, clearsky.ghi, clearsky.dni, clearsky.dhi)

    iam_sky, _ = diffuse_iam(0.0)
    daylight = geometry.apparent_elevation > 5.0

    # Undo the incidence-angle modifiers to recover geometric irradiance. At zero
    # tilt the ground-reflected term vanishes, since (1 - cos 0) / 2 is zero.
    geometric = np.zeros_like(poa.poa_beam)
    np.divide(poa.poa_beam, poa.iam_beam, out=geometric, where=poa.iam_beam > 0)
    geometric += poa.poa_diffuse / iam_sky

    np.testing.assert_allclose(geometric[daylight], clearsky.ghi[daylight], rtol=1e-10)


def test_shading_attenuates_beam_only(system: SystemSpec, summer_day: TimeArray) -> None:
    """Halving transmittance halves the beam and leaves the diffuse untouched.

    The whole validated result rests on this: a tree blocking the sun disk still
    passes most of the sky, so scaling total irradiance would over-penalise.
    """
    times, grid, weather = summer_day, grid_for(summer_day), clear_weather(summer_day)
    unshaded = forecast(system, grid, weather).array("east")

    shaded_system = SystemSpec(
        site=system.site,
        arrays=system.arrays,
        inverters=system.inverters,
        shading={"east": uniform_shading(0.5)},
    )
    shaded = forecast(shaded_system, grid, weather).array("east")

    transmittance = round(0.5 * 255) / 255.0
    np.testing.assert_allclose(shaded.poa_beam, unshaded.poa_beam * transmittance, rtol=1e-12)

    diffuse_unshaded = unshaded.poa_global - unshaded.poa_beam
    diffuse_shaded = shaded.poa_global - shaded.poa_beam
    np.testing.assert_allclose(diffuse_shaded, diffuse_unshaded, rtol=1e-12)
    assert times.size == shaded.poa_beam.size


def test_shading_never_increases_output(system: SystemSpec, summer_day: TimeArray) -> None:
    """Any transmittance below one can only reduce production."""
    grid, weather = grid_for(summer_day), clear_weather(summer_day)
    unshaded = forecast(system, grid, weather)

    shaded_system = SystemSpec(
        site=system.site,
        arrays=system.arrays,
        inverters=system.inverters,
        shading={array.name: uniform_shading(0.3) for array in system.arrays},
    )
    shaded = forecast(shaded_system, grid, weather)
    assert np.all(shaded.ac_power_w <= unshaded.ac_power_w + 1e-9)
    assert shaded.ac_power_w.sum() < unshaded.ac_power_w.sum()


def test_array_ac_sums_to_site_ac(system: SystemSpec, summer_day: TimeArray) -> None:
    """Per-array AC is attributed so the parts add up to the whole.

    Arrays on a shared inverter clip together, so per-array AC is a share of a
    joint result rather than an independent quantity. Users will put these on a
    dashboard next to the site total and expect them to agree.
    """
    result = forecast(system, grid_for(summer_day), clear_weather(summer_day))
    total = np.sum([entry.ac_power_w for entry in result.arrays], axis=0)
    np.testing.assert_allclose(total, result.ac_power_w, rtol=1e-12)


def test_inverter_clips_at_its_limit(summer_day: TimeArray) -> None:
    """AC output never exceeds the configured inverter limit."""
    limit = 10000.0
    system = SystemSpec(
        site=SITE,
        arrays=ARRAYS,
        inverters={"default": InverterSpec(name="default", ac_limit_w=limit)},
    )
    result = forecast(system, grid_for(summer_day), clear_weather(summer_day))
    assert result.ac_power_w.max() <= limit + 1e-6
    # 27.28 kWp against a 10 kW inverter must actually reach the limit, otherwise
    # this test would pass on a system that never clips.
    assert result.ac_power_w.max() > 0.99 * limit


def test_clipping_before_averaging_loses_more_energy(summer_day: TimeArray) -> None:
    """Clipping an interval mean understates the energy lost inside the interval.

    This is the concrete reason the pipeline clips on its fine grid and
    aggregates afterwards, rather than the other way round. Both paths below use
    exactly the same DC series, so the only difference is the order of the two
    operations -- averaging then clipping is what the incumbent integration does.
    """
    inverter = InverterSpec(name="default", ac_limit_w=12000.0)
    unlimited = SystemSpec(
        site=SITE,
        arrays=ARRAYS,
        inverters={"default": InverterSpec(name="default", ac_limit_w=1e9)},
    )
    dc = np.sum(
        [entry.dc_power_w for entry in forecast(unlimited, grid_for(summer_day), clear_weather(summer_day)).arrays],
        axis=0,
    )
    assert dc.max() > inverter.ac_limit_w, "test day must actually clip"

    hourly = dc.reshape(-1, 12)  # twelve five-minute samples per hour
    clip_then_average = float(ac_power(np.asarray(dc, dtype=np.float64), inverter).reshape(-1, 12).mean(axis=1).sum())
    average_then_clip = float(ac_power(np.asarray(hourly.mean(axis=1), dtype=np.float64), inverter).sum())

    assert average_then_clip > clip_then_average, (
        f"averaging first hides clipping losses: {average_then_clip:.1f} W vs {clip_then_average:.1f} W"
    )


def test_build_time_grid_shape() -> None:
    """The standard grid is fine near-term, coarse beyond, and a whole number of days.

    The whole-day requirement is not cosmetic: haeo pads a forecast series to
    whole days by wrapping its head onto its tail, so a series that stops
    mid-day gets this morning spliced onto its end.
    """
    start = np.datetime64("2026-01-15T04:35:00", "s")
    grid = build_time_grid(start)

    span = (grid.times[-1] + np.timedelta64(int(grid.step_seconds[-1]), "s") - grid.times[0]).astype("timedelta64[h]")
    assert int(span.astype(np.int64)) == DEFAULT_HORIZON_HOURS
    assert DEFAULT_HORIZON_HOURS % 24 == 0

    near = grid.step_seconds == NEAR_TERM_STEP_MINUTES * 60
    assert near.sum() == NEAR_TERM_HOURS * 60 // NEAR_TERM_STEP_MINUTES
    far = grid.step_seconds == FAR_TERM_STEP_MINUTES * 60
    assert far.sum() == (DEFAULT_HORIZON_HOURS - NEAR_TERM_HOURS) * 60 // FAR_TERM_STEP_MINUTES
    assert grid.times[0] == start


def test_weather_length_mismatch_is_rejected(system: SystemSpec, summer_day: TimeArray) -> None:
    """A short weather series would broadcast into a silently time-shifted forecast."""
    grid = grid_for(summer_day)
    weather = clear_weather(summer_day)
    truncated = WeatherSeries(
        ghi=weather.ghi[:-1],
        temp_air=weather.temp_air,
        wind_speed_10m=weather.wind_speed_10m,
    )
    with pytest.raises(ValueError, match="samples but the grid has"):
        forecast(system, grid, truncated)


def test_unknown_inverter_is_reported_with_context(summer_day: TimeArray) -> None:
    """A typo'd inverter name says which array and what was configured."""
    system = SystemSpec(
        site=SITE,
        arrays=(ArraySpec(name="east", azimuth_deg=84.0, tilt_deg=25.0, dc_capacity_w=7920.0, inverter="rooftop"),),
        inverters={"default": INVERTER},
    )
    with pytest.raises(KeyError, match="unknown inverter"):
        forecast(system, grid_for(summer_day), clear_weather(summer_day))


def test_decomposition_used_when_split_is_absent(system: SystemSpec, summer_day: TimeArray) -> None:
    """GHI alone is enough; the beam/diffuse split is derived when not supplied.

    This is the path taken whenever a satellite observation has contributed,
    since those products carry shortwave radiation only.
    """
    grid = grid_for(summer_day)
    full = clear_weather(summer_day)
    ghi_only = WeatherSeries(ghi=full.ghi, temp_air=full.temp_air, wind_speed_10m=full.wind_speed_10m)

    derived = forecast(system, grid, ghi_only)
    supplied = forecast(system, grid, full)

    assert derived.ac_power_w.sum() > 0.0
    # Erbs is a correlation, not an identity, so agreement is approximate. It
    # should still land within a fifth of the clear-sky answer on a clear day.
    ratio = derived.ac_power_w.sum() / supplied.ac_power_w.sum()
    assert 0.8 < ratio < 1.2, f"decomposed forecast is {ratio:.2f} of the supplied-split forecast"


@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 1.0])
def test_output_is_monotone_in_irradiance(system: SystemSpec, summer_day: TimeArray, scale: float) -> None:
    """Dimming the sky can never raise production."""
    grid = grid_for(summer_day)
    dimmed = forecast(system, grid, clear_weather(summer_day, scale=scale))
    full = forecast(system, grid, clear_weather(summer_day, scale=1.0))
    assert np.all(dimmed.ac_power_w <= full.ac_power_w + 1e-9)


def as_float(values: FloatArray) -> float:
    """Sum helper used to keep assertions readable."""
    return float(np.sum(values))
