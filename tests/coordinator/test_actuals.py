"""The learner's view of an array must match what the correction is applied against."""

import numpy as np

from custom_components.soular.coordinator.actuals import forecast_clearness, samples_from
from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import SystemSpec, forecast
from custom_components.soular.core.types import ArraySpec, TimeGrid
from tests.core.conftest import INVERTER, SITE

ARRAYS = (
    ArraySpec(name="east", azimuth_deg=84.0, tilt_deg=25.0, dc_capacity_w=7920.0),
    ArraySpec(name="west", azimuth_deg=264.0, tilt_deg=25.0, dc_capacity_w=7920.0),
    ArraySpec(name="north", azimuth_deg=354.0, tilt_deg=25.0, dc_capacity_w=5720.0),
)
SYSTEM = SystemSpec(site=SITE, arrays=ARRAYS, inverters={"default": INVERTER})


def test_every_array_learns_the_clearness_the_correction_is_applied_with() -> None:
    """On a clear day, every plane's clearness is the sky's, not its own geometry.

    Training on a plane-of-array index once taught the west plane that a clear
    sky ran from 0.15 to 1.11 over the day, while the correction was applied
    against a horizontal index of about 1.
    """
    hours = np.arange(
        np.datetime64("2026-11-14T18:00", "s"), np.datetime64("2026-11-15T12:00", "s"), np.timedelta64(3600, "s")
    ).astype("datetime64[s]")
    # Open-Meteo's hourly radiation is the mean over the hour ending at the label.
    midpoints = (hours - np.timedelta64(1800, "s")).astype("datetime64[s]")
    ghi = clear_sky(midpoints, SITE, solar_geometry(midpoints, SITE)).ghi
    times = np.arange(hours[1], hours[-1], np.timedelta64(900, "s")).astype("datetime64[s]")
    grid = TimeGrid(times=times, step_seconds=np.full(times.size, 900.0))
    weather = weather_from_hourly(
        hourly_times=hours,
        grid=grid,
        site=SITE,
        ghi=ghi,
        temperature=np.full(hours.size, 22.0),
        wind_speed_10m=np.full(hours.size, 2.0),
    )
    result = forecast(SYSTEM, grid, weather)
    daylight = np.flatnonzero(solar_geometry(times, SITE).apparent_elevation > 15.0)

    applied = forecast_clearness(times, SYSTEM, weather.ghi)[daylight]
    samples = samples_from(result, SYSTEM, weather.ghi, daylight)
    for name, entries in samples.items():
        learned = np.array([sample.clearness for sample in entries])
        np.testing.assert_allclose(learned, applied, err_msg=f"{name} learns a different clearness")
    assert np.all(np.abs(applied - 1.0) < 0.05), "a clear sky should read as clear"
