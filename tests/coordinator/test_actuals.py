"""The clearness the weather correction is keyed on."""

import numpy as np

from custom_components.soular.coordinator.actuals import forecast_clearness
from custom_components.soular.core.clearsky import clear_sky
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import SystemSpec
from custom_components.soular.core.types import ArraySpec, TimeGrid
from tests.core.conftest import INVERTER, SITE

ARRAYS = (
    ArraySpec(name="east", azimuth_deg=84.0, tilt_deg=25.0, dc_capacity_w=7920.0),
    ArraySpec(name="west", azimuth_deg=264.0, tilt_deg=25.0, dc_capacity_w=7920.0),
    ArraySpec(name="north", azimuth_deg=354.0, tilt_deg=25.0, dc_capacity_w=5720.0),
)
SYSTEM = SystemSpec(site=SITE, arrays=ARRAYS, inverters={"default": INVERTER})


def test_a_clear_sky_reads_as_clear() -> None:
    """The index is horizontal, so a clear day reads about 1 whatever the planes face.

    A plane-of-array index once read a clear sky as anything from 0.15 to 1.11
    over the day on the west plane, which would put a clear afternoon in the
    cloudy bins.
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
    daylight = np.flatnonzero(solar_geometry(times, SITE).apparent_elevation > 15.0)

    applied = forecast_clearness(times, SYSTEM, weather.ghi)[daylight]
    assert np.all(np.abs(applied - 1.0) < 0.05), "a clear sky should read as clear"
