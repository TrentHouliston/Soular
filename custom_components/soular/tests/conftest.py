"""Fixtures for the integration-level tests."""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import numpy as np
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.const import (
    CONF_ALBEDO,
    CONF_AZIMUTH,
    CONF_DC_CAPACITY,
    CONF_ELEVATION,
    CONF_GROUND_TYPE,
    CONF_TILT,
    DOMAIN,
    SUBENTRY_TYPE_ARRAY,
)
from custom_components.soular.sources.ensemble import EnsembleError, EnsembleForecast
from custom_components.soular.sources.open_meteo import HourlyWeather
from custom_components.soular.sources.satellite import SatelliteError, SatelliteObservations

SITE_DATA = {
    CONF_NAME: "Morisset Park",
    CONF_LATITUDE: -33.11915471966274,
    CONF_LONGITUDE: 151.53401076793673,
    CONF_ELEVATION: 10.0,
    CONF_GROUND_TYPE: "grass",
    CONF_ALBEDO: 0.20,
}

# The real site's four arrays. "north" and "south" share a plane and a capacity,
# so anything that distinguishes them is coming from shading, not geometry.
ARRAY_DATA: tuple[dict[str, Any], ...] = (
    {CONF_NAME: "east", CONF_AZIMUTH: 84.0, CONF_TILT: 25.0, CONF_DC_CAPACITY: 7920.0},
    {CONF_NAME: "west", CONF_AZIMUTH: 264.0, CONF_TILT: 25.0, CONF_DC_CAPACITY: 7920.0},
    {CONF_NAME: "north", CONF_AZIMUTH: 354.0, CONF_TILT: 25.0, CONF_DC_CAPACITY: 5720.0},
    {CONF_NAME: "south", CONF_AZIMUTH: 354.0, CONF_TILT: 25.0, CONF_DC_CAPACITY: 5720.0},
)


def synthetic_weather(hours: int = 96) -> HourlyWeather:
    """Build a plausible hourly forecast without touching the network.

    Anchored to *now* rather than a fixed date. A fixed date silently produces an
    all-zero forecast whenever the test clock drifts past it, and a structural
    assertion passes perfectly well against zeros -- so the anchor is what makes
    the value assertions elsewhere mean anything.

    A smooth diurnal shape rather than random noise, because several tests assert
    that energy and peak time come out sane, which needs a curve with a peak.
    """
    now = int(dt_util.utcnow().timestamp())
    start = np.datetime64(now - now % 3600 - 24 * 3600, "s")
    times = np.arange(start, start + np.timedelta64(hours * 3600, "s"), np.timedelta64(3600, "s")).astype(
        "datetime64[s]"
    )
    hour_of_day = (times.astype("datetime64[s]").astype(np.int64) // 3600) % 24
    # Local solar noon at this longitude is close to 02:00 UTC.
    daylight = np.clip(np.cos((hour_of_day - 2) / 24.0 * 2 * np.pi), 0.0, None)
    ghi = 950.0 * daylight
    return HourlyWeather(
        times=times,
        ghi=ghi,
        direct_horizontal=ghi * 0.78,
        diffuse=ghi * 0.22,
        temperature=np.full(times.size, 26.0),
        wind_speed_10m=np.full(times.size, 2.5),
    )


def synthetic_satellite(hours: int = 3) -> SatelliteObservations:
    """Recent observed irradiance, ending about half an hour ago.

    The lag is the point: the satellite is always published in arrears, so a
    fixture that pretends to be current would never exercise the code that
    handles latency.
    """
    now = int(dt_util.utcnow().timestamp())
    end = np.datetime64(now - now % 600 - 30 * 60, "s")
    times = np.arange(end - np.timedelta64(hours * 3600, "s"), end, np.timedelta64(600, "s")).astype("datetime64[s]")
    hour_of_day = (times.astype("datetime64[s]").astype(np.int64) / 3600.0) % 24
    daylight = np.clip(np.cos((hour_of_day - 2) / 24.0 * 2 * np.pi), 0.0, None)
    return SatelliteObservations(times=times, ghi=700.0 * daylight)


def synthetic_ensemble(members: int = 21, hours: int = 96) -> EnsembleForecast:
    """Build a three-hourly ensemble spanning the forecast, with correlated members.

    Each member carries a persistent bias rather than independent noise, because
    that is what makes a day's energy uncertain: real ensembles disagree about
    whether it will be cloudy, not about individual hours.
    """
    start = np.datetime64(int(dt_util.utcnow().timestamp()) // 3600 * 3600, "s")
    times = np.arange(start, start + np.timedelta64(hours * 3600, "s"), np.timedelta64(3 * 3600, "s")).astype(
        "datetime64[s]"
    )
    hour_of_day = (times.astype("datetime64[s]").astype(np.int64) / 3600.0) % 24
    daylight = np.clip(np.cos((hour_of_day - 2) / 24.0 * 2 * np.pi), 0.0, None)
    rng = np.random.default_rng(11)
    bias = rng.normal(1.0, 0.18, members)[:, None]
    return EnsembleForecast(times=times, members=np.clip(850.0 * daylight[None, :] * bias, 0.0, None))


@pytest.fixture
def mock_weather() -> Generator[Any]:
    """Serve synthetic weather and satellite data instead of calling the network."""
    weather = synthetic_weather()
    observations = synthetic_satellite()
    ensemble = synthetic_ensemble()

    async def _fetch(*_args: Any, **_kwargs: Any) -> HourlyWeather:
        return weather

    async def _fetch_satellite(*_args: Any, **_kwargs: Any) -> SatelliteObservations:
        return observations

    async def _fetch_ensemble(*_args: Any, **_kwargs: Any) -> EnsembleForecast:
        return ensemble

    with (
        patch("custom_components.soular.coordinator.coordinator.fetch", side_effect=_fetch) as mocked,
        patch(
            "custom_components.soular.coordinator.coordinator.satellite_source.fetch",
            side_effect=_fetch_satellite,
        ),
        patch(
            "custom_components.soular.coordinator.coordinator.ensemble_source.fetch",
            side_effect=_fetch_ensemble,
        ),
    ):
        yield mocked


@pytest.fixture
def no_satellite() -> Generator[Any]:
    """Make the satellite source fail, to exercise graceful degradation."""
    weather = synthetic_weather()

    async def _fetch(*_args: Any, **_kwargs: Any) -> HourlyWeather:
        return weather

    async def _fail(*_args: Any, **_kwargs: Any) -> SatelliteObservations:
        message = "satellite unavailable"
        raise SatelliteError(message)

    async def _no_ensemble(*_args: Any, **_kwargs: Any) -> EnsembleForecast:
        message = "ensemble unavailable"
        raise EnsembleError(message)

    with (
        patch("custom_components.soular.coordinator.coordinator.fetch", side_effect=_fetch),
        patch("custom_components.soular.coordinator.coordinator.satellite_source.fetch", side_effect=_fail) as mocked,
        patch("custom_components.soular.coordinator.coordinator.ensemble_source.fetch", side_effect=_no_ensemble),
    ):
        yield mocked


def build_entry(arrays: tuple[dict[str, Any], ...] = ARRAY_DATA) -> MockConfigEntry:
    """Create a config entry with array subentries."""
    return MockConfigEntry(
        domain=DOMAIN,
        title=str(SITE_DATA[CONF_NAME]),
        data=SITE_DATA,
        entry_id="soular_test_entry",
        unique_id="-33.11915_151.53401",
        subentries_data=[
            ConfigSubentryData(
                data=array,
                subentry_type=SUBENTRY_TYPE_ARRAY,
                title=str(array[CONF_NAME]),
                unique_id=None,
            )
            for array in arrays
        ],
    )


@pytest.fixture
async def configured(hass: HomeAssistant, mock_weather: Any) -> MockConfigEntry:
    """Set up the integration with the real site's four arrays.

    The instance is put in the site's own timezone, because a real one would be.
    Every daily total is defined against local midnight, so testing against a UTC
    instance ten hours away would exercise windows no user ever sees.
    """
    await hass.config.async_set_time_zone("Australia/Sydney")
    entry = build_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry
