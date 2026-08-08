"""Tests for the satellite observation source.

The request parameters matter more than usual here. Two of them are easy to get
subtly wrong and neither failure is loud: without ``temporal_resolution=native``
the response is silently resampled to hourly, throwing away the sub-hourly
variability the nowcast exists to capture; and the model identifier the API
accepts is not the one its documentation lists.
"""

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util
import numpy as np
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.soular.sources.satellite import ARCHIVE_URL, SatelliteError, fetch, parse

LATITUDE = -33.11915471966274
LONGITUDE = 151.53401076793673


def payload(values: list[float | None] | None = None) -> dict[str, Any]:
    """Build a well-formed ten-minute response."""
    readings = values if values is not None else [400.0, 520.0, 610.0, 580.0]
    times = [f"2026-01-15T00:{minute:02d}" for minute in range(0, 10 * len(readings), 10)]
    return {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": {"time": times, "shortwave_radiation": readings}}


def test_parse_reads_a_ten_minute_series() -> None:
    """The native cadence survives parsing."""
    observations = parse(payload())

    assert len(observations) == 4
    assert observations.times[1] - observations.times[0] == np.timedelta64(600, "s")
    assert observations.ghi[2] == 610.0


def test_latest_returns_the_most_recent_reading() -> None:
    """The freshest observation is what the nowcast persists from."""
    observations = parse(payload())
    latest = observations.latest()

    assert latest is not None
    stamp, value = latest
    assert stamp == np.datetime64("2026-01-15T00:30:00", "s")
    assert value == 580.0


def test_trailing_nulls_are_skipped() -> None:
    """The archive publishes in arrears, so the newest slots are routinely empty.

    A null tail is normal operation here, not an error. Treating it as one would
    disable the nowcast most of the time.
    """
    observations = parse(payload([400.0, 520.0, None, None]))
    latest = observations.latest()

    assert latest is not None
    stamp, value = latest
    assert stamp == np.datetime64("2026-01-15T00:10:00", "s")
    assert value == 520.0


def test_an_all_null_response_has_no_latest() -> None:
    """Nothing observed means no nowcast, not a crash."""
    observations = parse(payload([None, None]))
    assert observations.latest() is None


def test_a_response_without_a_series_is_rejected() -> None:
    """A malformed response is an error rather than an empty observation set."""
    with pytest.raises(SatelliteError, match="no series"):
        parse({})


def test_a_response_missing_the_variable_is_rejected() -> None:
    """A response with times but no radiation is not usable."""
    with pytest.raises(SatelliteError, match="missing shortwave_radiation"):
        parse({"hourly": {"time": ["2026-01-15T00:00"]}})


async def test_fetch_asks_for_native_resolution(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """Without the native flag the archive resamples to hourly.

    That would leave the nowcast persisting an hourly average, which is most of
    the sub-hourly variability it exists to capture thrown away before it arrives.
    """
    aioclient_mock.get(ARCHIVE_URL, json=payload())
    await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE, now=dt_util.utcnow())

    _, url, _, _ = aioclient_mock.mock_calls[0]
    assert url.query["temporal_resolution"] == "native"
    assert url.query["hourly"] == "shortwave_radiation"


async def test_fetch_raises_on_an_http_error(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """An outage surfaces as our own error type, for the caller to shrug off."""
    aioclient_mock.get(ARCHIVE_URL, status=503, text="unavailable")

    with pytest.raises(SatelliteError, match="HTTP 503"):
        await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE, now=dt_util.utcnow())


async def test_fetch_raises_on_an_api_level_error(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """The archive reports some failures with HTTP 200 and an error body."""
    aioclient_mock.get(ARCHIVE_URL, json={"error": True, "reason": "no coverage at this location"})

    with pytest.raises(SatelliteError, match="no coverage"):
        await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE, now=dt_util.utcnow())
