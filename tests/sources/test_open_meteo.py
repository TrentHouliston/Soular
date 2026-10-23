"""Tests for the Open-Meteo source.

Weighted towards the failure paths. A weather source fails routinely -- rate
limits, timeouts, partial coverage at a model's edge -- and how it fails decides
whether the integration degrades or falls over.
"""

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import numpy as np
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.soular.sources.open_meteo import FORECAST_URL, HOURLY_VARIABLES, OpenMeteoError, fetch, parse

LATITUDE = -33.11915471966274
LONGITUDE = 151.53401076793673


def payload(hours: int = 3, **overrides: Any) -> dict[str, Any]:
    """Build a well-formed response."""
    times = [f"2026-01-15T{hour:02d}:00" for hour in range(hours)]
    hourly: dict[str, Any] = {
        "time": times,
        "shortwave_radiation": [0.0, 420.0, 810.0][:hours],
        "direct_radiation": [0.0, 300.0, 640.0][:hours],
        "diffuse_radiation": [0.0, 120.0, 170.0][:hours],
        "temperature_2m": [18.0, 22.0, 27.0][:hours],
        "wind_speed_10m": [1.0, 2.0, 3.5][:hours],
    }
    hourly.update(overrides)
    return {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": hourly}


def test_parse_reads_every_variable() -> None:
    """A complete response becomes arrays of the right length."""
    weather = parse(payload())

    assert len(weather) == 3
    assert weather.times[0] == np.datetime64("2026-01-15T00:00:00", "s")
    assert weather.ghi[2] == 810.0
    assert weather.temperature[0] == 18.0
    assert weather.wind_speed_10m[2] == 3.5


def test_nulls_become_nan_not_zero() -> None:
    """A gap in coverage is unknown, not dark.

    Zero would be right for radiation and badly wrong for temperature, so the
    decision is left to whoever consumes it.
    """
    weather = parse(payload(temperature_2m=[18.0, None, 27.0]))

    assert np.isnan(weather.temperature[1])
    assert weather.temperature[0] == 18.0


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"hourly": {}},
        {"hourly": {"shortwave_radiation": [1.0]}},
    ],
)
def test_a_response_without_a_series_is_rejected(broken: dict[str, Any]) -> None:
    """A response with no usable time series is an error, not an empty forecast."""
    with pytest.raises(OpenMeteoError, match="no hourly series"):
        parse(broken)


def test_an_empty_series_is_rejected() -> None:
    """Zero timestamps would silently produce a forecast of nothing."""
    with pytest.raises(OpenMeteoError, match="empty forecast"):
        parse({"hourly": {"time": []}})


def test_a_missing_variable_is_named() -> None:
    """A response missing a variable says which one."""
    incomplete = payload()
    del incomplete["hourly"]["wind_speed_10m"]

    with pytest.raises(OpenMeteoError, match="missing wind_speed_10m"):
        parse(incomplete)


async def test_fetch_requests_horizontal_components(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """The request asks for GHI/direct/diffuse, not for tilted irradiance.

    Asking Open-Meteo for ``global_tilted_irradiance`` would hand the whole
    transposition to an isotropic sky model with a fixed albedo, which is the
    thing this integration exists to avoid.
    """
    aioclient_mock.get(FORECAST_URL, json=payload())
    await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE)

    assert len(aioclient_mock.mock_calls) == 1
    _, url, _, _ = aioclient_mock.mock_calls[0]
    requested = str(url.query["hourly"]).split(",")

    assert set(requested) == set(HOURLY_VARIABLES)
    assert "global_tilted_irradiance" not in requested
    # Wind defaults to km/h, which would over-cool the array by a factor of 3.6.
    assert url.query["wind_speed_unit"] == "ms"


async def test_fetch_raises_on_an_http_error(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """A rate limit or outage surfaces as our own error type."""
    aioclient_mock.get(FORECAST_URL, status=429, text="too many requests")
    with pytest.raises(OpenMeteoError, match="HTTP 429"):
        await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE)


async def test_fetch_raises_on_an_api_level_error(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """Open-Meteo reports some failures with HTTP 200 and an error body."""
    aioclient_mock.get(FORECAST_URL, json={"error": True, "reason": "latitude must be in range"})
    with pytest.raises(OpenMeteoError, match="latitude must be in range"):
        await fetch(async_get_clientsession(hass), LATITUDE, LONGITUDE)
