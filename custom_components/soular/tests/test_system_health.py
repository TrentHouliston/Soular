"""Tests for the system health report.

The point of this report is that the three sources fail independently. A
forecast that is still being produced says nothing about whether the satellite
or the ensemble is reachable, so the report has to name each one separately.
"""

import inspect
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry, get_system_health_info
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.soular.const import DOMAIN
from custom_components.soular.sources.ensemble import ENSEMBLE_URL
from custom_components.soular.sources.open_meteo import FORECAST_URL
from custom_components.soular.sources.satellite import ARCHIVE_URL

# Setting up ``system_health`` pulls in ``http``, and aiohttp warns about
# non-AppKey keys somewhere inside it. Scoped to this module so the project-wide
# error-on-warning still applies everywhere else.
pytestmark = pytest.mark.filterwarnings("ignore::aiohttp.web_exceptions.NotAppKeyWarning")


async def resolve(info: dict[str, Any]) -> dict[str, Any]:
    """Await the reachability probes, which the report returns unawaited.

    That is the documented contract: Home Assistant resolves them itself so a
    slow endpoint cannot block the rest of the panel.
    """
    return {key: await value if inspect.isawaitable(value) else value for key, value in info.items()}


async def test_it_reports_every_source(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, configured: MockConfigEntry
) -> None:
    """All three endpoints are probed, not just the one the forecast depends on."""
    aioclient_mock.get(FORECAST_URL, text="")
    aioclient_mock.get(ARCHIVE_URL, text="")
    aioclient_mock.get(ENSEMBLE_URL, text="")
    assert await async_setup_component(hass, "system_health", {})
    # Platform registration is a task, not part of setup returning.
    await hass.async_block_till_done()

    info = await resolve(await get_system_health_info(hass, DOMAIN))

    assert info["weather_api"] == "ok"
    assert info["satellite_api"] == "ok"
    assert info["ensemble_api"] == "ok"
    assert info[f"{configured.title} forecast"] == "ok"
    assert info[f"{configured.title} uncertainty"] == "available"


async def test_an_unreachable_source_is_named(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, configured: MockConfigEntry
) -> None:
    """A satellite outage must not be reported as a healthy integration."""
    del configured
    aioclient_mock.get(FORECAST_URL, text="")
    aioclient_mock.get(ARCHIVE_URL, exc=TimeoutError)
    aioclient_mock.get(ENSEMBLE_URL, text="")
    assert await async_setup_component(hass, "system_health", {})
    # Platform registration is a task, not part of setup returning.
    await hass.async_block_till_done()

    info = await resolve(await get_system_health_info(hass, DOMAIN))

    assert info["weather_api"] == "ok"
    assert info["satellite_api"] != "ok"
