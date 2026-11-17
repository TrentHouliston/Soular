"""Arrays added while the entry is still setting up all end up in the forecast."""

import asyncio
from typing import Any
from unittest.mock import patch

from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.soular.const import DOMAIN, SUBENTRY_TYPE_ARRAY
from custom_components.soular.sources.open_meteo import HourlyWeather
from tests.integration.conftest import ARRAY_DATA, build_entry, synthetic_weather


async def test_arrays_added_during_setup_are_all_forecast(hass: HomeAssistant) -> None:
    """An array added while setup is still waiting on its first forecast is not lost.

    On a real install the first refresh waits on the recorder and on Open-Meteo,
    which takes seconds, and adding arrays one after another lands the next one
    inside that wait. The coordinator had already been built from the arrays that
    existed when setup began, but the sensors were created from the arrays that
    existed when it finished -- so the late ones got entities with no forecast
    behind them, unavailable until something else reloaded the entry.
    """
    await hass.config.async_set_time_zone("Australia/Sydney")
    weather = synthetic_weather()
    gate = asyncio.Event()
    gate.set()
    waiting = asyncio.Event()

    async def fetch(*_args: Any, **_kwargs: Any) -> HourlyWeather:
        waiting.set()
        await gate.wait()
        return weather

    entry = build_entry(arrays=ARRAY_DATA[:1])
    entry.add_to_hass(hass)
    with patch("custom_components.soular.coordinator.coordinator.fetch", side_effect=fetch):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        # From here every setup stalls on its first forecast, as a real one does.
        gate.clear()
        waiting.clear()
        for array in ARRAY_DATA[1:]:
            flow = await hass.config_entries.subentries.async_init(
                (entry.entry_id, SUBENTRY_TYPE_ARRAY), context={"source": SOURCE_USER}
            )
            await hass.config_entries.subentries.async_configure(flow["flow_id"], array)
            # The first array's reload is now mid-setup; the rest land inside it.
            await waiting.wait()
        gate.set()
        await hass.async_block_till_done()

    forecast = {array.name for array in entry.runtime_data.coordinator.data.arrays}
    assert forecast == {str(array["name"]) for array in ARRAY_DATA}

    registry = er.async_get(hass)
    power = [
        entity.entity_id
        for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        if entity.platform == DOMAIN and entity.unique_id.endswith("power_production_now")
    ]
    assert len(power) == len(ARRAY_DATA)
    for entity_id in power:
        state = hass.states.get(entity_id)
        assert state is not None
        assert state.state != STATE_UNAVAILABLE, entity_id
