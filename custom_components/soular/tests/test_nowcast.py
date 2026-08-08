"""The nowcast, end to end through the integration.

The property that matters most is the one about failure. A nowcast is an
enhancement layered on a forecast that already worked; if losing the satellite
takes the forecast with it, the enhancement is a liability.
"""

from typing import Any

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.tests.conftest import build_entry

SITE_POWER = "sensor.morisset_park_estimated_power_production_now"
SITE_TOMORROW = "sensor.morisset_park_estimated_energy_production_tomorrow"


def coordinator(entry: MockConfigEntry) -> Any:
    """Return the entry's coordinator."""
    assert entry.runtime_data is not None
    return entry.runtime_data.coordinator


async def test_the_satellite_contributes_to_the_forecast(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """With observations available, some of the near-term answer is observed."""
    assert coordinator(configured).satellite_fetched_at is not None
    assert coordinator(configured).satellite_error is None


async def test_losing_the_satellite_leaves_a_working_forecast(hass: HomeAssistant, no_satellite: Any) -> None:
    """A satellite outage costs the nowcast and nothing else.

    This is the whole safety property. The weather model is the backbone; the
    satellite refines the next couple of hours. Losing the refinement must leave
    the backbone standing.
    """
    entry = build_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert coordinator(entry).satellite_error is not None
    assert coordinator(entry).observed_share == 0.0

    # The forecast is still a forecast.
    state = hass.states.get(SITE_TOMORROW)
    assert state is not None
    assert float(state.state) > 1.0


async def test_the_nowcast_share_is_reported(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """How much of the answer came from observation is answerable.

    Without this the user cannot tell a nowcast from the raw weather model, and
    neither can anyone reading a bug report.
    """
    registry = er.async_get(hass)
    entries = er.async_entries_for_config_entry(registry, configured.entry_id)
    keys = {entry.unique_id.rsplit("_", 1)[-1] for entry in entries}
    assert "share" in keys or any(entry.unique_id.endswith("nowcast_share") for entry in entries)

    share = coordinator(configured).observed_share
    assert 0.0 <= share <= 1.0


async def test_the_share_is_zero_without_observations(hass: HomeAssistant, no_satellite: Any) -> None:
    """No observations means the forecast is honestly labelled as model-only."""
    entry = build_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert coordinator(entry).observed_share == 0.0


async def test_diagnostics_include_the_satellite(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """A bug report should say whether the nowcast was working."""
    from custom_components.soular.diagnostics import async_get_config_entry_diagnostics  # noqa: PLC0415

    result = await async_get_config_entry_diagnostics(hass, configured)
    assert result["coordinator"]["last_update_success"] is True
    assert result["coordinator"]["satellite_error"] is None
    assert result["coordinator"]["satellite_fetched_at"] is not None
