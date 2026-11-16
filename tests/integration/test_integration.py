"""Integration-level behaviour: setup, entities, energy dashboard, diagnostics.

Several of these assert on *values* rather than only on shape. That is
deliberate: an earlier version of this suite passed completely against a
forecast that was identically zero, because every assertion was structural.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular import async_setup_entry
from custom_components.soular.const import CONF_ALBEDO, CONF_INVERTER_AC_LIMIT, DOMAIN
from custom_components.soular.diagnostics import async_get_config_entry_diagnostics
from custom_components.soular.energy import async_get_solar_forecast
from tests.integration.conftest import ARRAY_DATA, build_entry, synthetic_weather


def state_of(hass: HomeAssistant, entity_id: str) -> str:
    """Return an entity's state, failing clearly if it does not exist."""
    state = hass.states.get(entity_id)
    assert state is not None, f"{entity_id} does not exist"
    return state.state


SITE_POWER = "sensor.morisset_park_estimated_power_production_now"
SITE_TOMORROW = "sensor.morisset_park_estimated_energy_production_tomorrow"
SITE_PEAK_TOMORROW = "sensor.morisset_park_highest_power_peak_time_tomorrow"
SITE_TODAY = "sensor.morisset_park_estimated_energy_production_today"
SITE_REMAINING = "sensor.morisset_park_estimated_energy_production_remaining_today"
SITE_PEAK_TODAY = "sensor.morisset_park_highest_power_peak_time_today"


async def test_setup_creates_a_device_per_array_plus_the_site(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Each array is its own device, hung off the site."""
    assert configured.state is ConfigEntryState.LOADED

    registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(registry, configured.entry_id)
    names = {device.name for device in devices}

    assert "Morisset Park" in names
    for array in ARRAY_DATA:
        assert f"Morisset Park {array['name']}" in names
    assert len(devices) == len(ARRAY_DATA) + 1


async def test_arrays_hang_off_the_site_device(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Array devices declare the site as their parent."""
    registry = dr.async_get(hass)
    site = registry.async_get_device(identifiers={(DOMAIN, configured.entry_id)})
    assert site is not None

    arrays = [
        entry
        for entry in dr.async_entries_for_config_entry(registry, configured.entry_id)
        if entry.name != "Morisset Park"
    ]
    assert arrays
    for entry in arrays:
        assert entry.via_device_id == site.id


async def test_tomorrow_has_real_energy(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The forecast produces energy, and the parts sum to the whole.

    Tomorrow rather than today, because the test can run at any time of day and
    "today" may legitimately be over.
    """
    total = float(state_of(hass, SITE_TOMORROW))
    assert total > 1.0, "a full clear day on 27 kWp should not be near zero"

    per_array = [
        float(state_of(hass, f"sensor.morisset_park_{array['name']}_estimated_energy_production_tomorrow"))
        for array in ARRAY_DATA
    ]
    assert all(value > 0.0 for value in per_array)
    assert sum(per_array) == pytest.approx(total, rel=1e-6)


async def test_every_day_of_the_week_ahead_has_a_total(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Day after tomorrow through 7 days from now, as open_meteo_solar_forecast has.

    The synthetic sky is the same clear day every day, so each total is close to
    tomorrow's. A day that fell off the end of the horizon would come out short,
    not zero, which is why this compares against tomorrow rather than checking
    for energy at all.
    """
    tomorrow = float(state_of(hass, SITE_TOMORROW))
    names = ["day_after_tomorrow", *(f"{days}_days_from_now" for days in range(3, 8))]
    for name in names:
        total = float(state_of(hass, f"sensor.morisset_park_estimated_energy_production_{name}"))
        assert total == pytest.approx(tomorrow, rel=0.05), name


async def test_the_week_ahead_is_site_level_by_default(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Per array, the days past tomorrow exist but start disabled."""
    registry = er.async_get(hass)
    entity_id = registry.async_get_entity_id(
        "sensor", DOMAIN, f"{configured.entry_id}_{array_subentry(configured, 'east').subentry_id}_energy_production_d2"
    )
    assert entity_id is not None
    entry = registry.async_get(entity_id)
    assert entry is not None
    assert entry.disabled_by is er.RegistryEntryDisabler.INTEGRATION


def array_subentry(entry: MockConfigEntry, name: str) -> Any:
    """Return the array subentry with this title."""
    return next(subentry for subentry in entry.subentries.values() if subentry.title == name)


async def test_geometric_twins_produce_identically_without_shading(
    hass: HomeAssistant, configured: MockConfigEntry
) -> None:
    """North and south share a plane and a capacity, so nothing else may separate them.

    With no shading configured they must match exactly. If they ever diverge here
    something has leaked array identity into the physics.
    """
    north = float(state_of(hass, "sensor.morisset_park_north_estimated_energy_production_tomorrow"))
    south = float(state_of(hass, "sensor.morisset_park_south_estimated_energy_production_tomorrow"))
    assert north == pytest.approx(south, rel=1e-12)


async def test_peak_time_lands_in_daylight(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The peak is a real timestamp somewhere near the middle of the day."""
    from homeassistant.util import dt as dt_util  # noqa: PLC0415

    peak = dt_util.parse_datetime(state_of(hass, SITE_PEAK_TOMORROW))
    assert peak is not None
    local_hour = dt_util.as_local(peak).hour
    assert 8 <= local_hour <= 16, f"peak at {local_hour}:00 local is not plausible"


async def test_power_now_is_never_negative(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Power is zero at night, never negative."""
    assert float(state_of(hass, SITE_POWER)) >= 0.0


async def test_energy_dashboard_hook(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The energy dashboard gets hourly watt-hours that add up."""
    result = await async_get_solar_forecast(hass, configured.entry_id)
    assert result is not None
    hours = result["wh_hours"]
    assert hours

    total_kwh = sum(hours.values()) / 1000.0
    assert total_kwh > 1.0
    # Today from local midnight, then eight days ahead: at most 24 + 192 + 1 hours.
    assert len(hours) <= 217


async def test_today_counts_the_whole_day(hass: HomeAssistant, freezer: Any) -> None:
    """Mid-afternoon, "today" includes the morning that has already happened.

    An earlier version started the series at the present instant, so "today"
    equalled "remaining today", the peak time was always now, and the energy
    dashboard had no forecast for any hour before the integration last ran.
    """
    from homeassistant.util import dt as dt_util  # noqa: PLC0415

    await hass.config.async_set_time_zone("Australia/Sydney")
    # 15:00 local in AEDT: most of the day's sun is already behind it.
    freezer.move_to("2026-11-15T04:00:00+00:00")
    weather = synthetic_weather()

    async def _fetch(*_args: Any, **_kwargs: Any) -> Any:
        return weather

    with patch("custom_components.soular.coordinator.coordinator.fetch", side_effect=_fetch):
        entry = build_entry()
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        today = float(state_of(hass, SITE_TODAY))
        remaining = float(state_of(hass, SITE_REMAINING))
        assert today > 2 * remaining, f"today {today} kWh barely exceeds remaining {remaining} kWh"

        peak = dt_util.parse_datetime(state_of(hass, SITE_PEAK_TODAY))
        assert peak is not None
        assert peak < dt_util.utcnow(), "on a clear day the peak is behind us by 15:00"

        result = await async_get_solar_forecast(hass, entry.entry_id)
        assert result is not None
        morning = dt_util.as_local(datetime(2026, 11, 14, 23, tzinfo=UTC)).isoformat()
        assert result["wh_hours"].get(morning, 0.0) > 0.0, "the dashboard has no forecast for this morning"


async def test_energy_dashboard_returns_none_for_an_unknown_entry(hass: HomeAssistant) -> None:
    """An entry that is not ours yields nothing rather than raising."""
    assert await async_get_solar_forecast(hass, "does-not-exist") is None


async def test_diagnostics_report_the_configuration(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Diagnostics answer the questions a bug report needs."""
    result: dict[str, Any] = await async_get_config_entry_diagnostics(hass, configured)

    assert result["configured_array_count"] == len(ARRAY_DATA)
    assert {array["name"] for array in result["arrays"]} == {array["name"] for array in ARRAY_DATA}
    assert all(array["shading_loaded"] is False for array in result["arrays"])
    assert result["coordinator"]["last_update_success"] is True
    assert result["forecast"]["steps"] > 0
    assert result["forecast"]["site_ac_power_w"]["max"] > 0.0


async def test_diagnostics_do_not_leak_coordinates(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Latitude and longitude stay out of a shareable report."""
    result = await async_get_config_entry_diagnostics(hass, configured)
    assert "latitude" not in result["site"]
    assert "longitude" not in result["site"]


async def test_unload(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The entry unloads cleanly."""
    assert await hass.config_entries.async_unload(configured.entry_id)
    await hass.async_block_till_done()
    assert configured.state is ConfigEntryState.NOT_LOADED


async def test_diagnostic_sensors_are_disabled_by_default(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Twelve entities per array would drown a dashboard, so diagnostics start off."""
    registry = er.async_get(hass)
    entries = er.async_entries_for_config_entry(registry, configured.entry_id)

    assert any(entry.unique_id.endswith("poa_irradiance") for entry in entries), "diagnostics should be registered"
    for entry in entries:
        if entry.unique_id.endswith(("poa_irradiance", "cell_temperature", "shading_transmittance")):
            assert entry.disabled_by is not None, f"{entry.entity_id} should start disabled"


async def test_a_site_with_no_arrays_still_loads(hass: HomeAssistant, mock_weather: Any) -> None:
    """A site configured before any array is added must not break setup.

    This is the first thing a new user sees: the config flow creates the site,
    and arrays are added afterwards.
    """
    entry = build_entry(arrays=())
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert float(state_of(hass, SITE_POWER)) == 0.0


async def test_reconfiguring_a_loaded_site_sets_it_up_once(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The update listener reloads the entry, so the flow must not reload it too.

    Reloading twice rebuilds the coordinator twice and fetches the weather twice
    for one edit -- harmless once, but it is the kind of double work that hides.
    """
    with patch("custom_components.soular.async_setup_entry", wraps=async_setup_entry) as setup:
        result = await configured.start_reconfigure_flow(hass)
        result = await hass.config_entries.flow.async_configure(
            # The form takes the site's own fields; albedo is derived from ground type.
            result["flow_id"],
            user_input={
                **{key: value for key, value in configured.data.items() if key != CONF_ALBEDO},
                CONF_INVERTER_AC_LIMIT: 30000.0,
            },
        )
        await hass.async_block_till_done()

    assert result.get("reason") == "reconfigure_successful"
    assert setup.call_count == 1
    assert configured.state is ConfigEntryState.LOADED
    assert configured.data[CONF_INVERTER_AC_LIMIT] == 30000.0
