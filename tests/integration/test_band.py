"""The P10/P90 band as Home Assistant sees it."""

from datetime import timedelta
import math
from typing import Any

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.const import CONF_POWER_SENSOR
from custom_components.soular.core.band import MAX_TICK_SECONDS, PRIOR
from tests.integration.conftest import ARRAY_DATA, build_entry

SITE = "sensor.morisset_park_estimated"
POWER = f"{SITE}_power_production_now"


def attributes(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Return an entity's attributes, failing clearly if it does not exist."""
    state = hass.states.get(entity_id)
    assert state is not None, f"{entity_id} does not exist"
    return dict(state.attributes)


async def test_the_power_band_brackets_the_forecast(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """P10 and P90 carry whole series of their own, either side of the forecast."""
    middle = attributes(hass, POWER)["forecast"]
    low = attributes(hass, f"{POWER}_p10")["forecast"]
    high = attributes(hass, f"{POWER}_p90")["forecast"]
    assert len(low) == len(middle) == len(high)
    assert [p["time"] for p in low] == [p["time"] for p in middle]
    assert any(p["value"] > 0 for p in middle)
    for lo, mid, hi in zip(low, middle, high, strict=True):
        assert lo["value"] <= mid["value"] <= hi["value"]


async def test_the_power_band_is_one_coherent_day(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Within a day every point is scaled by the same ratio: a poor day, not every bad hour at once."""
    middle = attributes(hass, POWER)["forecast"]
    low = attributes(hass, f"{POWER}_p10")["forecast"]
    tomorrow = (dt_util.now() + timedelta(days=1)).date()

    def local_date(point: dict[str, Any]) -> Any:
        when = dt_util.parse_datetime(point["time"])
        assert when is not None
        return dt_util.as_local(when).date()

    ratios = {
        round(lo["value"] / mid["value"], 2)
        for lo, mid in zip(low, middle, strict=True)
        if mid["value"] > 100 and local_date(mid) == tomorrow
    }
    assert ratios == {PRIOR[1][0]}


async def test_daily_totals_carry_their_band(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Each site daily total names its P10 and P90 the way Solcast does."""
    for key, lead in (("today", 0), ("tomorrow", 1), ("d7", 7)):
        entity = f"{SITE}_energy_production_{key}" if key != "d7" else f"{SITE}_energy_production_7_days_from_now"
        state = hass.states.get(entity)
        assert state is not None, entity
        value = float(state.state)
        assert value > 0
        assert state.attributes["estimate10"] == pytest.approx(value * PRIOR[lead][0], abs=0.01)
        assert state.attributes["estimate90"] == pytest.approx(value * PRIOR[lead][1], abs=0.01)
        assert state.attributes["band_calibrated"] is False


async def test_arrays_carry_no_band(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The band is measured on the site's output, so it is not claimed per array."""
    registry = er.async_get(hass)
    keys = {entity.unique_id for entity in er.async_entries_for_config_entry(registry, configured.entry_id)}
    assert sum(key.endswith("power_production_now_p10") for key in keys) == 1
    tomorrow = attributes(hass, "sensor.morisset_park_east_estimated_energy_production_tomorrow")
    assert "estimate10" not in tomorrow


async def test_the_calibration_diagnostic_starts_disabled(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """How much history is behind the band is there for whoever asks."""
    registry = er.async_get(hass)
    entity_id = registry.async_get_entity_id("sensor", "soular", f"{configured.entry_id}_band_samples")
    assert entity_id is not None
    entry = registry.async_get(entity_id)
    assert entry is not None
    assert entry.disabled_by is er.RegistryEntryDisabler.INTEGRATION


async def test_a_measured_day_becomes_a_ratio(hass: HomeAssistant, mock_weather: Any, freezer: Any) -> None:
    """A day of readings, compared with the midnight forecast, yields that day's ratio.

    Setup and the first update after midnight go through the real update path,
    reading the power sensors; the readings in between are fed to the tracker
    directly, because recomputing the forecast every ten minutes of a simulated
    day is too slow for a CI runner.
    """
    await hass.config.async_set_time_zone("Australia/Sydney")
    freezer.move_to(dt_util.start_of_local_day() + timedelta(minutes=1))
    arrays = tuple({**array, CONF_POWER_SENSOR: f"sensor.pv_{array['name']}"} for array in ARRAY_DATA)
    for array in ARRAY_DATA:
        hass.states.async_set(f"sensor.pv_{array['name']}", "1000", {"unit_of_measurement": "W"})
    entry = build_entry(arrays)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    coordinator = entry.runtime_data.coordinator
    band = coordinator.band
    day = dt_util.now().date().toordinal()
    assert band.ledger is not None
    forecast_wh = band.snapshots[day].energy_wh(band.ledger.start, band.ledger.end)
    assert forecast_wh > 0

    # Daylight only: the night carries no forecast energy, so skipping it costs
    # the usable share nothing.
    midnight = dt_util.start_of_local_day()
    readings = {str(array["name"]): 1000.0 for array in ARRAY_DATA}
    freezer.move_to(midnight + timedelta(hours=5))
    for _ in range(15 * 3600 // MAX_TICK_SECONDS + 1):
        coordinator._track_band(coordinator.data, readings, None)
        freezer.tick(timedelta(seconds=MAX_TICK_SECONDS))
    freezer.move_to(dt_util.start_of_local_day(midnight.date() + timedelta(days=1)) + timedelta(minutes=1))
    await coordinator.async_refresh()

    residuals = [r for r in band.residuals if r.day == day and r.lead == 0]
    assert len(residuals) == 1
    # Four arrays at a steady kilowatt from five to eight, against the midnight forecast.
    assert math.exp(residuals[0].log_ratio) == pytest.approx(4 * 1000.0 * 15 / forecast_wh, rel=0.03)


async def test_the_band_is_saved_when_home_assistant_stops(
    hass: HomeAssistant, configured: MockConfigEntry, hass_storage: Any
) -> None:
    """What the band has learned survives a restart, like the correction does."""
    configured.runtime_data.coordinator.band.residuals.clear()
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    await hass.async_stop(force=True)
    assert "residuals" in hass_storage[f"soular.band.{configured.entry_id}"]["data"]
