"""The Soular solar forecasting integration."""

from dataclasses import dataclass, field

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.soular.const import CONF_BATTERY_SOC_SENSOR, CONF_POWER_SENSOR, SUBENTRY_TYPE_ARRAY
from custom_components.soular.coordinator.actuals import Learner
from custom_components.soular.coordinator.coordinator import SoularCoordinator
from custom_components.soular.core.shading import TransmittanceGrid
from custom_components.soular.learning.store import SAVE_DELAY_SECONDS, build_store, decode, encode
from custom_components.soular.system import build_system, load_all_shading, site_name

PLATFORMS: list[Platform] = [Platform.SENSOR]


@dataclass(slots=True)
class SoularRuntimeData:
    """Mutable per-entry state that must not survive a reload."""

    coordinator: SoularCoordinator | None = None
    shading: dict[str, TransmittanceGrid] = field(default_factory=dict)


type SoularConfigEntry = ConfigEntry[SoularRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> bool:
    """Set up Soular from a config entry."""
    # Reads shading files from disk, so it belongs off the event loop.
    shading = await hass.async_add_executor_job(load_all_shading, hass, entry)

    store = build_store(hass, entry.entry_id)
    learner = Learner(states=decode(await store.async_load()))

    power_sensors = {
        str(subentry.title): str(subentry.data[CONF_POWER_SENSOR])
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_ARRAY and subentry.data.get(CONF_POWER_SENSOR)
    }

    coordinator = SoularCoordinator(
        hass,
        site_name(entry),
        build_system(entry, shading),
        power_sensors=power_sensors,
        soc_sensor=entry.data.get(CONF_BATTERY_SOC_SENSOR),
        learner=learner,
    )
    entry.runtime_data = SoularRuntimeData(coordinator=coordinator, shading=shading)

    # Debounced: saving per sample would write every five minutes, and losing a
    # few minutes of learning to a hard restart costs nothing measurable.
    store.async_delay_save(lambda: encode(learner.states), SAVE_DELAY_SECONDS)
    entry.async_on_unload(lambda: store.async_delay_save(lambda: encode(learner.states), 0))

    await coordinator.async_config_entry_first_refresh()
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_reload_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> None:
    """Reload when configuration changes.

    Adding or editing an array changes the system spec the coordinator was built
    with, so the whole entry is rebuilt rather than patched in place.
    """
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
