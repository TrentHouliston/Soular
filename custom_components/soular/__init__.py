"""The Soular solar forecasting integration."""

from dataclasses import dataclass, field
from types import MappingProxyType

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, Platform
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_interval

from custom_components.soular.const import (
    CONF_BATTERY_SOC_SENSOR,
    CONF_POWER_SENSOR,
    SITE_SUBENTRY_TITLE,
    SUBENTRY_TYPE_ARRAY,
    SUBENTRY_TYPE_SITE,
)
from custom_components.soular.coordinator.actuals import Learner
from custom_components.soular.coordinator.coordinator import SoularCoordinator
from custom_components.soular.core.shading import TransmittanceGrid
from custom_components.soular.learning.store import SAVE_INTERVAL, build_store, decode, encode
from custom_components.soular.system import build_system, load_all_shading, site_name

PLATFORMS: list[Platform] = [Platform.SENSOR]


@dataclass(slots=True)
class SoularRuntimeData:
    """Mutable per-entry state that must not survive a reload."""

    coordinator: SoularCoordinator | None = None
    shading: dict[str, TransmittanceGrid] = field(default_factory=dict)


type SoularConfigEntry = ConfigEntry[SoularRuntimeData]


def ensure_site_subentry(hass: HomeAssistant, entry: ConfigEntry) -> ConfigSubentry:
    """Return the site's subentry, creating it on first setup or if it was deleted.

    Called before the update listener is registered, so creating it does not
    trigger a reload of the setup that created it.
    """
    for subentry in entry.subentries.values():
        if subentry.subentry_type == SUBENTRY_TYPE_SITE:
            return subentry
    subentry = ConfigSubentry(
        data=MappingProxyType({}),
        subentry_type=SUBENTRY_TYPE_SITE,
        title=SITE_SUBENTRY_TITLE,
        unique_id=SUBENTRY_TYPE_SITE,
    )
    hass.config_entries.async_add_subentry(entry, subentry)
    return subentry


async def async_setup_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> bool:
    """Set up Soular from a config entry."""
    ensure_site_subentry(hass, entry)

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

    # Saved on an interval rather than per sample, and on the way down. Home
    # Assistant does not unload entries when it stops, so the unload hook alone
    # would lose everything learned since setup on every restart.
    #
    # Both are @callback because the store is event-loop only: Home Assistant
    # runs a plain function handed to a timer or the bus in an executor thread.
    @callback
    def save(*_: object) -> None:
        store.async_delay_save(lambda: encode(learner.states), 0)

    @callback
    def save_on_stop(_event: Event) -> None:
        save()

    entry.async_on_unload(async_track_time_interval(hass, save, SAVE_INTERVAL))
    entry.async_on_unload(hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, save_on_stop))
    entry.async_on_unload(save)

    # Listening before the first refresh, not after it: that refresh reads the
    # recorder and Open-Meteo and can take seconds, and an array added in the
    # meantime must still trigger a rebuild. The reload waits on the setup lock,
    # so it runs once this setup has finished.
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    await coordinator.async_config_entry_first_refresh()
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_reload_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> None:
    """Reload when configuration changes.

    Adding or editing an array changes the system spec the coordinator was built
    with, so the whole entry is rebuilt rather than patched in place. Scheduled
    rather than awaited, so the listener returns at once and a change made during
    a reload queues another one behind it.
    """
    hass.config_entries.async_schedule_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
