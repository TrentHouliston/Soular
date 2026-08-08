"""The Soular solar forecasting integration."""

from dataclasses import dataclass, field

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.soular.coordinator.coordinator import SoularCoordinator
from custom_components.soular.core.shading import TransmittanceGrid
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

    coordinator = SoularCoordinator(hass, site_name(entry), build_system(entry, shading))
    entry.runtime_data = SoularRuntimeData(coordinator=coordinator, shading=shading)

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
