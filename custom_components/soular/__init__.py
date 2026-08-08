"""The Soular solar forecasting integration."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

PLATFORMS: list[Platform] = []


@dataclass(slots=True)
class SoularRuntimeData:
    """Mutable per-entry state that must not survive a reload."""


type SoularConfigEntry = ConfigEntry[SoularRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> bool:
    """Set up Soular from a config entry."""
    entry.runtime_data = SoularRuntimeData()
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SoularConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
