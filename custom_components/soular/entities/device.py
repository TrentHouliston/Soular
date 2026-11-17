"""Device identity for each array."""

from collections.abc import Callable

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo

from custom_components.soular.const import DOMAIN


def site_identifier(entry: ConfigEntry) -> tuple[str, str]:
    """Return the identifier the retired site device was registered under."""
    return (DOMAIN, entry.entry_id)


def array_identifier(entry: ConfigEntry, subentry: ConfigSubentry) -> tuple[str, str]:
    """Return the registry identifier for one array's device."""
    return (DOMAIN, f"{entry.entry_id}_{subentry.subentry_id}")


def array_device(entry: ConfigEntry, subentry: ConfigSubentry, site_name: str) -> DeviceInfo:
    """Describe one array's device.

    Each array is a device of its own: geometry, shading and the measured-power
    sensor all belong to an individual plane, and adding a fifth array should not
    disturb the other four.
    """
    return DeviceInfo(
        identifiers={array_identifier(entry, subentry)},
        name=f"{site_name} {subentry.title}",
        manufacturer="Soular",
    )


def find_device(registry: dr.DeviceRegistry, entry: ConfigEntry, identifier: tuple[str, str]) -> dr.DeviceEntry | None:
    """Return the entry's device with this identifier.

    Home Assistant 2026.10 deprecated the lookup that ignores the owning entry;
    Soular still supports releases from before it, so this picks whichever the
    running release has.
    """
    lookup: Callable[..., dr.DeviceEntry | None] | None = getattr(registry, "async_get_device_by_identifier", None)
    if lookup is not None:
        return lookup(identifier, entry.entry_id)
    return registry.async_get_device(identifiers={identifier})


def register_devices(hass: HomeAssistant, entry: ConfigEntry, site_name: str, subentries: list[ConfigSubentry]) -> None:
    """Create each array's device under its own subentry."""
    registry = dr.async_get(hass)
    for subentry in subentries:
        array = registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            config_subentry_id=subentry.subentry_id,
            **array_device(entry, subentry, site_name),
        )
        # Arrays used to hang off a site device that no longer exists.
        if array.via_device_id is not None:
            registry.async_update_device(array.id, via_device_id=None)
