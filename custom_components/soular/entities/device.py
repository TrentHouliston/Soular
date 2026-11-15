"""Device identity for the site and its arrays."""

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo

from custom_components.soular.const import DOMAIN


def site_identifier(entry: ConfigEntry) -> tuple[str, str]:
    """Return the registry identifier for the site device."""
    return (DOMAIN, entry.entry_id)


def array_identifier(entry: ConfigEntry, subentry: ConfigSubentry) -> tuple[str, str]:
    """Return the registry identifier for one array's device."""
    return (DOMAIN, f"{entry.entry_id}_{subentry.subentry_id}")


def site_device(entry: ConfigEntry, name: str) -> DeviceInfo:
    """Describe the site device."""
    return DeviceInfo(
        identifiers={site_identifier(entry)},
        name=name,
        manufacturer="Soular",
        entry_type=None,
    )


def array_device(entry: ConfigEntry, subentry: ConfigSubentry, site_name: str) -> DeviceInfo:
    """Describe one array's device.

    Arrays are separate devices rather than extra entities on the site because
    geometry, shading and eventually a measured-power sensor all belong to an
    individual plane, and adding a fifth array should not disturb the other four.

    The link to the site is made by :func:`register_devices`, not here: the
    ``via_device`` key is deprecated from Home Assistant 2026.8, and its
    replacement does not exist in the earlier releases Soular still supports.
    """
    return DeviceInfo(
        identifiers={array_identifier(entry, subentry)},
        name=f"{site_name} {subentry.title}",
        manufacturer="Soular",
    )


def register_devices(
    hass: HomeAssistant,
    entry: ConfigEntry,
    site_name: str,
    subentries: list[ConfigSubentry],
) -> None:
    """Create the site and array devices, and hang each array off the site.

    Linking by registry id through ``async_update_device`` works on every
    supported release, where neither of the two ``DeviceInfo`` spellings does.
    """
    registry = dr.async_get(hass)
    site = registry.async_get_or_create(config_entry_id=entry.entry_id, **site_device(entry, site_name))
    for subentry in subentries:
        array = registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            config_subentry_id=subentry.subentry_id,
            **array_device(entry, subentry, site_name),
        )
        if array.via_device_id != site.id:
            registry.async_update_device(array.id, via_device_id=site.id)
