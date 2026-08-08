"""Device identity for the site and its arrays."""

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
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
    """Describe one array's device, hung off the site.

    Arrays are separate devices rather than extra entities on the site because
    geometry, shading and eventually a measured-power sensor all belong to an
    individual plane, and adding a fifth array should not disturb the other four.
    """
    return DeviceInfo(
        identifiers={array_identifier(entry, subentry)},
        name=f"{site_name} {subentry.title}",
        manufacturer="Soular",
        via_device=site_identifier(entry),
    )
