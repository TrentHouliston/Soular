"""Device identity for the site and its arrays."""

from collections.abc import Callable

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


# Home Assistant 2026.10 gave each device a single owning entry and subentry,
# and deprecated the registry calls that assumed several; they stop working in
# 2027.8. Soular still supports releases from before the change, so the helpers
# below pick whichever form the running release has.


def single_owner(device: dr.DeviceEntry) -> bool:
    """Return whether this release gives a device a single owner."""
    return hasattr(device, "config_subentry_id")


def find_device(registry: dr.DeviceRegistry, entry: ConfigEntry, identifier: tuple[str, str]) -> dr.DeviceEntry | None:
    """Return the entry's device with this identifier."""
    lookup: Callable[..., dr.DeviceEntry | None] | None = getattr(registry, "async_get_device_by_identifier", None)
    if lookup is not None:
        return lookup(identifier, entry.entry_id)
    return registry.async_get_device(identifiers={identifier})


def device_subentries(device: dr.DeviceEntry, entry: ConfigEntry) -> set[str | None]:
    """Return the subentries of the entry that the device belongs to."""
    if single_owner(device):
        owner = getattr(device, "config_subentry_id", None)
        return {owner} if getattr(device, "config_entry_id", None) == entry.entry_id else set()
    return set(device.config_entries_subentries.get(entry.entry_id, set()))


def move_to_subentry(registry: dr.DeviceRegistry, device: dr.DeviceEntry, entry: ConfigEntry, subentry_id: str) -> None:
    """Move a device that belongs to the entry itself into one of its subentries."""
    update: Callable[..., dr.DeviceEntry | None] = registry.async_update_device
    if single_owner(device):
        update(device.id, new_config_subentry_id=subentry_id)
    else:
        update(
            device.id,
            add_config_entry_id=entry.entry_id,
            add_config_subentry_id=subentry_id,
            remove_config_entry_id=entry.entry_id,
            remove_config_subentry_id=None,
        )


def register_devices(
    hass: HomeAssistant,
    entry: ConfigEntry,
    site_name: str,
    site_subentry: ConfigSubentry,
    subentries: list[ConfigSubentry],
) -> None:
    """Create the site and array devices, and hang each array off the site.

    Linking by registry id through ``async_update_device`` works on every
    supported release, where neither of the two ``DeviceInfo`` spellings does.
    """
    registry = dr.async_get(hass)
    # Installs from before the site had a subentry registered its device against
    # the entry itself. Move it explicitly: re-registering it under the subentry
    # would move it implicitly, which 2026.11 warns about and 2027.8 refuses.
    existing = find_device(registry, entry, site_identifier(entry))
    if existing is not None and device_subentries(existing, entry) != {site_subentry.subentry_id}:
        move_to_subentry(registry, existing, entry, site_subentry.subentry_id)
    site = registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        config_subentry_id=site_subentry.subentry_id,
        **site_device(entry, site_name),
    )
    for subentry in subentries:
        array = registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            config_subentry_id=subentry.subentry_id,
            **array_device(entry, subentry, site_name),
        )
        if array.via_device_id != site.id:
            registry.async_update_device(array.id, via_device_id=site.id)
