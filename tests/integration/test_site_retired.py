"""Earlier versions had a site-wide device; setup removes it."""

from types import MappingProxyType
from typing import Any

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.const import DOMAIN, SUBENTRY_TYPE_ARRAY, SUBENTRY_TYPE_SITE
from custom_components.soular.entities.device import find_device, site_identifier
from tests.integration.conftest import build_entry


def owner(device: dr.DeviceEntry, entry: MockConfigEntry) -> str | None:
    """Return the subentry that owns a device, on releases before and after single ownership."""
    if hasattr(device, "config_subentry_id"):
        return getattr(device, "config_subentry_id", None)
    (subentry,) = device.config_entries_subentries[entry.entry_id]
    return subentry


async def setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Set the entry up and let it settle."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


async def test_an_install_with_a_site_subentry_loses_it(hass: HomeAssistant, mock_weather: Any) -> None:
    """The site subentry, its device and its sensors all go; the arrays stay."""
    await hass.config.async_set_time_zone("Australia/Sydney")
    entry = build_entry()
    entry.add_to_hass(hass)
    site = ConfigSubentry(
        data=MappingProxyType({}), subentry_type=SUBENTRY_TYPE_SITE, title="Total", unique_id=SUBENTRY_TYPE_SITE
    )
    hass.config_entries.async_add_subentry(entry, site)
    devices = dr.async_get(hass)
    old = devices.async_get_or_create(
        config_entry_id=entry.entry_id, config_subentry_id=site.subentry_id, identifiers={site_identifier(entry)}
    )
    old_entity = er.async_get(hass).async_get_or_create(
        "sensor",
        DOMAIN,
        f"{entry.entry_id}_power_production_now",
        config_entry=entry,
        config_subentry_id=site.subentry_id,
        device_id=old.id,
    )

    await setup(hass, entry)

    assert {s.subentry_type for s in entry.subentries.values()} == {SUBENTRY_TYPE_ARRAY}
    assert devices.async_get(old.id) is None
    assert er.async_get(hass).async_get(old_entity.entity_id) is None
    for device in dr.async_entries_for_config_entry(devices, entry.entry_id):
        assert device.via_device_id is None


async def test_an_install_from_before_the_subentry_loses_its_device(hass: HomeAssistant, mock_weather: Any) -> None:
    """A site device registered on the entry itself is removed too."""
    await hass.config.async_set_time_zone("Australia/Sydney")
    entry = build_entry()
    entry.add_to_hass(hass)
    devices = dr.async_get(hass)
    old = devices.async_get_or_create(config_entry_id=entry.entry_id, identifiers={site_identifier(entry)})

    await setup(hass, entry)

    assert devices.async_get(old.id) is None
    assert find_device(devices, entry, site_identifier(entry)) is None


async def test_a_fresh_install_has_only_arrays(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Every device belongs to an array, and every sensor to one of them."""
    devices = dr.async_entries_for_config_entry(dr.async_get(hass), configured.entry_id)
    arrays = {s.subentry_id for s in configured.subentries.values() if s.subentry_type == SUBENTRY_TYPE_ARRAY}
    assert {owner(device, configured) for device in devices} == arrays
    entities = er.async_entries_for_config_entry(er.async_get(hass), configured.entry_id)
    assert entities
    assert {entity.config_subentry_id for entity in entities} <= arrays
