"""The site's totals live under a subentry of their own, beside the arrays."""

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.const import DOMAIN, SITE_SUBENTRY_TITLE, SUBENTRY_TYPE_SITE
from custom_components.soular.entities.device import site_identifier
from tests.integration.conftest import build_entry


def site_subentries(entry: MockConfigEntry) -> list[Any]:
    """Return the entry's site subentries."""
    return [subentry for subentry in entry.subentries.values() if subentry.subentry_type == SUBENTRY_TYPE_SITE]


def site_entities(hass: HomeAssistant, entry: MockConfigEntry) -> list[er.RegistryEntry]:
    """Return the entities on the site device."""
    device = dr.async_get(hass).async_get_device(identifiers={site_identifier(entry)})
    assert device is not None
    return er.async_entries_for_device(er.async_get(hass), device.id)


async def test_setup_puts_the_site_under_its_own_subentry(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """Without one, the totals show as "devices that don't belong to a sub-entry"."""
    (site,) = site_subentries(configured)
    assert site.title == SITE_SUBENTRY_TITLE

    device = dr.async_get(hass).async_get_device(identifiers={site_identifier(configured)})
    assert device is not None
    assert device.config_entries_subentries[configured.entry_id] == {site.subentry_id}

    entities = site_entities(hass, configured)
    assert entities
    assert {entity.config_subentry_id for entity in entities} == {site.subentry_id}


async def test_a_reload_does_not_add_a_second_site(hass: HomeAssistant, configured: MockConfigEntry) -> None:
    """The site subentry is created once and then found."""
    assert await hass.config_entries.async_reload(configured.entry_id)
    await hass.async_block_till_done()
    assert len(site_subentries(configured)) == 1


async def test_an_existing_install_moves_its_site_across(hass: HomeAssistant, mock_weather: Any) -> None:
    """A site device and entities registered before the subentry existed are moved, not duplicated."""
    await hass.config.async_set_time_zone("Australia/Sydney")
    entry = build_entry()
    entry.add_to_hass(hass)
    devices = dr.async_get(hass)
    old = devices.async_get_or_create(config_entry_id=entry.entry_id, identifiers={site_identifier(entry)})
    old_entity = er.async_get(hass).async_get_or_create(
        "sensor",
        DOMAIN,
        f"{entry.entry_id}_power_production_now",
        config_entry=entry,
        device_id=old.id,
    )
    assert old_entity.config_subentry_id is None

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    (site,) = site_subentries(entry)
    device = devices.async_get(old.id)
    assert device is not None
    assert device.config_entries_subentries[entry.entry_id] == {site.subentry_id}
    moved = er.async_get(hass).async_get(old_entity.entity_id)
    assert moved is not None
    assert moved.config_subentry_id == site.subentry_id
