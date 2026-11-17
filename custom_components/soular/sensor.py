"""Sensor platform setup."""

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from custom_components.soular import SoularConfigEntry, ensure_site_subentry
from custom_components.soular.const import SUBENTRY_TYPE_ARRAY
from custom_components.soular.entities import ARRAY_DIAGNOSTICS, SENSOR_DESCRIPTIONS, SITE_DIAGNOSTICS
from custom_components.soular.entities.device import register_devices
from custom_components.soular.entities.sensor import (
    SoularArrayDiagnosticSensor,
    SoularForecastSensor,
    SoularSiteDiagnosticSensor,
)
from custom_components.soular.system import site_name

# The coordinator pushes; entities never poll independently.
PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SoularConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Create the site's sensors, and one set per array."""
    runtime = entry.runtime_data
    coordinator = runtime.coordinator
    if coordinator is None:  # pragma: no cover - setup guarantees this
        return

    name = site_name(entry)
    # The arrays the coordinator was built with, not whatever the entry holds by
    # now: one added during setup has no forecast behind it until the reload it
    # triggered, and an entity for it would sit unavailable until then.
    modelled = {array.name for array in coordinator.system.arrays}
    arrays = [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_ARRAY and str(subentry.title) in modelled
    ]
    site = ensure_site_subentry(hass, entry)
    register_devices(hass, entry, name, site, arrays)

    entities: list[SensorEntity] = [
        SoularForecastSensor(coordinator, description, entry, name) for description in SENSOR_DESCRIPTIONS
    ]
    entities.extend(
        SoularSiteDiagnosticSensor(coordinator, description, entry, name) for description in SITE_DIAGNOSTICS
    )
    # Under the site's own subentry, alongside the arrays rather than apart from them.
    async_add_entities(entities, config_subentry_id=site.subentry_id)

    for subentry in arrays:
        array_entities: list[SensorEntity] = [
            SoularForecastSensor(coordinator, description, entry, name, subentry)
            for description in SENSOR_DESCRIPTIONS
            if not description.site_only
        ]
        array_entities.extend(
            SoularArrayDiagnosticSensor(coordinator, description, entry, name, subentry)
            for description in ARRAY_DIAGNOSTICS
        )
        # Entities created against a subentry must be registered to it, or they
        # are orphaned when that array is removed.
        async_add_entities(array_entities, config_subentry_id=subentry.subentry_id)
