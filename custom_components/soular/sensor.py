"""Sensor platform setup."""

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from custom_components.soular import SoularConfigEntry
from custom_components.soular.const import SUBENTRY_TYPE_ARRAY
from custom_components.soular.entities import ARRAY_DIAGNOSTICS, SENSOR_DESCRIPTIONS
from custom_components.soular.entities.sensor import SoularArrayDiagnosticSensor, SoularForecastSensor
from custom_components.soular.system import site_name

# The coordinator pushes; entities never poll independently.
PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001 - fixed by the sensor platform signature
    entry: SoularConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Create the site's sensors, and one set per array."""
    runtime = entry.runtime_data
    coordinator = runtime.coordinator
    if coordinator is None:  # pragma: no cover - setup guarantees this
        return

    name = site_name(entry)
    entities: list[SensorEntity] = [
        SoularForecastSensor(coordinator, description, entry, name) for description in SENSOR_DESCRIPTIONS
    ]
    async_add_entities(entities)

    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_ARRAY:
            continue
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
