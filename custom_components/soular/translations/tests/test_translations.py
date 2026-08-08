"""Every user-visible string exists, and nothing unused lingers.

Asserted in both directions. A missing key shows the user a raw slug; an unused
key is a rename someone only half finished. Neither fails any other test.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from custom_components.soular.entities import ALL_SENSOR_KEYS
from custom_components.soular.flows.array import array_schema
from custom_components.soular.flows.hub import site_schema

TRANSLATIONS = Path(__file__).parent.parent / "en.json"
ICONS = Path(__file__).parent.parent.parent / "icons.json"


@pytest.fixture(scope="module")
def translations() -> dict[str, Any]:
    """Load the shipped English translations."""
    return json.loads(TRANSLATIONS.read_text())


@pytest.fixture(scope="module")
def icons() -> dict[str, Any]:
    """Load the shipped icon map."""
    return json.loads(ICONS.read_text())


def test_every_sensor_has_a_name(translations: dict[str, Any]) -> None:
    """A sensor without a translation shows its key to the user."""
    named = set(translations["entity"]["sensor"])
    missing = ALL_SENSOR_KEYS - named
    assert not missing, f"no translation for {sorted(missing)}"


def test_no_unused_sensor_translations(translations: dict[str, Any]) -> None:
    """A leftover key is a rename that was not finished."""
    named = set(translations["entity"]["sensor"])
    unused = named - ALL_SENSOR_KEYS
    assert not unused, f"translations for sensors that do not exist: {sorted(unused)}"


def test_every_sensor_has_an_icon(icons: dict[str, Any]) -> None:
    """Every sensor gets an icon, so none falls back to a generic one."""
    iconed = set(icons["entity"]["sensor"])
    missing = ALL_SENSOR_KEYS - iconed
    assert not missing, f"no icon for {sorted(missing)}"


def test_no_unused_icons(icons: dict[str, Any]) -> None:
    """An icon for a sensor that no longer exists is dead weight."""
    iconed = set(icons["entity"]["sensor"])
    unused = iconed - ALL_SENSOR_KEYS
    assert not unused, f"icons for sensors that do not exist: {sorted(unused)}"


def schema_keys(schema: Any) -> set[str]:
    """Return the field names of a voluptuous schema."""
    return {str(marker.schema) for marker in schema.schema}


def test_site_form_fields_are_described(translations: dict[str, Any]) -> None:
    """Every site field has both a label and a description.

    Descriptions rather than only labels because several of these fields are
    genuinely ambiguous -- azimuth conventions differ between integrations, and
    an inverter limit set wrongly is invisible until a sunny day.
    """
    step = translations["config"]["step"]["user"]
    fields = schema_keys(site_schema())

    assert fields <= set(step["data"]), f"unlabelled: {sorted(fields - set(step['data']))}"
    assert fields <= set(step["data_description"]), f"undescribed: {sorted(fields - set(step['data_description']))}"


@pytest.mark.parametrize("step_id", ["user", "reconfigure"])
def test_array_form_fields_are_described(translations: dict[str, Any], step_id: str) -> None:
    """Every array field has a label and a description, on both steps."""
    step = translations["config_subentries"]["array"]["step"][step_id]
    fields = schema_keys(array_schema())

    assert fields <= set(step["data"]), f"unlabelled: {sorted(fields - set(step['data']))}"
    assert fields <= set(step["data_description"]), f"undescribed: {sorted(fields - set(step['data_description']))}"


def test_array_flow_errors_are_translated(translations: dict[str, Any]) -> None:
    """Every error the array flow can raise has a message."""
    declared = set(translations["config_subentries"]["array"]["error"])
    raised = {"name_required", "invalid_shading_file", "shading_file_not_found"}
    assert raised <= declared, f"untranslated errors: {sorted(raised - declared)}"


def test_ground_type_options_are_translated(translations: dict[str, Any]) -> None:
    """The ground-surface dropdown shows words, not slugs."""
    from custom_components.soular.const import GROUND_TYPE_ALBEDO  # noqa: PLC0415

    options = set(translations["selector"]["ground_type"]["options"])
    assert set(GROUND_TYPE_ALBEDO) == options
