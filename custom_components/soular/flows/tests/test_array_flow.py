"""Tests for the array subentry flow, including shading-file validation."""

from pathlib import Path
from typing import Any

from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import numpy as np
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.soular.const import (
    CONF_AZIMUTH,
    CONF_DC_CAPACITY,
    CONF_SHADING_FILE,
    CONF_TILT,
    DOMAIN,
    SHADING_DIRECTORY,
    SUBENTRY_TYPE_ARRAY,
)
from custom_components.soular.tests.conftest import build_entry

ARRAY_INPUT: dict[str, Any] = {
    CONF_NAME: "east",
    CONF_AZIMUTH: 84.0,
    CONF_TILT: 25.0,
    CONF_DC_CAPACITY: 7920.0,
}


def write_grid(hass: HomeAssistant, filename: str, arrays: tuple[str, ...] = ("east",)) -> Path:
    """Write a small but structurally real transmittance grid."""
    directory = Path(hass.config.path(SHADING_DIRECTORY))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    payload: dict[str, Any] = {
        "azimuth_deg": np.arange(0.0, 361.0, 10.0),
        "elevation_deg": np.arange(0.0, 91.0, 10.0),
    }
    for name in arrays:
        payload[f"T_{name}"] = np.full((37, 10), 0.8)
    np.savez(path, **payload)
    return path


async def start_array_flow(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    """Begin an array subentry flow and return its id."""
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_ARRAY), context={"source": SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    return result["flow_id"]


@pytest.fixture
async def entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a site entry with no arrays yet."""
    created = build_entry(arrays=())
    created.add_to_hass(hass)
    return created


async def test_adding_an_array_creates_a_subentry(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """A complete form creates a subentry titled after the array."""
    flow_id = await start_array_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(flow_id, ARRAY_INPUT)

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("title") == "east"
    assert result.get("data", {})[CONF_AZIMUTH] == 84.0


async def test_a_blank_name_is_rejected(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """An unnamed array would produce unusable entity names."""
    flow_id = await start_array_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(flow_id, {**ARRAY_INPUT, CONF_NAME: "   "})

    assert result.get("type") is FlowResultType.FORM
    assert result.get("errors") == {CONF_NAME: "name_required"}


async def test_a_valid_shading_file_is_accepted(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """A grid containing this array's name loads."""
    await hass.async_add_executor_job(write_grid, hass, "east.npz")

    flow_id = await start_array_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        flow_id, {**ARRAY_INPUT, CONF_SHADING_FILE: "east.npz"}
    )

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("data", {})[CONF_SHADING_FILE] == "east.npz"


async def test_a_missing_shading_file_is_rejected(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """A typo'd filename is caught at configuration time.

    It matters that this is caught here. A shading file that quietly fails to
    load leaves the forecast unshaded, which on a treed site is a double-digit
    error with nothing anywhere to indicate it.
    """
    flow_id = await start_array_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        flow_id, {**ARRAY_INPUT, CONF_SHADING_FILE: "nope.npz"}
    )

    assert result.get("type") is FlowResultType.FORM
    assert result.get("errors") == {CONF_SHADING_FILE: "invalid_shading_file"}


async def test_a_grid_without_this_array_is_rejected(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """A grid that has no series for this array name is a misconfiguration."""
    await hass.async_add_executor_job(write_grid, hass, "others.npz", ("west", "north"))

    flow_id = await start_array_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        flow_id, {**ARRAY_INPUT, CONF_SHADING_FILE: "others.npz"}
    )

    assert result.get("type") is FlowResultType.FORM
    assert result.get("errors") == {CONF_SHADING_FILE: "invalid_shading_file"}


async def test_a_horizon_file_is_accepted(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """The two-column horizon format the incumbent uses still works."""

    def write() -> None:
        directory = Path(hass.config.path(SHADING_DIRECTORY))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "horizon-east.txt").write_text("\n".join(f"{azimuth}\t{15.0}" for azimuth in range(0, 361, 2)))

    await hass.async_add_executor_job(write)

    flow_id = await start_array_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        flow_id, {**ARRAY_INPUT, CONF_SHADING_FILE: "horizon-east.txt"}
    )

    assert result.get("type") is FlowResultType.CREATE_ENTRY


async def test_shading_is_optional(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """An array with no shading file configures fine."""
    flow_id = await start_array_flow(hass, entry)
    result = await hass.config_entries.subentries.async_configure(flow_id, ARRAY_INPUT)

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("data", {})[CONF_SHADING_FILE] == ""


async def test_subentry_type_is_offered(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """The integration advertises the array subentry type."""
    from custom_components.soular.config_flow import SoularConfigFlow  # noqa: PLC0415

    assert set(SoularConfigFlow.async_get_supported_subentry_types(entry)) == {SUBENTRY_TYPE_ARRAY}
    assert entry.domain == DOMAIN
