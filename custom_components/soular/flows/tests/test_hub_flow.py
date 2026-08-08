"""Tests for the hub (site) config flow."""

from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.soular.const import CONF_ALBEDO, CONF_ELEVATION, CONF_GROUND_TYPE, DOMAIN, GROUND_TYPE_ALBEDO

SITE_INPUT = {
    CONF_NAME: "Morisset Park",
    CONF_LATITUDE: -33.11915471966274,
    CONF_LONGITUDE: 151.53401076793673,
    CONF_ELEVATION: 10.0,
    CONF_GROUND_TYPE: "grass",
}


async def test_user_flow_creates_entry(hass: HomeAssistant) -> None:
    """A complete site form creates an entry titled after the site."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "user"

    result = await hass.config_entries.flow.async_configure(result["flow_id"], user_input=SITE_INPUT)
    await hass.async_block_till_done()

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("title") == "Morisset Park"
    assert result.get("data") == {**SITE_INPUT, CONF_ALBEDO: GROUND_TYPE_ALBEDO["grass"]}


async def test_ground_type_resolves_to_albedo(hass: HomeAssistant) -> None:
    """The ground-surface dropdown is stored as a number, not a label.

    Downstream transposition needs a reflectance; keeping the label as well would let
    the two drift apart on a reconfigure.
    """
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={**SITE_INPUT, CONF_GROUND_TYPE: "snow"}
    )
    await hass.async_block_till_done()

    assert result.get("data", {})[CONF_ALBEDO] == 0.65


async def test_same_location_aborts(hass: HomeAssistant) -> None:
    """A second entry for the same coordinates aborts rather than duplicating the site."""
    first = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    await hass.config_entries.flow.async_configure(first["flow_id"], user_input=SITE_INPUT)
    await hass.async_block_till_done()

    second = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(second["flow_id"], user_input=SITE_INPUT)

    assert result.get("type") is FlowResultType.ABORT
    assert result.get("reason") == "already_configured"
