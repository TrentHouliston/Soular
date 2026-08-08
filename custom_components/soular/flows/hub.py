"""Hub config flow: the site every array belongs to.

One config entry per site. Arrays are added afterwards as subentries, because a
site's location is shared while geometry, shading and the actual-power sensor all
belong to an individual array.
"""

from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, ConfigSubentryFlow
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers import selector
import voluptuous as vol

from custom_components.soular.const import (
    CONF_ALBEDO,
    CONF_BATTERY_SOC_SENSOR,
    CONF_ELEVATION,
    CONF_GROUND_TYPE,
    CONF_INVERTER_AC_LIMIT,
    DEFAULT_GROUND_TYPE,
    DOMAIN,
    GROUND_TYPE_ALBEDO,
    SUBENTRY_TYPE_ARRAY,
)
from custom_components.soular.flows.array import ArraySubentryFlow


def site_schema(
    latitude: float = 0.0,
    longitude: float = 0.0,
    elevation: float = 0.0,
) -> vol.Schema:
    """Build the site form.

    Defaults are passed in rather than read from ``hass`` so the schema can be
    built and inspected without a running instance -- which is what lets a test
    assert that every field it declares actually has a translation.
    """
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default="Solar"): selector.TextSelector(),
            vol.Required(CONF_LATITUDE, default=latitude): vol.Coerce(float),
            vol.Required(CONF_LONGITUDE, default=longitude): vol.Coerce(float),
            vol.Required(CONF_ELEVATION, default=elevation): vol.Coerce(float),
            vol.Required(CONF_GROUND_TYPE, default=DEFAULT_GROUND_TYPE): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=sorted(GROUND_TYPE_ALBEDO),
                    translation_key=CONF_GROUND_TYPE,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            # Optional, and left empty by default. Plenty of systems never reach
            # their inverter's limit, and a limit guessed too low truncates the
            # forecast at exactly the sunniest moment.
            vol.Optional(CONF_INVERTER_AC_LIMIT): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, step=100, unit_of_measurement="W")
            ),
            # Optional. A full battery curtails the array at exactly the sunniest
            # part of the day, and those samples would otherwise teach the
            # correction that the array had lost efficiency.
            vol.Optional(CONF_BATTERY_SOC_SENSOR): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="battery")
            ),
        }
    )


class HubConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial site setup."""

    VERSION = 1
    MINOR_VERSION = 1

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls,
        config_entry: ConfigEntry,  # noqa: ARG003 - fixed by the config flow signature
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Arrays are added as subentries, one device each."""
        return {SUBENTRY_TYPE_ARRAY: ArraySubentryFlow}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Collect the site location."""
        if user_input is not None:
            ground_type: str = user_input[CONF_GROUND_TYPE]
            latitude: float = user_input[CONF_LATITUDE]
            longitude: float = user_input[CONF_LONGITUDE]
            name: str = user_input[CONF_NAME]
            data = {**user_input, CONF_ALBEDO: GROUND_TYPE_ALBEDO[ground_type]}
            # Five decimal places is ~1 m, well inside the resolution of any weather
            # model, so two entries this close would be forecasting the same site.
            await self.async_set_unique_id(f"{latitude:.5f}_{longitude:.5f}")
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title=name, data=data)

        # Default to wherever Home Assistant thinks it is; most people's panels
        # are on the roof above it.
        return self.async_show_form(
            step_id="user",
            data_schema=site_schema(
                latitude=self.hass.config.latitude,
                longitude=self.hass.config.longitude,
                elevation=float(self.hass.config.elevation),
            ),
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Change the site after the fact.

        The inverter limit is the field this exists for. It is optional at setup
        because a guess that is too low truncates the forecast at the sunniest
        moment, so the honest answer for most people is to leave it empty and fill
        it in once they know -- which requires being able to come back.
        """
        entry = self._get_reconfigure_entry()
        if user_input is not None:
            ground_type: str = user_input[CONF_GROUND_TYPE]
            data = {**user_input, CONF_ALBEDO: GROUND_TYPE_ALBEDO[ground_type]}
            latitude: float = user_input[CONF_LATITUDE]
            longitude: float = user_input[CONF_LONGITUDE]
            await self.async_set_unique_id(f"{latitude:.5f}_{longitude:.5f}")
            self._abort_if_unique_id_mismatch(reason="cannot_move_site")
            return self.async_update_reload_and_abort(entry, data=data, title=str(user_input[CONF_NAME]))

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(site_schema(), entry.data),
        )
