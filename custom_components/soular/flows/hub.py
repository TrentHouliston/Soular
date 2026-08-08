"""Hub config flow: the site every array belongs to.

One config entry per site. Arrays are added afterwards as subentries, because a
site's location is shared while geometry, shading and the actual-power sensor all
belong to an individual array.
"""

from typing import Any

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.helpers import selector
import voluptuous as vol

from custom_components.soular.const import (
    CONF_ALBEDO,
    CONF_ELEVATION,
    CONF_GROUND_TYPE,
    DEFAULT_GROUND_TYPE,
    DOMAIN,
    GROUND_TYPE_ALBEDO,
)


class HubConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial site setup."""

    VERSION = 1
    MINOR_VERSION = 1

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

        return self.async_show_form(step_id="user", data_schema=self._schema())

    def _schema(self) -> vol.Schema:
        """Build the site form, defaulting to the Home Assistant install's own location."""
        return vol.Schema(
            {
                vol.Required(CONF_NAME, default="Solar"): selector.TextSelector(),
                vol.Required(CONF_LATITUDE, default=self.hass.config.latitude): vol.Coerce(float),
                vol.Required(CONF_LONGITUDE, default=self.hass.config.longitude): vol.Coerce(float),
                vol.Required(CONF_ELEVATION, default=float(self.hass.config.elevation)): vol.Coerce(float),
                vol.Required(CONF_GROUND_TYPE, default=DEFAULT_GROUND_TYPE): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=sorted(GROUND_TYPE_ALBEDO),
                        translation_key=CONF_GROUND_TYPE,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )
