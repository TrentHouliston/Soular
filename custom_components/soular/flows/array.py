"""Subentry flow for adding a solar array to a site."""

from typing import Any

from homeassistant.config_entries import ConfigSubentryFlow, SubentryFlowResult
from homeassistant.const import CONF_NAME
from homeassistant.helpers import selector
import voluptuous as vol

from custom_components.soular.const import (
    CONF_AZIMUTH,
    CONF_DC_CAPACITY,
    CONF_DC_LOSS,
    CONF_SHADING_FILE,
    CONF_TEMPERATURE_COEFFICIENT,
    CONF_TILT,
    DEFAULT_DC_LOSS_PERCENT,
    DEFAULT_TEMPERATURE_COEFFICIENT,
)
from custom_components.soular.core.shading import ShadingFormatError
from custom_components.soular.system import load_shading_file, shading_directory


def array_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    """Build the array form.

    Azimuth is compass degrees from north, matching both the shading files and
    every other Home Assistant solar integration. Tilt is from horizontal.
    """
    values = defaults or {}
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=values.get(CONF_NAME, "")): selector.TextSelector(),
            vol.Required(CONF_AZIMUTH, default=values.get(CONF_AZIMUTH, 0.0)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=360, step=1, unit_of_measurement="°")
            ),
            vol.Required(CONF_TILT, default=values.get(CONF_TILT, 25.0)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=90, step=1, unit_of_measurement="°")
            ),
            vol.Required(CONF_DC_CAPACITY, default=values.get(CONF_DC_CAPACITY, 5000.0)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, step=1, unit_of_measurement="W")
            ),
            vol.Optional(
                CONF_TEMPERATURE_COEFFICIENT,
                default=values.get(CONF_TEMPERATURE_COEFFICIENT, DEFAULT_TEMPERATURE_COEFFICIENT),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=-1, max=0, step=0.01, unit_of_measurement="%/°C")
            ),
            vol.Optional(
                CONF_DC_LOSS, default=values.get(CONF_DC_LOSS, DEFAULT_DC_LOSS_PERCENT)
            ): selector.NumberSelector(selector.NumberSelectorConfig(min=0, max=50, step=0.5, unit_of_measurement="%")),
            vol.Optional(CONF_SHADING_FILE, default=values.get(CONF_SHADING_FILE, "")): selector.TextSelector(),
        }
    )


class ArraySubentryFlow(ConfigSubentryFlow):
    """Add or reconfigure one array."""

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Collect the array's geometry and optional shading file."""
        return await self._async_form(user_input)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Edit an existing array."""
        return await self._async_form(user_input, existing=dict(self._get_reconfigure_subentry().data))

    async def _async_form(
        self,
        user_input: dict[str, Any] | None,
        existing: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        """Show and handle the array form."""
        errors: dict[str, str] = {}

        if user_input is not None:
            name = str(user_input[CONF_NAME]).strip()
            if not name:
                errors[CONF_NAME] = "name_required"

            filename = str(user_input.get(CONF_SHADING_FILE, "")).strip()
            if filename and not errors:
                # Validate by actually parsing it. A shading file that silently
                # fails to load leaves the forecast quietly unshaded, which on
                # this kind of site is a 12% error nobody would notice.
                error = await self.hass.async_add_executor_job(self._validate_shading, filename, name)
                if error:
                    errors[CONF_SHADING_FILE] = error

            if not errors:
                data = {**user_input, CONF_NAME: name, CONF_SHADING_FILE: filename}
                if existing is not None:
                    return self.async_update_and_abort(
                        self._get_entry(), self._get_reconfigure_subentry(), data=data, title=name
                    )
                return self.async_create_entry(title=name, data=data)

        return self.async_show_form(
            step_id="reconfigure" if existing is not None else "user",
            data_schema=array_schema(user_input or existing),
            errors=errors,
        )

    def _validate_shading(self, filename: str, array_name: str) -> str | None:
        """Try to load a shading file; return an error key or None."""
        path = shading_directory(self.hass) / filename
        try:
            load_shading_file(path, array_name)
        except ShadingFormatError:
            return "invalid_shading_file"
        except OSError:
            return "shading_file_not_found"
        return None
