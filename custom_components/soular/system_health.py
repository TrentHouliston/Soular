"""System health for Soular.

Answers the question a user has when the forecast looks wrong: which of the
three upstream services is actually reachable right now. They fail
independently and the integration keeps working without any of the optional
ones, so "it is still producing a forecast" is not evidence that everything is
up.
"""

from typing import Any

from homeassistant.components import system_health
from homeassistant.core import HomeAssistant, callback

from custom_components.soular.const import DOMAIN
from custom_components.soular.sources.ensemble import ENSEMBLE_URL
from custom_components.soular.sources.open_meteo import FORECAST_URL
from custom_components.soular.sources.satellite import ARCHIVE_URL


@callback
def async_register(
    hass: HomeAssistant,  # noqa: ARG001 - fixed by the system health registration signature
    register: system_health.SystemHealthRegistration,
) -> None:
    """Register the health callback."""
    register.async_register_info(async_system_health_info)


async def async_system_health_info(hass: HomeAssistant) -> dict[str, Any]:
    """Report reachability of each source, and what the forecast currently has."""
    info: dict[str, Any] = {
        "weather_api": system_health.async_check_can_reach_url(hass, FORECAST_URL),
        "satellite_api": system_health.async_check_can_reach_url(hass, ARCHIVE_URL),
        "ensemble_api": system_health.async_check_can_reach_url(hass, ENSEMBLE_URL),
    }

    entries = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        runtime = getattr(entry, "runtime_data", None)
        coordinator = runtime.coordinator if runtime else None
        if coordinator is None:
            continue
        prefix = entry.title or entry.entry_id
        info[f"{prefix} forecast"] = "ok" if coordinator.last_update_success else "failing"
        # The two optional refinements, reported separately from the backbone:
        # either can be absent for hours without the forecast noticing.
        info[f"{prefix} nowcast"] = f"{coordinator.observed_share * 100:.0f}% observed"
        info[f"{prefix} uncertainty"] = "available" if coordinator.uncertainty_available else "unavailable"

    return info
