"""System health for Soular.

Answers the question a user has when the forecast looks wrong: is the weather
service reachable, and is each site still producing a forecast.
"""

from typing import Any

from homeassistant.components import system_health
from homeassistant.core import HomeAssistant, callback

from custom_components.soular.const import DOMAIN
from custom_components.soular.sources.open_meteo import FORECAST_URL


@callback
def async_register(
    hass: HomeAssistant,  # noqa: ARG001 - fixed by the system health registration signature
    register: system_health.SystemHealthRegistration,
) -> None:
    """Register the health callback."""
    register.async_register_info(async_system_health_info)


async def async_system_health_info(hass: HomeAssistant) -> dict[str, Any]:
    """Report reachability of the weather service, and each site's forecast."""
    info: dict[str, Any] = {
        "weather_api": system_health.async_check_can_reach_url(hass, FORECAST_URL),
    }

    entries = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        runtime = getattr(entry, "runtime_data", None)
        coordinator = runtime.coordinator if runtime else None
        if coordinator is None:
            continue
        prefix = entry.title or entry.entry_id
        info[f"{prefix} forecast"] = "ok" if coordinator.last_update_success else "failing"

    return info
