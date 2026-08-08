"""Solar production forecast for Home Assistant's energy dashboard.

Discovered by platform lookup on this module's name, so the filename matters.
"""

import logging

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular import SoularConfigEntry
from custom_components.soular.core.series import hourly_energy

_LOGGER = logging.getLogger(__name__)

WATT_HOURS_PER_KILOWATT_HOUR = 1000.0


async def async_get_solar_forecast(hass: HomeAssistant, config_entry_id: str) -> dict[str, dict[str, float]] | None:
    """Return hourly forecast energy for the energy dashboard.

    Watt-hours per clock hour, keyed by the hour's start in local time, for the
    whole site.
    """
    entry: SoularConfigEntry | None = hass.config_entries.async_get_entry(config_entry_id)  # type: ignore[assignment]
    # An entry that is not loaded has no runtime_data attribute at all, so its
    # state is the guard rather than a None check on the attribute.
    if entry is None or entry.state is not ConfigEntryState.LOADED:
        return None

    coordinator = entry.runtime_data.coordinator
    if coordinator is None:
        return None

    result = coordinator.data
    return {
        "wh_hours": {
            dt_util.as_local(
                dt_util.utc_from_timestamp(int(hour.astype("datetime64[s]").astype(np.int64)))
            ).isoformat(): energy * WATT_HOURS_PER_KILOWATT_HOUR
            for hour, energy in hourly_energy(result.times, result.ac_power_w).items()
        }
    }
