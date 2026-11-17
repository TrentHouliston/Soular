"""Read measured output and the instants it can be compared at.

Nothing here decides how the forecast is corrected; that is
:mod:`custom_components.soular.core.weather_bias`, which is pure so the backtest
applies exactly the same rules.
"""

import logging

from homeassistant.core import HomeAssistant, State
import numpy as np

from custom_components.soular.core.clearsky import clear_sky, clear_sky_index
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.pipeline import SystemSpec

_LOGGER = logging.getLogger(__name__)


def forecast_clearness(times: np.ndarray, system: SystemSpec, ghi: np.ndarray) -> np.ndarray:
    """Return the forecast's horizontal clear-sky index, which the weather correction is keyed on."""
    clearsky = clear_sky(times, system.site, solar_geometry(times, system.site)).ghi
    return np.nan_to_num(clear_sky_index(ghi, clearsky), nan=1.0)


def read_power(hass: HomeAssistant, entity_id: str | None) -> float | None:
    """Read a power sensor in watts, or nothing if it is not usable.

    Kilowatts are converted: plenty of inverter integrations report kW, and a
    factor of a thousand in the training signal would drive the correction
    straight into its clamp.
    """
    if not entity_id:
        return None
    state: State | None = hass.states.get(entity_id)
    if state is None or state.state in ("unknown", "unavailable", ""):
        return None
    try:
        value = float(state.state)
    except (TypeError, ValueError):
        return None

    unit = str(state.attributes.get("unit_of_measurement", "")).strip()
    if unit in ("kW", "kilowatt"):
        value *= 1000.0
    elif unit not in ("W", "watt", ""):
        _LOGGER.debug("Ignoring %s: unexpected power unit %r", entity_id, unit)
        return None
    return value


def read_percentage(hass: HomeAssistant, entity_id: str | None) -> float | None:
    """Read a percentage sensor, or nothing if it is not usable."""
    if not entity_id:
        return None
    state = hass.states.get(entity_id)
    if state is None or state.state in ("unknown", "unavailable", ""):
        return None
    try:
        return float(state.state)
    except (TypeError, ValueError):
        return None
