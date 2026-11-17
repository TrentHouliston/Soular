"""Persist what the correction has learned.

``basis`` is the versioning lever. The correction used to be a per-array
regression on eight weather features; that state means nothing to the weather
correction, so anything stored under another basis is discarded and the
recorder backfill starts the record again from the last week.
"""

from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from custom_components.soular.const import DOMAIN
from custom_components.soular.core import weather_bias
from custom_components.soular.core.weather_bias import WeatherBias

STORAGE_VERSION = 1
STORAGE_MINOR_VERSION = 1

# What the stored state means. Bumped whenever a stored state stops meaning the
# same thing to the code reading it.
BASIS: str = "weather-v1"

# Saving on every sample would write to disk every five minutes for no benefit;
# losing a quarter of an hour of learning to a crash costs nothing measurable.
SAVE_INTERVAL = timedelta(minutes=15)


def encode(bias: WeatherBias) -> dict[str, Any]:
    """Serialise the weather correction's record."""
    return {"basis": BASIS, "weather": weather_bias.encode(bias)}


def decode(payload: Any) -> WeatherBias:
    """Restore the record, discarding anything stored under another basis."""
    if not isinstance(payload, dict) or payload.get("basis") != BASIS:
        return WeatherBias()
    return weather_bias.decode(payload.get("weather"))


def build_store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, Any]]:
    """Create the store for one config entry."""
    return Store[dict[str, Any]](
        hass,
        STORAGE_VERSION,
        f"{DOMAIN}.learning.{entry_id}",
        minor_version=STORAGE_MINOR_VERSION,
    )


def build_band_store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, Any]]:
    """Create the store for one config entry's forecast band record."""
    return Store[dict[str, Any]](
        hass, STORAGE_VERSION, f"{DOMAIN}.band.{entry_id}", minor_version=STORAGE_MINOR_VERSION
    )
