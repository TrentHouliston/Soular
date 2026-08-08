"""Entity descriptions and device wiring.

The sensor keys deliberately mirror ``forecast_solar`` and
``open_meteo_solar_forecast``. Anyone migrating brings dashboards, blueprints and
automations with them, and a gratuitously different name would cost them a
rewrite for nothing.
"""

from collections.abc import Callable
from dataclasses import dataclass
import datetime as dt
from typing import Final

from homeassistant.components.sensor import SensorDeviceClass, SensorEntityDescription, SensorStateClass
from homeassistant.const import EntityCategory, UnitOfEnergy, UnitOfPower, UnitOfTemperature
import numpy as np

from custom_components.soular.core.series import energy_between, peak_time, power_at
from custom_components.soular.core.types import FloatArray, TimeArray

# What a sensor is handed to compute itself: the series, and the local day
# boundaries it may need.
type SeriesValue = float | dt.datetime | None


@dataclass(frozen=True, slots=True)
class Window:
    """Local day boundaries, as UTC instants on the forecast grid."""

    now: np.datetime64
    today_start: np.datetime64
    today_end: np.datetime64
    tomorrow_end: np.datetime64


@dataclass(frozen=True, kw_only=True)
class SoularSensorEntityDescription(SensorEntityDescription):
    """A sensor, and how to compute it from a power series."""

    value: Callable[[TimeArray, FloatArray, Window], SeriesValue]
    # Only the power sensor carries the full series; attaching it to every sensor
    # would multiply a 20 kB attribute across a dozen entities.
    carries_forecast: bool = False
    # Site-only sensors would be meaningless or duplicated per array.
    site_only: bool = False


def _peak(times: TimeArray, power: FloatArray, start: np.datetime64, end: np.datetime64) -> dt.datetime | None:
    """Return a peak time as an aware datetime, or None."""
    peak = peak_time(times, power, start, end)
    if peak is None:
        return None
    return dt.datetime.fromtimestamp(int(peak.astype("datetime64[s]").astype(np.int64)), tz=dt.UTC)


SENSOR_DESCRIPTIONS: Final[tuple[SoularSensorEntityDescription, ...]] = (
    SoularSensorEntityDescription(
        key="power_production_now",
        translation_key="power_production_now",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        suggested_display_precision=0,
        carries_forecast=True,
        value=lambda times, power, window: power_at(times, power, window.now),
    ),
    SoularSensorEntityDescription(
        key="energy_production_today",
        translation_key="energy_production_today",
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=1,
        value=lambda times, power, window: energy_between(times, power, window.today_start, window.today_end),
    ),
    SoularSensorEntityDescription(
        key="energy_production_today_remaining",
        translation_key="energy_production_today_remaining",
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=1,
        value=lambda times, power, window: energy_between(times, power, window.now, window.today_end),
    ),
    SoularSensorEntityDescription(
        key="energy_production_tomorrow",
        translation_key="energy_production_tomorrow",
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=1,
        value=lambda times, power, window: energy_between(times, power, window.today_end, window.tomorrow_end),
    ),
    SoularSensorEntityDescription(
        key="energy_current_hour",
        translation_key="energy_current_hour",
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=2,
        entity_registry_enabled_default=False,
        value=lambda times, power, window: energy_between(
            times,
            power,
            window.now.astype("datetime64[h]").astype("datetime64[s]"),
            window.now.astype("datetime64[h]").astype("datetime64[s]") + np.timedelta64(3600, "s"),
        ),
    ),
    SoularSensorEntityDescription(
        key="energy_next_hour",
        translation_key="energy_next_hour",
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=2,
        entity_registry_enabled_default=False,
        value=lambda times, power, window: energy_between(
            times,
            power,
            window.now.astype("datetime64[h]").astype("datetime64[s]") + np.timedelta64(3600, "s"),
            window.now.astype("datetime64[h]").astype("datetime64[s]") + np.timedelta64(7200, "s"),
        ),
    ),
    SoularSensorEntityDescription(
        key="power_highest_peak_time_today",
        translation_key="power_highest_peak_time_today",
        device_class=SensorDeviceClass.TIMESTAMP,
        site_only=True,
        value=lambda times, power, window: _peak(times, power, window.today_start, window.today_end),
    ),
    SoularSensorEntityDescription(
        key="power_highest_peak_time_tomorrow",
        translation_key="power_highest_peak_time_tomorrow",
        device_class=SensorDeviceClass.TIMESTAMP,
        site_only=True,
        value=lambda times, power, window: _peak(times, power, window.today_end, window.tomorrow_end),
    ),
)

SENSOR_KEYS: Final[frozenset[str]] = frozenset(entry.key for entry in SENSOR_DESCRIPTIONS)


@dataclass(frozen=True, kw_only=True)
class ArrayDiagnosticDescription(SensorEntityDescription):
    """A per-array diagnostic, computed from the array's own forecast fields."""

    field: str


ARRAY_DIAGNOSTICS: Final[tuple[ArrayDiagnosticDescription, ...]] = (
    ArrayDiagnosticDescription(
        key="poa_irradiance",
        translation_key="poa_irradiance",
        device_class=SensorDeviceClass.IRRADIANCE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement="W/m²",
        suggested_display_precision=0,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        field="poa_global",
    ),
    ArrayDiagnosticDescription(
        key="cell_temperature",
        translation_key="cell_temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        suggested_display_precision=1,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        field="cell_temperature",
    ),
    ArrayDiagnosticDescription(
        key="shading_transmittance",
        translation_key="shading_transmittance",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement="%",
        suggested_display_precision=0,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        field="transmittance",
    ),
)

DIAGNOSTIC_KEYS: Final[frozenset[str]] = frozenset(entry.key for entry in ARRAY_DIAGNOSTICS)
ALL_SENSOR_KEYS: Final[frozenset[str]] = SENSOR_KEYS | DIAGNOSTIC_KEYS
