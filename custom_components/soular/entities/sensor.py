"""Sensor entities, one set per array."""

from datetime import timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular.coordinator.coordinator import SoularCoordinator
from custom_components.soular.core.band import scale_by_day
from custom_components.soular.core.series import power_at
from custom_components.soular.core.types import FloatArray, TimeArray
from custom_components.soular.entities import (
    FURTHER_DAYS,
    ArrayDiagnosticDescription,
    SoularSensorEntityDescription,
    Window,
)
from custom_components.soular.entities.device import array_device, array_identifier

# An eight-day mixed-resolution series is ~550 points and about 30 kB of JSON.
# Recording that every five minutes would add several gigabytes a year to the
# recorder database for no benefit: it is a forecast, not a measurement.
FORECAST_UNRECORDED_ATTRIBUTES = frozenset({"forecast", "watts"})


def local_window(times: TimeArray) -> Window:
    """Build the local-day boundaries the sensors are defined against.

    Local, not UTC: "today" means the day the user is living in, and getting this
    wrong shifts every daily total by the site's offset.
    """
    now = dt_util.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def instant(value: Any) -> np.datetime64:
        return np.datetime64(int(value.timestamp()), "s")

    # Wall-clock arithmetic on an aware datetime, so each is a true local
    # midnight even across a daylight-saving change.
    return Window(
        now=instant(now) if times.size == 0 else max(instant(now), times[0]),
        midnights=tuple(instant(midnight + timedelta(days=offset)) for offset in range(max(FURTHER_DAYS) + 2)),
    )


class SoularSensorBase(CoordinatorEntity[SoularCoordinator], SensorEntity):
    """Shared plumbing for every Soular sensor."""

    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(self, coordinator: SoularCoordinator, unique_id: str) -> None:
        """Set up the entity's identity."""
        super().__init__(coordinator)
        self._attr_unique_id = unique_id


class SoularForecastSensor(SoularSensorBase):
    """A sensor derived from one array's power series.

    There is deliberately no site-wide copy. The energy dashboard and the
    recorder both want one entity per array, and haeo sums several sensors
    itself, so a total would only duplicate what the arrays already say.
    """

    _unrecorded_attributes = FORECAST_UNRECORDED_ATTRIBUTES
    entity_description: SoularSensorEntityDescription

    def __init__(
        self,
        coordinator: SoularCoordinator,
        description: SoularSensorEntityDescription,
        entry: ConfigEntry,
        site_name: str,
        subentry: ConfigSubentry,
    ) -> None:
        """Set up an array's forecast sensor."""
        identifier = array_identifier(entry, subentry)
        super().__init__(coordinator, f"{identifier[1]}_{description.key}")
        self.entity_description = description
        self._array = str(subentry.title)
        self._attr_device_info = array_device(entry, subentry, site_name)
        if description.carries_forecast:
            self._attr_extra_state_attributes = {}

    @property
    def _series(self) -> tuple[TimeArray, FloatArray] | None:
        """Return the times and power this entity reports on."""
        result = self.coordinator.data
        try:
            power = result.array(self._array).ac_power_w
        except KeyError:
            return None
        bound = self.entity_description.bound
        if bound is None:
            return result.times, power
        ratios = tuple(b.high if bound else b.low for b in self.coordinator.band_bounds)
        return result.times, power * scale_by_day(result.times, local_window(result.times).midnights, ratios)

    @property
    def available(self) -> bool:
        """Report availability, which requires an actual series to read."""
        return super().available and self._series is not None

    @property
    def native_value(self) -> float | Any:
        """Compute this sensor's value from the current forecast."""
        series = self._series
        if series is None:
            return None
        times, power = series
        return self.entity_description.value(times, power, local_window(times))

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Attach the forecast series, in the shape haeo's extractor accepts.

        Emitted strictly: haeo's detector rejects the whole entity if a single
        point is malformed, and then silently falls back to reading the plain
        state, which drops the forecast without any error anywhere.

        The series starts at or before now, because when that extractor matches,
        haeo discards the sensor's own state and takes the first interval from
        the series. It spans a whole number of days because haeo pads a series to
        whole days by wrapping its head onto its tail -- so a series ending
        mid-day gets this morning spliced onto its end.
        """
        if not self.entity_description.carries_forecast:
            return self._band_attributes()
        series = self._series
        if series is None:
            return None
        times, power = series
        # From now: haeo reads interval zero as the present, and wants the
        # horizon ahead, not the elapsed part of today.
        start = self.coordinator.data.now_index
        times, power = times[start:], power[start:]

        points = [
            {
                "time": dt_util.utc_from_timestamp(int(stamp.astype("datetime64[s]").astype(np.int64))).isoformat(),
                "value": round(float(value), 1),
            }
            for stamp, value in zip(times, power, strict=True)
        ]
        return {
            "forecast": points,
            "unit_of_measurement": "W",
            "device_class": "power",
            "interpolation_mode": "linear",
        }

    def _band_attributes(self) -> dict[str, Any] | None:
        """Attach a daily total's P10 and P90, as Solcast names them.

        The ratios are measured on the sensed arrays together and applied to
        each array alike, so the arrays' bounds add up to the site's: one poor
        day for the whole roof, which is what weather does.
        """
        lead = self.entity_description.lead
        if lead is None:
            return None
        value = self.native_value
        if not isinstance(value, float):
            return None
        bounds = self.coordinator.band_bounds[lead]
        return {
            "estimate10": round(value * bounds.low, 2),
            "estimate90": round(value * bounds.high, 2),
            "band_calibrated": bounds.calibrated,
        }


class SoularArrayDiagnosticSensor(SoularSensorBase):
    """A per-array diagnostic sampled at the current instant."""

    entity_description: ArrayDiagnosticDescription

    def __init__(
        self,
        coordinator: SoularCoordinator,
        description: ArrayDiagnosticDescription,
        entry: ConfigEntry,
        site_name: str,
        subentry: ConfigSubentry,
    ) -> None:
        """Set up an array diagnostic sensor."""
        identifier = array_identifier(entry, subentry)
        super().__init__(coordinator, f"{identifier[1]}_{description.key}")
        self.entity_description = description
        self._array = str(subentry.title)
        self._attr_device_info = array_device(entry, subentry, site_name)

    @property
    def native_value(self) -> Any:
        """Interpolate the underlying field at the current time."""
        result = self.coordinator.data
        try:
            entry = result.array(self._array)
        except KeyError:
            return None

        # Two of these report on the learner rather than on the forecast: what
        # the correction is currently doing, and how much evidence is behind it.
        field = self.entity_description.field
        if field == "__efficiency__":
            spec = next((array for array in self.coordinator.system.arrays if array.name == self._array), None)
            grid = self.coordinator.system.shading.get(self._array)
            if spec is None or grid is None or grid.efficiency is None:
                return None
            return grid.efficiency_for(spec.dc_loss_fraction) * 100.0
        if field == "__weather_correction__":
            return self.coordinator.weather_correction_now * 100.0
        if field == "__weather_fetched_at__":
            return self.coordinator.weather_fetched_at

        values: FloatArray = getattr(entry, self.entity_description.field)
        now = np.datetime64(int(dt_util.utcnow().timestamp()), "s")
        value = power_at(result.times, values, now)
        # Transmittance is a fraction in the model and a percentage on a dashboard.
        if self.entity_description.key == "shading_transmittance":
            return value * 100.0
        return value
