"""Sensor entities for the site and each array."""

from datetime import timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular.coordinator.coordinator import SoularCoordinator
from custom_components.soular.core.series import power_at
from custom_components.soular.core.types import FloatArray, TimeArray
from custom_components.soular.entities import (
    ArrayDiagnosticDescription,
    QuantileDescription,
    SiteDiagnosticDescription,
    SoularSensorEntityDescription,
    Window,
)
from custom_components.soular.entities.device import array_device, array_identifier, site_device, site_identifier

# A 48-hour mixed-resolution series is ~350 points and about 20 kB of JSON.
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

    return Window(
        now=instant(now) if times.size == 0 else max(instant(now), times[0]),
        today_start=instant(midnight),
        today_end=instant(midnight + timedelta(days=1)),
        tomorrow_end=instant(midnight + timedelta(days=2)),
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
    """A sensor derived from a power series."""

    entity_description: SoularSensorEntityDescription

    def __init__(
        self,
        coordinator: SoularCoordinator,
        description: SoularSensorEntityDescription,
        entry: ConfigEntry,
        site_name: str,
        subentry: ConfigSubentry | None = None,
    ) -> None:
        """Set up a site-level or array-level forecast sensor."""
        identifier = array_identifier(entry, subentry) if subentry else site_identifier(entry)
        super().__init__(coordinator, f"{identifier[1]}_{description.key}")
        self.entity_description = description
        self._array = str(subentry.title) if subentry else None
        self._attr_device_info = array_device(entry, subentry, site_name) if subentry else site_device(entry, site_name)
        if description.carries_forecast:
            self._attr_extra_state_attributes = {}

    @property
    def _series(self) -> tuple[TimeArray, FloatArray] | None:
        """Return the times and power this entity reports on."""
        result = self.coordinator.data
        if self._array is None:
            return result.times, result.ac_power_w
        try:
            return result.times, result.array(self._array).ac_power_w
        except KeyError:
            return None

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
            return None
        series = self._series
        if series is None:
            return None
        times, power = series

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


class SoularQuantileSensor(SoularSensorBase):
    """A confidence bound on the forecast, supplied by the NWP ensemble.

    When no ensemble is available this reports ``unknown`` rather than falling
    back to the median. A band that quietly collapses onto the point forecast
    reads as certainty, which is the opposite of what has happened, and a
    consumer hedging against it would hedge against nothing.
    """

    entity_description: QuantileDescription

    def __init__(
        self,
        coordinator: SoularCoordinator,
        description: QuantileDescription,
        entry: ConfigEntry,
        site_name: str,
    ) -> None:
        """Set up a site-level quantile sensor."""
        identifier = site_identifier(entry)
        super().__init__(coordinator, f"{identifier[1]}_{description.key}")
        self.entity_description = description
        self._attr_device_info = site_device(entry, site_name)

    @property
    def native_value(self) -> float | None:
        """Read the requested quantile, or nothing if there is no spread."""
        description = self.entity_description
        if description.day is not None:
            return self.coordinator.quantile_energy.get(description.day, {}).get(description.level)

        power = self.coordinator.quantile_power.get(description.level)
        if power is None:
            return None
        times = self.coordinator.data.times
        return power_at(times, power, local_window(times).now)


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
    def native_value(self) -> float | None:
        """Interpolate the underlying field at the current time."""
        result = self.coordinator.data
        try:
            entry = result.array(self._array)
        except KeyError:
            return None

        # Two of these report on the learner rather than on the forecast: what
        # the correction is currently doing, and how much evidence is behind it.
        state = self.coordinator.learner.states.get(self._array)
        if self.entity_description.field == "__correction__":
            return None if state is None else state.efficiency() * 100.0
        if self.entity_description.field == "__samples__":
            return 0.0 if state is None else float(state.samples)

        values: FloatArray = getattr(entry, self.entity_description.field)
        now = np.datetime64(int(dt_util.utcnow().timestamp()), "s")
        value = power_at(result.times, values, now)
        # Transmittance is a fraction in the model and a percentage on a dashboard.
        if self.entity_description.key == "shading_transmittance":
            return value * 100.0
        return value


class SoularSiteDiagnosticSensor(SoularSensorBase):
    """A site-level diagnostic read straight off the coordinator.

    These answer "why does the forecast look like that?" rather than "what will
    it produce". They start disabled; a user who never asks the question should
    not have to see the answer.
    """

    entity_description: SiteDiagnosticDescription

    def __init__(
        self,
        coordinator: SoularCoordinator,
        description: SiteDiagnosticDescription,
        entry: ConfigEntry,
        site_name: str,
    ) -> None:
        """Set up a site diagnostic sensor."""
        identifier = site_identifier(entry)
        super().__init__(coordinator, f"{identifier[1]}_{description.key}")
        self.entity_description = description
        self._attr_device_info = site_device(entry, site_name)

    @property
    def native_value(self) -> Any:
        """Read the coordinator attribute this sensor reports."""
        value = getattr(self.coordinator, self.entity_description.attribute, None)
        # A share is a fraction in the model and a percentage on a dashboard.
        if isinstance(value, float) and self.entity_description.key == "nowcast_share":
            return value * 100.0
        return value
