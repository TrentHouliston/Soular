"""Read measured output and fold it into the correction.

Samples live state as it goes. An array with nothing learned yet is first caught
up from the recorder's history (:mod:`.backfill`), once, so a fresh install is
calibrated from the start rather than after days of daylight.

Nothing here decides whether a sample is trustworthy; that is
:mod:`custom_components.soular.core.mask`, which is pure so the backtest applies
exactly the same rules.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging

from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular.core.clearsky import clear_sky, clear_sky_index
from custom_components.soular.core.correction import CorrectionState, build_features, observation_target, sample_weight
from custom_components.soular.core.geometry import solar_geometry
from custom_components.soular.core.mask import MaskInputs, usable
from custom_components.soular.core.pipeline import ForecastResult, SystemSpec

_LOGGER = logging.getLogger(__name__)

# Below this fraction of capacity the ratio measures sunrise geometry rather
# than efficiency.
MIN_LEARNING_FRACTION = 0.03


@dataclass(slots=True)
class ArraySample:
    """One instant's worth of everything the learner needs about one array."""

    predicted_w: float
    capacity_w: float
    transmittance: float
    elevation_deg: float
    clearness: float
    day_of_year: float
    temperature_c: float


def samples_from(result: ForecastResult, system: SystemSpec, index: np.ndarray) -> dict[str, list[ArraySample]]:
    """Build the learner's view of each array at the given points of a forecast.

    Shared by live sampling and the recorder backfill, so the two cannot drift
    into training the correction on different features.
    """
    times = result.times[index]
    geometry = solar_geometry(times, system.site)
    clearsky = clear_sky(times, system.site, geometry).ghi
    days = (times.astype("datetime64[D]") - times.astype("datetime64[Y]").astype("datetime64[D]")).astype(np.int64) + 1
    capacity = {array.name: array.dc_capacity_w for array in system.arrays}

    out: dict[str, list[ArraySample]] = {}
    for entry in result.arrays:
        # Clear-sky index of the forecast at each instant, which is the same
        # quantity the correction is applied against later.
        clearness = np.nan_to_num(clear_sky_index(entry.poa_global[index], clearsky), nan=1.0)
        out[entry.name] = [
            ArraySample(
                predicted_w=float(entry.dc_power_w[i]),
                capacity_w=capacity.get(entry.name, 0.0),
                transmittance=float(entry.transmittance[i]),
                elevation_deg=float(geometry.apparent_elevation[k]),
                clearness=float(clearness[k]),
                day_of_year=float(days[k]),
                temperature_c=float(entry.cell_temperature[i]),
            )
            for k, i in enumerate(index)
        ]
    return out


@dataclass(slots=True)
class Learner:
    """Holds one estimator per array and folds in measured output."""

    states: dict[str, CorrectionState] = field(default_factory=dict)
    last_sample_at: datetime | None = None
    masked: int = 0
    accepted: int = 0

    def state_for(self, name: str) -> CorrectionState:
        """Return the estimator for an array, creating it on first use."""
        return self.states.setdefault(name, CorrectionState())

    @property
    def masked_fraction(self) -> float:
        """Share of otherwise-usable samples the mask rejected."""
        total = self.masked + self.accepted
        return self.masked / total if total else 0.0

    def observe(self, name: str, sample: ArraySample, measured_w: float, soc_pct: float | None) -> None:
        """Fold one measured sample into the estimator for an array."""
        keep = usable(
            MaskInputs(
                predicted_w=np.array([sample.predicted_w]),
                actual_w=np.array([measured_w]),
                elevation_deg=np.array([sample.elevation_deg]),
                capacity_w=sample.capacity_w,
                soc_pct=None if soc_pct is None else np.array([soc_pct]),
            )
        )
        if not bool(keep[0]):
            self.masked += 1
            return

        target = observation_target(measured_w, sample.predicted_w, floor_w=MIN_LEARNING_FRACTION * sample.capacity_w)
        if target is None:
            self.masked += 1
            return

        features = build_features(
            elevation_deg=np.array([sample.elevation_deg]),
            clearness=np.array([sample.clearness]),
            day_of_year=np.array([sample.day_of_year]),
            transmittance=np.array([sample.transmittance]),
            temperature_c=np.array([sample.temperature_c]),
        )
        self.state_for(name).update(features[0], target, sample_weight(sample.predicted_w, sample.capacity_w))
        self.accepted += 1
        self.last_sample_at = dt_util.utcnow()


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
