"""Read measured output and fold it into the correction.

Samples live state rather than replaying the recorder. That keeps setup fast and
avoids a heavy history query on every restart, at the cost of taking a couple of
days of daylight to reach a useful correction on a fresh install -- which the
ramp-in already imposes anyway.

Nothing here decides whether a sample is trustworthy; that is
:mod:`custom_components.soular.core.mask`, shared with the nowcast so a curtailed
sample cannot mean one thing to the learner and another to the forecast.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging

from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util
import numpy as np

from custom_components.soular.core.correction import CorrectionState, build_features, observation_target, sample_weight
from custom_components.soular.core.mask import MaskInputs, usable

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
