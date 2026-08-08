"""Decide which measured samples are telling the truth about irradiance.

Used by two things that must agree: the online correction, which learns from
measured output, and the nowcast, which reads the array as an irradiance sensor.
One function, two consumers -- because a curtailed sample would tell the nowcast
the sky had just gone dark, and tell the correction the array had just lost a
tenth of its efficiency, and both would be wrong in the same way.

Curtailment is the dominant problem on a battery site. When the battery is full
and export is limited, the inverter throttles the array, and it does so at the
sunniest part of the day -- exactly the samples that carry the most weight. On the
system this was built for, DC-coupled EV charging is a second path that can draw
from the array without the AC meter ever seeing it.

Rules that need no configuration come first, so masking works out of the box.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from custom_components.soular.core.types import FloatArray

# A battery this full is either curtailing or about to.
CURTAILMENT_SOC_PCT: Final = 97.0

# Within this of the inverter's limit, output is being held rather than produced.
AC_PLATEAU_FRACTION: Final = 0.97

# A flat top: measured output barely moving while the model says it should be.
# Three five-minute samples is fifteen minutes, long enough to be a plateau and
# short enough not to swallow a genuinely steady hour.
FLAT_TOP_WINDOW: Final = 3
FLAT_TOP_MAX_VARIATION: Final = 0.005
FLAT_TOP_MIN_MODELLED_SLOPE: Final = 0.03

# Producing almost nothing while the model expects real output is a fault -- a
# tripped string, a comms dropout -- not a lesson about efficiency.
OUTAGE_ACTUAL_FRACTION: Final = 0.02
OUTAGE_PREDICTED_FRACTION: Final = 0.15

# Whole-day gate: a day this far below expectation was not a sunny day the model
# got wrong, it was an outage or a snow-covered array.
DAY_MIN_RATIO: Final = 0.35

# Below these the ratio is dominated by everything except what is being measured.
MIN_ELEVATION_DEG: Final = 8.0
MIN_PREDICTED_FRACTION: Final = 0.03


@dataclass(frozen=True, slots=True)
class MaskInputs:
    """Everything the mask can use. Optional fields sharpen it; none are required."""

    predicted_w: FloatArray
    actual_w: FloatArray
    elevation_deg: FloatArray
    capacity_w: float
    soc_pct: FloatArray | None = None
    ac_limit_w: float | None = None
    day_index: FloatArray | None = None


def _flat_top(predicted: FloatArray, actual: FloatArray) -> np.ndarray:
    """Flag runs where output is pinned while the model says it should be moving.

    This is what curtailment looks like without a battery sensor to confirm it:
    a horizontal line drawn across the top of a curve that ought to be rising.
    """
    flagged = np.zeros(actual.size, dtype=bool)
    if actual.size < FLAT_TOP_WINDOW:
        return flagged

    for start in range(actual.size - FLAT_TOP_WINDOW + 1):
        stop = start + FLAT_TOP_WINDOW
        window = actual[start:stop]
        mean = float(np.mean(window))
        if mean <= 0.0:
            continue
        if float(np.std(window)) / mean > FLAT_TOP_MAX_VARIATION:
            continue
        modelled = predicted[start:stop]
        modelled_mean = float(np.mean(modelled))
        if modelled_mean <= 0.0:
            continue
        if float(np.max(modelled) - np.min(modelled)) / modelled_mean > FLAT_TOP_MIN_MODELLED_SLOPE:
            flagged[start:stop] = True
    return flagged


def usable(inputs: MaskInputs) -> np.ndarray:
    """Return which samples may be learned from.

    Every rule is a veto: a sample survives only if none of them fires.
    """
    predicted = np.asarray(inputs.predicted_w, dtype=np.float64)
    actual = np.asarray(inputs.actual_w, dtype=np.float64)
    capacity = max(inputs.capacity_w, 1.0)

    keep = np.isfinite(predicted) & np.isfinite(actual)
    keep &= inputs.elevation_deg >= MIN_ELEVATION_DEG
    keep &= predicted >= MIN_PREDICTED_FRACTION * capacity

    # A fault, not a lesson.
    keep &= ~((actual < OUTAGE_ACTUAL_FRACTION * capacity) & (predicted > OUTAGE_PREDICTED_FRACTION * capacity))

    keep &= ~_flat_top(predicted, actual)

    if inputs.soc_pct is not None:
        keep &= np.asarray(inputs.soc_pct, dtype=np.float64) < CURTAILMENT_SOC_PCT

    if inputs.ac_limit_w:
        keep &= actual < AC_PLATEAU_FRACTION * inputs.ac_limit_w

    if inputs.day_index is not None:
        keep &= _whole_days(predicted, actual, np.asarray(inputs.day_index))

    return keep


def _whole_days(predicted: FloatArray, actual: FloatArray, day_index: np.ndarray) -> np.ndarray:
    """Drop entire days that produced far less than the model expected."""
    keep = np.ones(actual.size, dtype=bool)
    for day in np.unique(day_index):
        window = day_index == day
        expected = float(np.sum(predicted[window]))
        if expected <= 0.0:
            continue
        if float(np.sum(actual[window])) / expected < DAY_MIN_RATIO:
            keep[window] = False
    return keep


def masked_fraction(keep: np.ndarray, eligible: np.ndarray) -> float:
    """Share of otherwise-eligible samples the mask removed.

    Surfaced as a diagnostic. Above roughly a third and the training signal is
    thin enough that the user should know why.
    """
    total = int(np.count_nonzero(eligible))
    if total == 0:
        return 0.0
    return 1.0 - int(np.count_nonzero(keep & eligible)) / total
