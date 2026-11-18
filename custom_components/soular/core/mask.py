"""When measured output is not telling the truth about the sky.

Two things make the panels read low on a day the sky was fine: the inverter
throttling them, and equipment that is off.

Curtailment is the dominant problem on a battery site. When the battery is full
and export is limited, the inverter throttles the array, and it does so at the
sunniest part of the day. Counted, that reads as the weather forecast promising
sun that never came, and the correction would mark every sunny day down.
"""

from collections.abc import Sequence
from typing import Final

# A battery this full is either curtailing or about to. Instants above it are
# left out of both sides of every comparison with measured output.
CURTAILMENT_SOC_PCT: Final = 97.0

# An array reading under this share of its capacity while the model gives it this
# much is not under a cloud: diffuse light alone keeps a working array well above it.
DEAD_FRACTION: Final = 0.02
DEAD_PREDICTED_FRACTION: Final = 0.15

# An array delivering under this share of what the others deliver, against their
# own forecasts, is off while they are on: a tripped string, not the sky.
ARRAY_OUT_OF_STEP: Final = 0.1
ARRAY_STEP_MIN_RATIO: Final = 0.3


def outage(measured_w: Sequence[float], predicted_w: Sequence[float], capacity_w: Sequence[float]) -> bool:
    """Return whether an instant's readings look like equipment off, not weather.

    Either an array reads next to nothing while the model expects it to be well
    lit, or one array is far out of step with the rest. A storm darkens every
    array together and keeps diffuse light on all of them; an outage does not.
    Counted, an outage would teach the weather correction that the forecast
    promises far more sun than arrives.
    """
    ratios = []
    for measured, predicted, capacity in zip(measured_w, predicted_w, capacity_w, strict=True):
        if capacity <= 0.0:
            continue
        if measured < DEAD_FRACTION * capacity and predicted > DEAD_PREDICTED_FRACTION * capacity:
            return True
        if predicted > DEAD_PREDICTED_FRACTION * capacity:
            ratios.append(measured / predicted)
    if len(ratios) < 2:  # noqa: PLR2004 - "out of step" needs something to be out of step with
        return False
    for index, ratio in enumerate(ratios):
        others = ratios[:index] + ratios[index + 1 :]
        typical = sorted(others)[len(others) // 2]
        if typical > ARRAY_STEP_MIN_RATIO and ratio < ARRAY_OUT_OF_STEP * typical:
            return True
    return False
