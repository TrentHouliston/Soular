"""How far the weather forecast's sky tends to be from the one that arrives.

The panels' own efficiency comes measured with the shading map (see
:class:`custom_components.soular.core.shading.TransmittanceGrid`); nothing here
touches it. What is left between a forecast and the meter is the weather, and
the weather forecast is not unbiased: at this site it promises more sun than
arrives, by a margin that depends on how much sun it promised. A clear forecast
is usually right; a broken-cloud forecast usually overshoots.

So the forecast is corrected by how clear it said the sky would be. The
forecast's clear-sky index is binned, and each bin carries the ratio of energy
that arrived to energy forecast over the last :data:`WINDOW_DAYS` days, for the
arrays together. One correction for the site, because cloud covers the whole
roof at once, and per array it would only learn the arrays' differences again.

**A strong prior, not a ramp.** Each bin's ratio is pulled to 1 by
:data:`PRIOR_WH` of imaginary perfectly-forecast energy, so a bin moves only
once it has seen about two good days' worth of its own. Without it a cold start
learns a week of cloudy weather as the site's permanent climate, which is how
the previous learner came to cut every sunny day by a quarter.

**Measured, not modelled.** Curtailed and unread instants are left out of both
sides, as everywhere else.

A year's replay at the reference site, day-ahead, put the typical day's error
at 17.7% against 19.0% for the regression it replaces, and well ahead of it
from a cold start.
"""

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from custom_components.soular.core.types import FloatArray

# Edges between bins of the forecast clear-sky index.
BIN_EDGES: Final = (0.2, 0.4, 0.6, 0.75, 0.9)
BINS: Final = len(BIN_EDGES) + 1

WINDOW_DAYS: Final = 60

# Imaginary perfectly-forecast energy in every bin, in watt-hours.
PRIOR_WH: Final = 200_000.0

# A gap between readings longer than this adds nothing.
MAX_TICK_SECONDS: Final = 600

# The correction is a refinement. Outside this it is a broken sensor.
MIN_SCALE: Final = 0.5
MAX_SCALE: Final = 1.2

SECONDS_PER_HOUR: Final = 3600.0


def bin_of(clearness: FloatArray) -> np.ndarray:
    """Return each instant's bin from the forecast clear-sky index."""
    return np.digitize(np.nan_to_num(np.asarray(clearness, dtype=np.float64), nan=1.0), BIN_EDGES)


def ratios(predicted_wh: np.ndarray, actual_wh: np.ndarray) -> FloatArray:
    """Return each bin's correction from its totals, pulled towards 1 by the prior."""
    raw = (np.asarray(actual_wh) + PRIOR_WH) / (np.asarray(predicted_wh) + PRIOR_WH)
    return np.clip(raw, MIN_SCALE, MAX_SCALE)


@dataclass(slots=True)
class WeatherBias:
    """Per-day energy totals by forecast clearness, and the instant last counted."""

    # day ordinal -> (predicted Wh per bin, actual Wh per bin)
    days: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)
    last_tick: int | None = None

    def add(self, day: int, clearness: FloatArray, predicted_w: FloatArray, actual_w: FloatArray, hours: float) -> None:
        """Fold in instants of one day at once, each lasting ``hours``."""
        bins = bin_of(clearness)
        predicted, actual = self.days.setdefault(day, ([0.0] * BINS, [0.0] * BINS))
        add_p = np.bincount(bins, weights=np.asarray(predicted_w) * hours, minlength=BINS)
        add_a = np.bincount(bins, weights=np.asarray(actual_w) * hours, minlength=BINS)
        for index in range(BINS):
            predicted[index] += float(add_p[index])
            actual[index] += float(add_a[index])

    def tick(self, day: int, now: int, clearness: float, predicted_w: float, measured_w: float | None) -> None:
        """Fold in one live instant, or only note its time if it cannot count."""
        previous, self.last_tick = self.last_tick, now
        if previous is None or measured_w is None:
            return
        seconds = now - previous
        if not 0 < seconds <= MAX_TICK_SECONDS:
            return
        hours = seconds / SECONDS_PER_HOUR
        self.add(day, np.array([clearness]), np.array([predicted_w]), np.array([measured_w]), hours)
        oldest = day - WINDOW_DAYS
        for stale in [d for d in self.days if d < oldest]:
            del self.days[stale]

    def bin_ratios(self, today: int) -> FloatArray:
        """Return each bin's correction from the days before ``today``."""
        predicted = np.zeros(BINS)
        actual = np.zeros(BINS)
        for day, (p, a) in self.days.items():
            if today - WINDOW_DAYS <= day < today:
                predicted += p
                actual += a
        return ratios(predicted, actual)

    def scale(self, clearness: FloatArray, today: int) -> FloatArray:
        """Return the correction for each instant of a forecast made ``today``."""
        return self.bin_ratios(today)[bin_of(clearness)]

    @property
    def day_count(self) -> int:
        """How many days of record the correction holds."""
        return len(self.days)


def encode(bias: WeatherBias) -> dict[str, Any]:
    """Serialise the record."""
    return {
        "days": [
            [day, [round(v, 1) for v in predicted], [round(v, 1) for v in actual]]
            for day, (predicted, actual) in sorted(bias.days.items())
        ],
        "last_tick": bias.last_tick,
    }


def decode(payload: Any) -> WeatherBias:
    """Restore the record, starting afresh from anything unreadable."""
    if not isinstance(payload, dict):
        return WeatherBias()
    try:
        days = {
            int(day): ([float(v) for v in p], [float(v) for v in a])
            for day, p, a in payload.get("days", [])
            if len(p) == BINS and len(a) == BINS
        }
        last = payload.get("last_tick")
        return WeatherBias(days=days, last_tick=None if last is None else int(last))
    except (TypeError, ValueError):
        return WeatherBias()
