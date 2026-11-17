"""A P10/P90 band around the daily totals, calibrated on the site's own record.

The usual source of a band is a weather ensemble: the spread between fifty runs
of the model. Open-Meteo serves only the current ensemble run and has no archive,
so nothing could ever show such a band was calibrated -- and a band nobody has
checked is worse than none to something that hedges against it.

This one is built from what can be checked: how far this site's actual daily
production has landed from Soular's own forecast of it. For each lead -- today
as forecast at midnight, tomorrow as forecast a day earlier, and so on to a week
out -- it keeps the ratio of actual to forecast over the last
:data:`WINDOW_DAYS` days and reports that ratio's 10th and 90th percentiles. A
year's walk-forward replay at the reference site, one to seven days ahead and
using only outcomes known before each forecast, put 10% of days below P10 and
11% above P90.

Four choices are load-bearing.

**Rolling, not fixed.** The spread is seasonal: a band fitted on summer and
applied to winter left 10% of days outside it instead of 20%, and the reverse
left 43% outside. Sixty days follows the season without starving the quantiles.

**Each lead borrows from its neighbours.** Sixty days of one lead is forty-odd
ratios once curtailed days are dropped, and tenth percentiles of samples that
small come out too narrow: alone, each lead let 13% through either side. Lead
barely changes the spread over a week, so pooling a lead with the one either
side of it triples the sample at little cost in sharpness.

**Curtailed samples are left out on both sides.** A full battery throttles the
array on the sunniest days, so the inverter's daily total says less than the sky
delivered. Counted as forecast error, that made sunny days look a third
over-forecast and stretched the lower bound accordingly. Only instants when the
battery was not full are summed, for the actual and for each forecast alike.

**The bounds are whole-day ratios, not pointwise percentiles.** P10 of a day's
energy is not the integral of each interval's P10, which would assume every hour
goes badly at once. A whole day measured against a whole day avoids the question.
For the same reason a P10 power series is the forecast scaled by its day's ratio:
one coherent poor day, not the envelope of every bad hour.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Final

import numpy as np

from custom_components.soular.core.series import energy_between
from custom_components.soular.core.types import FloatArray, TimeArray

# Today, as forecast at its own midnight, then one through seven days ahead.
BAND_LEADS: Final = 8

QUANTILES: Final = (0.1, 0.9)

# Long enough for stable tenth percentiles, short enough to follow the season.
WINDOW_DAYS: Final = 60

# A lead is pooled with this many leads either side of it.
NEIGHBOURS: Final = 1

# A lead with fewer ratios than this reports the starting band instead.
MIN_SAMPLES: Final = 15

# A day mostly spent curtailed or without readings says little about the sky.
MIN_USABLE_SHARE: Final = 0.6

# Below this a ratio is dominated by rounding, not by the weather.
MIN_DAY_WH: Final = 500.0

# A gap longer than this between readings is a gap, not a long sample: it adds
# nothing, so a day with an outage fails the usable-share test honestly.
MAX_TICK_SECONDS: Final = 600

SECONDS_PER_HOUR: Final = 3600

# Starting band, as actual/forecast ratios at (P10, P90), for each lead: a year
# of replay at the reference site, pooled with neighbours as the live band is,
# curtailed samples excluded. A site replaces it with its own record once a lead
# has MIN_SAMPLES ratios. The replay has no lead-0 day, so today starts from
# tomorrow's ratios alone.
PRIOR: Final[tuple[tuple[float, float], ...]] = (
    (0.62, 1.36),
    (0.64, 1.33),
    (0.63, 1.38),
    (0.63, 1.45),
    (0.60, 1.52),
    (0.59, 1.54),
    (0.60, 1.64),
    (0.61, 1.52),
)


@dataclass(slots=True)
class Snapshot:
    """The forecast as it stood at the start of a local day, hour by hour."""

    issued_day: int
    start: int
    hourly_w: list[float]

    def power_at(self, when: int) -> float | None:
        """Return the forecast mean power for the hour holding ``when``."""
        index = (when - self.start) // SECONDS_PER_HOUR
        if index < 0 or index >= len(self.hourly_w):
            return None
        return self.hourly_w[index]

    def energy_wh(self, start: int, end: int) -> float:
        """Return the forecast energy between two instants on hour boundaries."""
        first = max(0, (start - self.start) // SECONDS_PER_HOUR)
        last = min(len(self.hourly_w), (end - self.start) // SECONDS_PER_HOUR)
        return float(sum(self.hourly_w[first:last]))


@dataclass(slots=True)
class Ledger:
    """What one local day has accumulated so far."""

    day: int
    start: int
    end: int
    actual_wh: float = 0.0
    predicted_wh: dict[int, float] = field(default_factory=dict)
    last_tick: int | None = None


@dataclass(frozen=True, slots=True)
class Residual:
    """How one day landed against its forecast at one lead."""

    day: int
    lead: int
    log_ratio: float


@dataclass(frozen=True, slots=True)
class Bounds:
    """The band for one lead, as ratios to the forecast."""

    low: float
    high: float
    samples: int

    @property
    def calibrated(self) -> bool:
        """Whether the bounds come from this site's record rather than the prior."""
        return self.samples >= MIN_SAMPLES


def hourly_snapshot(issued_day: int, start: int, times: TimeArray, power_w: FloatArray, hours: int) -> Snapshot:
    """Reduce a forecast series to hourly mean power from ``start``.

    Hourly is enough to tell which hours were usable, and keeps a week of
    snapshots small enough to persist.
    """
    hourly: list[float] = []
    for hour in range(hours):
        begin = np.datetime64(start + hour * SECONDS_PER_HOUR, "s")
        end = begin + np.timedelta64(SECONDS_PER_HOUR, "s")
        hourly.append(energy_between(times, power_w, begin, end) * 1000.0)
    return Snapshot(issued_day=issued_day, start=start, hourly_w=hourly)


def bounds_for(residuals: list[Residual], lead: int, today: int) -> Bounds:
    """Return the band for a lead from the days before ``today``.

    Only outcomes known before today count, so the band is what could have been
    published at the time -- the property the walk-forward replay measured.
    """
    ratios = [
        r.log_ratio for r in residuals if abs(r.lead - lead) <= NEIGHBOURS and today - WINDOW_DAYS <= r.day < today
    ]
    if len(ratios) < MIN_SAMPLES:
        low, high = PRIOR[min(lead, len(PRIOR) - 1)]
        return Bounds(low=low, high=high, samples=len(ratios))
    low, high = np.quantile(np.array(ratios), QUANTILES)
    return Bounds(low=math.exp(float(low)), high=math.exp(float(high)), samples=len(ratios))


@dataclass(slots=True)
class BandTracker:
    """Snapshots forecasts, compares them with what was produced, and reports a band."""

    snapshots: dict[int, Snapshot] = field(default_factory=dict)
    ledger: Ledger | None = None
    residuals: list[Residual] = field(default_factory=list)

    def start_day(self, day: int, start: int, end: int, snapshot: Snapshot | None) -> None:
        """Begin a local day, closing the previous one and keeping its snapshot.

        Called on every recompute; does nothing once the day has begun. The
        snapshot is only taken if the day has none, so a restart mid-afternoon
        does not replace the midnight forecast with a better-informed one.
        """
        if self.ledger is not None and self.ledger.day == day:
            return
        if self.ledger is not None and self.ledger.day < day:
            self._close(self.ledger)
        if snapshot is not None and day not in self.snapshots:
            self.snapshots[day] = snapshot
        self.ledger = Ledger(day=day, start=start, end=end)
        for issued in [issued for issued in self.snapshots if issued <= day - BAND_LEADS]:
            del self.snapshots[issued]
        oldest = day - WINDOW_DAYS - BAND_LEADS
        self.residuals = [r for r in self.residuals if r.day >= oldest]

    def tick(self, now: int, measured_w: float | None) -> None:
        """Fold in one reading of the site's output, or ``None`` if unusable."""
        ledger = self.ledger
        if ledger is None or not ledger.start <= now < ledger.end:
            return
        previous, ledger.last_tick = ledger.last_tick, now
        if previous is None or measured_w is None:
            return
        seconds = now - previous
        if not 0 < seconds <= MAX_TICK_SECONDS:
            return
        hours = seconds / SECONDS_PER_HOUR
        ledger.actual_wh += measured_w * hours
        for lead in range(BAND_LEADS):
            snapshot = self.snapshots.get(ledger.day - lead)
            if snapshot is None:
                continue
            power = snapshot.power_at(now)
            if power is not None:
                ledger.predicted_wh[lead] = ledger.predicted_wh.get(lead, 0.0) + power * hours

    def _close(self, ledger: Ledger) -> None:
        """Turn a finished day into one residual per lead that saw enough of it."""
        if ledger.actual_wh < MIN_DAY_WH:
            return
        for lead, predicted in ledger.predicted_wh.items():
            snapshot = self.snapshots.get(ledger.day - lead)
            if snapshot is None or predicted < MIN_DAY_WH:
                continue
            total = snapshot.energy_wh(ledger.start, ledger.end)
            if total <= 0.0 or predicted / total < MIN_USABLE_SHARE:
                continue
            self.residuals.append(Residual(day=ledger.day, lead=lead, log_ratio=math.log(ledger.actual_wh / predicted)))

    def bounds(self, today: int) -> tuple[Bounds, ...]:
        """Return the band for every lead, as of ``today``."""
        return tuple(bounds_for(self.residuals, lead, today) for lead in range(BAND_LEADS))


def encode(tracker: BandTracker) -> dict[str, Any]:
    """Serialise the tracker."""
    ledger = tracker.ledger
    return {
        "snapshots": [
            {"issued_day": s.issued_day, "start": s.start, "hourly_w": [round(v, 1) for v in s.hourly_w]}
            for s in tracker.snapshots.values()
        ],
        "ledger": None
        if ledger is None
        else {
            "day": ledger.day,
            "start": ledger.start,
            "end": ledger.end,
            "actual_wh": ledger.actual_wh,
            "predicted_wh": {str(lead): value for lead, value in ledger.predicted_wh.items()},
            "last_tick": ledger.last_tick,
        },
        "residuals": [[r.day, r.lead, round(r.log_ratio, 5)] for r in tracker.residuals],
    }


def decode(payload: Any) -> BandTracker:
    """Restore the tracker, starting afresh from anything unreadable."""
    if not isinstance(payload, dict):
        return BandTracker()
    try:
        snapshots = {
            int(s["issued_day"]): Snapshot(
                issued_day=int(s["issued_day"]), start=int(s["start"]), hourly_w=[float(v) for v in s["hourly_w"]]
            )
            for s in payload.get("snapshots", [])
        }
        raw = payload.get("ledger")
        ledger = (
            None
            if raw is None
            else Ledger(
                day=int(raw["day"]),
                start=int(raw["start"]),
                end=int(raw["end"]),
                actual_wh=float(raw["actual_wh"]),
                predicted_wh={int(lead): float(value) for lead, value in raw["predicted_wh"].items()},
                last_tick=None if raw.get("last_tick") is None else int(raw["last_tick"]),
            )
        )
        residuals = [
            Residual(day=int(day), lead=int(lead), log_ratio=float(value))
            for day, lead, value in payload.get("residuals", [])
        ]
    except (KeyError, TypeError, ValueError, AttributeError):
        return BandTracker()
    return BandTracker(snapshots=snapshots, ledger=ledger, residuals=residuals)


def scale_by_day(times: TimeArray, midnights: tuple[np.datetime64, ...], ratios: tuple[float, ...]) -> FloatArray:
    """Return each instant's ratio, taken from the local day it falls in.

    ``midnights`` starts at today's; instants before it take today's ratio and
    instants past the last day take the last one.
    """
    days = np.searchsorted(np.array(midnights, dtype="datetime64[s]"), times.astype("datetime64[s]"), side="right") - 1
    return np.array(ratios, dtype=np.float64)[np.clip(days, 0, len(ratios) - 1)]
