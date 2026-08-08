"""Derived quantities a dashboard asks for: energy over a window, peak time.

Kept in the core so the arithmetic is testable without a running Home Assistant,
and so "energy today" means the same thing to the integration, the tests and the
backtest.

Power is sampled at instants, so energy is a trapezoidal integral rather than a
sum of samples times a step. On a mixed-resolution grid the two differ, and the
difference lands exactly at sunrise and sunset where the curve is steepest.
"""

import numpy as np

from custom_components.soular.core.types import FloatArray, TimeArray

SECONDS_PER_HOUR = 3600.0
WATT_HOURS_PER_KILOWATT_HOUR = 1000.0


def energy_between(
    times: TimeArray,
    power_w: FloatArray,
    start: np.datetime64,
    end: np.datetime64,
) -> float:
    """Integrate power over a window, in kilowatt-hours.

    The window's edges are interpolated rather than snapped to the nearest
    sample, so "remaining today" does not jump by a sample's worth of energy each
    time it is recomputed.
    """
    if times.size == 0 or end <= start:
        return 0.0

    seconds = times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    lo = float(start.astype("datetime64[s]").astype(np.int64))
    hi = float(end.astype("datetime64[s]").astype(np.int64))
    lo = max(lo, seconds[0])
    hi = min(hi, seconds[-1])
    if hi <= lo:
        return 0.0

    inside = (seconds > lo) & (seconds < hi)
    edge_seconds = np.concatenate([[lo], seconds[inside], [hi]])
    edge_power = np.concatenate(
        [
            [float(np.interp(lo, seconds, power_w))],
            power_w[inside],
            [float(np.interp(hi, seconds, power_w))],
        ]
    )

    watt_seconds = float(np.trapezoid(edge_power, edge_seconds))
    return watt_seconds / SECONDS_PER_HOUR / WATT_HOURS_PER_KILOWATT_HOUR


def power_at(times: TimeArray, power_w: FloatArray, when: np.datetime64) -> float:
    """Interpolate power at an instant, clamped to the ends of the series."""
    if times.size == 0:
        return 0.0
    seconds = times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    return float(np.interp(float(when.astype("datetime64[s]").astype(np.int64)), seconds, power_w))


def peak_time(
    times: TimeArray,
    power_w: FloatArray,
    start: np.datetime64,
    end: np.datetime64,
) -> np.datetime64 | None:
    """Return when power peaks inside a window, or None if nothing is produced."""
    if times.size == 0:
        return None
    window = (times >= start) & (times < end)
    if not window.any():
        return None
    values = power_w[window]
    if float(np.max(values)) <= 0.0:
        return None
    return times[window][int(np.argmax(values))]


def hourly_energy(times: TimeArray, power_w: FloatArray) -> dict[np.datetime64, float]:
    """Integrate into whole clock hours, keyed by the hour's start.

    This is the shape Home Assistant's energy dashboard consumes.
    """
    if times.size == 0:
        return {}

    first = times[0].astype("datetime64[h]").astype("datetime64[s]")
    last = times[-1].astype("datetime64[h]").astype("datetime64[s]")
    hours = np.arange(first, last + np.timedelta64(3600, "s"), np.timedelta64(3600, "s")).astype("datetime64[s]")

    result: dict[np.datetime64, float] = {}
    for hour in hours:
        energy = energy_between(times, power_w, hour, hour + np.timedelta64(3600, "s"))
        if energy > 0.0:
            result[hour] = energy
    return result
