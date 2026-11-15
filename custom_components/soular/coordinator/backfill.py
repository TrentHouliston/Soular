"""Catch a fresh array's correction up from the recorder.

Live sampling alone takes days of daylight to calibrate a new install, but Home
Assistant has usually been recording the array's output for far longer than
Soular has been installed. This replays that history through the same learner,
once, for arrays that have learned nothing yet -- not on every restart, which
is what would make the history query a cost.

The recorder's five-minute statistics are used rather than raw state history:
one mean per sensor per five minutes is exactly the learner's own cadence, a
week of it is a couple of thousand rows instead of hundreds of thousands, and
the recorder converts kW to W on the way out.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.core import HomeAssistant
import numpy as np

from custom_components.soular.coordinator.actuals import Learner, samples_from
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import SystemSpec, forecast
from custom_components.soular.core.types import TimeGrid
from custom_components.soular.sources.open_meteo import HourlyWeather

_LOGGER = logging.getLogger(__name__)

# The recorder keeps ten days by default. A week leaves margin at both ends and
# is already past the correction's ramp-in.
BACKFILL_DAYS = 7
PERIOD_SECONDS = 300

type Series = dict[int, float]


def arrays_to_backfill(learner: Learner, power_sensors: Mapping[str, str]) -> dict[str, str]:
    """Return the arrays with a power sensor and nothing learned yet."""
    return {
        name: entity
        for name, entity in power_sensors.items()
        if name not in learner.states or learner.states[name].samples == 0
    }


async def async_read_history(
    hass: HomeAssistant,
    entity_ids: set[str],
    start: datetime,
    end: datetime,
) -> dict[str, Series]:
    """Read five-minute means from the recorder, keyed by period start in epoch seconds.

    Returns nothing, rather than raising, when the recorder is not running: the
    backfill is a head start, never a requirement.
    """
    if "recorder" not in hass.config.components:
        return {}

    from homeassistant.components.recorder.statistics import statistics_during_period  # noqa: PLC0415
    from homeassistant.helpers.recorder import get_instance  # noqa: PLC0415

    rows: dict[str, list[Any]] = await get_instance(hass).async_add_executor_job(
        statistics_during_period, hass, start, end, entity_ids, "5minute", {"power": "W"}, {"mean"}
    )
    out: dict[str, Series] = {}
    for entity_id, points in rows.items():
        series: Series = {}
        for point in points:
            mean = point.get("mean")
            if mean is None:
                continue
            stamp = point["start"]
            seconds = int(stamp.timestamp()) if isinstance(stamp, datetime) else int(stamp)
            series[seconds] = float(mean)
        out[entity_id] = series
    return out


def replay(
    learner: Learner,
    system: SystemSpec,
    weather: HourlyWeather,
    arrays: Mapping[str, str],
    history: Mapping[str, Series],
    soc: Series | None,
) -> int:
    """Fold recorded output into the learner, oldest first. Blocking.

    Each five-minute mean is compared against the model at the middle of its
    period, run on the weather Open-Meteo has for that time. Returns how many
    samples were offered to the learner (the mask still decides which count).
    """
    stamps = sorted({stamp for entity in arrays.values() for stamp in history.get(entity, {})})
    if not stamps:
        return 0

    first, last = stamps[0], stamps[-1]
    starts = np.arange(first, last + PERIOD_SECONDS, PERIOD_SECONDS, dtype=np.int64)
    times = (starts + PERIOD_SECONDS // 2).astype("datetime64[s]")
    covered = (times >= weather.times[0]) & (times <= weather.times[-1])
    starts, times = starts[covered], times[covered]
    if times.size == 0:
        return 0

    grid = TimeGrid(times=times, step_seconds=np.full(times.size, float(PERIOD_SECONDS)))
    series = weather_from_hourly(
        hourly_times=weather.times,
        grid=grid,
        site=system.site,
        ghi=weather.ghi,
        temperature=weather.temperature,
        wind_speed_10m=weather.wind_speed_10m,
        direct_horizontal=weather.direct_horizontal,
        diffuse=weather.diffuse,
    )
    result = forecast(system, grid, series)
    samples = samples_from(result, system, np.arange(times.size))

    offered = 0
    for k, start in enumerate(starts.tolist()):
        charge = None if soc is None else soc.get(start)
        for name, entity in arrays.items():
            measured = history.get(entity, {}).get(start)
            if measured is None or name not in samples:
                continue
            learner.observe(name, samples[name][k], measured, charge)
            offered += 1
    return offered


def backfill_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the period to replay: the last week, up to the current five minutes."""
    end = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
    return end - timedelta(days=BACKFILL_DAYS), end
