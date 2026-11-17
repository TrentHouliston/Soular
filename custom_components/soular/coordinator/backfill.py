"""Catch a fresh weather correction up from the recorder.

Live sampling alone takes weeks to say anything about the weather forecast's
bias, but Home Assistant has usually been recording the arrays' output for far
longer than Soular has been installed. This replays that history into the same
record, once, while the record is empty -- not on every restart, which is what
would make the history query a cost.

The recorder's five-minute statistics are used rather than raw state history:
one mean per sensor per five minutes is exactly the learner's own cadence, a
week of it is a couple of thousand rows instead of hundreds of thousands, and
the recorder converts kW to W on the way out.
"""

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.core import HomeAssistant
import numpy as np

from custom_components.soular.coordinator.actuals import forecast_clearness
from custom_components.soular.core.mask import CURTAILMENT_SOC_PCT
from custom_components.soular.core.nwp import weather_from_hourly
from custom_components.soular.core.pipeline import SystemSpec, forecast
from custom_components.soular.core.types import TimeGrid
from custom_components.soular.core.weather_bias import WeatherBias
from custom_components.soular.sources.open_meteo import HourlyWeather

_LOGGER = logging.getLogger(__name__)

# The recorder keeps ten days by default. A week leaves margin at both ends and
# is already past the correction's ramp-in.
BACKFILL_DAYS = 7
PERIOD_SECONDS = 300

type Series = dict[int, float]


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
    bias: WeatherBias,
    system: SystemSpec,
    weather: HourlyWeather,
    arrays: Mapping[str, str],
    history: Mapping[str, Series],
    soc: Series | None,
    local_day: Callable[[int], int],
) -> int:
    """Fold recorded output into the weather correction, by local day. Blocking.

    Each five-minute mean is compared against the model at the middle of its
    period, run on the weather Open-Meteo has for that time, for the arrays
    together. An instant counts only with every array read and the battery not
    full. Returns how many instants counted.
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
    predicted = np.sum([entry.dc_power_w for entry in result.arrays if entry.name in arrays], axis=0)
    clearness = forecast_clearness(times, system, series.ghi)

    measured = np.full(times.size, np.nan)
    readings = np.array([[history.get(entity, {}).get(int(s), np.nan) for s in starts] for entity in arrays.values()])
    complete = np.isfinite(readings).all(axis=0)
    measured[complete] = readings[:, complete].sum(axis=0)
    if soc is not None:
        charge = np.array([soc.get(int(s), np.nan) for s in starts])
        complete &= ~(charge >= CURTAILMENT_SOC_PCT)

    days = np.array([local_day(int(s)) for s in starts])
    hours = PERIOD_SECONDS / 3600.0
    for day in np.unique(days[complete]):
        keep = complete & (days == day)
        bias.add(int(day), clearness[keep], predicted[keep], measured[keep], hours)
    return int(complete.sum())


def backfill_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the period to replay: the last week, up to the current five minutes."""
    end = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
    return end - timedelta(days=BACKFILL_DAYS), end
